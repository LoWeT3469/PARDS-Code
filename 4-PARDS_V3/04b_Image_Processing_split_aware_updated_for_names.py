#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04b_Image_Processing_split_aware_updated_for_names.py

Leakage-safe image pipeline for teacher-student PARDS design, with
image-eligible-subset stratification enforced on the FINAL filtered V1V2 rows
used by the image pipeline.

What this script does:
1) Read V1V2
2) Apply image-eligibility filters to V1V2
3) Rebuild a grouped, class-balanced split on the FINAL filtered V1V2 subset
4) Fine-tune image encoder on filtered V1V2 TRAIN ONLY
5) Extract CXR embeddings for filtered V1V2 only
6) Merge embeddings back into filtered V1V2 vent table
7) Save split-aware V1V2 multimodal dataset

Important design choice:
- V3 is NOT required to have CXR.
- Teacher uses vent + CXR on V1V2 train.
- Student uses vent only.
- External validation on V3 is vent-only in downstream 05a/05b scripts.

Run this script twice if you want separate encoders:
  - FINETUNE_MODE = "regression"
  - FINETUNE_MODE = "classification"
"""

import os
import json
import random
import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except Exception:
    HAS_STRATIFIED_GROUP_KFOLD = False

try:
    import timm
except Exception:
    timm = None

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None


BASE_DATA_DIR = "/nfs/turbo/med-kayvan-lab/Projects/PARDS/02-Data/PARDS_Risk_V3"

V1V2_PATH = os.path.join(BASE_DATA_DIR, "vent_window_with_best_cxr_V1V2.csv")

ENCODER_NAME = "vit_base_patch16_224"
USE_LORA = True
DO_FINETUNE = True
# FINETUNE_MODE = "regression"   # change to "classification" for second run
FINETUNE_MODE = "classification"   
CLASS_THRESHOLD = 7.5

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4
SEED = 3469
TEST_SIZE = 0.20
MAX_SPLIT_TRIALS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["qkv", "proj", "fc1", "fc2"]

FT_EPOCHS = 8
FT_PATIENCE = 2
FT_LR = 1e-4
FT_WEIGHT_DECAY = 1e-4
FT_INNER_VAL_RATIO = 0.15

TARGET_CANDIDATES_V1V2 = ["OSI_V2_12th_avg", "OSI_12th_avg", "OSI_12th"]


def set_seed(seed: int = 3469) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def mrn_9(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    s = s.str.replace(r"^tensor\((.*)\)$", r"\1", regex=True)
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"\D", "", regex=True)
    s = s.replace("", pd.NA)
    s = s.str.zfill(9)
    return s


def clean_acc(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    s = s.str.replace(r"^tensor\((.*)\)$", r"\1", regex=True)
    s = s.str.strip()
    return s


def choose_target_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find target column from {candidates}")


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_encoder_tag(encoder_name: str, use_lora: bool, do_finetune: bool, finetune_mode: str) -> str:
    base = encoder_name.lower().replace("/", "_")
    if base == "resnet50":
        return "resnet50"
    if do_finetune:
        if use_lora:
            return f"{base}_lora_ft_{finetune_mode}"
        return f"{base}_fullft_{finetune_mode}"
    return f"{base}_frozen"


def grouped_split_with_balance(df, y_row, groups, test_size=0.20, seed=3469, max_trials=200):
    if HAS_STRATIFIED_GROUP_KFOLD:
        n_splits = int(round(1 / test_size))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, test_idx = next(sgkf.split(df, y_row, groups))
        return train_idx, test_idx, "StratifiedGroupKFold", seed, 1

    best = None
    best_gap = np.inf
    best_seed = None

    for s in range(seed, seed + max_trials):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=s)
        train_idx, test_idx = next(gss.split(df, y_row, groups=groups))

        y_train = y_row[train_idx]
        y_test = y_row[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        gap = abs(y_train.mean() - y_test.mean())

        if gap < best_gap:
            best_gap = gap
            best = (train_idx, test_idx)
            best_seed = s

    if best is None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(df, y_row, groups=groups))
        return train_idx, test_idx, "GroupShuffleSplit_fallback", seed, max_trials

    return best[0], best[1], "GroupShuffleSplit_balanced_search", best_seed, max_trials


ENCODER_TAG = make_encoder_tag(
    encoder_name=ENCODER_NAME,
    use_lora=USE_LORA,
    do_finetune=DO_FINETUNE,
    finetune_mode=FINETUNE_MODE,
)

OUT_EMB_V1V2_PATH = os.path.join(BASE_DATA_DIR, f"v1v2_cxr_embeddings_{ENCODER_TAG}.csv")
OUT_MULTI_V1V2_PATH = os.path.join(BASE_DATA_DIR, f"v1v2_multimodal_dataset_{ENCODER_TAG}.csv")
OUT_IMAGE_SPLIT_PATH = os.path.join(BASE_DATA_DIR, f"shared_patient_split_v1v2_imageeligible_{ENCODER_TAG}.csv")
OUT_MODEL_DIR = os.path.join(BASE_DATA_DIR, f"image_encoder_{ENCODER_TAG}")
Path(OUT_MODEL_DIR).mkdir(parents=True, exist_ok=True)


# =========================
# Build filtered image-eligible subset
# =========================
v1v2_df = pd.read_csv(V1V2_PATH)
v1v2_df["MRN"] = mrn_9(v1v2_df["MRN"])
if "ACC" in v1v2_df.columns:
    v1v2_df["ACC"] = clean_acc(v1v2_df["ACC"])


def common_filters_v1v2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "within_24h" in out.columns:
        out["within_24h"] = out["within_24h"].astype(str).str.lower().map({"true": True, "false": False})
        out = out[out["within_24h"] == True].copy()
    out = out[out["dicom_path"].notna()].copy()
    out = out.drop_duplicates().reset_index(drop=True)
    return out


manifest_v1v2 = common_filters_v1v2(v1v2_df)
target_v1v2 = choose_target_col(manifest_v1v2, TARGET_CANDIDATES_V1V2)
manifest_v1v2[target_v1v2] = pd.to_numeric(manifest_v1v2[target_v1v2], errors="coerce")
manifest_v1v2 = manifest_v1v2[manifest_v1v2[target_v1v2].notna()].copy().reset_index(drop=True)

print("Filtered V1V2 rows:", len(manifest_v1v2))

groups = manifest_v1v2["MRN"].astype(str).to_numpy()
y_row = (manifest_v1v2[target_v1v2].to_numpy(dtype=np.float32) >= CLASS_THRESHOLD).astype(np.int32)

train_idx, test_idx, split_method, seed_used, max_trials_used = grouped_split_with_balance(
    df=manifest_v1v2,
    y_row=y_row,
    groups=groups,
    test_size=TEST_SIZE,
    seed=SEED,
    max_trials=MAX_SPLIT_TRIALS,
)

manifest_v1v2["split"] = "test"
manifest_v1v2.iloc[train_idx, manifest_v1v2.columns.get_loc("split")] = "train"

train_patients = set(manifest_v1v2.iloc[train_idx]["MRN"].astype(str).unique())
test_patients = set(manifest_v1v2.iloc[test_idx]["MRN"].astype(str).unique())

split_df = pd.DataFrame({"MRN": sorted(train_patients.union(test_patients))})
split_df["split"] = np.where(split_df["MRN"].isin(train_patients), "train", "test")
split_df.to_csv(OUT_IMAGE_SPLIT_PATH, index=False)

filtered_train_rate = float(y_row[train_idx].mean()) if len(train_idx) else np.nan
filtered_test_rate = float(y_row[test_idx].mean()) if len(test_idx) else np.nan
filtered_gap = abs(filtered_train_rate - filtered_test_rate)

print("Filtered train rows:", len(train_idx))
print("Filtered test rows :", len(test_idx))
print("Filtered train positive rate:", filtered_train_rate)
print("Filtered test positive rate :", filtered_test_rate)
print("Filtered class gap          :", filtered_gap)
print("Image-eligible split saved to:", OUT_IMAGE_SPLIT_PATH)


# =========================
# Image helpers
# =========================
def load_dicom_image(dicom_path: str) -> np.ndarray:
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    try:
        img = apply_voi_lut(img, ds)
    except Exception:
        pass
    img = img.astype(np.float32)
    photometric = getattr(ds, "PhotometricInterpretation", "")
    if photometric == "MONOCHROME1":
        img = img.max() - img
    img = img - img.min()
    max_val = img.max()
    if max_val > 0:
        img = img / max_val
    return img


def preprocess_image_array(img: np.ndarray, image_size: int = 224, to_3ch: bool = True) -> np.ndarray:
    img_uint8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img_uint8).convert("L")
    pil = pil.resize((image_size, image_size))
    arr = np.array(pil, dtype=np.float32) / 255.0
    if to_3ch:
        arr = np.stack([arr, arr, arr], axis=0)
    else:
        arr = arr[None, ...]
    return arr


class CXRDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, image_size: int = 224, to_3ch: bool = True):
        self.df = manifest_df.reset_index(drop=True).copy()
        self.image_size = image_size
        self.to_3ch = to_3ch

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        dicom_path = row["dicom_path"]
        img = load_dicom_image(dicom_path)
        img = preprocess_image_array(img, image_size=self.image_size, to_3ch=self.to_3ch)
        return {
            "image": torch.tensor(img, dtype=torch.float32),
            "MRN": row["MRN"],
            "ACC": row["ACC"] if "ACC" in row.index else "",
            "dicom_path": dicom_path,
        }


class CXRLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_col: str, image_size: int = 224, mode: str = "regression", threshold: float = 7.5):
        self.df = df.reset_index(drop=True).copy()
        self.target_col = target_col
        self.image_size = image_size
        self.mode = mode
        self.threshold = threshold

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = load_dicom_image(row["dicom_path"])
        img = preprocess_image_array(img, image_size=self.image_size, to_3ch=True)

        y_raw = float(row[self.target_col])
        y = float(y_raw >= self.threshold) if self.mode == "classification" else y_raw

        return torch.tensor(img, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)


class CXRTaskModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, out_dim: int = 1):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, out_dim)

    def forward(self, x):
        feat = self.backbone(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        return self.head(feat)


def infer_vit_feature_dim(model: nn.Module, image_size: int = 224, device: str = "cpu") -> int:
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        feat = model(dummy)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        return int(feat.shape[-1])


def build_image_encoder(
    encoder_name: str = "resnet50",
    use_lora: bool = False,
    image_size: int = 224,
    lora_weights_path: str = None,
    load_for_training: bool = False,
):
    encoder_name = encoder_name.lower()

    if encoder_name == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        feature_extractor = backbone
        if not load_for_training:
            for p in feature_extractor.parameters():
                p.requires_grad = False
        return feature_extractor, feature_dim

    if timm is None:
        raise ImportError("timm is required for ViT models.")

    vit = timm.create_model(
        encoder_name,
        pretrained=True,
        num_classes=0,
        global_pool="avg",
    )
    feature_dim = infer_vit_feature_dim(vit, image_size=image_size, device=DEVICE)

    if use_lora:
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("peft is required for LoRA.")
        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
        )
        vit = get_peft_model(vit, config)

        if lora_weights_path is not None and os.path.exists(lora_weights_path):
            state = torch.load(lora_weights_path, map_location="cpu")
            vit.load_state_dict(state, strict=False)

        if not load_for_training:
            vit.eval()

        return vit, feature_dim

    if not load_for_training:
        for p in vit.parameters():
            p.requires_grad = False

    return vit, feature_dim


def save_encoder_state(obj, path: str) -> None:
    if isinstance(obj, OrderedDict):
        torch.save(obj, path)
    else:
        torch.save(obj.state_dict(), path)
    print("Saved model state to:", path)


# =========================
# Fine-tuning on filtered TRAIN ONLY
# =========================
ft_df = manifest_v1v2[manifest_v1v2["split"] == "train"].copy()

print("Fine-tuning eligible TRAIN rows:", len(ft_df))
print("Target column:", target_v1v2)

feature_extractor, feature_dim = build_image_encoder(
    encoder_name=ENCODER_NAME,
    use_lora=USE_LORA and ENCODER_NAME.lower() != "resnet50",
    image_size=IMAGE_SIZE,
    lora_weights_path=None,
    load_for_training=DO_FINETUNE,
)
feature_extractor = feature_extractor.to(DEVICE)

if DO_FINETUNE:
    groups_ft = ft_df["MRN"].astype(str).to_numpy() if "MRN" in ft_df.columns else np.arange(len(ft_df)).astype(str)
    y_ft = ft_df[target_v1v2].to_numpy(dtype=np.float32)
    if FINETUNE_MODE == "classification":
        y_ft = (y_ft >= CLASS_THRESHOLD).astype(np.float32)

    gss = GroupShuffleSplit(n_splits=1, test_size=FT_INNER_VAL_RATIO, random_state=SEED)
    tr_idx, va_idx = next(gss.split(ft_df, y_ft, groups=groups_ft))

    df_tr = ft_df.iloc[tr_idx].copy().reset_index(drop=True)
    df_va = ft_df.iloc[va_idx].copy().reset_index(drop=True)

    tr_ds = CXRLabelDataset(df_tr, target_col=target_v1v2, image_size=IMAGE_SIZE, mode=FINETUNE_MODE, threshold=CLASS_THRESHOLD)
    va_ds = CXRLabelDataset(df_va, target_col=target_v1v2, image_size=IMAGE_SIZE, mode=FINETUNE_MODE, threshold=CLASS_THRESHOLD)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    task_model = CXRTaskModel(feature_extractor, feature_dim, out_dim=1).to(DEVICE)
    optimizer = torch.optim.AdamW(task_model.parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss() if FINETUNE_MODE == "classification" else nn.MSELoss()

    best_loss = np.inf
    best_state = None
    wait = 0

    for epoch in range(FT_EPOCHS):
        task_model.train()
        train_losses = []

        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = task_model(x)
            loss = criterion(pred, y)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(task_model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        task_model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = task_model(x)
                val_losses.append(criterion(pred, y).item())

        tr_loss = float(np.mean(train_losses)) if train_losses else np.nan
        va_loss = float(np.mean(val_losses)) if val_losses else np.nan
        print(f"[FT][{epoch+1}/{FT_EPOCHS}] train={tr_loss:.6f} val={va_loss:.6f}")

        if va_loss < best_loss:
            best_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in task_model.backbone.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= FT_PATIENCE:
                break

    if best_state is not None:
        task_model.backbone.load_state_dict(best_state, strict=False)

    encoder_weights_path = os.path.join(OUT_MODEL_DIR, "encoder_state.pt")
    save_encoder_state(task_model.backbone, encoder_weights_path)
    feature_extractor = task_model.backbone
else:
    encoder_weights_path = None

feature_extractor.eval()
feature_extractor = feature_extractor.to(DEVICE)

total_params, trainable_params = count_parameters(feature_extractor)
print(f"Encoder tag: {ENCODER_TAG}")
print(f"Encoder params: total={total_params:,} trainable={trainable_params:,}")


# =========================
# Embedding extraction
# =========================
def extract_embeddings(manifest_df: pd.DataFrame, feature_extractor: nn.Module, out_csv_path: str) -> pd.DataFrame:
    dataset = CXRDataset(manifest_df, image_size=IMAGE_SIZE, to_3ch=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_mrn, all_acc, all_path, all_emb = [], [], [], []

    feature_extractor.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            feat = feature_extractor(images)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = feat.detach().cpu().numpy()

            all_emb.append(feat)
            all_mrn.extend(batch["MRN"])
            all_acc.extend(batch["ACC"])
            all_path.extend(batch["dicom_path"])

    emb = np.concatenate(all_emb, axis=0)
    print(f"Embedding shape for {os.path.basename(out_csv_path)}:", emb.shape)

    emb_df = pd.DataFrame(emb, columns=[f"cxr_emb_{i}" for i in range(emb.shape[1])])
    emb_df.insert(0, "dicom_path", all_path)
    emb_df.insert(0, "ACC", all_acc)
    emb_df.insert(0, "MRN", all_mrn)
    emb_df.insert(0, "encoder_tag", ENCODER_TAG)
    emb_df.insert(1, "encoder_name", ENCODER_NAME)
    emb_df.insert(2, "use_lora", bool(USE_LORA and ENCODER_NAME.lower() != "resnet50"))
    emb_df.insert(3, "did_finetune", bool(DO_FINETUNE))
    emb_df.insert(4, "finetune_mode", FINETUNE_MODE if DO_FINETUNE else "none")

    emb_df["MRN"] = mrn_9(emb_df["MRN"])
    if "ACC" in emb_df.columns:
        emb_df["ACC"] = clean_acc(emb_df["ACC"])

    emb_df.to_csv(out_csv_path, index=False)
    print("Saved embeddings:", out_csv_path)
    return emb_df


emb_v1v2 = extract_embeddings(manifest_v1v2, feature_extractor, OUT_EMB_V1V2_PATH)


# =========================
# Merge with filtered V1V2 vent table
# =========================
def merge_multimodal_v1v2(vent_df: pd.DataFrame, emb_df: pd.DataFrame, out_path: str) -> pd.DataFrame:
    out = vent_df.copy()
    out["MRN"] = mrn_9(out["MRN"])
    emb_df = emb_df.copy()
    emb_df["MRN"] = mrn_9(emb_df["MRN"])

    if "ACC" in out.columns:
        out["ACC"] = clean_acc(out["ACC"])
    if "ACC" in emb_df.columns:
        emb_df["ACC"] = clean_acc(emb_df["ACC"])

    merge_keys = [c for c in ["MRN", "ACC", "dicom_path"] if c in out.columns and c in emb_df.columns]
    if not merge_keys:
        merge_keys = [c for c in ["MRN", "ACC"] if c in out.columns and c in emb_df.columns]
    if not merge_keys:
        raise ValueError("Could not find merge keys between vent_df and emb_df.")

    df_multi = out.merge(
        emb_df,
        on=merge_keys,
        how="inner",
        suffixes=("", "_img"),
    )

    key_cols = [c for c in ["MRN", "ACC", "1st_Time_Start", "dicom_path"] if c in df_multi.columns]
    if key_cols:
        print("Duplicate rows by key before dropping:", int(df_multi.duplicated(subset=key_cols).sum()))
        df_multi = df_multi.drop_duplicates(subset=key_cols).copy()

    df_multi.to_csv(out_path, index=False)
    print("Saved multimodal dataset to:", out_path)
    print("Rows:", len(df_multi))
    return df_multi


df_multi_v1v2 = merge_multimodal_v1v2(manifest_v1v2, emb_v1v2, OUT_MULTI_V1V2_PATH)

metadata = {
    "encoder_tag": ENCODER_TAG,
    "encoder_name": ENCODER_NAME,
    "use_lora": bool(USE_LORA and ENCODER_NAME.lower() != "resnet50"),
    "did_finetune": bool(DO_FINETUNE),
    "finetune_mode": FINETUNE_MODE if DO_FINETUNE else "none",
    "weights_path": encoder_weights_path,
    "v1v2_embeddings_path": OUT_EMB_V1V2_PATH,
    "v1v2_multimodal_path": OUT_MULTI_V1V2_PATH,
    "imageeligible_split_path": OUT_IMAGE_SPLIT_PATH,
    "n_v1v2_rows_filtered": int(len(manifest_v1v2)),
    "n_v1v2_train_ft_rows": int(len(ft_df)),
    "target_col": target_v1v2,
    "split_method": split_method,
    "seed_used": int(seed_used),
    "max_trials": int(max_trials_used),
    "train_test_class_gap": float(filtered_gap),
    "row_positive_rate_train_filtered": float(filtered_train_rate),
    "row_positive_rate_test_filtered": float(filtered_test_rate),
    "note": "V3 intentionally not processed for CXR because downstream external validation is vent-only.",
}

with open(os.path.join(OUT_MODEL_DIR, "run_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(json.dumps(metadata, indent=2))