#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05a_Multi_Modalities_Models_Regression_leakage_safe.py

Requirements:
  - v1v2_multimodal_dataset_{ENCODER_TAG}.csv already exists and contains a split column

Design:
  - teacher trains on V1V2 train with vent + CXR
  - baseline trains on V1V2 train with vent only
  - student trains on V1V2 train with vent only using teacher distillation
  - teacher is evaluated on V1V2 test only
  - baseline/student are evaluated on V1V2 test and vent-only V3 external validation
"""

import os
import math
import json
import random
import warnings
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

HAS_MAMBA = True
try:
    from mamba_ssm import Mamba
except Exception:
    HAS_MAMBA = False
    print("WARNING: mamba_ssm not available. Mamba experiments will be skipped.")

SEED = 3469
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENCODER_TAG = "vit_base_patch16_224_lora_ft_regression"

BASE_DATA_DIR = "/nfs/turbo/med-kayvan-lab/Projects/PARDS/02-Data/PARDS_Risk_V3"
BASE_OUT_DIR = "/nfs/turbo/med-kayvan-lab/Projects/PARDS/04-Results/PARDS_Risk_V3"

V1V2_MULTI_PATH = os.path.join(BASE_DATA_DIR, f"v1v2_multimodal_dataset_{ENCODER_TAG}.csv")
V3_VENT_PATH = os.path.join(BASE_DATA_DIR, "vent_window_with_best_cxr_V3.csv")

OUT_DIR = os.path.join(BASE_OUT_DIR, f"MMReg_VentALL_CXR_{ENCODER_TAG}")
os.makedirs(OUT_DIR, exist_ok=True)

N_FOLDS = 5

CXR_REDUCER_CANDIDATES = [
    {"name": "pca", "fit_dim": 64, "latent_dim": 20},
    {"name": "autoencoder", "latent_dim": 20, "hidden_dim": 128, "epochs": 80, "lr": 1e-3, "weight_decay": 1e-5},
]

EPOCHS = 120
PATIENCE = 12
BATCH_SIZE = 64

DISTILLATION_CANDIDATES = [
    {"alpha": 0.9, "beta": 0.1},
    {"alpha": 0.8, "beta": 0.2},
    {"alpha": 0.7, "beta": 0.3},
]

FINAL_VAL_STRATEGY = "holdout"
FINAL_VAL_RATIO = 0.15
FINAL_TRAIN_SEEDS = [SEED, SEED + 17, SEED + 29]

RUN_BOOTSTRAP_CI_EVAL = True
RUN_LEARNING_CURVES = True
RUN_CXR_REDUCER_SENSITIVITY = True
BOOTSTRAP_B = 1000
LEARNING_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
LEARNING_SEEDS = [SEED]
CXR_REDUCER_SENSITIVITY_CANDIDATES = [
    {"name": "pca", "fit_dim": 64, "latent_dim": 16},
    {"name": "pca", "fit_dim": 64, "latent_dim": 20},
    {"name": "pca", "fit_dim": 64, "latent_dim": 32},
    {"name": "autoencoder", "latent_dim": 16, "hidden_dim": 128, "epochs": 80, "lr": 1e-3, "weight_decay": 1e-5},
    {"name": "autoencoder", "latent_dim": 20, "hidden_dim": 128, "epochs": 80, "lr": 1e-3, "weight_decay": 1e-5},
    {"name": "autoencoder", "latent_dim": 32, "hidden_dim": 128, "epochs": 80, "lr": 1e-3, "weight_decay": 1e-5},
]

FAST_DEBUG = False

TARGET_TRAIN_CANDIDATES = ["OSI_V2_12th_avg", "OSI_12th_avg", "OSI_12th"]
TARGET_VAL_CANDIDATES = ["OSI_V3_12th_avg", "OSI_V2_12th_avg", "OSI_12th_avg", "OSI_12th"]

base_template = [
    "OSI_mean_TW{}", "OSI_std_TW{}", "PIP_mean_TW{}", "PIP_std_TW{}",
    "PEEP_mean_TW{}", "PEEP_std_TW{}", "TV_mean_TW{}(mL/Kg)", "TV_std_TW{}(mL/Kg)",
    "Avg_NegFlowDur_TW{}", "Std_NegFlowDur_TW{}", "Avg_PeakInterval_TW{}", "Std_PeakInterval_TW{}"
]
tw_features = {
    i: [f.format(i) for f in base_template] + [f"w{i}_SubBandEnergy_row{j}" for j in range(1, 17)]
    for i in range(1, 7)
}

MODEL_FAMILIES = ["MLP", "RNN", "LSTM", "GRU", "Transformer"] + (["Mamba"] if HAS_MAMBA else [])
ENSEMBLE_TOP_K = min(3, len(MODEL_FAMILIES))

if FAST_DEBUG:
    PARAM_GRID = {
        "MLP": [{"hidden_dim": 64, "dropout": 0.2, "num_layers": 2, "lr": 1e-3, "weight_decay": 1e-4}],
        "RNN": [{"hidden_dim": 64, "dropout": 0.2, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-4, "bidirectional": False}],
        "LSTM": [{"hidden_dim": 64, "dropout": 0.2, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-4, "bidirectional": False}],
        "GRU": [{"hidden_dim": 64, "dropout": 0.2, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-4, "bidirectional": False}],
        "Transformer": [{"hidden_dim": 64, "dropout": 0.2, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-4}],
        "Mamba": [{"hidden_dim": 64, "dropout": 0.2, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-4}] if HAS_MAMBA else [],
    }
else:
    PARAM_GRID = {
        "MLP": [
            {"hidden_dim": 64, "dropout": 0.0, "num_layers": 2, "lr": 1e-3, "weight_decay": 1e-5},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 2, "lr": 1e-3, "weight_decay": 1e-4},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 3, "lr": 3e-4, "weight_decay": 1e-3},
        ],
        "RNN": [
            {"hidden_dim": 64, "dropout": 0.0, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-5, "bidirectional": False},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-4, "bidirectional": False},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 2, "lr": 3e-4, "weight_decay": 1e-3, "bidirectional": True},
        ],
        "LSTM": [
            {"hidden_dim": 64, "dropout": 0.0, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-5, "bidirectional": False},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-4, "bidirectional": False},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 2, "lr": 3e-4, "weight_decay": 1e-3, "bidirectional": True},
        ],
        "GRU": [
            {"hidden_dim": 64, "dropout": 0.0, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-5, "bidirectional": False},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-4, "bidirectional": False},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 2, "lr": 3e-4, "weight_decay": 1e-3, "bidirectional": True},
        ],
        "Transformer": [
            {"hidden_dim": 64, "dropout": 0.2, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-5},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-4},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 2, "lr": 3e-4, "weight_decay": 1e-3},
        ],
        "Mamba": [
            {"hidden_dim": 64, "dropout": 0.2, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-5},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-4},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 2, "lr": 3e-4, "weight_decay": 1e-3},
        ] if HAS_MAMBA else [],
    }

def set_seed(seed=3469):
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

def choose_target_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find target column from {candidates}")

def detect_cxr_emb_cols(df):
    cols = [c for c in df.columns if c.startswith("cxr_emb_")]
    if len(cols) == 0:
        raise ValueError("No CXR embedding columns found.")
    return cols

def detect_group_col(df):
    for c in ["ResearchID", "MRN", "PatientID"]:
        if c in df.columns:
            return c
    return None

def get_all_vent_feature_cols():
    cols = []
    for tw in range(1, 7):
        cols.extend(tw_features[tw])
    return cols

VENT_FEATURE_COLS = get_all_vent_feature_cols()

def validate_required_vent_columns(df: pd.DataFrame):
    missing_all = []
    for tw in range(1, 7):
        missing = [c for c in tw_features[tw] if c not in df.columns]
        missing_all.extend(missing)
    if missing_all:
        raise ValueError(f"Missing vent columns. First 20: {missing_all[:20]}")

def fit_vent_imputer_train_only(df_train: pd.DataFrame):
    validate_required_vent_columns(df_train)
    tmp = df_train[VENT_FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    fill_values = tmp.median(axis=0)
    fill_values = fill_values.fillna(0.0)
    return fill_values

def apply_vent_imputer(df: pd.DataFrame, fill_values: pd.Series):
    validate_required_vent_columns(df)
    df2 = df.copy()
    tmp = df2[VENT_FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    tmp = tmp.fillna(fill_values)
    df2[VENT_FEATURE_COLS] = tmp
    return df2

def build_vent_seq_from_df(df: pd.DataFrame):
    validate_required_vent_columns(df)
    x_list = [df[tw_features[tw]].to_numpy(dtype=np.float32) for tw in range(1, 7)]
    return np.stack(x_list, axis=1)

def scale_vent_train_only(X_train, X_other_list):
    scaler = StandardScaler()
    N, T, D = X_train.shape
    scaler.fit(X_train.reshape(-1, D))
    X_train_s = scaler.transform(X_train.reshape(-1, D)).reshape(N, T, D).astype(np.float32)

    outs = [X_train_s]
    for Xo in X_other_list:
        No = Xo.shape[0]
        outs.append(scaler.transform(Xo.reshape(-1, D)).reshape(No, T, D).astype(np.float32))
    return scaler, outs

class TabularAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=20, hidden_dim=128):
        super().__init__()
        hidden_dim = max(hidden_dim, latent_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.apply(init_weights)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def fit_autoencoder_latent_train_only(X_train, X_other_list, cfg):
    set_seed(SEED)
    latent_dim = int(cfg.get("latent_dim", 20))
    hidden_dim = int(cfg.get("hidden_dim", 128))
    epochs = int(cfg.get("epochs", 80))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-5))
    batch_size = int(cfg.get("batch_size", BATCH_SIZE))

    model = TabularAutoEncoder(X_train.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    perm = torch.randperm(X_tensor.shape[0])
    val_count = min(max(1, int(round(0.15 * len(X_tensor)))), max(len(X_tensor) - 1, 1))
    if len(X_tensor) <= 8:
        val_count = max(1, len(X_tensor) // 4)
    train_idx = perm[val_count:]
    val_idx = perm[:val_count]
    if len(train_idx) == 0:
        train_idx = perm
        val_idx = perm[:1]

    ds_train = torch.utils.data.TensorDataset(X_tensor[train_idx])
    ds_val = torch.utils.data.TensorDataset(X_tensor[val_idx])
    dl_train = DataLoader(ds_train, batch_size=min(batch_size, len(ds_train)), shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=min(batch_size, len(ds_val)), shuffle=False)

    best_state = None
    best_loss = np.inf
    wait = 0
    for _ in range(epochs):
        model.train()
        for (xb,) in dl_train:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            recon, _ = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (xb,) in dl_val:
                xb = xb.to(DEVICE)
                recon, _ = model(xb)
                val_losses.append(criterion(recon, xb).item())
        val_loss = float(np.mean(val_losses)) if val_losses else np.inf
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= max(5, PATIENCE // 2):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def encode(arr):
        with torch.no_grad():
            xb = torch.tensor(arr, dtype=torch.float32, device=DEVICE)
            _, z = model(xb)
            return z.detach().cpu().numpy().astype(np.float32)

    outs = [encode(X_train)]
    for Xo in X_other_list:
        outs.append(encode(Xo))
    artifact = {
        "name": "autoencoder",
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "best_recon_loss": best_loss,
    }
    return artifact, outs

def fit_cxr_transform_train_only(X_train_img, X_other_list, reducer_cfg=None):
    reducer_cfg = reducer_cfg or {"name": "pca", "fit_dim": 64, "latent_dim": 32}
    img_scaler = StandardScaler()
    X_train_img_s = img_scaler.fit_transform(X_train_img)
    reducer_name = reducer_cfg.get("name", "pca").lower()

    if reducer_name == "pca":
        fit_dim = int(reducer_cfg.get("fit_dim", 64))
        use_dim = int(reducer_cfg.get("latent_dim", 32))
        reducer = PCA(n_components=min(fit_dim, X_train_img_s.shape[0], X_train_img_s.shape[1]), random_state=SEED)
        X_train_img_t = reducer.fit_transform(X_train_img_s)[:, :min(use_dim, reducer.n_components_)].astype(np.float32)
        outs = [X_train_img_t]
        for Xo in X_other_list:
            Xo_s = img_scaler.transform(Xo)
            outs.append(reducer.transform(Xo_s)[:, :min(use_dim, reducer.n_components_)].astype(np.float32))
        artifact = {"name": "pca", "model": reducer, "latent_dim": X_train_img_t.shape[1], "fit_dim": fit_dim}
        return img_scaler, artifact, outs

    if reducer_name == "autoencoder":
        artifact, outs = fit_autoencoder_latent_train_only(X_train_img_s.astype(np.float32), [img_scaler.transform(Xo).astype(np.float32) for Xo in X_other_list], reducer_cfg)
        artifact["latent_dim"] = outs[0].shape[1]
        return img_scaler, artifact, outs

    if reducer_name == "none":
        outs = [X_train_img_s.astype(np.float32)]
        for Xo in X_other_list:
            outs.append(img_scaler.transform(Xo).astype(np.float32))
        artifact = {"name": "none", "latent_dim": X_train_img_s.shape[1]}
        return img_scaler, artifact, outs

    raise ValueError(f"Unknown CXR reducer: {reducer_name}")

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def regression_metrics(y_true, y_pred):
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred), "R2": float(r2_score(y_true, y_pred)), "N": int(len(y_true))}

def bootstrap_ci_regression_grouped(y_true, y_pred, groups, B=1000, seed=0):
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    groups = np.asarray(groups).astype(str)
    unique_groups = np.unique(groups)
    rng = np.random.RandomState(seed)
    rows = []
    metric_vals = {"RMSE": [], "MAE": [], "R2": []}
    point = regression_metrics(y_true, y_pred)
    for _ in range(B):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        yt = y_true[idx]
        yp = y_pred[idx]
        m = regression_metrics(yt, yp)
        for k in metric_vals:
            if np.isfinite(m[k]):
                metric_vals[k].append(m[k])
    for metric, vals in metric_vals.items():
        if not vals:
            continue
        rows.append({"metric": metric, "point": point[metric], "ci_low": float(np.percentile(vals, 2.5)), "ci_high": float(np.percentile(vals, 97.5)), "n_valid_bootstrap": int(len(vals))})
    return pd.DataFrame(rows)

def scale_targets_train_only(y_train, y_other_list):
    scaler = StandardScaler()
    scaler.fit(y_train.reshape(-1, 1))
    y_train_s = scaler.transform(y_train.reshape(-1, 1)).astype(np.float32)
    outs = [y_train_s]
    for yo in y_other_list:
        outs.append(scaler.transform(yo.reshape(-1, 1)).astype(np.float32))
    return scaler, outs


class VentOnlyDataset(Dataset):
    def __init__(self, Xv, y):
        self.Xv = torch.tensor(Xv, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.Xv)

    def __getitem__(self, idx):
        return self.Xv[idx], self.y[idx]

class MultiDataset(Dataset):
    def __init__(self, Xv, Xi, y):
        self.Xv = torch.tensor(Xv, dtype=torch.float32)
        self.Xi = torch.tensor(Xi, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.Xv)

    def __getitem__(self, idx):
        return self.Xv[idx], self.Xi[idx], self.y[idx]

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(w * x, dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, nhead, dropout):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True,
                                         dim_feedforward=hidden_dim * 4, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=1)

    def forward(self, x):
        return self.encoder(x)

class MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(d_model=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.mamba(x))

class VentMLP(nn.Module):
    def __init__(self, vent_dim_per_tw, timesteps=6, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        in_dim = vent_dim_per_tw * timesteps
        layers, d = [], in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, xv):
        return self.net(xv.flatten(start_dim=1))

class VentRecurrent(nn.Module):
    def __init__(self, rnn_type, vent_dim_per_tw, hidden_dim=128, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__()
        if num_layers == 1:
            dropout = 0.0
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            vent_dim_per_tw,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.pool = AttentionPool(out_dim)
        self.fc = nn.Linear(out_dim, 1)
        self.apply(init_weights)

    def forward(self, xv):
        h, _ = self.rnn(xv)
        return self.fc(self.pool(h))

class VentRNN(VentRecurrent):
    def __init__(self, vent_dim_per_tw, hidden_dim=128, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__("RNN", vent_dim_per_tw, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

class VentLSTM(VentRecurrent):
    def __init__(self, vent_dim_per_tw, hidden_dim=128, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__("LSTM", vent_dim_per_tw, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

class VentGRU(VentRecurrent):
    def __init__(self, vent_dim_per_tw, hidden_dim=128, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__("GRU", vent_dim_per_tw, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

class VentTransformer(nn.Module):
    def __init__(self, vent_dim_per_tw, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(vent_dim_per_tw, hidden_dim), nn.LayerNorm(hidden_dim))
        self.pe = PositionalEncoding(hidden_dim)
        if hidden_dim % 8 == 0:
            nhead = max(1, hidden_dim // 8)
        elif hidden_dim % 4 == 0:
            nhead = max(1, hidden_dim // 4)
        else:
            nhead = 1
        self.blocks = nn.Sequential(*[TransformerBlock(hidden_dim, nhead=nhead, dropout=dropout) for _ in range(num_layers)])
        self.pool = AttentionPool(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)

    def forward(self, xv):
        x = self.in_proj(xv)
        x = self.pe(x)
        x = self.blocks(x)
        return self.fc(self.pool(x))

class VentMamba(nn.Module):
    def __init__(self, vent_dim_per_tw, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(vent_dim_per_tw, hidden_dim), nn.LayerNorm(hidden_dim))
        self.blocks = nn.Sequential(*[nn.Sequential(MambaBlock(hidden_dim), nn.Dropout(dropout)) for _ in range(num_layers)])
        self.pool = AttentionPool(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)

    def forward(self, xv):
        x = self.in_proj(xv)
        x = self.blocks(x)
        return self.fc(self.pool(x))

class EarlyFusionMLP(nn.Module):
    def __init__(self, vent_dim_per_tw, img_dim=32, timesteps=6, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        in_dim = (vent_dim_per_tw + img_dim) * timesteps
        layers, d = [], in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, xv, xi):
        xi_rep = xi.unsqueeze(1).repeat(1, xv.size(1), 1)
        return self.net(torch.cat([xv, xi_rep], dim=-1).flatten(start_dim=1))

class EarlyFusionRecurrent(nn.Module):
    def __init__(self, rnn_type, vent_dim_per_tw, img_dim=32, hidden_dim=128, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__()
        if num_layers == 1:
            dropout = 0.0
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            vent_dim_per_tw + img_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.pool = AttentionPool(out_dim)
        self.fc = nn.Linear(out_dim, 1)
        self.apply(init_weights)

    def forward(self, xv, xi):
        xi_rep = xi.unsqueeze(1).repeat(1, xv.size(1), 1)
        h, _ = self.rnn(torch.cat([xv, xi_rep], dim=-1))
        return self.fc(self.pool(h))

class EarlyFusionRNN(EarlyFusionRecurrent):
    def __init__(self, vent_dim_per_tw, img_dim=32, hidden_dim=128, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__("RNN", vent_dim_per_tw, img_dim=img_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

class EarlyFusionLSTM(EarlyFusionRecurrent):
    def __init__(self, vent_dim_per_tw, img_dim=32, hidden_dim=128, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__("LSTM", vent_dim_per_tw, img_dim=img_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

class EarlyFusionGRU(EarlyFusionRecurrent):
    def __init__(self, vent_dim_per_tw, img_dim=32, hidden_dim=128, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__("GRU", vent_dim_per_tw, img_dim=img_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

class EarlyFusionTransformer(nn.Module):
    def __init__(self, vent_dim_per_tw, img_dim=32, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(vent_dim_per_tw + img_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.pe = PositionalEncoding(hidden_dim)
        if hidden_dim % 8 == 0:
            nhead = max(1, hidden_dim // 8)
        elif hidden_dim % 4 == 0:
            nhead = max(1, hidden_dim // 4)
        else:
            nhead = 1
        self.blocks = nn.Sequential(*[TransformerBlock(hidden_dim, nhead=nhead, dropout=dropout) for _ in range(num_layers)])
        self.pool = AttentionPool(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)

    def forward(self, xv, xi):
        xi_rep = xi.unsqueeze(1).repeat(1, xv.size(1), 1)
        x = self.in_proj(torch.cat([xv, xi_rep], dim=-1))
        x = self.pe(x)
        x = self.blocks(x)
        return self.fc(self.pool(x))

class EarlyFusionMamba(nn.Module):
    def __init__(self, vent_dim_per_tw, img_dim=32, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(vent_dim_per_tw + img_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.blocks = nn.Sequential(*[nn.Sequential(MambaBlock(hidden_dim), nn.Dropout(dropout)) for _ in range(num_layers)])
        self.pool = AttentionPool(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)

    def forward(self, xv, xi):
        xi_rep = xi.unsqueeze(1).repeat(1, xv.size(1), 1)
        x = self.in_proj(torch.cat([xv, xi_rep], dim=-1))
        x = self.blocks(x)
        return self.fc(self.pool(x))

class CrossFusionMLP(nn.Module):
    def __init__(self, vent_dim_per_tw, img_dim=32, timesteps=6, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        vent_in = vent_dim_per_tw * timesteps
        self.vent_branch = nn.Sequential(
            nn.Linear(vent_in, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.img_branch = nn.Sequential(
            nn.Linear(img_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        fusion_in = hidden_dim * 3
        layers, d = [], fusion_in
        for _ in range(max(1, num_layers - 1)):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, 1)]
        self.head = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, xv, xi):
        v = self.vent_branch(xv.flatten(start_dim=1))
        i = self.img_branch(xi)
        return self.head(torch.cat([v, i, v * i], dim=1))

class CrossFusionRecurrent(nn.Module):
    def __init__(self, rnn_type, vent_dim_per_tw, img_dim=32, hidden_dim=128, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__()
        if num_layers == 1:
            dropout = 0.0
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.vent_encoder = rnn_cls(
            vent_dim_per_tw,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.vent_pool = AttentionPool(out_dim)
        self.img_branch = nn.Sequential(
            nn.Linear(img_dim, out_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(out_dim, 1)
        )
        self.apply(init_weights)

    def forward(self, xv, xi):
        h, _ = self.vent_encoder(xv)
        v = self.vent_pool(h)
        i = self.img_branch(xi)
        return self.head(torch.cat([v, i, v * i], dim=1))

class CrossFusionRNN(CrossFusionRecurrent):
    def __init__(self, vent_dim_per_tw, img_dim=32, hidden_dim=128, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__("RNN", vent_dim_per_tw, img_dim=img_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

class CrossFusionLSTM(CrossFusionRecurrent):
    def __init__(self, vent_dim_per_tw, img_dim=32, hidden_dim=128, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__("LSTM", vent_dim_per_tw, img_dim=img_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

class CrossFusionGRU(CrossFusionRecurrent):
    def __init__(self, vent_dim_per_tw, img_dim=32, hidden_dim=128, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__("GRU", vent_dim_per_tw, img_dim=img_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

class CrossFusionTransformer(nn.Module):
    def __init__(self, vent_dim_per_tw, img_dim=32, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.vent_in = nn.Sequential(nn.Linear(vent_dim_per_tw, hidden_dim), nn.LayerNorm(hidden_dim))
        self.pe = PositionalEncoding(hidden_dim)
        if hidden_dim % 8 == 0:
            nhead = max(1, hidden_dim // 8)
        elif hidden_dim % 4 == 0:
            nhead = max(1, hidden_dim // 4)
        else:
            nhead = 1
        self.vent_blocks = nn.Sequential(*[TransformerBlock(hidden_dim, nhead=nhead, dropout=dropout) for _ in range(num_layers)])
        self.vent_pool = AttentionPool(hidden_dim)
        self.img_branch = nn.Sequential(
            nn.Linear(img_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(init_weights)

    def forward(self, xv, xi):
        x = self.vent_in(xv)
        x = self.pe(x)
        x = self.vent_blocks(x)
        v = self.vent_pool(x)
        i = self.img_branch(xi)
        return self.head(torch.cat([v, i, v * i], dim=1))

class CrossFusionMamba(nn.Module):
    def __init__(self, vent_dim_per_tw, img_dim=32, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.vent_in = nn.Sequential(nn.Linear(vent_dim_per_tw, hidden_dim), nn.LayerNorm(hidden_dim))
        self.vent_blocks = nn.Sequential(*[nn.Sequential(MambaBlock(hidden_dim), nn.Dropout(dropout)) for _ in range(num_layers)])
        self.vent_pool = AttentionPool(hidden_dim)
        self.img_branch = nn.Sequential(
            nn.Linear(img_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(init_weights)

    def forward(self, xv, xi):
        x = self.vent_in(xv)
        x = self.vent_blocks(x)
        v = self.vent_pool(x)
        i = self.img_branch(xi)
        return self.head(torch.cat([v, i, v * i], dim=1))

def build_baseline_model(family, vent_dim_per_tw, cfg):
    if family == "MLP":
        return VentMLP(vent_dim_per_tw, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    if family == "RNN":
        return VentRNN(vent_dim_per_tw, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"], bidirectional=cfg.get("bidirectional", False))
    if family == "LSTM":
        return VentLSTM(vent_dim_per_tw, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"], bidirectional=cfg.get("bidirectional", False))
    if family == "GRU":
        return VentGRU(vent_dim_per_tw, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"], bidirectional=cfg.get("bidirectional", False))
    if family == "Transformer":
        return VentTransformer(vent_dim_per_tw, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    if family == "Mamba":
        return VentMamba(vent_dim_per_tw, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    raise ValueError(f"Unknown family: {family}")

def build_teacher_model(family, fusion_type, vent_dim_per_tw, img_dim, cfg):
    if fusion_type == "early":
        if family == "MLP":
            return EarlyFusionMLP(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
        if family == "RNN":
            return EarlyFusionRNN(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"], bidirectional=cfg.get("bidirectional", False))
        if family == "LSTM":
            return EarlyFusionLSTM(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"], bidirectional=cfg.get("bidirectional", False))
        if family == "GRU":
            return EarlyFusionGRU(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"], bidirectional=cfg.get("bidirectional", False))
        if family == "Transformer":
            return EarlyFusionTransformer(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
        if family == "Mamba":
            return EarlyFusionMamba(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    if fusion_type == "cross":
        if family == "MLP":
            return CrossFusionMLP(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
        if family == "RNN":
            return CrossFusionRNN(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"], bidirectional=cfg.get("bidirectional", False))
        if family == "LSTM":
            return CrossFusionLSTM(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"], bidirectional=cfg.get("bidirectional", False))
        if family == "GRU":
            return CrossFusionGRU(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"], bidirectional=cfg.get("bidirectional", False))
        if family == "Transformer":
            return CrossFusionTransformer(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
        if family == "Mamba":
            return CrossFusionMamba(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    raise ValueError(f"Unknown family/fusion: {family}, {fusion_type}")

@dataclass
class FitResult:
    model: nn.Module
    best_val_loss: float

def train_baseline(model, train_loader, val_loader, cfg):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    criterion = nn.MSELoss()
    best_loss = np.inf
    best_state = None
    wait = 0
    for _ in range(EPOCHS):
        model.train()
        for xv, y in train_loader:
            xv, y = xv.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xv)
            loss = criterion(pred, y)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if val_loader is None:
            val_loss = float(loss.item())
        else:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xv, y in val_loader:
                    xv, y = xv.to(DEVICE), y.to(DEVICE)
                    pred = model(xv)
                    val_losses.append(criterion(pred, y).item())
            val_loss = float(np.mean(val_losses))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return FitResult(model=model, best_val_loss=best_loss)

def train_teacher(model, train_loader, val_loader, cfg):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    criterion = nn.MSELoss()
    best_loss = np.inf
    best_state = None
    wait = 0
    for _ in range(EPOCHS):
        model.train()
        for xv, xi, y in train_loader:
            xv, xi, y = xv.to(DEVICE), xi.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xv, xi)
            loss = criterion(pred, y)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if val_loader is None:
            val_loss = float(loss.item())
        else:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xv, xi, y in val_loader:
                    xv, xi, y = xv.to(DEVICE), xi.to(DEVICE), y.to(DEVICE)
                    pred = model(xv, xi)
                    val_losses.append(criterion(pred, y).item())
            val_loss = float(np.mean(val_losses))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return FitResult(model=model, best_val_loss=best_loss)

def train_student(student, teacher, train_loader, val_loader, cfg):
    student = student.to(DEVICE)
    teacher = teacher.to(DEVICE)
    teacher.eval()
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    mse = nn.MSELoss()
    alpha = float(cfg.get("alpha", 0.8))
    beta = float(cfg.get("beta", 0.2))
    best_loss = np.inf
    best_state = None
    wait = 0
    for _ in range(EPOCHS):
        student.train()
        for xv, xi, y in train_loader:
            xv, xi, y = xv.to(DEVICE), xi.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                t_pred = teacher(xv, xi)
            s_pred = student(xv)
            hard_loss = mse(s_pred, y)
            soft_loss = mse(s_pred, t_pred)
            loss = alpha * hard_loss + beta * soft_loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
        if val_loader is None:
            val_loss = float(loss.item())
        else:
            student.eval()
            val_losses = []
            with torch.no_grad():
                for xv, xi, y in val_loader:
                    xv, xi, y = xv.to(DEVICE), xi.to(DEVICE), y.to(DEVICE)
                    t_pred = teacher(xv, xi)
                    s_pred = student(xv)
                    hard_loss = mse(s_pred, y)
                    soft_loss = mse(s_pred, t_pred)
                    val_losses.append((alpha * hard_loss + beta * soft_loss).item())
            val_loss = float(np.mean(val_losses))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    student.load_state_dict(best_state if best_state is not None else student.state_dict())
    return FitResult(model=student, best_val_loss=best_loss)

def predict_baseline(model, Xv):
    model.eval()
    with torch.no_grad():
        xv = torch.tensor(Xv, dtype=torch.float32, device=DEVICE)
        pred = model(xv).detach().cpu().numpy()
    return pred

def predict_teacher(model, Xv, Xi):
    model.eval()
    with torch.no_grad():
        xv = torch.tensor(Xv, dtype=torch.float32, device=DEVICE)
        xi = torch.tensor(Xi, dtype=torch.float32, device=DEVICE)
        pred = model(xv, xi).detach().cpu().numpy()
    return pred

def get_inner_train_val_split(df, y, groups, val_ratio=0.15, seed=3469):
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    return next(splitter.split(df, y, groups=groups))

def mean_stack(arr_list):
    return np.mean(np.stack(arr_list, axis=0), axis=0)

def save_regression_posthoc(prefix, y_true, y_pred, groups, out_dir):
    pd.DataFrame([regression_metrics(y_true, y_pred)]).to_csv(os.path.join(out_dir, f"{prefix}_point_metrics.csv"), index=False)
    if RUN_BOOTSTRAP_CI_EVAL:
        bootstrap_ci_regression_grouped(y_true, y_pred, groups, B=BOOTSTRAP_B, seed=SEED).to_csv(os.path.join(out_dir, f"{prefix}_bootstrap_ci.csv"), index=False)

def train_eval_single_run_regression(experiment_type, family, run_seed, reducer_cfg=None, distill_cfg=None, fraction=1.0):
    set_seed(run_seed)
    idx_all = np.arange(len(df_train))
    if fraction < 1.0:
        rng = np.random.RandomState(run_seed)
        sample_n = max(8, int(round(len(idx_all) * fraction)))
        chosen = np.sort(rng.choice(idx_all, size=sample_n, replace=False))
    else:
        chosen = idx_all
    df_train_sub = df_train.iloc[chosen].copy().reset_index(drop=True)
    groups_sub = df_train_sub[detect_group_col(df_train_sub)].astype(str).to_numpy() if detect_group_col(df_train_sub) else np.arange(len(df_train_sub)).astype(str)
    Xv_sub_raw = build_vent_seq_from_df(df_train_sub)
    _, [Xv_sub, Xv_te, Xv_v3_sub] = scale_vent_train_only(Xv_sub_raw, [Xv_test_raw, Xv_v3_raw])
    y_sub_raw = df_train_sub[target_train].to_numpy(dtype=np.float32)
    y_scaler_sub, [y_sub] = scale_targets_train_only(y_sub_raw, [])
    inner_tr_idx, inner_va_idx = get_inner_train_val_split(df=df_train_sub, y=y_sub_raw, groups=groups_sub, val_ratio=FINAL_VAL_RATIO, seed=run_seed)
    Xv_tr, Xv_va = Xv_sub[inner_tr_idx], Xv_sub[inner_va_idx]
    y_tr, y_va = y_sub[inner_tr_idx], y_sub[inner_va_idx]
    baseline_tr_loader = DataLoader(VentOnlyDataset(Xv_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    baseline_va_loader = DataLoader(VentOnlyDataset(Xv_va, y_va), batch_size=BATCH_SIZE, shuffle=False)
    cfg = best_baseline[family]["cfg"] if experiment_type == "baseline" else best_student[family]["cfg"] if experiment_type == "student" else best_teacher[family]["cfg"]
    if experiment_type == "baseline":
        model = train_baseline(build_baseline_model(family, VENT_DIM, cfg), baseline_tr_loader, baseline_va_loader, cfg).model
        return y_scaler_sub.inverse_transform(predict_baseline(model, Xv_te)).ravel(), y_scaler_sub.inverse_transform(predict_baseline(model, Xv_v3_sub)).ravel()
    Xi_sub_raw = df_train_sub[cxr_emb_cols].to_numpy(dtype=np.float32)
    _, _, [Xi_sub, Xi_te] = fit_cxr_transform_train_only(Xi_sub_raw, [Xi_test_raw], reducer_cfg=reducer_cfg)
    Xi_tr, Xi_va = Xi_sub[inner_tr_idx], Xi_sub[inner_va_idx]
    multi_tr_loader = DataLoader(MultiDataset(Xv_tr, Xi_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    multi_va_loader = DataLoader(MultiDataset(Xv_va, Xi_va, y_va), batch_size=BATCH_SIZE, shuffle=False)
    if experiment_type == "teacher":
        model = train_teacher(build_teacher_model(family, best_teacher[family]["fusion"], VENT_DIM, Xi_sub.shape[1], cfg), multi_tr_loader, multi_va_loader, cfg).model
        return y_scaler_sub.inverse_transform(predict_teacher(model, Xv_te, Xi_te)).ravel(), None
    teacher_cfg = best_teacher[family]["cfg"]
    teacher_model = train_teacher(build_teacher_model(family, best_teacher[family]["fusion"], VENT_DIM, Xi_sub.shape[1], teacher_cfg), multi_tr_loader, multi_va_loader, teacher_cfg).model
    student_cfg = {**best_student[family]["cfg"], **(distill_cfg or best_student[family]["distill_cfg"])}
    student_model = train_student(build_baseline_model(family, VENT_DIM, best_student[family]["cfg"]), teacher_model, multi_tr_loader, multi_va_loader, student_cfg).model
    return y_scaler_sub.inverse_transform(predict_baseline(student_model, Xv_te)).ravel(), y_scaler_sub.inverse_transform(predict_baseline(student_model, Xv_v3_sub)).ravel()


df_multi = pd.read_csv(V1V2_MULTI_PATH)
df_v3 = pd.read_csv(V3_VENT_PATH)

if "MRN" in df_multi.columns:
    df_multi["MRN"] = mrn_9(df_multi["MRN"])
if "MRN" in df_v3.columns:
    df_v3["MRN"] = mrn_9(df_v3["MRN"])

if "split" not in df_multi.columns:
    raise RuntimeError("The multimodal V1V2 file does not contain a split column. Re-run 04b first.")

target_train = choose_target_col(df_multi, TARGET_TRAIN_CANDIDATES)
target_val = choose_target_col(df_v3, TARGET_VAL_CANDIDATES)
cxr_emb_cols = detect_cxr_emb_cols(df_multi)

df_multi[target_train] = pd.to_numeric(df_multi[target_train], errors="coerce")
df_v3[target_val] = pd.to_numeric(df_v3[target_val], errors="coerce")

valid_multi_mask = (
    df_multi[target_train].notna() &
    np.isfinite(df_multi[cxr_emb_cols].to_numpy(dtype=np.float32)).all(axis=1)
)
valid_v3_mask = df_v3[target_val].notna()

df_multi = df_multi.loc[valid_multi_mask].copy().reset_index(drop=True)
df_v3 = df_v3.loc[valid_v3_mask].copy().reset_index(drop=True)

df_train_raw = df_multi[df_multi["split"] == "train"].copy().reset_index(drop=True)
df_test_raw = df_multi[df_multi["split"] == "test"].copy().reset_index(drop=True)

if len(df_train_raw) == 0 or len(df_test_raw) == 0:
    raise RuntimeError("The multimodal V1V2 split produced empty train or test set.")

group_col_train = detect_group_col(df_train_raw)
groups_train = df_train_raw[group_col_train].astype(str).to_numpy() if group_col_train else np.arange(len(df_train_raw)).astype(str)

print("Train rows:", len(df_train_raw), "Test rows:", len(df_test_raw), "V3 rows:", len(df_v3))
VENT_DIM = len(tw_features[1])

def prepare_fold_data(df_tr_raw, df_va_raw):
    vent_fill = fit_vent_imputer_train_only(df_tr_raw)
    df_tr = apply_vent_imputer(df_tr_raw, vent_fill)
    df_va = apply_vent_imputer(df_va_raw, vent_fill)

    Xv_tr_raw = build_vent_seq_from_df(df_tr)
    Xv_va_raw = build_vent_seq_from_df(df_va)
    _, [Xv_tr, Xv_va] = scale_vent_train_only(Xv_tr_raw, [Xv_va_raw])

    Xi_tr_raw = df_tr[cxr_emb_cols].to_numpy(dtype=np.float32)
    Xi_va_raw = df_va[cxr_emb_cols].to_numpy(dtype=np.float32)
    reducer_cfg = df_tr_raw.attrs.get("cxr_reducer_cfg", CXR_REDUCER_CANDIDATES[0])
    _, _, [Xi_tr, Xi_va] = fit_cxr_transform_train_only(Xi_tr_raw, [Xi_va_raw], reducer_cfg=reducer_cfg)

    y_tr_raw = df_tr[target_train].to_numpy(dtype=np.float32)
    y_va_raw = df_va[target_train].to_numpy(dtype=np.float32)
    y_scaler, [y_tr, y_va] = scale_targets_train_only(y_tr_raw, [y_va_raw])

    return {"Xv_tr": Xv_tr, "Xv_va": Xv_va, "Xi_tr": Xi_tr, "Xi_va": Xi_va, "y_tr": y_tr, "y_va": y_va, "y_va_raw": y_va_raw, "y_scaler": y_scaler}

def evaluate_cv_baseline(family, cfg, df_train_raw, groups_train):
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_rmses = []
    dummy_y = df_train_raw[target_train].to_numpy(dtype=np.float32)
    for tr_i, va_i in gkf.split(df_train_raw, dummy_y, groups=groups_train):
        df_tr_raw = df_train_raw.iloc[tr_i].copy()
        df_va_raw = df_train_raw.iloc[va_i].copy()
        prep = prepare_fold_data(df_tr_raw, df_va_raw)
        tr_loader = DataLoader(VentOnlyDataset(prep["Xv_tr"], prep["y_tr"]), batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(VentOnlyDataset(prep["Xv_va"], prep["y_va"]), batch_size=BATCH_SIZE, shuffle=False)
        model = build_baseline_model(family, VENT_DIM, cfg)
        result = train_baseline(model, tr_loader, va_loader, cfg)
        pred_scaled = predict_baseline(result.model, prep["Xv_va"])
        pred = prep["y_scaler"].inverse_transform(pred_scaled).ravel()
        truth = prep["y_va_raw"].ravel()
        fold_rmses.append(rmse(truth, pred))
    return float(np.mean(fold_rmses)), float(np.std(fold_rmses))

def evaluate_cv_teacher(family, fusion_type, cfg, reducer_cfg, df_train_raw, groups_train):
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_rmses = []
    dummy_y = df_train_raw[target_train].to_numpy(dtype=np.float32)
    for tr_i, va_i in gkf.split(df_train_raw, dummy_y, groups=groups_train):
        df_tr_raw = df_train_raw.iloc[tr_i].copy()
        df_va_raw = df_train_raw.iloc[va_i].copy()
        df_tr_raw.attrs["cxr_reducer_cfg"] = reducer_cfg
        prep = prepare_fold_data(df_tr_raw, df_va_raw)
        tr_loader = DataLoader(MultiDataset(prep["Xv_tr"], prep["Xi_tr"], prep["y_tr"]), batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(MultiDataset(prep["Xv_va"], prep["Xi_va"], prep["y_va"]), batch_size=BATCH_SIZE, shuffle=False)
        model = build_teacher_model(family, fusion_type, VENT_DIM, prep["Xi_tr"].shape[1], cfg)
        result = train_teacher(model, tr_loader, va_loader, cfg)
        pred_scaled = predict_teacher(result.model, prep["Xv_va"], prep["Xi_va"])
        pred = prep["y_scaler"].inverse_transform(pred_scaled).ravel()
        truth = prep["y_va_raw"].ravel()
        fold_rmses.append(rmse(truth, pred))
    return float(np.mean(fold_rmses)), float(np.std(fold_rmses))

def evaluate_cv_student(family, fusion_type, teacher_cfg, student_cfg, reducer_cfg, distill_cfg, df_train_raw, groups_train):
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_rmses = []
    dummy_y = df_train_raw[target_train].to_numpy(dtype=np.float32)
    for tr_i, va_i in gkf.split(df_train_raw, dummy_y, groups=groups_train):
        df_tr_raw = df_train_raw.iloc[tr_i].copy()
        df_va_raw = df_train_raw.iloc[va_i].copy()
        df_tr_raw.attrs["cxr_reducer_cfg"] = reducer_cfg
        prep = prepare_fold_data(df_tr_raw, df_va_raw)
        tr_loader = DataLoader(MultiDataset(prep["Xv_tr"], prep["Xi_tr"], prep["y_tr"]), batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(MultiDataset(prep["Xv_va"], prep["Xi_va"], prep["y_va"]), batch_size=BATCH_SIZE, shuffle=False)
        teacher = build_teacher_model(family, fusion_type, VENT_DIM, prep["Xi_tr"].shape[1], teacher_cfg)
        teacher = train_teacher(teacher, tr_loader, va_loader, teacher_cfg).model
        student = build_baseline_model(family, VENT_DIM, student_cfg)
        student_fit_cfg = {**student_cfg, **distill_cfg}
        student = train_student(student, teacher, tr_loader, va_loader, student_fit_cfg).model
        pred_scaled = predict_baseline(student, prep["Xv_va"])
        pred = prep["y_scaler"].inverse_transform(pred_scaled).ravel()
        truth = prep["y_va_raw"].ravel()
        fold_rmses.append(rmse(truth, pred))
    return float(np.mean(fold_rmses)), float(np.std(fold_rmses))

baseline_cv_rows = []
best_baseline = {}
for family in MODEL_FAMILIES:
    best_rmse, best_cfg = np.inf, None
    for cfg in PARAM_GRID[family]:
        mean_rmse, std_rmse = evaluate_cv_baseline(family, cfg, df_train_raw, groups_train)
        baseline_cv_rows.append({"family": family, "hidden_dim": cfg["hidden_dim"], "dropout": cfg["dropout"], "num_layers": cfg["num_layers"], "lr": cfg["lr"], "cv_rmse_mean": mean_rmse, "cv_rmse_std": std_rmse})
        print(f"[BASELINE][{family}] {cfg} -> RMSE {mean_rmse:.4f} ± {std_rmse:.4f}")
        if mean_rmse < best_rmse:
            best_rmse, best_cfg = mean_rmse, cfg
    best_baseline[family] = {"cfg": best_cfg, "cv_rmse": best_rmse}
pd.DataFrame(baseline_cv_rows).to_csv(os.path.join(OUT_DIR, "baseline_cv_results.csv"), index=False)

teacher_cv_rows = []
best_teacher = {}
for family in MODEL_FAMILIES:
    best_rmse, best_cfg, best_fusion = np.inf, None, None
    for fusion_type in ["early", "cross"]:
        for reducer_cfg in CXR_REDUCER_CANDIDATES:
            for cfg in PARAM_GRID[family]:
                mean_rmse, std_rmse = evaluate_cv_teacher(family, fusion_type, cfg, reducer_cfg, df_train_raw, groups_train)
                teacher_cv_rows.append({"family": family, "fusion": fusion_type, "cxr_reducer": reducer_cfg["name"], "cxr_latent_dim": reducer_cfg.get("latent_dim"), "hidden_dim": cfg["hidden_dim"], "dropout": cfg["dropout"], "num_layers": cfg["num_layers"], "lr": cfg["lr"], "weight_decay": cfg.get("weight_decay"), "bidirectional": cfg.get("bidirectional", False), "cv_rmse_mean": mean_rmse, "cv_rmse_std": std_rmse})
                print(f"[TEACHER][{family}][{fusion_type}][{reducer_cfg['name']}] {cfg} -> RMSE {mean_rmse:.4f} ± {std_rmse:.4f}")
                if mean_rmse < best_rmse:
                    best_rmse, best_cfg, best_fusion = mean_rmse, cfg, fusion_type
                    best_teacher[family] = {"cfg": best_cfg, "fusion": best_fusion, "reducer_cfg": reducer_cfg, "cv_rmse": best_rmse}
pd.DataFrame(teacher_cv_rows).to_csv(os.path.join(OUT_DIR, "teacher_cv_results.csv"), index=False)

student_cv_rows = []
best_student = {}
for family in MODEL_FAMILIES:
    teacher_cfg = best_teacher[family]["cfg"]
    teacher_fusion = best_teacher[family]["fusion"]
    reducer_cfg = best_teacher[family]["reducer_cfg"]
    best_rmse, best_cfg, best_distill = np.inf, None, None
    for cfg in PARAM_GRID[family]:
        for distill_cfg in DISTILLATION_CANDIDATES:
            mean_rmse, std_rmse = evaluate_cv_student(family, teacher_fusion, teacher_cfg, cfg, reducer_cfg, distill_cfg, df_train_raw, groups_train)
            student_cv_rows.append({"family": family, "teacher_fusion": teacher_fusion, "cxr_reducer": reducer_cfg["name"], "student_hidden_dim": cfg["hidden_dim"], "student_dropout": cfg["dropout"], "student_num_layers": cfg["num_layers"], "student_lr": cfg["lr"], "student_weight_decay": cfg.get("weight_decay"), "student_bidirectional": cfg.get("bidirectional", False), "alpha": distill_cfg["alpha"], "beta": distill_cfg["beta"], "cv_rmse_mean": mean_rmse, "cv_rmse_std": std_rmse})
            print(f"[STUDENT][{family}][alpha={distill_cfg['alpha']:.2f},beta={distill_cfg['beta']:.2f}] {cfg} -> RMSE {mean_rmse:.4f} ± {std_rmse:.4f}")
            if mean_rmse < best_rmse:
                best_rmse, best_cfg, best_distill = mean_rmse, cfg, distill_cfg
    best_student[family] = {"cfg": best_cfg, "distill_cfg": best_distill, "cv_rmse": best_rmse}
pd.DataFrame(student_cv_rows).to_csv(os.path.join(OUT_DIR, "student_cv_results.csv"), index=False)

with open(os.path.join(OUT_DIR, "best_baseline.json"), "w") as f:
    json.dump(best_baseline, f, indent=2)
with open(os.path.join(OUT_DIR, "best_teacher.json"), "w") as f:
    json.dump(best_teacher, f, indent=2)
with open(os.path.join(OUT_DIR, "best_student.json"), "w") as f:
    json.dump(best_student, f, indent=2)

vent_fill_values = fit_vent_imputer_train_only(df_train_raw)
df_train = apply_vent_imputer(df_train_raw, vent_fill_values)
df_test = apply_vent_imputer(df_test_raw, vent_fill_values)
df_v3_imp = apply_vent_imputer(df_v3, vent_fill_values)

Xv_train_raw = build_vent_seq_from_df(df_train)
Xv_test_raw = build_vent_seq_from_df(df_test)
Xv_v3_raw = build_vent_seq_from_df(df_v3_imp)

Xi_train_raw = df_train[cxr_emb_cols].to_numpy(dtype=np.float32)
Xi_test_raw = df_test[cxr_emb_cols].to_numpy(dtype=np.float32)

y_train_raw = df_train[target_train].to_numpy(dtype=np.float32)
y_test_raw = df_test[target_train].to_numpy(dtype=np.float32)
y_v3 = df_v3_imp[target_val].to_numpy(dtype=np.float32)
test_groups = df_test[detect_group_col(df_test)].astype(str).to_numpy() if detect_group_col(df_test) else np.arange(len(df_test)).astype(str)
v3_groups = df_v3_imp[detect_group_col(df_v3_imp)].astype(str).to_numpy() if detect_group_col(df_v3_imp) else np.arange(len(df_v3_imp)).astype(str)

vent_scaler, [Xv_train, Xv_test, Xv_v3_scaled] = scale_vent_train_only(Xv_train_raw, [Xv_test_raw, Xv_v3_raw])
y_scaler, [y_train, y_test, y_v3_scaled] = scale_targets_train_only(y_train_raw, [y_test_raw, y_v3])

joblib.dump(vent_fill_values, os.path.join(OUT_DIR, "vent_fill_values.joblib"))
joblib.dump(vent_scaler, os.path.join(OUT_DIR, "vent_scaler.joblib"))
joblib.dump(y_scaler, os.path.join(OUT_DIR, "y_scaler.joblib"))

results_rows = []
predictions = {}
ensemble_members = {}
run_metadata = {
    "encoder_tag": ENCODER_TAG,
    "cxr_reducer_candidates": CXR_REDUCER_CANDIDATES,
    "distillation_candidates": DISTILLATION_CANDIDATES,
    "final_val_strategy": FINAL_VAL_STRATEGY,
    "final_val_ratio": FINAL_VAL_RATIO,
    "final_train_seeds": FINAL_TRAIN_SEEDS,
    "single_family_predictions_are_seed_ensembles": True,
    "ensemble_top_k": ENSEMBLE_TOP_K,
}
baseline_family_preds = {}
teacher_family_preds = {}
student_family_preds = {}

for family in MODEL_FAMILIES:
    teacher_choice = best_teacher[family]
    student_choice = best_student[family]
    reducer_cfg = teacher_choice["reducer_cfg"]
    img_scaler, img_reducer, [Xi_train, Xi_test] = fit_cxr_transform_train_only(Xi_train_raw, [Xi_test_raw], reducer_cfg=reducer_cfg)
    joblib.dump(img_scaler, os.path.join(OUT_DIR, f"img_scaler_{family}.joblib"))
    if img_reducer["name"] == "pca":
        joblib.dump(img_reducer["model"], os.path.join(OUT_DIR, f"img_reducer_{family}_pca.joblib"))
    else:
        torch.save(img_reducer, os.path.join(OUT_DIR, f"img_reducer_{family}_{img_reducer['name']}.pt"))

    baseline_test_preds, baseline_v3_preds = [], []
    teacher_test_preds, student_test_preds, student_v3_preds = [], [], []

    for run_seed in FINAL_TRAIN_SEEDS:
        set_seed(run_seed)
        if FINAL_VAL_STRATEGY == "holdout":
            inner_tr_idx, inner_va_idx = get_inner_train_val_split(df=df_train, y=y_train_raw, groups=groups_train, val_ratio=FINAL_VAL_RATIO, seed=run_seed)
            Xv_tr, Xv_va = Xv_train[inner_tr_idx], Xv_train[inner_va_idx]
            Xi_tr, Xi_va = Xi_train[inner_tr_idx], Xi_train[inner_va_idx]
            y_tr, y_va = y_train[inner_tr_idx], y_train[inner_va_idx]
            baseline_va_loader = DataLoader(VentOnlyDataset(Xv_va, y_va), batch_size=BATCH_SIZE, shuffle=False)
            multi_va_loader = DataLoader(MultiDataset(Xv_va, Xi_va, y_va), batch_size=BATCH_SIZE, shuffle=False)
        else:
            Xv_tr, Xi_tr, y_tr = Xv_train, Xi_train, y_train
            baseline_va_loader = None
            multi_va_loader = None

        baseline_tr_loader = DataLoader(VentOnlyDataset(Xv_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
        multi_tr_loader = DataLoader(MultiDataset(Xv_tr, Xi_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

        baseline_cfg = best_baseline[family]["cfg"]
        teacher_cfg = teacher_choice["cfg"]
        teacher_fusion = teacher_choice["fusion"]
        student_cfg = {**student_choice["cfg"], **student_choice["distill_cfg"]}

        baseline_model = train_baseline(build_baseline_model(family, VENT_DIM, baseline_cfg), baseline_tr_loader, baseline_va_loader, baseline_cfg).model
        teacher_model = train_teacher(build_teacher_model(family, teacher_fusion, VENT_DIM, Xi_train.shape[1], teacher_cfg), multi_tr_loader, multi_va_loader, teacher_cfg).model
        student_model = train_student(build_baseline_model(family, VENT_DIM, student_choice["cfg"]), teacher_model, multi_tr_loader, multi_va_loader, student_cfg).model

        baseline_test_preds.append(y_scaler.inverse_transform(predict_baseline(baseline_model, Xv_test)).ravel())
        baseline_v3_preds.append(y_scaler.inverse_transform(predict_baseline(baseline_model, Xv_v3_scaled)).ravel())
        teacher_test_preds.append(y_scaler.inverse_transform(predict_teacher(teacher_model, Xv_test, Xi_test)).ravel())
        student_test_preds.append(y_scaler.inverse_transform(predict_baseline(student_model, Xv_test)).ravel())
        student_v3_preds.append(y_scaler.inverse_transform(predict_baseline(student_model, Xv_v3_scaled)).ravel())

    baseline_test = mean_stack(baseline_test_preds)
    baseline_v3 = mean_stack(baseline_v3_preds)
    teacher_test = mean_stack(teacher_test_preds)
    student_test = mean_stack(student_test_preds)
    student_v3 = mean_stack(student_v3_preds)
    baseline_family_preds[family] = {"test": baseline_test, "v3": baseline_v3}
    teacher_family_preds[family] = {"test": teacher_test}
    student_family_preds[family] = {"test": student_test, "v3": student_v3}

    test_m = regression_metrics(y_test_raw, baseline_test)
    v3_m = regression_metrics(y_v3, baseline_v3)
    results_rows.append({"experiment": "baseline", "family": family, "fusion": "vent_only", "ensemble_type": "seed", "train_seeds": len(FINAL_TRAIN_SEEDS), **{f"test_{k}": v for k, v in test_m.items()}, **{f"v3_{k}": v for k, v in v3_m.items()}})
    predictions[f"baseline_{family}"] = {"test_true": y_test_raw.tolist(), "test_pred": baseline_test.tolist(), "v3_true": y_v3.tolist(), "v3_pred": baseline_v3.tolist()}

    test_m = regression_metrics(y_test_raw, teacher_test)
    results_rows.append({"experiment": "teacher", "family": family, "fusion": teacher_choice["fusion"], "ensemble_type": "seed", "cxr_reducer": reducer_cfg["name"], "cxr_latent_dim": reducer_cfg.get("latent_dim"), "train_seeds": len(FINAL_TRAIN_SEEDS), **{f"test_{k}": v for k, v in test_m.items()}})
    predictions[f"teacher_{family}_{teacher_choice['fusion']}"] = {"test_true": y_test_raw.tolist(), "test_pred": teacher_test.tolist()}

    test_m = regression_metrics(y_test_raw, student_test)
    v3_m = regression_metrics(y_v3, student_v3)
    results_rows.append({"experiment": "student", "family": family, "fusion": "vent_only_distilled", "ensemble_type": "seed", "teacher_fusion": teacher_choice["fusion"], "cxr_reducer": reducer_cfg["name"], "cxr_latent_dim": reducer_cfg.get("latent_dim"), "alpha": student_choice["distill_cfg"]["alpha"], "beta": student_choice["distill_cfg"]["beta"], "train_seeds": len(FINAL_TRAIN_SEEDS), **{f"test_{k}": v for k, v in test_m.items()}, **{f"v3_{k}": v for k, v in v3_m.items()}})
    predictions[f"student_{family}"] = {"test_true": y_test_raw.tolist(), "test_pred": student_test.tolist(), "v3_true": y_v3.tolist(), "v3_pred": student_v3.tolist()}

baseline_top_families = sorted(MODEL_FAMILIES, key=lambda fam: best_baseline[fam]["cv_rmse"])[:ENSEMBLE_TOP_K]
baseline_test_ensemble = mean_stack([baseline_family_preds[fam]["test"] for fam in baseline_top_families])
baseline_v3_ensemble = mean_stack([baseline_family_preds[fam]["v3"] for fam in baseline_top_families])
results_rows.append({"experiment": "baseline_ensemble", "family": "+".join(baseline_top_families), "fusion": "vent_only", "ensemble_type": "family", "ensemble_size": len(baseline_top_families), "train_seeds": len(FINAL_TRAIN_SEEDS), **{f"test_{k}": v for k, v in regression_metrics(y_test_raw, baseline_test_ensemble).items()}, **{f"v3_{k}": v for k, v in regression_metrics(y_v3, baseline_v3_ensemble).items()}})
predictions["baseline_family_ensemble"] = {"members": baseline_top_families, "test_true": y_test_raw.tolist(), "test_pred": baseline_test_ensemble.tolist(), "v3_true": y_v3.tolist(), "v3_pred": baseline_v3_ensemble.tolist()}
ensemble_members["baseline_family_ensemble"] = baseline_top_families

teacher_top_families = sorted(MODEL_FAMILIES, key=lambda fam: best_teacher[fam]["cv_rmse"])[:ENSEMBLE_TOP_K]
teacher_test_ensemble = mean_stack([teacher_family_preds[fam]["test"] for fam in teacher_top_families])
results_rows.append({"experiment": "teacher_ensemble", "family": "+".join(teacher_top_families), "fusion": "multimodal", "ensemble_type": "family", "ensemble_size": len(teacher_top_families), "train_seeds": len(FINAL_TRAIN_SEEDS), **{f"test_{k}": v for k, v in regression_metrics(y_test_raw, teacher_test_ensemble).items()}})
predictions["teacher_family_ensemble"] = {"members": teacher_top_families, "test_true": y_test_raw.tolist(), "test_pred": teacher_test_ensemble.tolist()}
ensemble_members["teacher_family_ensemble"] = teacher_top_families

student_top_families = sorted(MODEL_FAMILIES, key=lambda fam: best_student[fam]["cv_rmse"])[:ENSEMBLE_TOP_K]
student_test_ensemble = mean_stack([student_family_preds[fam]["test"] for fam in student_top_families])
student_v3_ensemble = mean_stack([student_family_preds[fam]["v3"] for fam in student_top_families])
results_rows.append({"experiment": "student_ensemble", "family": "+".join(student_top_families), "fusion": "vent_only_distilled", "ensemble_type": "family", "ensemble_size": len(student_top_families), "train_seeds": len(FINAL_TRAIN_SEEDS), **{f"test_{k}": v for k, v in regression_metrics(y_test_raw, student_test_ensemble).items()}, **{f"v3_{k}": v for k, v in regression_metrics(y_v3, student_v3_ensemble).items()}})
predictions["student_family_ensemble"] = {"members": student_top_families, "test_true": y_test_raw.tolist(), "test_pred": student_test_ensemble.tolist(), "v3_true": y_v3.tolist(), "v3_pred": student_v3_ensemble.tolist()}
ensemble_members["student_family_ensemble"] = student_top_families

results_df = pd.DataFrame(results_rows)
results_df.to_csv(os.path.join(OUT_DIR, "all_results_summary.csv"), index=False)

with open(os.path.join(OUT_DIR, "all_predictions.json"), "w") as f:
    json.dump(predictions, f)
with open(os.path.join(OUT_DIR, "ensemble_members.json"), "w") as f:
    json.dump(ensemble_members, f, indent=2)
with open(os.path.join(OUT_DIR, "run_metadata.json"), "w") as f:
    json.dump(run_metadata, f, indent=2)

posthoc_dir = os.path.join(OUT_DIR, "posthoc_analysis")
os.makedirs(posthoc_dir, exist_ok=True)
for key, payload in predictions.items():
    if "test_pred" in payload:
        save_regression_posthoc(f"{key}_test", np.asarray(payload["test_true"]), np.asarray(payload["test_pred"]), test_groups, posthoc_dir)
    if "v3_pred" in payload:
        save_regression_posthoc(f"{key}_v3", np.asarray(payload["v3_true"]), np.asarray(payload["v3_pred"]), v3_groups, posthoc_dir)

if RUN_LEARNING_CURVES:
    learning_rows = []
    learning_specs = [
        ("baseline", min(best_baseline, key=lambda fam: best_baseline[fam]["cv_rmse"])),
        ("teacher", min(best_teacher, key=lambda fam: best_teacher[fam]["cv_rmse"])),
        ("student", min(best_student, key=lambda fam: best_student[fam]["cv_rmse"])),
    ]
    for experiment_type, family in learning_specs:
        reducer_cfg = best_teacher[family]["reducer_cfg"] if experiment_type in {"teacher", "student"} else None
        distill_cfg = best_student[family]["distill_cfg"] if experiment_type == "student" else None
        for frac in LEARNING_FRACTIONS:
            for learn_seed in LEARNING_SEEDS:
                test_pred, v3_pred = train_eval_single_run_regression(experiment_type, family, learn_seed, reducer_cfg=reducer_cfg, distill_cfg=distill_cfg, fraction=frac)
                row = {"experiment": experiment_type, "family": family, "train_fraction": frac, "seed": learn_seed, **{f"test_{k}": v for k, v in regression_metrics(y_test_raw, test_pred).items()}}
                if v3_pred is not None:
                    row.update({f"v3_{k}": v for k, v in regression_metrics(y_v3, v3_pred).items()})
                learning_rows.append(row)
    pd.DataFrame(learning_rows).to_csv(os.path.join(OUT_DIR, "learning_curve_results.csv"), index=False)

if RUN_CXR_REDUCER_SENSITIVITY:
    sensitivity_rows = []
    teacher_family = min(best_teacher, key=lambda fam: best_teacher[fam]["cv_rmse"])
    student_family = min(best_student, key=lambda fam: best_student[fam]["cv_rmse"])
    for reducer_cfg in CXR_REDUCER_SENSITIVITY_CANDIDATES:
        test_pred, _ = train_eval_single_run_regression("teacher", teacher_family, SEED, reducer_cfg=reducer_cfg, fraction=1.0)
        sensitivity_rows.append({"experiment": "teacher", "family": teacher_family, "reducer": reducer_cfg["name"], "latent_dim": reducer_cfg.get("latent_dim"), **{f"test_{k}": v for k, v in regression_metrics(y_test_raw, test_pred).items()}})
        test_pred, v3_pred = train_eval_single_run_regression("student", student_family, SEED, reducer_cfg=reducer_cfg, distill_cfg=best_student[student_family]["distill_cfg"], fraction=1.0)
        sensitivity_rows.append({"experiment": "student", "family": student_family, "reducer": reducer_cfg["name"], "latent_dim": reducer_cfg.get("latent_dim"), **{f"test_{k}": v for k, v in regression_metrics(y_test_raw, test_pred).items()}, **({f"v3_{k}": v for k, v in regression_metrics(y_v3, v3_pred).items()} if v3_pred is not None else {})})
    pd.DataFrame(sensitivity_rows).to_csv(os.path.join(OUT_DIR, "cxr_reducer_sensitivity.csv"), index=False)

print("\n================ FINAL SUMMARY ================")
print(results_df)
print("\nSaved to:", OUT_DIR)
