#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# =========================
# Optional Mamba
# =========================
HAS_MAMBA = True
try:
    from mamba_ssm import Mamba
except Exception:
    HAS_MAMBA = False
    print("WARNING: mamba_ssm not available. Mamba experiments will be skipped.")

# =========================
# Config
# =========================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

V1V2_MULTI_PATH = "/nfs/turbo/med-kayvan-lab/Projects/PARDS/02-Data/PARDS_Risk_V3/v1v2_multimodal_dataset.csv"
V3_PATH = "/nfs/turbo/med-kayvan-lab/Projects/PARDS/02-Data/PARDS_Risk_V3/PARDS_Risk_V3_df.xlsx"
V3_SHEET = "Sheet15"

OUT_DIR = "/nfs/turbo/med-kayvan-lab/Projects/PARDS/04-Results/PARDS_Risk_V3/TRAM_Regression_VentALL_CXR"
os.makedirs(OUT_DIR, exist_ok=True)

TEST_SIZE = 0.20
N_FOLDS = 5

CXR_PCA_FIT_DIM = 64
CXR_PCA_USE_DIM = 32

EPOCHS = 120
PATIENCE = 12
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4

ALPHA = 0.7
BETA = 0.3

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

MODEL_FAMILIES = ["MLP", "GRU", "Transformer"] + (["Mamba"] if HAS_MAMBA else [])

if FAST_DEBUG:
    PARAM_GRID = {
        "MLP": [{"hidden_dim": 64, "dropout": 0.2, "num_layers": 2, "lr": 1e-3}],
        "GRU": [{"hidden_dim": 64, "dropout": 0.2, "num_layers": 1, "lr": 1e-3}],
        "Transformer": [{"hidden_dim": 64, "dropout": 0.2, "num_layers": 1, "lr": 1e-3}],
        "Mamba": [{"hidden_dim": 64, "dropout": 0.2, "num_layers": 1, "lr": 1e-3}] if HAS_MAMBA else [],
    }
else:
    PARAM_GRID = {
        "MLP": [
            {"hidden_dim": 64, "dropout": 0.0, "num_layers": 2, "lr": 1e-3},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 2, "lr": 1e-3},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 3, "lr": 3e-4},
        ],
        "GRU": [
            {"hidden_dim": 64, "dropout": 0.0, "num_layers": 1, "lr": 1e-3},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 1, "lr": 1e-3},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 2, "lr": 3e-4},
        ],
        "Transformer": [
            {"hidden_dim": 64, "dropout": 0.2, "num_layers": 1, "lr": 1e-3},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 1, "lr": 1e-3},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 2, "lr": 3e-4},
        ],
        "Mamba": [
            {"hidden_dim": 64, "dropout": 0.2, "num_layers": 1, "lr": 1e-3},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 1, "lr": 1e-3},
            {"hidden_dim": 128, "dropout": 0.2, "num_layers": 2, "lr": 3e-4},
        ] if HAS_MAMBA else [],
    }

# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# =========================
# Helpers
# =========================
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
        outs.append(
            scaler.transform(Xo.reshape(-1, D)).reshape(No, T, D).astype(np.float32)
        )
    return scaler, outs

def fit_cxr_pca_train_only(X_train_img, X_other_list, fit_dim=64, use_dim=32):
    img_scaler = StandardScaler()
    X_train_img_s = img_scaler.fit_transform(X_train_img)

    pca = PCA(n_components=fit_dim, random_state=SEED)
    X_train_img_p = pca.fit_transform(X_train_img_s)[:, :use_dim].astype(np.float32)

    outs = [X_train_img_p]
    for Xo in X_other_list:
        Xo_s = img_scaler.transform(Xo)
        outs.append(pca.transform(Xo_s)[:, :use_dim].astype(np.float32))

    return img_scaler, pca, outs

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def regression_metrics(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "N": int(len(y_true)),
    }

def scale_targets_train_only(y_train, y_other_list):
    scaler = StandardScaler()
    scaler.fit(y_train.reshape(-1, 1))
    y_train_s = scaler.transform(y_train.reshape(-1, 1)).astype(np.float32)

    outs = [y_train_s]
    for yo in y_other_list:
        outs.append(scaler.transform(yo.reshape(-1, 1)).astype(np.float32))
    return scaler, outs

# =========================
# Datasets
# =========================
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

# =========================
# Blocks
# =========================
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
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
        )
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

# =========================
# Vent-only models
# =========================
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

class VentGRU(nn.Module):
    def __init__(self, vent_dim_per_tw, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        if num_layers == 1:
            dropout = 0.0
        self.gru = nn.GRU(vent_dim_per_tw, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.pool = AttentionPool(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)

    def forward(self, xv):
        h, _ = self.gru(xv)
        return self.fc(self.pool(h))

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

# =========================
# Teacher models
# =========================
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

class EarlyFusionGRU(nn.Module):
    def __init__(self, vent_dim_per_tw, img_dim=32, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        if num_layers == 1:
            dropout = 0.0
        self.gru = nn.GRU(vent_dim_per_tw + img_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.pool = AttentionPool(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)

    def forward(self, xv, xi):
        xi_rep = xi.unsqueeze(1).repeat(1, xv.size(1), 1)
        h, _ = self.gru(torch.cat([xv, xi_rep], dim=-1))
        return self.fc(self.pool(h))

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

class CrossFusionGRU(nn.Module):
    def __init__(self, vent_dim_per_tw, img_dim=32, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        if num_layers == 1:
            dropout = 0.0
        self.vent_encoder = nn.GRU(vent_dim_per_tw, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
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
        h, _ = self.vent_encoder(xv)
        v = self.vent_pool(h)
        i = self.img_branch(xi)
        return self.head(torch.cat([v, i, v * i], dim=1))

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
    if family == "GRU":
        return VentGRU(vent_dim_per_tw, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    if family == "Transformer":
        return VentTransformer(vent_dim_per_tw, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    if family == "Mamba":
        return VentMamba(vent_dim_per_tw, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    raise ValueError(f"Unknown family: {family}")

def build_teacher_model(family, fusion_type, vent_dim_per_tw, img_dim, cfg):
    if fusion_type == "early":
        if family == "MLP":
            return EarlyFusionMLP(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
        if family == "GRU":
            return EarlyFusionGRU(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
        if family == "Transformer":
            return EarlyFusionTransformer(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
        if family == "Mamba":
            return EarlyFusionMamba(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    if fusion_type == "cross":
        if family == "MLP":
            return CrossFusionMLP(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
        if family == "GRU":
            return CrossFusionGRU(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
        if family == "Transformer":
            return CrossFusionTransformer(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
        if family == "Mamba":
            return CrossFusionMamba(vent_dim_per_tw, img_dim=img_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    raise ValueError(f"Unknown family/fusion: {family}, {fusion_type}")

# =========================
# Train helpers
# =========================
@dataclass
class FitResult:
    model: nn.Module
    best_val_loss: float

def train_baseline(model, train_loader, val_loader, cfg):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
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

    optimizer = torch.optim.Adam(student.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()

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
            loss = ALPHA * hard_loss + BETA * soft_loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

        student.eval()
        val_losses = []
        with torch.no_grad():
            for xv, xi, y in val_loader:
                xv, xi, y = xv.to(DEVICE), xi.to(DEVICE), y.to(DEVICE)
                t_pred = teacher(xv, xi)
                s_pred = student(xv)

                hard_loss = mse(s_pred, y)
                soft_loss = mse(s_pred, t_pred)
                val_losses.append((ALPHA * hard_loss + BETA * soft_loss).item())

        val_loss = float(np.mean(val_losses))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    student.load_state_dict(best_state)
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

# =========================
# Load data
# =========================
df_multi = pd.read_csv(V1V2_MULTI_PATH)
df_v3 = pd.read_excel(V3_PATH, sheet_name=V3_SHEET)

if "MRN" in df_multi.columns:
    df_multi["MRN"] = mrn_9(df_multi["MRN"])
if "MRN" in df_v3.columns:
    df_v3["MRN"] = mrn_9(df_v3["MRN"])

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

print("Original V1V2 rows:", len(df_multi))
print("Original V3 rows  :", len(df_v3))
print("After target+CXR filter V1V2:", int(valid_multi_mask.sum()))
print("After target filter V3      :", int(valid_v3_mask.sum()))

df_multi = df_multi.loc[valid_multi_mask].copy().reset_index(drop=True)
df_v3 = df_v3.loc[valid_v3_mask].copy().reset_index(drop=True)

group_col_train = detect_group_col(df_multi)
group_col_v3 = detect_group_col(df_v3)

groups_multi = (
    df_multi[group_col_train].astype(str).to_numpy()
    if group_col_train else np.arange(len(df_multi)).astype(str)
)
groups_v3 = (
    df_v3[group_col_v3].astype(str).to_numpy()
    if group_col_v3 else np.arange(len(df_v3)).astype(str)
)

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
train_idx, test_idx = next(gss.split(df_multi, df_multi[target_train].to_numpy(), groups=groups_multi))

df_train_raw = df_multi.iloc[train_idx].copy().reset_index(drop=True)
df_test_raw = df_multi.iloc[test_idx].copy().reset_index(drop=True)

groups_train = (
    df_train_raw[group_col_train].astype(str).to_numpy()
    if group_col_train else np.arange(len(df_train_raw)).astype(str)
)
groups_test = (
    df_test_raw[group_col_train].astype(str).to_numpy()
    if group_col_train else np.arange(len(df_test_raw)).astype(str)
)

print("Train rows:", len(df_train_raw), "Test rows:", len(df_test_raw))

VENT_DIM = len(tw_features[1])

# =========================
# CV helpers
# =========================
def prepare_fold_data(df_tr_raw, df_va_raw):
    vent_fill = fit_vent_imputer_train_only(df_tr_raw)
    df_tr = apply_vent_imputer(df_tr_raw, vent_fill)
    df_va = apply_vent_imputer(df_va_raw, vent_fill)

    Xv_tr_raw = build_vent_seq_from_df(df_tr)
    Xv_va_raw = build_vent_seq_from_df(df_va)

    _, [Xv_tr, Xv_va] = scale_vent_train_only(Xv_tr_raw, [Xv_va_raw])

    Xi_tr_raw = df_tr[cxr_emb_cols].to_numpy(dtype=np.float32)
    Xi_va_raw = df_va[cxr_emb_cols].to_numpy(dtype=np.float32)
    _, _, [Xi_tr, Xi_va] = fit_cxr_pca_train_only(
        Xi_tr_raw, [Xi_va_raw], fit_dim=CXR_PCA_FIT_DIM, use_dim=CXR_PCA_USE_DIM
    )

    y_tr_raw = df_tr[target_train].to_numpy(dtype=np.float32)
    y_va_raw = df_va[target_train].to_numpy(dtype=np.float32)
    y_scaler, [y_tr, y_va] = scale_targets_train_only(y_tr_raw, [y_va_raw])

    return {
        "Xv_tr": Xv_tr,
        "Xv_va": Xv_va,
        "Xi_tr": Xi_tr,
        "Xi_va": Xi_va,
        "y_tr": y_tr,
        "y_va": y_va,
        "y_tr_raw": y_tr_raw,
        "y_va_raw": y_va_raw,
        "y_scaler": y_scaler,
    }

def evaluate_cv_baseline(family, cfg, df_train_raw, groups_train):
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_rmses = []
    dummy_y = df_train_raw[target_train].to_numpy(dtype=np.float32)

    for tr_i, va_i in gkf.split(df_train_raw, dummy_y, groups=groups_train):
        df_tr_raw = df_train_raw.iloc[tr_i].copy()
        df_va_raw = df_train_raw.iloc[va_i].copy()

        prep = prepare_fold_data(df_tr_raw, df_va_raw)

        tr_loader = DataLoader(
            VentOnlyDataset(prep["Xv_tr"], prep["y_tr"]),
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        va_loader = DataLoader(
            VentOnlyDataset(prep["Xv_va"], prep["y_va"]),
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        model = build_baseline_model(family, VENT_DIM, cfg)
        result = train_baseline(model, tr_loader, va_loader, cfg)

        pred_scaled = predict_baseline(result.model, prep["Xv_va"])
        pred = prep["y_scaler"].inverse_transform(pred_scaled).ravel()
        truth = prep["y_va_raw"].ravel()

        fold_rmses.append(rmse(truth, pred))

    return float(np.mean(fold_rmses)), float(np.std(fold_rmses))

def evaluate_cv_teacher(family, fusion_type, cfg, df_train_raw, groups_train):
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_rmses = []
    dummy_y = df_train_raw[target_train].to_numpy(dtype=np.float32)

    for tr_i, va_i in gkf.split(df_train_raw, dummy_y, groups=groups_train):
        df_tr_raw = df_train_raw.iloc[tr_i].copy()
        df_va_raw = df_train_raw.iloc[va_i].copy()

        prep = prepare_fold_data(df_tr_raw, df_va_raw)

        tr_loader = DataLoader(
            MultiDataset(prep["Xv_tr"], prep["Xi_tr"], prep["y_tr"]),
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        va_loader = DataLoader(
            MultiDataset(prep["Xv_va"], prep["Xi_va"], prep["y_va"]),
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        model = build_teacher_model(family, fusion_type, VENT_DIM, CXR_PCA_USE_DIM, cfg)
        result = train_teacher(model, tr_loader, va_loader, cfg)

        pred_scaled = predict_teacher(result.model, prep["Xv_va"], prep["Xi_va"])
        pred = prep["y_scaler"].inverse_transform(pred_scaled).ravel()
        truth = prep["y_va_raw"].ravel()

        fold_rmses.append(rmse(truth, pred))

    return float(np.mean(fold_rmses)), float(np.std(fold_rmses))

# =========================
# Tune baselines
# =========================
baseline_cv_rows = []
best_baseline = {}

for family in MODEL_FAMILIES:
    best_rmse, best_cfg = np.inf, None
    for cfg in PARAM_GRID[family]:
        mean_rmse, std_rmse = evaluate_cv_baseline(family, cfg, df_train_raw, groups_train)
        baseline_cv_rows.append({
            "family": family,
            "hidden_dim": cfg["hidden_dim"],
            "dropout": cfg["dropout"],
            "num_layers": cfg["num_layers"],
            "lr": cfg["lr"],
            "cv_rmse_mean": mean_rmse,
            "cv_rmse_std": std_rmse,
        })
        print(f"[BASELINE][{family}] {cfg} -> RMSE {mean_rmse:.4f} ± {std_rmse:.4f}")
        if mean_rmse < best_rmse:
            best_rmse, best_cfg = mean_rmse, cfg

    best_baseline[family] = {"cfg": best_cfg, "cv_rmse": best_rmse}

pd.DataFrame(baseline_cv_rows).to_csv(os.path.join(OUT_DIR, "baseline_cv_results.csv"), index=False)

# =========================
# Tune teachers
# =========================
teacher_cv_rows = []
best_teacher = {}

for family in MODEL_FAMILIES:
    best_rmse, best_cfg, best_fusion = np.inf, None, None
    for fusion_type in ["early", "cross"]:
        for cfg in PARAM_GRID[family]:
            mean_rmse, std_rmse = evaluate_cv_teacher(family, fusion_type, cfg, df_train_raw, groups_train)
            teacher_cv_rows.append({
                "family": family,
                "fusion": fusion_type,
                "hidden_dim": cfg["hidden_dim"],
                "dropout": cfg["dropout"],
                "num_layers": cfg["num_layers"],
                "lr": cfg["lr"],
                "cv_rmse_mean": mean_rmse,
                "cv_rmse_std": std_rmse,
            })
            print(f"[TEACHER][{family}][{fusion_type}] {cfg} -> RMSE {mean_rmse:.4f} ± {std_rmse:.4f}")
            if mean_rmse < best_rmse:
                best_rmse, best_cfg, best_fusion = mean_rmse, cfg, fusion_type

    best_teacher[family] = {"cfg": best_cfg, "fusion": best_fusion, "cv_rmse": best_rmse}

pd.DataFrame(teacher_cv_rows).to_csv(os.path.join(OUT_DIR, "teacher_cv_results.csv"), index=False)

with open(os.path.join(OUT_DIR, "best_baseline.json"), "w") as f:
    json.dump(best_baseline, f, indent=2)
with open(os.path.join(OUT_DIR, "best_teacher.json"), "w") as f:
    json.dump(best_teacher, f, indent=2)

# =========================
# Final preprocessors
# =========================
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

vent_scaler, [Xv_train, Xv_test, Xv_v3_scaled] = scale_vent_train_only(
    Xv_train_raw, [Xv_test_raw, Xv_v3_raw]
)
img_scaler, img_pca, [Xi_train, Xi_test] = fit_cxr_pca_train_only(
    Xi_train_raw, [Xi_test_raw], fit_dim=CXR_PCA_FIT_DIM, use_dim=CXR_PCA_USE_DIM
)
y_scaler, [y_train, y_test, y_v3_scaled] = scale_targets_train_only(
    y_train_raw, [y_test_raw, y_v3]
)

joblib.dump(vent_fill_values, os.path.join(OUT_DIR, "vent_fill_values.joblib"))
joblib.dump(vent_scaler, os.path.join(OUT_DIR, "vent_scaler.joblib"))
joblib.dump(img_scaler, os.path.join(OUT_DIR, "img_scaler.joblib"))
joblib.dump(img_pca, os.path.join(OUT_DIR, "img_pca_fit64_use32.joblib"))
joblib.dump(y_scaler, os.path.join(OUT_DIR, "y_scaler.joblib"))

inner_gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
inner_tr_idx, inner_va_idx = next(
    inner_gss.split(df_train, y_train_raw, groups=groups_train)
)

Xv_tr, Xv_va = Xv_train[inner_tr_idx], Xv_train[inner_va_idx]
Xi_tr, Xi_va = Xi_train[inner_tr_idx], Xi_train[inner_va_idx]
y_tr, y_va = y_train[inner_tr_idx], y_train[inner_va_idx]

# =========================
# Final baselines
# =========================
results_rows = []
predictions = {}

for family in MODEL_FAMILIES:
    cfg = best_baseline[family]["cfg"]

    tr_loader = DataLoader(VentOnlyDataset(Xv_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(VentOnlyDataset(Xv_va, y_va), batch_size=BATCH_SIZE, shuffle=False)

    model = build_baseline_model(family, VENT_DIM, cfg)
    fit = train_baseline(model, tr_loader, va_loader, cfg)
    model = fit.model

    pred_test_scaled = predict_baseline(model, Xv_test)
    pred_test = y_scaler.inverse_transform(pred_test_scaled).ravel()

    pred_v3_scaled = predict_baseline(model, Xv_v3_scaled)
    pred_v3 = y_scaler.inverse_transform(pred_v3_scaled).ravel()

    test_m = regression_metrics(y_test_raw, pred_test)
    v3_m = regression_metrics(y_v3, pred_v3)

    results_rows.append({
        "experiment": "baseline",
        "family": family,
        "fusion": "vent_only",
        **{f"test_{k}": v for k, v in test_m.items()},
        **{f"v3_{k}": v for k, v in v3_m.items()},
    })

    predictions[f"baseline_{family}"] = {
        "test_true": y_test_raw.tolist(),
        "test_pred": pred_test.tolist(),
        "v3_true": y_v3.tolist(),
        "v3_pred": pred_v3.tolist(),
    }

    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"baseline_{family}.pt"))

# =========================
# Final teachers
# =========================
teacher_models = {}

for family in MODEL_FAMILIES:
    cfg = best_teacher[family]["cfg"]
    fusion = best_teacher[family]["fusion"]

    tr_loader = DataLoader(MultiDataset(Xv_tr, Xi_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(MultiDataset(Xv_va, Xi_va, y_va), batch_size=BATCH_SIZE, shuffle=False)

    model = build_teacher_model(family, fusion, VENT_DIM, CXR_PCA_USE_DIM, cfg)
    fit = train_teacher(model, tr_loader, va_loader, cfg)
    model = fit.model
    teacher_models[family] = model

    pred_test_scaled = predict_teacher(model, Xv_test, Xi_test)
    pred_test = y_scaler.inverse_transform(pred_test_scaled).ravel()
    test_m = regression_metrics(y_test_raw, pred_test)

    results_rows.append({
        "experiment": "teacher",
        "family": family,
        "fusion": fusion,
        **{f"test_{k}": v for k, v in test_m.items()},
    })

    predictions[f"teacher_{family}_{fusion}"] = {
        "test_true": y_test_raw.tolist(),
        "test_pred": pred_test.tolist(),
    }

    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"teacher_{family}_{fusion}.pt"))

# =========================
# Final students
# =========================
for family in MODEL_FAMILIES:
    teacher = teacher_models[family]
    cfg = best_teacher[family]["cfg"]

    tr_loader = DataLoader(MultiDataset(Xv_tr, Xi_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(MultiDataset(Xv_va, Xi_va, y_va), batch_size=BATCH_SIZE, shuffle=False)

    student = build_baseline_model(family, VENT_DIM, cfg)
    fit = train_student(student, teacher, tr_loader, va_loader, cfg)
    student = fit.model

    pred_test_scaled = predict_baseline(student, Xv_test)
    pred_test = y_scaler.inverse_transform(pred_test_scaled).ravel()

    pred_v3_scaled = predict_baseline(student, Xv_v3_scaled)
    pred_v3 = y_scaler.inverse_transform(pred_v3_scaled).ravel()

    test_m = regression_metrics(y_test_raw, pred_test)
    v3_m = regression_metrics(y_v3, pred_v3)

    results_rows.append({
        "experiment": "student",
        "family": family,
        "fusion": "vent_only_distilled",
        **{f"test_{k}": v for k, v in test_m.items()},
        **{f"v3_{k}": v for k, v in v3_m.items()},
    })

    predictions[f"student_{family}"] = {
        "test_true": y_test_raw.tolist(),
        "test_pred": pred_test.tolist(),
        "v3_true": y_v3.tolist(),
        "v3_pred": pred_v3.tolist(),
    }

    torch.save(student.state_dict(), os.path.join(OUT_DIR, f"student_{family}.pt"))

# =========================
# Save
# =========================
results_df = pd.DataFrame(results_rows)
results_df.to_csv(os.path.join(OUT_DIR, "all_results_summary.csv"), index=False)

with open(os.path.join(OUT_DIR, "all_predictions.json"), "w") as f:
    json.dump(predictions, f)

print("\n================ FINAL SUMMARY ================")
print(results_df)
print("\nSaved to:", OUT_DIR)