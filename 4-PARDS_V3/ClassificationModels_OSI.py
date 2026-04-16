# ======================================================================================
# TBME Major Revision Utilities (OSI-only feature set)
# - Save BEST config+weights per model type (RNN/LSTM/GRU/Transformer/Mamba)
# - Save TRAINING CV predictions per config (for downstream ROC/PRC/F1 plots)
# - Evaluate BEST per model on TEST + TEMPORAL VALIDATION and save y_true/y_prob
# - Learning curve (20/40/60/80/100% of training patients)
# - Bootstrap CIs + Sens@fixed specificity + Calibration metrics/curve
#
# NOTE: OSI-only => NO PCA64 (disabled). Keeps the rest the same.
# OSI features per TW: ["OSI_mean_TW{tw}", "OSI_std_TW{tw}"]
# ======================================================================================

import os
import math
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    roc_curve, brier_score_loss
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from mamba_ssm import Mamba

# ======================================================================================
# ================================ USER CONTROLS ======================================
# ======================================================================================

RUN_GRID_SEARCH       = True     # trains many configs and saves best-per-model-type
RUN_EVAL_BEST_MODELS  = True     # runs TEST+VALIDATION for best-per-model-type (no retraining)
RUN_LEARNING_CURVES   = True     # trains best-config per model_type on fractions of patients
RUN_PCA_SENSITIVITY   = False    # OSI-only => disabled (no PCA64)
RUN_BOOTSTRAP_CI_EVAL = True     # compute patient-level CI + sens@spec + calibration on TEST/VAL preds

# For learning curves, you probably don't want to redo all 5 models.
LEARNING_MODELS = ["RNN", "LSTM", "GRU", "Transformer", "Mamba"]   # or ["Mamba"] to keep it light
LEARNING_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
# LEARNING_SEEDS     = [0, 1, 2]              # repeats per fraction
LEARNING_SEEDS     = [0]              # repeats per fraction

# Bootstrap settings
BOOTSTRAP_B = 2000
SPEC_TARGETS = [0.90, 0.95]   # sensitivity at these specificities
CALIB_BINS = 10

# Stratified sampling inside CV (same logic as your previous pipeline)
SUBSAMPLE_RATIO = 0.5
SUBSAMPLE_POS_RATIO = 1/3

# ======================================================================================
# ================================ PATHS / I/O ========================================
# ======================================================================================

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training data
xlsx_train = "/nfs/turbo/med-kayvan-lab/Projects/PARDS/02-Data/PARDS_Risk_V2/PARDS_Risk_RNN/PARDS_Risk_V1V2_df.xlsx"
sheet_train = "Sheet5"

# Temporal validation data
xlsx_val = "/nfs/turbo/med-kayvan-lab/Projects/PARDS/02-Data/PARDS_Risk_V3/PARDS_Risk_V3_df.xlsx"
sheet_val = "Sheet15"

# Output roots
out_root_models = "/nfs/turbo/med-kayvan-lab/Projects/PARDS/04-Results/PARDS_Risk_V2/Fig8_ClassificationModels"
save_root = os.path.join(out_root_models, "SavedModels_OSI_8")  # NEW folder
os.makedirs(save_root, exist_ok=True)

out_revision_dir = "/nfs/turbo/med-kayvan-lab/Projects/PARDS/04-Results/PARDS_Risk_V2/20251105_TBME_Submission/20260301_Major_Revision"
os.makedirs(out_revision_dir, exist_ok=True)

# ======================================================================================
# ================================= DATA LOADING ======================================
# ======================================================================================

target_col = "OSI_V2_12th_avg"
threshold = 7.5

# OSI features per TW (requested format)
osi_features = {tw: [f"OSI_mean_TW{tw}", f"OSI_std_TW{tw}"] for tw in range(1, 7)}

def build_sequences_from_df_osi_only(
    df: pd.DataFrame,
    label_col: str,
    osi_features: dict,
    threshold: float
):
    """
    Build X6 with shape (N, 6, D_osi) where D_osi=2 (mean,std).
    """
    X, y, groups = [], [], []
    for idx, row in df.iterrows():
        seq = []
        ok = True
        for tw in range(1, 7):
            cols = osi_features[tw]
            # If any required feature is missing for this TW, drop this sample
            if any([(c not in df.columns) for c in cols]):
                missing = [c for c in cols if c not in df.columns]
                raise ValueError(f"Missing OSI feature columns in TRAIN sheet: {missing}")
            if row[cols].isnull().any():
                ok = False
                break
            seq.append(row[cols].values.astype(np.float32))
        if ok and (not pd.isna(row[label_col])):
            X.append(seq)
            y.append(1 if float(row[label_col]) >= threshold else 0)
            groups.append(row["ResearchID"] if "ResearchID" in df.columns else idx)
    X = np.array(X, dtype=np.float32)  # (N, 6, 2)
    y = np.array(y, dtype=np.float32)  # (N,)
    groups = np.array(groups)
    return X, y, groups

def add_delta_window(X6: np.ndarray) -> np.ndarray:
    # X6: (N, 6, D)
    delta = (X6[:, -1, :] - X6[:, 0, :])[:, np.newaxis, :]  # (N,1,D)
    return np.concatenate([X6, delta], axis=1)              # (N,7,D)

# Load train
df = pd.read_excel(xlsx_train, sheet_name=sheet_train)
X6, y, groups = build_sequences_from_df_osi_only(df, target_col, osi_features, threshold)
X = add_delta_window(X6)  # (N,7,2)

print(f"✅ Total usable samples: {len(X)} | Positives: {int(y.sum())} | Negatives: {int((1-y).sum())}")
print(f"✅ OSI-only feature dim per TW: {X.shape[2]} (expected 2)")

# ======================================================================================
# ============================= PREPROCESSING HELPERS =================================
# ======================================================================================

def fit_scaler_on_train(X_train: np.ndarray) -> StandardScaler:
    # Fit per-feature scaler on flattened time dimension
    scaler = StandardScaler()
    N, T, D = X_train.shape
    scaler.fit(X_train.reshape(-1, D))
    return scaler

def apply_scaler(scaler: StandardScaler, X_in: np.ndarray) -> np.ndarray:
    N, T, D = X_in.shape
    return scaler.transform(X_in.reshape(-1, D)).reshape(N, T, D)

def augment_data(X_in, y_in, groups_in, noise_std=0.005, seed=0):
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_std, X_in.shape).astype(np.float32)
    return (
        np.concatenate([X_in, X_in + noise]),
        np.concatenate([y_in, y_in]),
        np.concatenate([groups_in, groups_in])
    )

# ======================================================================================
# ================================== MODELS ===========================================
# ======================================================================================

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)  # (B,T,1)
        return torch.sum(w * x, dim=1)          # (B,H)

class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_type="RNN", num_layers=1, dropout=0.0):
        super().__init__()
        if num_layers == 1:
            dropout = 0.0
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.attn = AttentionLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.attn(out)
        return self.fc(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, nhead, dropout):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout,
            batch_first=True, layer_norm_eps=1e-5
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=1)

    def forward(self, x):
        return self.encoder(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0, nhead=None):
        super().__init__()
        if nhead is None:
            if hidden_dim % 8 == 0:
                nhead = max(1, hidden_dim // 8)
            elif hidden_dim % 4 == 0:
                nhead = max(1, hidden_dim // 4)
            else:
                nhead = 1

        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.pe = PositionalEncoding(hidden_dim)

        if num_layers == 1:
            self.stack = nn.Sequential(TransformerBlock(hidden_dim, nhead, 0.0))
        else:
            self.stack = nn.Sequential(*[
                nn.Sequential(TransformerBlock(hidden_dim, nhead, dropout), nn.Dropout(dropout))
                for _ in range(num_layers)
            ])

        self.attn = AttentionLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pe(x)
        x = self.stack(x)
        x = self.attn(x)
        return self.fc(x)

class MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(d_model=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.mamba(x))

class MambaClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        if num_layers == 1:
            self.stack = nn.Sequential(MambaBlock(hidden_dim))
        else:
            self.stack = nn.Sequential(*[
                nn.Sequential(MambaBlock(hidden_dim), nn.Dropout(dropout))
                for _ in range(num_layers)
            ])
        self.attn = AttentionLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.norm(x)
        x = self.stack(x)
        x = self.attn(x)
        return self.fc(x)

def build_model(model_type, input_dim, hidden_dim, num_layers, dropout):
    if model_type == "Mamba":
        return MambaClassifier(input_dim, hidden_dim, num_layers, dropout)
    if model_type == "Transformer":
        return TransformerModel(input_dim, hidden_dim, num_layers, dropout)
    return RNNClassifier(input_dim, hidden_dim, model_type, num_layers, dropout)

def get_loader(X_in, y_in, batch_size, shuffle=True):
    X_tensor = torch.tensor(X_in, dtype=torch.float32)
    y_tensor = torch.tensor(y_in, dtype=torch.float32).view(-1, 1)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)

def auprc_score(y_true, y_prob):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return auc(r, p)

# ======================================================================================
# ============================ CLINICAL METRICS HELPERS ===============================
# ======================================================================================

def sens_at_fixed_spec(y_true, y_prob, spec_target=0.90):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr
    idx = np.where(spec >= spec_target)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    best_i = idx[np.argmax(tpr[idx])]
    return float(tpr[best_i]), float(thr[best_i])

def calibration_curve_bins(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = (bin_ids == b)
        if mask.sum() == 0:
            rows.append((b, 0, np.nan, np.nan))
        else:
            rows.append((b, int(mask.sum()), float(np.mean(y_prob[mask])), float(np.mean(y_true[mask]))))
    return pd.DataFrame(rows, columns=["bin", "count", "mean_pred", "frac_pos"])

def eval_point_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out = {}
    out["AUC"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    out["AUPRC"] = float(auprc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    out["F1"] = float(f1_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan
    out["Brier"] = float(brier_score_loss(y_true, y_prob))
    return out

def bootstrap_ci_metrics(y_true, y_prob, B=2000, seed=0, spec_targets=(0.90,)):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)

    # Point estimates
    point = eval_point_metrics(y_true, y_prob, threshold=0.5)
    for s in spec_targets:
        sens, thr = sens_at_fixed_spec(y_true, y_prob, spec_target=s)
        point[f"Sens@Spec{int(s*100)}"] = sens
        point[f"Thr@Spec{int(s*100)}"] = thr

    dist = {k: [] for k in point.keys()}
    valid = 0

    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        valid += 1
        m = eval_point_metrics(yt, yp, threshold=0.5)
        for s in spec_targets:
            sens, thr = sens_at_fixed_spec(yt, yp, spec_target=s)
            m[f"Sens@Spec{int(s*100)}"] = sens
            m[f"Thr@Spec{int(s*100)}"] = thr
        for k, v in m.items():
            dist[k].append(v)

    rows = []
    for k, v_point in point.items():
        arr = np.asarray(dist[k], dtype=float)
        if arr.size == 0:
            rows.append((k, v_point, np.nan, np.nan, valid))
        else:
            lo = float(np.nanpercentile(arr, 2.5))
            hi = float(np.nanpercentile(arr, 97.5))
            rows.append((k, v_point, lo, hi, valid))
    return pd.DataFrame(rows, columns=["metric", "point", "ci_low", "ci_high", "n_valid_bootstrap"])

# ======================================================================================
# ============================== GRID SEARCH TRAINING =================================
# ======================================================================================

def stratified_sample(y_train, ratio=0.5, pos_ratio=1/3, seed=0):
    rng = np.random.RandomState(seed)
    y_flat = y_train.flatten()
    idx_0 = np.where(y_flat == 0)[0]
    idx_1 = np.where(y_flat == 1)[0]

    total = int(len(y_flat) * ratio)
    n1 = int(total * pos_ratio)
    n0 = total - n1
    if len(idx_0) < n0 or len(idx_1) < n1:
        return None

    idx = np.concatenate([
        rng.choice(idx_0, n0, replace=False),
        rng.choice(idx_1, n1, replace=False)
    ])
    rng.shuffle(idx)
    return idx

def train_one_config_cv(
    model_type, X_train, y_train, groups_train,
    hidden_dim, dropout, num_layers, batch_size, lr, wd, optimizer_name,
    epochs=200, n_repeats=5, n_splits=3, early_stop_patience=10,
    save_cv_records=True
):
    """
    Returns:
      summary: dict
      combo_best_state: best fold weights among all folds (by train loss proxy)
      cv_records: list of dicts (repeat/fold y_true/y_prob + config) if save_cv_records
    """
    expected_metrics = n_repeats * n_splits
    all_aucs, all_auprcs, all_f1s = [], [], []
    used_epochs_all = []
    combo_best_val_loss = np.inf
    combo_best_state = None
    cv_records = []

    for rep in range(n_repeats):
        sampled_idx = stratified_sample(
            y_train, ratio=SUBSAMPLE_RATIO, pos_ratio=SUBSAMPLE_POS_RATIO, seed=rep
        )
        if sampled_idx is None:
            continue

        X_sub = X_train[sampled_idx]
        y_sub = y_train[sampled_idx]
        g_sub = groups_train[sampled_idx]

        gkf = GroupKFold(n_splits=n_splits)
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_sub, y_sub, g_sub)):

            model = build_model(model_type, X_train.shape[2], hidden_dim, num_layers, dropout).to(device)
            opt_cls = torch.optim.Adam if optimizer_name == "Adam" else torch.optim.SGD
            optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
            criterion = nn.BCEWithLogitsLoss()

            loader = get_loader(X_sub[tr_idx], y_sub[tr_idx], batch_size, shuffle=True)

            best_loss = float("inf")
            wait = 0
            used_epochs = 0
            best_state_fold = None

            for _ in range(epochs):
                model.train()
                losses = []
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    losses.append(loss.item())

                avg_loss = float(np.mean(losses)) if len(losses) else np.inf
                scheduler.step(avg_loss)
                used_epochs += 1

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    wait = 0
                    best_state_fold = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= early_stop_patience:
                        break

            used_epochs_all.append(used_epochs)
            if best_state_fold is not None:
                model.load_state_dict(best_state_fold)

            # Validation metrics + optional saving y_true/y_prob for plotting
            model.eval()
            with torch.no_grad():
                val_x = torch.tensor(X_sub[val_idx], dtype=torch.float32).to(device)
                logits = model(val_x).cpu().flatten()
                prob = torch.sigmoid(logits).numpy()
                yt = y_sub[val_idx].flatten().astype(int)

            valid_mask = np.isfinite(prob)
            if valid_mask.sum() == 0:
                continue
            prob = prob[valid_mask]
            yt = yt[valid_mask]

            if len(np.unique(yt)) < 2:
                continue

            all_aucs.append(float(roc_auc_score(yt, prob)))
            all_auprcs.append(float(auprc_score(yt, prob)))
            all_f1s.append(float(f1_score(yt, (prob >= 0.5).astype(int))))

            if save_cv_records:
                cv_records.append({
                    "model": model_type,
                    "hidden_dim": int(hidden_dim),
                    "dropout": float(dropout),
                    "num_layers": int(num_layers),
                    "batch_size": int(batch_size),
                    "lr": float(lr),
                    "weight_decay": float(wd),
                    "optimizer": optimizer_name,
                    "repeat": int(rep),
                    "fold": int(fold),
                    "y_true": yt.tolist(),
                    "y_prob": prob.tolist()
                })

            if best_loss < combo_best_val_loss and best_state_fold is not None:
                combo_best_val_loss = best_loss
                combo_best_state = best_state_fold

    is_complete = (len(all_aucs) == expected_metrics)
    summary = {
        "AUC_mean": float(np.mean(all_aucs)) if is_complete else np.nan,
        "AUC_std":  float(np.std(all_aucs))  if is_complete else np.nan,
        "AUPRC_mean": float(np.mean(all_auprcs)) if is_complete else np.nan,
        "AUPRC_std":  float(np.std(all_auprcs))  if is_complete else np.nan,
        "F1_mean": float(np.mean(all_f1s)) if is_complete else np.nan,
        "F1_std":  float(np.std(all_f1s))  if is_complete else np.nan,
        "n_metrics": int(len(all_aucs)),
        "expected_metrics": int(expected_metrics),
        "used_epochs_mean": float(np.mean(used_epochs_all)) if used_epochs_all else np.nan,
        "used_epochs_std": float(np.std(used_epochs_all)) if used_epochs_all else np.nan,
        "best_val_loss": float(combo_best_val_loss) if np.isfinite(combo_best_val_loss) else np.inf,
        "has_state": combo_best_state is not None
    }
    return summary, combo_best_state, cv_records

# ======================================================================================
# ================================ TRAIN/TEST SPLIT ===================================
# ======================================================================================

# Split BEFORE scaling to avoid leakage
X_train_raw, X_test_raw, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42, stratify=y
)
print(f"Train N={len(X_train_raw)} | Test N={len(X_test_raw)}")

# Fit scaler on train only, then apply to train/test
scaler_x = fit_scaler_on_train(X_train_raw)
X_train = apply_scaler(scaler_x, X_train_raw)
X_test  = apply_scaler(scaler_x, X_test_raw)

# Augment train only (do NOT augment test/validation)
X_train_aug, y_train_aug, groups_train_aug = augment_data(X_train, y_train, groups_train, seed=0)
print(f"Train after augmentation N={len(X_train_aug)}")

# ======================================================================================
# ============================ GRID SEARCH + SAVE BEST PER MODEL =======================
# ======================================================================================

model_types = ["RNN", "LSTM", "GRU", "Transformer", "Mamba"]

best_per_model = {m: {"auc": -np.inf, "cfg": None, "state": None} for m in model_types}
grid_summary_rows = []
all_cv_predictions = []  # <-- for downstream plots

if RUN_GRID_SEARCH:
    epochs = 200
    n_repeats = 5
    n_splits = 3

    for model_type in model_types:
        for hidden_dim in [16, 32, 64, 128]:
            for dropout in [0.0, 0.2]:
                for num_layers in [1, 2, 3]:
                    for batch_size in [16, 32]:
                        for lr in [0.001, 0.01]:
                            for wd in [0, 1e-4]:
                                for optimizer_name in ["Adam", "SGD"]:

                                    summary, best_state, cv_records = train_one_config_cv(
                                        model_type,
                                        X_train_aug, y_train_aug, groups_train_aug,
                                        hidden_dim, dropout, num_layers, batch_size, lr, wd, optimizer_name,
                                        epochs=epochs, n_repeats=n_repeats, n_splits=n_splits,
                                        save_cv_records=True
                                    )

                                    # keep CV fold preds for plotting scripts
                                    all_cv_predictions.extend(cv_records)

                                    row = {
                                        "model": model_type,
                                        "hidden_dim": hidden_dim,
                                        "dropout": dropout,
                                        "num_layers": num_layers,
                                        "batch_size": batch_size,
                                        "lr": lr,
                                        "weight_decay": wd,
                                        "optimizer": optimizer_name,
                                        "epochs": epochs,
                                        "repeats": n_repeats,
                                        "folds": n_splits,
                                        **summary
                                    }
                                    grid_summary_rows.append(row)

                                    auc_mean = row["AUC_mean"]
                                    if np.isfinite(auc_mean) and auc_mean > best_per_model[model_type]["auc"] and summary["has_state"]:
                                        best_per_model[model_type]["auc"] = float(auc_mean)
                                        best_per_model[model_type]["cfg"] = {
                                            "model": model_type,
                                            "hidden_dim": int(hidden_dim),
                                            "dropout": float(dropout),
                                            "num_layers": int(num_layers),
                                            "batch_size": int(batch_size),
                                            "lr": float(lr),
                                            "weight_decay": float(wd),
                                            "optimizer": optimizer_name,
                                            "epochs": int(epochs),
                                            "repeats": int(n_repeats),
                                            "folds": int(n_splits),
                                            "threshold": float(threshold),
                                            "target_col": target_col,
                                            "timesteps": int(X_train_aug.shape[1]),
                                            "input_dim": int(X_train_aug.shape[2]),
                                            "feature_repr": "OSI_ONLY",
                                            "pca_components": None
                                        }
                                        best_per_model[model_type]["state"] = best_state
                                        print(f"\n🏆 BEST for {model_type}: AUC_mean={auc_mean:.4f}")

        # Save best artifacts per model_type
        bm = best_per_model[model_type]
        if bm["cfg"] is not None:
            model_dir = os.path.join(save_root, f"best_{model_type}")
            os.makedirs(model_dir, exist_ok=True)
            torch.save({"state_dict": bm["state"], "config": bm["cfg"]}, os.path.join(model_dir, "best_model_state.pt"))
            with open(os.path.join(model_dir, "best_model_config.json"), "w") as f:
                json.dump(bm["cfg"], f, indent=2)
            joblib.dump(scaler_x, os.path.join(model_dir, "scaler_x.joblib"))
            print(f"✅ Saved best artifacts for {model_type} -> {model_dir}")

    # Save grid summary
    df_grid = pd.DataFrame(grid_summary_rows)
    grid_csv_out = os.path.join(out_root_models, "Classification_grid_search_results(OSI)_8.csv")
    df_grid.to_csv(grid_csv_out, index=False)
    print("✅ Grid search summary saved:", grid_csv_out)

    # Save CV predictions (for your 1x3 plot script + top-config-per-model plots)
    cv_json_out = os.path.join(out_root_models, "Classification_cv_predictions(OSI)_8.json")
    with open(cv_json_out, "w") as f:
        json.dump(all_cv_predictions, f)
    print(f"✅ CV predictions saved: {len(all_cv_predictions)} records -> {cv_json_out}")

# ======================================================================================
# ============================= LOAD BEST-PER-MODEL ARTIFACTS ==========================
# ======================================================================================

def load_best_for_model(model_type: str):
    model_dir = os.path.join(save_root, f"best_{model_type}")
    ckpt = os.path.join(model_dir, "best_model_state.pt")
    cfgp = os.path.join(model_dir, "best_model_config.json")
    sxp  = os.path.join(model_dir, "scaler_x.joblib")
    if not (os.path.exists(ckpt) and os.path.exists(cfgp) and os.path.exists(sxp)):
        raise FileNotFoundError(f"Missing best artifacts for {model_type} in {model_dir}")
    with open(cfgp, "r") as f:
        cfg = json.load(f)
    sx = joblib.load(sxp)
    state = torch.load(ckpt, map_location="cpu")
    return cfg, sx, state["state_dict"]

# ======================================================================================
# =============================== VALIDATION BUILD (V3) ================================
# ======================================================================================

def load_validation_set_osi_only(scaler_train: StandardScaler):
    df_val = pd.read_excel(xlsx_val, sheet_name=sheet_val)
    train_target_col = target_col
    val_target_col = "OSI_V3_12th_avg"

    # choose label column
    if train_target_col not in df_val.columns:
        if val_target_col in df_val.columns:
            label_col = val_target_col
        else:
            raise ValueError(f"Missing target in {sheet_val}: neither {train_target_col} nor {val_target_col}")
    else:
        label_col = train_target_col

    all_feature_cols = [c for tw in range(1, 7) for c in osi_features[tw]]
    missing = [c for c in all_feature_cols if c not in df_val.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} OSI feature cols in {sheet_val}. First 20: {missing[:20]}")

    df_val_model = df_val.dropna(subset=[label_col] + all_feature_cols).copy()
    if len(df_val_model) == 0:
        raise ValueError("No validation rows left after dropna.")

    # X6: (N,6,2)
    X_list = [df_val_model[osi_features[tw]].to_numpy(dtype=np.float32) for tw in range(1, 7)]
    X6_val = np.stack(X_list, axis=1)
    X7_val = add_delta_window(X6_val)  # (N,7,2)

    y_val = (df_val_model[label_col].to_numpy(dtype=np.float32) >= threshold).astype(int)

    # scale using training scaler (no refit)
    X7_val_scaled = apply_scaler(scaler_train, X7_val)
    return X7_val_scaled, y_val, label_col

# ======================================================================================
# =============================== EVALUATION: TEST + VAL ===============================
# ======================================================================================

def predict_probs(model: nn.Module, X_in: np.ndarray, y_in: np.ndarray, batch_size=32):
    loader = get_loader(X_in, y_in, batch_size=batch_size, shuffle=False)
    probs, ys = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb).detach().cpu().flatten()
            p = torch.sigmoid(logits).numpy()
            probs.append(p)
            ys.append(yb.numpy().flatten())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(ys).astype(int)
    return y_true, y_prob

def evaluate_and_save_best_per_model():
    test_payloads = []
    val_payloads  = []
    metrics_rows_test = []
    metrics_rows_val  = []

    for m in model_types:
        cfg, sx, state_dict = load_best_for_model(m)

        model = build_model(
            cfg["model"],
            input_dim=int(cfg["input_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"])
        ).to(device)
        model.load_state_dict(state_dict)

        bs = int(cfg.get("batch_size", 32))

        # TEST
        y_true_test, y_prob_test = predict_probs(model, X_test, y_test, batch_size=bs)
        test_payloads.append({"model": m, "config": cfg, "y_true": y_true_test.tolist(), "y_prob": y_prob_test.tolist()})

        pm = eval_point_metrics(y_true_test, y_prob_test)
        rowt = {"model": m, **pm, "N": int(len(y_true_test))}
        for s in SPEC_TARGETS:
            sens, thr = sens_at_fixed_spec(y_true_test, y_prob_test, s)
            rowt[f"Sens@Spec{int(s*100)}"] = sens
            rowt[f"Thr@Spec{int(s*100)}"] = thr
        metrics_rows_test.append(rowt)

        # VALIDATION (V3)
        X_val_scaled, y_val, label_col_used = load_validation_set_osi_only(sx)
        y_true_val, y_prob_val = predict_probs(model, X_val_scaled, y_val, batch_size=bs)
        val_payloads.append({
            "model": m, "config": cfg, "label_col_used": label_col_used,
            "y_true": y_true_val.tolist(), "y_prob": y_prob_val.tolist()
        })

        pmv = eval_point_metrics(y_true_val, y_prob_val)
        rowv = {"model": m, **pmv, "N": int(len(y_true_val)), "label_col_used": label_col_used}
        for s in SPEC_TARGETS:
            sens, thr = sens_at_fixed_spec(y_true_val, y_prob_val, s)
            rowv[f"Sens@Spec{int(s*100)}"] = sens
            rowv[f"Thr@Spec{int(s*100)}"] = thr
        metrics_rows_val.append(rowv)

    test_preds_json = os.path.join(out_root_models, "Classification_best_model_test_predictions(OSI)_8.json")
    val_preds_json  = os.path.join(out_revision_dir, "Classification_best_model_validation_predictions(OSI)_8.json")

    with open(test_preds_json, "w") as f:
        json.dump(test_payloads, f, indent=2)
    with open(val_preds_json, "w") as f:
        json.dump(val_payloads, f, indent=2)

    pd.DataFrame(metrics_rows_test).to_csv(
        os.path.join(out_root_models, "Classification_best_models_TEST_metrics_point(OSI)_8.csv"),
        index=False
    )
    pd.DataFrame(metrics_rows_val).to_csv(
        os.path.join(out_revision_dir, "Classification_best_models_VALIDATION_metrics_point(OSI)_8.csv"),
        index=False
    )

    print("✅ Saved ALL5 TEST preds:", test_preds_json)
    print("✅ Saved ALL5 VAL preds:",  val_preds_json)
    return test_preds_json, val_preds_json

# ======================================================================================
# ============================ BOOTSTRAP CI + CALIBRATION ==============================
# ======================================================================================

def run_bootstrap_and_calibration(preds_json_path: str, split_name: str, out_dir: str):
    with open(preds_json_path, "r") as f:
        payloads = json.load(f)

    ci_rows = []
    calib_rows = []

    for item in payloads:
        model_name = item["model"]
        y_true = np.array(item["y_true"], dtype=int)
        y_prob = np.array(item["y_prob"], dtype=float)

        ci_df = bootstrap_ci_metrics(
            y_true, y_prob, B=BOOTSTRAP_B, seed=0, spec_targets=tuple(SPEC_TARGETS)
        )
        ci_df.insert(0, "split", split_name)
        ci_df.insert(1, "model", model_name)
        ci_rows.append(ci_df)

        cal_df = calibration_curve_bins(y_true, y_prob, n_bins=CALIB_BINS)
        cal_df.insert(0, "split", split_name)
        cal_df.insert(1, "model", model_name)
        calib_rows.append(cal_df)

    ci_all = pd.concat(ci_rows, ignore_index=True)
    cal_all = pd.concat(calib_rows, ignore_index=True)

    ci_csv  = os.path.join(out_dir, f"Classification_BootstrapCI_{split_name}_OSI_8.csv")
    cal_csv = os.path.join(out_dir, f"Classification_CalibrationCurve_{split_name}_OSI_8.csv")

    ci_all.to_csv(ci_csv, index=False)
    cal_all.to_csv(cal_csv, index=False)

    print(f"✅ Bootstrap CI saved: {ci_csv}")
    print(f"✅ Calibration curve bins saved: {cal_csv}")

# ======================================================================================
# ================================ LEARNING CURVES =====================================
# ======================================================================================

def train_single_model_once(
    model_type: str,
    cfg: dict,
    X_train_in: np.ndarray,
    y_train_in: np.ndarray,
    groups_train_in: np.ndarray,
    X_test_in: np.ndarray,
    y_test_in: np.ndarray,
    X_val_in: np.ndarray,
    y_val_in: np.ndarray,
    seed: int = 0
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(
        model_type,
        input_dim=X_train_in.shape[2],
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"])
    ).to(device)

    optimizer_name = cfg["optimizer"]
    lr = float(cfg["lr"])
    wd = float(cfg["weight_decay"])
    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])

    opt_cls = torch.optim.Adam if optimizer_name == "Adam" else torch.optim.SGD
    optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
    criterion = nn.BCEWithLogitsLoss()

    loader = get_loader(X_train_in, y_train_in, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    wait = 0
    best_state = None

    for _ in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if len(losses) else np.inf
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 10:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    yt_test, yp_test = predict_probs(model, X_test_in, y_test_in, batch_size=batch_size)
    pm_test = eval_point_metrics(yt_test, yp_test)
    for s in SPEC_TARGETS:
        sens, thr = sens_at_fixed_spec(yt_test, yp_test, s)
        pm_test[f"Sens@Spec{int(s*100)}"] = sens
        pm_test[f"Thr@Spec{int(s*100)}"] = thr

    yt_val, yp_val = predict_probs(model, X_val_in, y_val_in, batch_size=batch_size)
    pm_val = eval_point_metrics(yt_val, yp_val)
    for s in SPEC_TARGETS:
        sens, thr = sens_at_fixed_spec(yt_val, yp_val, s)
        pm_val[f"Sens@Spec{int(s*100)}"] = sens
        pm_val[f"Thr@Spec{int(s*100)}"] = thr

    return pm_test, pm_val

def run_learning_curves():
    X_val_scaled, y_val, label_col_used = load_validation_set_osi_only(scaler_x)

    rows = []
    unique_groups = np.unique(groups_train)

    for model_type in LEARNING_MODELS:
        cfg, _, _ = load_best_for_model(model_type)

        for frac in LEARNING_FRACTIONS:
            n_groups = max(2, int(len(unique_groups) * frac))
            for seed in LEARNING_SEEDS:
                rng = np.random.RandomState(seed)
                chosen_groups = rng.choice(unique_groups, size=n_groups, replace=False)
                mask = np.isin(groups_train, chosen_groups)

                X_sub = X_train[mask]
                y_sub = y_train[mask]
                g_sub = groups_train[mask]

                X_sub_aug, y_sub_aug, g_sub_aug = augment_data(X_sub, y_sub, g_sub, seed=seed)

                pm_test, pm_val = train_single_model_once(
                    model_type, cfg,
                    X_sub_aug, y_sub_aug, g_sub_aug,
                    X_test, y_test,
                    X_val_scaled, y_val,
                    seed=seed
                )

                rows.append({
                    "model": model_type,
                    "fraction": frac,
                    "seed": seed,
                    "label_col_val": label_col_used,
                    **{f"test_{k}": v for k, v in pm_test.items()},
                    **{f"val_{k}": v for k, v in pm_val.items()},
                    "n_train_patients": int(n_groups),
                    "n_train_samples": int(len(X_sub))
                })

                print(f"✅ LearningCurve {model_type} frac={frac} seed={seed} "
                      f"test_AUC={pm_test['AUC']:.3f} val_AUC={pm_val['AUC']:.3f}")

    df_lc = pd.DataFrame(rows)
    lc_csv = os.path.join(out_revision_dir, "Classification_LearningCurve_OSI_8.csv")
    df_lc.to_csv(lc_csv, index=False)
    print("✅ Learning curve results saved:", lc_csv)

# ======================================================================================
# ===================================== RUN ===========================================
# ======================================================================================

test_preds_json = None
val_preds_json = None

if RUN_EVAL_BEST_MODELS:
    test_preds_json, val_preds_json = evaluate_and_save_best_per_model()

if RUN_BOOTSTRAP_CI_EVAL:
    if test_preds_json is None or val_preds_json is None:
        test_preds_json = os.path.join(out_root_models, "Classification_best_model_test_predictions(OSI)_8.json")
        val_preds_json  = os.path.join(out_revision_dir, "Classification_best_model_validation_predictions(OSI)_8.json")
    run_bootstrap_and_calibration(test_preds_json, "TEST", out_root_models)
    run_bootstrap_and_calibration(val_preds_json, "VALIDATION", out_revision_dir)

if RUN_LEARNING_CURVES:
    run_learning_curves()

print("✅ All done.")