# ======================================================================================
# TBME Major Revision Utilities (REGRESSION: CNN-ONLY feature set)
# - Save BEST config+weights per model type (RNN/LSTM/GRU/Transformer/Mamba)
# - Save TRAINING CV predictions + summary (like your previous *_7 regression code)
# - Evaluate BEST per model on TEST + TEMPORAL VALIDATION and save y_true/y_pred
# - Learning curve (20/40/60/80/100% of training patients)
# - PCA sensitivity:
#     * CNN-only: PCA64 is applied to the FULL feature vector (same as before)
# - Patient-level bootstrap CIs for RMSE/MAE on TEST/VAL preds
# ======================================================================================

import os
import math
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
RUN_PCA_SENSITIVITY   = True     # trains best-config per model_type with PCA-64 (and optional PCA-32)
RUN_BOOTSTRAP_CI_EVAL = True     # patient-level bootstrap CI for RMSE/MAE on TEST/VAL preds

LEARNING_MODELS = ["RNN", "LSTM", "GRU", "Transformer", "Mamba"]    # or ["Mamba"]
PCA_MODELS      = ["RNN", "LSTM", "GRU", "Transformer", "Mamba"]    # or ["Mamba"]
PCA_COMPONENTS_LIST = [64]  # can set [64, 32] if you want both

LEARNING_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
LEARNING_SEEDS     = [0]  # repeats per fraction

BOOTSTRAP_B = 2000

# ---------------------------
# CNN-only toggles
# ---------------------------
USE_OSI_FEATURES = False   # <-- CNN ONLY
CNN_DIM_PER_TW = 256       # f1..f256 per TW

# ======================================================================================
# ================================ PATHS / I/O ========================================
# ======================================================================================

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training data
xlsx_train  = "/nfs/turbo/med-kayvan-lab/Projects/PARDS/02-Data/PARDS_Risk_V2/PARDS_Risk_RNN/PARDS_Risk_V1V2_df.xlsx"
sheet_train = "Sheet5"

# Temporal validation data
xlsx_val  = "/nfs/turbo/med-kayvan-lab/Projects/PARDS/02-Data/PARDS_Risk_V3/PARDS_Risk_V3_df.xlsx"
sheet_val = "Sheet15"

# Output roots
out_root_models = "/nfs/turbo/med-kayvan-lab/Projects/PARDS/04-Results/PARDS_Risk_V2/Fig7_RegressionModels"
save_root = os.path.join(out_root_models, "SavedModels_CNN_8")  # NEW folder
os.makedirs(save_root, exist_ok=True)

out_revision_dir = "/nfs/turbo/med-kayvan-lab/Projects/PARDS/04-Results/PARDS_Risk_V2/20251105_TBME_Submission/20260301_Major_Revision"
os.makedirs(out_revision_dir, exist_ok=True)

# ======================================================================================
# ================================= DATA LOADING ======================================
# ======================================================================================

target_col = "OSI_V2_12th_avg"

# CNN features (per TW): f1_TW1..f256_TW6
tw_cnn_features = {tw: [f"f{i}_TW{tw}" for i in range(1, 257)] for tw in range(1, 7)}

# (CNN-only) keep osi_features defined but unused; makes code robust if you flip the toggle later
osi_features = {tw: [f"OSI_mean_TW{tw}", f"OSI_std_TW{tw}"] for tw in range(1, 7)}

def build_sequences_from_df(
    df: pd.DataFrame,
    label_col: str,
    cnn_features: dict,
    use_osi: bool = False,
    osi_features: dict = None
):
    """
    Returns:
      X6: (N,6,D_perTW)
      y:  (N,)
      groups: (N,)
    """
    if osi_features is None:
        osi_features = {tw: [] for tw in range(1, 7)}

    # Pre-check required columns (faster + clearer error)
    required_cols = [label_col]
    for tw in range(1, 7):
        required_cols += list(cnn_features[tw])
        if use_osi:
            required_cols += list(osi_features.get(tw, []))

    missing_global = [c for c in required_cols if c not in df.columns]
    if missing_global:
        raise ValueError(f"Missing {len(missing_global)} columns in dataframe. First 30: {missing_global[:30]}")

    # Drop rows with any missing in required columns (vectorized)
    df2 = df.dropna(subset=required_cols).copy()
    if len(df2) == 0:
        raise ValueError("No usable rows after dropna on required columns.")

    X_list = []
    for tw in range(1, 7):
        cols = list(cnn_features[tw])
        if use_osi:
            cols += list(osi_features.get(tw, []))
        X_list.append(df2[cols].to_numpy(dtype=np.float32))

    X6 = np.stack(X_list, axis=1)  # (N,6,D)
    y = df2[label_col].to_numpy(dtype=np.float32)
    groups = df2["ResearchID"].to_numpy() if "ResearchID" in df2.columns else np.arange(len(df2))

    return X6.astype(np.float32), y.astype(np.float32), np.asarray(groups)

def add_delta_window(X6: np.ndarray) -> np.ndarray:
    # X6: (N, 6, D)
    delta = (X6[:, -1, :] - X6[:, 0, :])[:, np.newaxis, :]  # (N,1,D)
    return np.concatenate([X6, delta], axis=1).astype(np.float32)  # (N,7,D)

# Load train dataframe
df = pd.read_excel(xlsx_train, sheet_name=sheet_train)

X6, y, groups = build_sequences_from_df(
    df,
    target_col,
    tw_cnn_features,
    use_osi=USE_OSI_FEATURES,
    osi_features=osi_features
)
X = add_delta_window(X6)  # (N,7,D_total)

print(f"✅ Total usable samples: {len(X)}")
print(f"✅ Input shape: {X.shape} (N,T,D_total)")
print(f"✅ CNN-only: D_total_perTW={X.shape[2]} (expected 256)")

# ======================================================================================
# ============================= PREPROCESSING HELPERS =================================
# ======================================================================================

def fit_scaler_on_train_X(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    N, T, D = X_train.shape
    scaler.fit(X_train.reshape(-1, D))
    return scaler

def apply_scaler_X(scaler: StandardScaler, X_in: np.ndarray) -> np.ndarray:
    N, T, D = X_in.shape
    return scaler.transform(X_in.reshape(-1, D)).reshape(N, T, D).astype(np.float32)

def fit_scaler_on_train_y(y_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(y_train.reshape(-1, 1))
    return scaler

def apply_scaler_y(scaler: StandardScaler, y_in: np.ndarray) -> np.ndarray:
    return scaler.transform(y_in.reshape(-1, 1)).flatten().astype(np.float32)

def invert_scaler_y(scaler: StandardScaler, y_scaled: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten().astype(np.float32)

def augment_data(X_in, y_in, groups_in, noise_std=0.005, seed=0):
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_std, X_in.shape).astype(np.float32)
    return (
        np.concatenate([X_in, X_in + noise], axis=0),
        np.concatenate([y_in, y_in], axis=0),
        np.concatenate([groups_in, groups_in], axis=0)
    )

# --- PCA (full vector) ---
def fit_pca_on_train_full(X_train_scaled: np.ndarray, n_components: int) -> PCA:
    N, T, D = X_train_scaled.shape
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(X_train_scaled.reshape(-1, D))
    return pca

def apply_pca_full(pca: PCA, X_scaled: np.ndarray) -> np.ndarray:
    N, T, D = X_scaled.shape
    X2 = pca.transform(X_scaled.reshape(-1, D))  # (N*T, K)
    return X2.reshape(N, T, -1).astype(np.float32)

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

class RNNRegressor(nn.Module):
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

class MambaRegressor(nn.Module):
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
        return MambaRegressor(input_dim, hidden_dim, num_layers, dropout)
    if model_type == "Transformer":
        return TransformerModel(input_dim, hidden_dim, num_layers, dropout)
    return RNNRegressor(input_dim, hidden_dim, model_type, num_layers, dropout)

def get_loader(X_in, y_in, batch_size, shuffle=True):
    X_tensor = torch.tensor(X_in, dtype=torch.float32)
    y_tensor = torch.tensor(y_in, dtype=torch.float32).view(-1, 1)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)

# ======================================================================================
# ============================== REGRESSION METRICS ===================================
# ======================================================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def eval_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred), "N": int(len(y_true))}

def bootstrap_ci_regression_grouped(y_true, y_pred, groups, B=2000, seed=0):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)

    uniq = np.unique(groups)
    point = eval_regression_metrics(y_true, y_pred)

    dist_rmse, dist_mae = [], []
    for _ in range(B):
        sampled_groups = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(groups, sampled_groups)
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) < 2:
            continue
        dist_rmse.append(rmse(yt, yp))
        dist_mae.append(mae(yt, yp))

    def ci(arr):
        if len(arr) == 0:
            return (np.nan, np.nan)
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    rmse_lo, rmse_hi = ci(dist_rmse)
    mae_lo, mae_hi = ci(dist_mae)

    return pd.DataFrame([
        {"metric": "RMSE", "point": point["RMSE"], "ci_low": rmse_lo, "ci_high": rmse_hi, "n_boot": len(dist_rmse)},
        {"metric": "MAE",  "point": point["MAE"],  "ci_low": mae_lo,  "ci_high": mae_hi,  "n_boot": len(dist_mae)},
    ])

# ======================================================================================
# ================================ TRAIN/TEST SPLIT ===================================
# ======================================================================================

# Split BEFORE scaling to avoid leakage (X and y)
X_train_raw, X_test_raw, y_train_raw, y_test_raw, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42
)
print(f"Train N={len(X_train_raw)} | Test N={len(X_test_raw)}")

# Fit scalers on TRAIN only
scaler_x = fit_scaler_on_train_X(X_train_raw)
scaler_y = fit_scaler_on_train_y(y_train_raw)

# Apply scaling
X_train = apply_scaler_X(scaler_x, X_train_raw)
X_test  = apply_scaler_X(scaler_x, X_test_raw)

y_train = apply_scaler_y(scaler_y, y_train_raw)
y_test  = apply_scaler_y(scaler_y, y_test_raw)

# Augment train only (do NOT augment test/validation)
X_train_aug, y_train_aug, groups_train_aug = augment_data(X_train, y_train, groups_train, seed=0)
print(f"Train after augmentation N={len(X_train_aug)}")

# ======================================================================================
# ============================== GRID SEARCH TRAINING =================================
# ======================================================================================

def train_one_config_cv_regression(
    model_type, X_train_in, y_train_in, groups_train_in,
    hidden_dim, dropout, num_layers, batch_size, lr, wd, optimizer_name,
    epochs=200, n_splits=5, early_stop_patience=10,
    collect_cv_preds=True,
    scaler_y_for_inverse=None
):
    """
    - Early stop + scheduler on VALIDATION LOSS
    - Save per-fold CV predictions in ORIGINAL y domain (inverse-scaled)
    - Return summary metrics (RMSE/MAE mean/std in ORIGINAL y domain),
      plus "best state dict for this hyperparam combo" chosen by lowest val loss.
    """
    gkf = GroupKFold(n_splits=n_splits)

    rmse_list, mae_list = [], []
    used_epochs_all = []

    combo_best_val_loss = np.inf
    combo_best_state = None
    cv_pred_records = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train_in, y_train_in, groups_train_in)):

        model = build_model(model_type, X_train_in.shape[2], hidden_dim, num_layers, dropout).to(device)

        opt_cls = torch.optim.Adam if optimizer_name == "Adam" else torch.optim.SGD
        optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
        criterion = nn.MSELoss()

        train_loader = get_loader(X_train_in[tr_idx], y_train_in[tr_idx], batch_size=batch_size, shuffle=True)

        val_x_t = torch.tensor(X_train_in[val_idx], dtype=torch.float32).to(device)
        val_y_t = torch.tensor(y_train_in[val_idx], dtype=torch.float32).view(-1, 1).to(device)

        best_val_loss = float("inf")
        wait = 0
        used_epochs = 0
        best_state_fold = None

        for _ in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(val_x_t)
                val_loss = float(criterion(val_pred, val_y_t).item())

            scheduler.step(val_loss)
            used_epochs += 1

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                best_state_fold = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= early_stop_patience:
                    break

        used_epochs_all.append(used_epochs)

        if best_state_fold is not None:
            model.load_state_dict(best_state_fold)

        model.eval()
        with torch.no_grad():
            preds_scaled = model(val_x_t).squeeze().detach().cpu().numpy()

        yt_scaled = y_train_in[val_idx].reshape(-1)

        if scaler_y_for_inverse is None:
            raise ValueError("scaler_y_for_inverse is required to compute RMSE/MAE in original domain.")

        y_pred = invert_scaler_y(scaler_y_for_inverse, preds_scaled)
        y_true = invert_scaler_y(scaler_y_for_inverse, yt_scaled)

        rmse_list.append(rmse(y_true, y_pred))
        mae_list.append(mae(y_true, y_pred))

        if collect_cv_preds:
            cv_pred_records.append({
                "model": model_type,
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "num_layers": int(num_layers),
                "batch_size": int(batch_size),
                "lr": float(lr),
                "weight_decay": float(wd),
                "optimizer": optimizer_name,
                "fold": int(fold),
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist()
            })

        if best_val_loss < combo_best_val_loss and best_state_fold is not None:
            combo_best_val_loss = best_val_loss
            combo_best_state = best_state_fold

    summary = {
        "RMSE_mean": float(np.mean(rmse_list)) if len(rmse_list) else np.inf,
        "RMSE_std":  float(np.std(rmse_list))  if len(rmse_list) else np.nan,
        "MAE_mean":  float(np.mean(mae_list))  if len(mae_list)  else np.inf,
        "MAE_std":   float(np.std(mae_list))   if len(mae_list)  else np.nan,
        "n_folds_done": int(len(rmse_list)),
        "expected_folds": int(n_splits),
        "used_epochs_mean": float(np.mean(used_epochs_all)) if used_epochs_all else np.nan,
        "used_epochs_std":  float(np.std(used_epochs_all))  if used_epochs_all else np.nan,
        "combo_best_val_loss": float(combo_best_val_loss),
        "has_state": combo_best_state is not None
    }
    return summary, combo_best_state, cv_pred_records

# ======================================================================================
# ============================ GRID SEARCH + SAVE BEST PER MODEL =======================
# ======================================================================================

model_types = ["RNN", "LSTM", "GRU", "Transformer", "Mamba"]

best_per_model = {m: {"rmse": np.inf, "cfg": None, "state": None} for m in model_types}
grid_summary_rows = []
cv_predictions_all = []

if RUN_GRID_SEARCH:
    epochs = 200
    n_splits = 5

    for model_type in model_types:
        for hidden_dim in [16, 32, 64, 128]:
            for dropout in [0.0, 0.2]:
                for num_layers in [1, 2, 3]:
                    for batch_size in [16, 32]:
                        for lr in [0.001, 0.01]:
                            for wd in [0, 1e-4]:
                                for optimizer_name in ["Adam", "SGD"]:

                                    summary, best_state, cv_pred_records = train_one_config_cv_regression(
                                        model_type,
                                        X_train_aug, y_train_aug, groups_train_aug,
                                        hidden_dim, dropout, num_layers, batch_size, lr, wd, optimizer_name,
                                        epochs=epochs, n_splits=n_splits, early_stop_patience=10,
                                        collect_cv_preds=True,
                                        scaler_y_for_inverse=scaler_y
                                    )

                                    if cv_pred_records:
                                        cv_predictions_all.extend(cv_pred_records)

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
                                        "folds": n_splits,
                                        **summary
                                    }
                                    grid_summary_rows.append(row)

                                    rmse_mean = row["RMSE_mean"]
                                    if np.isfinite(rmse_mean) and rmse_mean < best_per_model[model_type]["rmse"] and summary["has_state"]:
                                        best_per_model[model_type]["rmse"] = float(rmse_mean)
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
                                            "folds": int(n_splits),
                                            "target_col": target_col,
                                            "timesteps": int(X_train_aug.shape[1]),
                                            "input_dim": int(X_train_aug.shape[2]),
                                            "feature_repr": "CNN_ONLY",
                                            "pca_components": None,
                                            "pca_mode": None,     # "full"
                                            "cnn_dim_per_tw": int(CNN_DIM_PER_TW),
                                        }
                                        best_per_model[model_type]["state"] = best_state
                                        print(f"\n🏆 BEST for {model_type}: RMSE_mean={rmse_mean:.4f}")

        bm = best_per_model[model_type]
        if bm["cfg"] is not None:
            model_dir = os.path.join(save_root, f"best_{model_type}")
            os.makedirs(model_dir, exist_ok=True)

            torch.save({"state_dict": bm["state"], "config": bm["cfg"]}, os.path.join(model_dir, "best_model_state.pt"))
            with open(os.path.join(model_dir, "best_model_config.json"), "w") as f:
                json.dump(bm["cfg"], f, indent=2)

            joblib.dump(scaler_x, os.path.join(model_dir, "scaler_x.joblib"))
            joblib.dump(scaler_y, os.path.join(model_dir, "scaler_y.joblib"))

            print(f"✅ Saved best artifacts for {model_type} -> {model_dir}")

    df_grid = pd.DataFrame(grid_summary_rows)
    grid_csv_out = os.path.join(out_root_models, "Regression_grid_search_results(CNN)_8.csv")
    df_grid.to_csv(grid_csv_out, index=False)
    print("✅ Grid search summary saved:", grid_csv_out)

    cv_json_out = os.path.join(out_root_models, "Regression_cv_predictions(CNN)_8.json")
    with open(cv_json_out, "w") as f:
        json.dump(cv_predictions_all, f)
    print(f"✅ CV predictions saved: {len(cv_predictions_all)} records -> {cv_json_out}")

# ======================================================================================
# ============================= LOAD BEST-PER-MODEL ARTIFACTS ==========================
# ======================================================================================

def load_best_for_model(model_type: str):
    model_dir = os.path.join(save_root, f"best_{model_type}")
    ckpt = os.path.join(model_dir, "best_model_state.pt")
    cfgp = os.path.join(model_dir, "best_model_config.json")
    sxp  = os.path.join(model_dir, "scaler_x.joblib")
    syp  = os.path.join(model_dir, "scaler_y.joblib")

    if not (os.path.exists(ckpt) and os.path.exists(cfgp) and os.path.exists(sxp) and os.path.exists(syp)):
        raise FileNotFoundError(f"Missing best artifacts for {model_type} in {model_dir}")

    with open(cfgp, "r") as f:
        cfg = json.load(f)

    sx = joblib.load(sxp)
    sy = joblib.load(syp)
    state = torch.load(ckpt, map_location="cpu")

    return cfg, sx, sy, state["state_dict"]

# ======================================================================================
# =============================== VALIDATION BUILD (V3) ================================
# ======================================================================================

def load_validation_set(scaler_train_x: StandardScaler, scaler_train_y: StandardScaler):
    df_val = pd.read_excel(xlsx_val, sheet_name=sheet_val)
    train_target_col = target_col
    val_target_col = "OSI_V3_12th_avg"

    if train_target_col in df_val.columns:
        label_col = train_target_col
    elif val_target_col in df_val.columns:
        print(f"NOTE: '{train_target_col}' not found in {sheet_val}; using '{val_target_col}' instead.")
        label_col = val_target_col
    else:
        raise ValueError(f"Missing target in {sheet_val}: neither {train_target_col} nor {val_target_col}")

    # CNN-only feature columns
    all_cols = [c for tw in range(1, 7) for c in tw_cnn_features[tw]]
    missing = [c for c in all_cols if c not in df_val.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} CNN feature cols in {sheet_val}. First 20: {missing[:20]}")

    df_val_model = df_val.dropna(subset=[label_col] + all_cols).copy()
    if len(df_val_model) == 0:
        raise ValueError("No validation rows left after dropna.")

    # Build X6
    X_list = [df_val_model[tw_cnn_features[tw]].to_numpy(dtype=np.float32) for tw in range(1, 7)]
    X6_val = np.stack(X_list, axis=1)  # (N,6,256)
    X7_val = add_delta_window(X6_val)  # (N,7,256)

    y_val_raw = df_val_model[label_col].to_numpy(dtype=np.float32)

    X7_val_scaled = apply_scaler_X(scaler_train_x, X7_val)
    y_val_scaled  = apply_scaler_y(scaler_train_y, y_val_raw)

    g_val = df_val_model["ResearchID"].to_numpy() if "ResearchID" in df_val_model.columns else np.arange(len(df_val_model))
    return X7_val_scaled, y_val_scaled, y_val_raw, g_val, label_col

# ======================================================================================
# =============================== EVALUATION: TEST + VAL ===============================
# ======================================================================================

def predict_scaled(model: nn.Module, X_in: np.ndarray, batch_size=32):
    X_tensor = torch.tensor(X_in, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            yhat = model(xb).detach().cpu().numpy().reshape(-1)
            preds.append(yhat)
    return np.concatenate(preds, axis=0)

def evaluate_and_save_best_per_model():
    test_payloads = []
    val_payloads  = []
    metrics_rows_test = []
    metrics_rows_val  = []

    y_test_raw_local = invert_scaler_y(scaler_y, y_test)

    for m in model_types:
        cfg, sx, sy, state_dict = load_best_for_model(m)

        model = build_model(
            cfg["model"],
            input_dim=int(cfg["input_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"])
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        bs = int(cfg.get("batch_size", 32))

        # TEST (use the globally scaled X_test; sx should match scaler_x)
        y_pred_test_scaled = predict_scaled(model, X_test, batch_size=bs)
        y_pred_test = invert_scaler_y(sy, y_pred_test_scaled)
        y_true_test = y_test_raw_local

        test_payloads.append({
            "model": m,
            "config": cfg,
            "y_true": y_true_test.tolist(),
            "y_pred": y_pred_test.tolist(),
            "groups": groups_test.tolist()
        })
        metrics_rows_test.append({"model": m, **eval_regression_metrics(y_true_test, y_pred_test)})

        # VALIDATION
        X_val_scaled, y_val_scaled, y_val_raw, g_val, label_col_used = load_validation_set(sx, sy)

        y_pred_val_scaled = predict_scaled(model, X_val_scaled, batch_size=bs)
        y_pred_val = invert_scaler_y(sy, y_pred_val_scaled)
        y_true_val = y_val_raw

        val_payloads.append({
            "model": m,
            "config": cfg,
            "label_col_used": label_col_used,
            "y_true": y_true_val.tolist(),
            "y_pred": y_pred_val.tolist(),
            "groups": g_val.tolist()
        })
        metrics_rows_val.append({"model": m, **eval_regression_metrics(y_true_val, y_pred_val), "label_col_used": label_col_used})

    test_preds_json = os.path.join(out_root_models, "Regression_best_model_test_predictions(CNN)_8.json")
    val_preds_json  = os.path.join(out_revision_dir, "Regression_best_model_validation_predictions(CNN)_8.json")

    with open(test_preds_json, "w") as f:
        json.dump(test_payloads, f, indent=2)
    with open(val_preds_json, "w") as f:
        json.dump(val_payloads, f, indent=2)

    pd.DataFrame(metrics_rows_test).to_csv(
        os.path.join(out_root_models, "Regression_best_models_TEST_metrics_point(CNN)_8.csv"),
        index=False
    )
    pd.DataFrame(metrics_rows_val).to_csv(
        os.path.join(out_revision_dir, "Regression_best_models_VALIDATION_metrics_point(CNN)_8.csv"),
        index=False
    )

    print("✅ Saved ALL5 TEST preds:", test_preds_json)
    print("✅ Saved ALL5 VAL preds:",  val_preds_json)
    return test_preds_json, val_preds_json

# ======================================================================================
# ============================ BOOTSTRAP CI (GROUPED) =================================
# ======================================================================================

def run_bootstrap_ci(preds_json_path: str, split_name: str, out_dir: str):
    with open(preds_json_path, "r") as f:
        payloads = json.load(f)

    all_rows = []
    for item in payloads:
        model_name = item["model"]
        y_true = np.array(item["y_true"], dtype=float)
        y_pred = np.array(item["y_pred"], dtype=float)
        groups = np.array(item.get("groups", np.arange(len(y_true))), dtype=object)

        ci_df = bootstrap_ci_regression_grouped(y_true, y_pred, groups, B=BOOTSTRAP_B, seed=0)
        ci_df.insert(0, "split", split_name)
        ci_df.insert(1, "model", model_name)
        all_rows.append(ci_df)

    out = pd.concat(all_rows, ignore_index=True)
    out_csv = os.path.join(out_dir, f"Regression_BootstrapCI_{split_name}_CNN_8.csv")
    out.to_csv(out_csv, index=False)
    print(f"✅ Bootstrap CI saved: {out_csv}")

# ======================================================================================
# ================================ LEARNING CURVES =====================================
# ======================================================================================

def train_single_model_once_regression(
    model_type: str,
    cfg: dict,
    X_train_in: np.ndarray,
    y_train_in: np.ndarray,      # scaled y
    groups_train_in: np.ndarray,
    X_test_in: np.ndarray,
    y_test_raw: np.ndarray,      # raw y
    X_val_in: np.ndarray,
    y_val_raw: np.ndarray,       # raw y
    sy: StandardScaler,          # y scaler for inverse
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
    criterion = nn.MSELoss()

    loader = get_loader(X_train_in, y_train_in, batch_size=batch_size, shuffle=True)

    best_proxy = float("inf")
    wait = 0
    best_state = None

    for _ in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if len(losses) else np.inf
        scheduler.step(avg_loss)

        if avg_loss < best_proxy:
            best_proxy = avg_loss
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 10:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_pred_test_scaled = predict_scaled(model, X_test_in, batch_size=batch_size)
    y_pred_test = invert_scaler_y(sy, y_pred_test_scaled)
    m_test = eval_regression_metrics(y_test_raw, y_pred_test)

    y_pred_val_scaled = predict_scaled(model, X_val_in, batch_size=batch_size)
    y_pred_val = invert_scaler_y(sy, y_pred_val_scaled)
    m_val = eval_regression_metrics(y_val_raw, y_pred_val)

    return m_test, m_val

def run_learning_curves():
    X_val_scaled, y_val_scaled, y_val_raw, g_val, label_col_used = load_validation_set(scaler_x, scaler_y)
    y_test_raw_local = invert_scaler_y(scaler_y, y_test)

    rows = []
    unique_groups = np.unique(groups_train)

    for model_type in LEARNING_MODELS:
        cfg, _, sy, _ = load_best_for_model(model_type)

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

                m_test, m_val = train_single_model_once_regression(
                    model_type, cfg,
                    X_sub_aug, y_sub_aug, g_sub_aug,
                    X_test, y_test_raw_local,
                    X_val_scaled, y_val_raw,
                    sy=sy,
                    seed=seed
                )

                rows.append({
                    "model": model_type,
                    "fraction": frac,
                    "seed": seed,
                    "label_col_val": label_col_used,
                    **{f"test_{k}": v for k, v in m_test.items()},
                    **{f"val_{k}": v for k, v in m_val.items()},
                    "n_train_patients": int(n_groups),
                    "n_train_samples": int(len(X_sub))
                })

                print(f"✅ LearningCurve {model_type} frac={frac} seed={seed} "
                      f"test_RMSE={m_test['RMSE']:.3f} val_RMSE={m_val['RMSE']:.3f}")

    df_lc = pd.DataFrame(rows)
    lc_csv = os.path.join(out_revision_dir, "Regression_LearningCurve_CNN_8.csv")
    df_lc.to_csv(lc_csv, index=False)
    print("✅ Learning curve results saved:", lc_csv)

# ======================================================================================
# ================================ PCA SENSITIVITY =====================================
# ======================================================================================

def run_pca_sensitivity():
    X_val_scaled, y_val_scaled, y_val_raw, g_val, label_col_used = load_validation_set(scaler_x, scaler_y)
    y_test_raw_local = invert_scaler_y(scaler_y, y_test)

    rows = []
    for model_type in PCA_MODELS:
        cfg, _, sy, _ = load_best_for_model(model_type)

        for k in PCA_COMPONENTS_LIST:
            # CNN-only => PCA on FULL feature vector
            pca = fit_pca_on_train_full(X_train, n_components=int(k))

            X_train_pca = apply_pca_full(pca, X_train)
            X_test_pca  = apply_pca_full(pca, X_test)
            X_val_pca   = apply_pca_full(pca, X_val_scaled)

            cfg_pca = dict(cfg)
            cfg_pca["pca_components"] = int(k)
            cfg_pca["pca_mode"] = "full"

            X_train_pca_aug, y_train_aug2, g_train_aug2 = augment_data(X_train_pca, y_train, groups_train, seed=0)

            m_test, m_val = train_single_model_once_regression(
                model_type, cfg_pca,
                X_train_pca_aug, y_train_aug2, g_train_aug2,
                X_test_pca, y_test_raw_local,
                X_val_pca, y_val_raw,
                sy=sy,
                seed=0
            )

            rows.append({
                "model": model_type,
                "pca_components": int(k),
                "pca_mode": "full",
                "label_col_val": label_col_used,
                "input_dim_after_pca": int(X_train_pca.shape[2]),
                **{f"test_{k2}": v for k2, v in m_test.items()},
                **{f"val_{k2}": v for k2, v in m_val.items()},
            })

            print(f"✅ PCA{k} {model_type} (full): "
                  f"test_RMSE={m_test['RMSE']:.3f} val_RMSE={m_val['RMSE']:.3f} | input_dim={X_train_pca.shape[2]}")

    df_pca = pd.DataFrame(rows)
    pca_csv = os.path.join(out_revision_dir, "Regression_PCASensitivity_CNN_8.csv")
    df_pca.to_csv(pca_csv, index=False)
    print("✅ PCA sensitivity results saved:", pca_csv)

# ======================================================================================
# ===================================== RUN ===========================================
# ======================================================================================

test_preds_json = None
val_preds_json = None

if RUN_EVAL_BEST_MODELS:
    test_preds_json, val_preds_json = evaluate_and_save_best_per_model()

if RUN_BOOTSTRAP_CI_EVAL:
    if test_preds_json is None or val_preds_json is None:
        test_preds_json = os.path.join(out_root_models, "Regression_best_model_test_predictions(CNN)_8.json")
        val_preds_json  = os.path.join(out_revision_dir, "Regression_best_model_validation_predictions(CNN)_8.json")
    run_bootstrap_ci(test_preds_json, "TEST", out_root_models)
    run_bootstrap_ci(val_preds_json, "VALIDATION", out_revision_dir)

if RUN_LEARNING_CURVES:
    run_learning_curves()

if RUN_PCA_SENSITIVITY:
    run_pca_sensitivity()

print("✅ All done.")