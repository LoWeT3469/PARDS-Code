# === Try more

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
from mamba_ssm import Mamba
import math
import json

# ========== Device ==========
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== Load Data ==========
df = pd.read_excel(
    "/nfs/turbo/med-kayvan-lab/Projects/PARDS/02-Data/PARDS_Risk_V2/PARDS_Risk_RNN/PARDS_Risk_V1V2_df.xlsx",
    sheet_name="Sheet3"
)

target_col = "OSI_V2_12th_avg"

base_template = [
    "OSI_mean_TW{}", "OSI_std_TW{}", "PIP_mean_TW{}", "PIP_std_TW{}",
    "PEEP_mean_TW{}", "PEEP_std_TW{}", "TV_mean_TW{}(mL/Kg)", "TV_std_TW{}(mL/Kg)",
    "Avg_NegFlowDur_TW{}", "Std_NegFlowDur_TW{}", "Avg_PeakInterval_TW{}", "Std_PeakInterval_TW{}"
]

tw_features = {
    i: [f.format(i) for f in base_template] + [f"w{i}_SubBandEnergy_row{j}" for j in range(1, 17)]
    for i in range(1, 7)
}

X, y, groups = [], [], []

for idx, row in df.iterrows():
    sequence = []
    for tw in range(1, 7):
        cols = tw_features[tw]
        if row[cols].isnull().any():
            break
        sequence.append(row[cols].values)
    if len(sequence) == 6 and not pd.isna(row[target_col]):
        X.append(sequence)
        y.append(1 if row[target_col] >= 7.5 else 0)
        groups.append(row['ResearchID'] if 'ResearchID' in row else idx)

X = np.array(X)
y = np.array(y)
groups = np.array(groups)

feature_dim = X.shape[2]

# ========== Feature Engineering ==========
delta_features = X[:, -1, :] - X[:, 0, :]
delta_features = delta_features[:, np.newaxis, :]
X = np.concatenate([X, delta_features], axis=1)

# ========== Scaling ==========
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

# ========== Train/Test Split ==========
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X_scaled, y, groups, test_size=0.2, random_state=42
)

print(f"Training samples before augmentation: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ========== Data Augmentation ==========
def augment_data(X, y, groups):
    noise = np.random.normal(0, 0.005, X.shape)
    return (
        np.concatenate([X, X + noise]),
        np.concatenate([y, y]),
        np.concatenate([groups, groups])
    )

X_train, y_train, groups_train = augment_data(X_train, y_train, groups_train)

print(f"Training samples after augmentation: {len(X_train)}")

# ========== Model Definitions ==========
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
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)

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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            layer_norm_eps=1e-5
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
    def forward(self, x):
        return self.encoder(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0, nhead=None):
        super().__init__()
        if nhead is None:
            if hidden_dim % 8 == 0:
                nhead = hidden_dim // 8
            elif hidden_dim % 4 == 0:
                nhead = hidden_dim // 4
            else:
                nhead = 1

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.pe = PositionalEncoding(hidden_dim)

        if num_layers == 1:
            self.stack = nn.Sequential(*[TransformerBlock(hidden_dim, nhead, 0.0)])
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
            self.stack = nn.Sequential(*[MambaBlock(hidden_dim)])
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

# ========== Utility ==========
def get_loader(X, y, batch_size):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

def auprc(y_true, y_prob):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return auc(r, p)

def stratified_sample(y_train, ratio=0.5, pos_ratio=1/3):
# def stratified_sample(y_train, ratio=0.8, pos_ratio=1/3):
    y_flat = y_train.flatten()
    idx_0 = np.where(y_flat == 0)[0]
    idx_1 = np.where(y_flat == 1)[0]
    total = int(len(y_flat) * ratio)
    n1 = int(total * pos_ratio)
    n0 = total - n1
    if len(idx_0) < n0 or len(idx_1) < n1:
        return None, None
    idx = np.concatenate([
        np.random.choice(idx_0, n0, replace=False),
        np.random.choice(idx_1, n1, replace=False)
    ])
    np.random.shuffle(idx)
    return idx, total

# ========== Train Loop ==========
epochs = 200
results = {"cv_predictions": [], "summary": []}
model_types = ["RNN", "LSTM", "GRU", "Transformer", "Mamba"]

for model_type in model_types:
    for hidden_dim in [16, 32, 64, 128]:
        for dropout in [0.0, 0.2]:
            for num_layers in [1, 2, 3]:
                for batch_size in [16, 32]:
                    for lr in [0.001, 0.01]:
                        for wd in [0, 1e-4]:
                            for optimizer_name in ["Adam", "SGD"]:
                                all_aucs, all_auprcs, all_f1s = [], [], []
                                used_epochs_all = []
                                for repeat in range(10):
                                    sampled_idx, _ = stratified_sample(y_train)
                                    if sampled_idx is None:
                                        print(f"[{model_type}] Skipping due to not enough positive or negative samples.")
                                        continue

                                    X_sub = X_train[sampled_idx]
                                    y_sub = y_train[sampled_idx]
                                    groups_sub = groups_train[sampled_idx]  # ✅ GroupKFold needs corresponding groups

                                    gkf = GroupKFold(n_splits=5)

                                    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_sub, y_sub, groups_sub)):
                                        print(f"[Repeat {repeat+1} | Fold {fold+1}] Train size: {len(tr_idx)} | Val size: {len(val_idx)} | Test size: {len(y_test)}")

                                        if model_type == "Mamba":
                                            model = MambaClassifier(X_train.shape[2], hidden_dim, num_layers, dropout).to(device)
                                        elif model_type == "Transformer":
                                            model = TransformerModel(X_train.shape[2], hidden_dim, num_layers, dropout).to(device)
                                        else:
                                            model = RNNClassifier(X_train.shape[2], hidden_dim, model_type, num_layers, dropout).to(device)
                                        optimizer_cls = torch.optim.Adam if optimizer_name == "Adam" else torch.optim.SGD
                                        optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=wd)
                                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
                                        criterion = nn.BCEWithLogitsLoss()
                                        loader = get_loader(X_sub[tr_idx], y_sub[tr_idx], batch_size)

                                        best_loss = float('inf')
                                        wait = 0
                                        used_epochs = 0
                                        for epoch in range(epochs):
                                            model.train()
                                            epoch_losses = []
                                            for xb, yb in loader:
                                                xb, yb = xb.to(device), yb.to(device)
                                                optimizer.zero_grad()
                                                pred = model(xb)
                                                loss = criterion(pred, yb)
                                                if torch.isnan(loss) or torch.isinf(loss):
                                                    continue
                                                loss.backward()
                                                epoch_losses.append(loss.item())
                                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                                optimizer.step()
                                            avg_loss = np.mean(epoch_losses)
                                            scheduler.step(avg_loss)
                                            used_epochs += 1
                                            if avg_loss < best_loss:
                                                best_loss = avg_loss
                                                wait = 0
                                            else:
                                                wait += 1
                                                if wait >= 10:
                                                    break
                                        used_epochs_all.append(used_epochs)

                                        model.eval()
                                        with torch.no_grad():
                                            val_input = torch.tensor(X_sub[val_idx], dtype=torch.float32).to(device)
                                            logits_tensor = model(val_input).cpu().flatten()
                                            probs = torch.sigmoid(logits_tensor).numpy()
                                            y_true = y_sub[val_idx].flatten()
                                            valid_mask = np.isfinite(probs)
                                            if valid_mask.sum() == 0:
                                                continue
                                            probs = probs[valid_mask]
                                            y_true = y_true[valid_mask]
                                            preds = (probs >= 0.5).astype(int)

                                            all_aucs.append(roc_auc_score(y_true, probs))
                                            all_auprcs.append(auprc(y_true, probs))
                                            all_f1s.append(f1_score(y_true, preds))

                                            if "cv_predictions" not in results:
                                                results["cv_predictions"] = []

                                            results["cv_predictions"].append({
                                                "model": model_type,
                                                "hidden_dim": hidden_dim,
                                                "dropout": dropout,
                                                "num_layers": num_layers,
                                                "batch_size": batch_size,
                                                "lr": lr,
                                                "weight_decay": wd,
                                                "optimizer": optimizer_name,
                                                "repeat": repeat,
                                                "fold": fold,
                                                "y_true": y_true.tolist(),
                                                "y_prob": probs.tolist()
                                            })

                                results["summary"].append({
                                    "model": model_type,
                                    "hidden_dim": hidden_dim,
                                    "dropout": dropout,
                                    "num_layers": num_layers,
                                    "batch_size": batch_size,
                                    "lr": lr,
                                    "weight_decay": wd,
                                    "optimizer": optimizer_name,
                                    "epochs": epochs,
                                    "AUC_mean": np.mean(all_aucs) if len(all_aucs) == 50 else np.nan,
                                    "AUC_std": np.std(all_aucs) if len(all_aucs) == 50 else np.nan,
                                    "AUPRC_mean": np.mean(all_auprcs) if len(all_auprcs) == 50 else np.nan,
                                    "AUPRC_std": np.std(all_auprcs) if len(all_auprcs) == 50 else np.nan,
                                    "F1_mean": np.mean(all_f1s) if len(all_f1s) == 50 else np.nan,
                                    "F1_std": np.std(all_f1s) if len(all_f1s) == 50 else np.nan,
                                    "used_epochs_mean": np.mean(used_epochs_all) if len(all_aucs) == 50 else np.nan,
                                    "used_epochs_std": np.std(used_epochs_all) if len(all_aucs) == 50 else np.nan,
                                })

# ========== Save Results ==========
df_results = pd.DataFrame(results["summary"])
df_results.to_csv("/nfs/turbo/med-kayvan-lab/Projects/PARDS/04-Results/PARDS_Risk_V2/Fig8_ClassificationModels/Classification_grid_search_results(Stats)_5.csv", index=False)
print("✅ Grid search completed and saved.")

with open ("/nfs/turbo/med-kayvan-lab/Projects/PARDS/04-Results/PARDS_Risk_V2/Fig8_ClassificationModels/Classification_cv_predictions(Stats)_5.json", "w") as f:
    json.dump(results["cv_predictions"], f)
print(f"✅ CV predictions saved: {len(results['cv_predictions'])} records.")
