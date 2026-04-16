# === Try more

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from mamba_ssm import Mamba
import math
import json

# ========== Device ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== Load Data ==========
df = pd.read_excel(
    "/nfs/turbo/med-kayvan-lab/Projects/PARDS/02-Data/PARDS_Risk_V2/PARDS_Risk_RNN/PARDS_Risk_V1V2_df.xlsx",
    sheet_name="Sheet3"
)

target = "OSI_V2_12th_avg"

base_template = ["OSI_mean_TW{}", "OSI_std_TW{}"]

tw_features = {
    tw: [f.format(tw) for f in base_template] + [f"f{i}_TW{tw}" for i in range(1, 257)]
    for tw in range(1, 7)
}

X, y, groups = [], [], []
for idx, row in df.iterrows():
    sequence = []
    for tw in range(1, 7):
        cols = tw_features[tw]
        if row[cols].isnull().any():
            break
        sequence.append(row[cols].values)
    if len(sequence) == 6 and not pd.isna(row[target]):
        X.append(sequence)
        y.append(row[target])
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

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# ========== Train/Test Split ==========
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X_scaled, y_scaled, groups, test_size=0.2, random_state=42
)

# ========== Data Augmentation ==========
def augment_data(X, y, groups):
    noise = np.random.normal(0, 0.005, X.shape)
    return (
        np.concatenate([X, X + noise]),
        np.concatenate([y, y]),
        np.concatenate([groups, groups])
    )

X_train, y_train, groups_train = augment_data(X_train, y_train, groups_train)

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

class RNNModel(nn.Module):
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

class MambaModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        if num_layers == 1:
            self.stack = nn.Sequential(*[MambaBlock(hidden_dim) for _ in range(num_layers)])
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

# ========== Train and Evaluate ==========
def get_loader(X, y, batch_size):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

def run_grid_search():
    models = ["RNN", "LSTM", "GRU", "Transformer", "Mamba"]
    hidden_dims = [16, 32, 64, 128]
    dropouts = [0.0, 0.2]
    num_layers_list = [1, 2, 3]
    batch_sizes = [16, 32]
    lrs = [0.001, 0.01]
    weight_decays = [0, 1e-4]
    optimizers = ["Adam", "SGD"]
    epochs = 200
    results = {"cv_predictions": [], "summary": []}
    gkf = GroupKFold(n_splits=5)

    for model_type in models:
        for hidden_dim in hidden_dims:
            for dropout in dropouts:
                for num_layers in num_layers_list:
                    for batch_size in batch_sizes:
                        for lr in lrs:
                            for wd in weight_decays:
                                for opt_name in optimizers:
                                    rmse_list, mae_list = [], []
                                    used_epochs_all = []
                                    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups_train)):
                                        if model_type == "Mamba":
                                            model = MambaModel(X_train.shape[2], hidden_dim, num_layers, dropout).to(device)
                                        elif model_type == "Transformer":
                                            model = TransformerModel(X_train.shape[2], hidden_dim, num_layers, dropout).to(device)
                                        else:
                                            model = RNNModel(X_train.shape[2], hidden_dim, model_type, num_layers, dropout).to(device)
                                        optimizer_cls = torch.optim.Adam if opt_name == "Adam" else torch.optim.SGD
                                        optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=wd)
                                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
                                        criterion = nn.MSELoss()
                                        loader = get_loader(X_train[train_idx], y_train[train_idx], batch_size)

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
                                        val_x = torch.tensor(X_train[val_idx], dtype=torch.float32).to(device)
                                        preds = model(val_x).squeeze().detach().cpu().numpy()
                                        preds = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
                                        y_true = scaler_y.inverse_transform(y_train[val_idx].reshape(-1, 1)).flatten()

                                        results["cv_predictions"].append({
                                            "model": model_type,
                                            "hidden_dim": hidden_dim,
                                            "dropout": dropout,
                                            "num_layers": num_layers,
                                            "batch_size": batch_size,
                                            "lr": lr,
                                            "weight_decay": wd,
                                            "optimizer": opt_name,
                                            "fold": fold,
                                            "y_true": y_true.tolist(),
                                            "y_pred": preds.tolist()
                                        })

                                        rmse = np.sqrt(mean_squared_error(y_true, preds))
                                        mae = mean_absolute_error(y_true, preds)
                                        rmse_list.append(rmse)
                                        mae_list.append(mae)

                                    results["summary"].append({
                                        "model": model_type, "hidden_dim": hidden_dim, "dropout": dropout,
                                        "num_layers": num_layers, "batch_size": batch_size, "lr": lr,
                                        "weight_decay": wd, "optimizer": opt_name, "epochs": epochs,
                                        "rmse_mean": np.mean(rmse_list), "rmse_std": np.std(rmse_list),
                                        "mae_mean": np.mean(mae_list), "mae_std": np.std(mae_list),
                                        "used_epochs_mean": np.mean(used_epochs_all), "used_epochs_std": np.std(used_epochs_all)
                                    })
                                    print(f"{model_type} HD={hidden_dim} DO={dropout} NL={num_layers} EP={epochs} OPT={opt_name} => RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}, MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
    return pd.DataFrame(results["summary"]), results["cv_predictions"]

# ========== Run ==========
results_df, cv_predictions = run_grid_search()
results_df.to_csv("/nfs/turbo/med-kayvan-lab/Projects/PARDS/04-Results/PARDS_Risk_V2/Fig7_RegressionModels/Regression_grid_search_results(Stats_CNN)_5.csv", index=False)
print("✅ Grid search completed and results saved.")

with open("/nfs/turbo/med-kayvan-lab/Projects/PARDS/04-Results/PARDS_Risk_V2/Fig7_RegressionModels/Regression_cv_predictions(Stats_CNN)_5.json", "w") as f:
    json.dump(cv_predictions, f)
print(f"✅ CV predictions saved: {len(cv_predictions)} records.")
