#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Water Level Forecasting with LSTM + SHAP
# - Loads multiple CSVs from a folder
# - Cleans, splits per-file (80/20), standardizes
# - Builds rolling sequences
# - Trains a PyTorch LSTM
# - Evaluates (NSE, KGE)
# - Explains with SHAP (summary, bar, heatmap)

import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def nash_sutcliffe_efficiency(observed, predicted):
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    obs_mean = np.mean(observed)
    denom = np.sum((observed - obs_mean) ** 2)
    if denom == 0:
        return np.nan
    num = np.sum((observed - predicted) ** 2)
    return 1.0 - (num / denom)

def kling_gupta_efficiency(observed, predicted):
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    if observed.size < 2:
        return np.nan
    obs_mean = np.mean(observed)
    pred_mean = np.mean(predicted)
    std_obs = np.std(observed)
    std_pred = np.std(predicted)
    if std_obs == 0 or obs_mean == 0:
        return np.nan
    r = np.corrcoef(observed, predicted)[0, 1]
    alpha = std_pred / std_obs
    beta = pred_mean / obs_mean
    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        return np.nan
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

def create_sequences(df: pd.DataFrame, seq_len: int, target_col: str):
    feats = [c for c in df.columns if c not in ['date', target_col]]
    Xs, ys = [], []
    for i in range(len(df) - seq_len):
        seq_x = df.iloc[i:i+seq_len][feats + [target_col]].values
        seq_y = df.iloc[i+seq_len][target_col]
        Xs.append(seq_x)
        ys.append(seq_y)
    return np.array(Xs), np.array(ys)

class WaterLevelLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def main():
    parser = argparse.ArgumentParser(description="Water Level Forecasting with LSTM + SHAP")
    parser.add_argument('--data_dir', type=str, required=True, help='Folder of CSV files')
    parser.add_argument('--target', type=str, default='wl', help='Target column name (default: wl)')
    parser.add_argument('--seq_length', type=int, default=3, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='LSTM hidden dim')
    parser.add_argument('--layers', type=int, default=1, help='LSTM layers')
    parser.add_argument('--sample_shap', type=int, default=10000, help='Background sample size for SHAP')
    parser.add_argument('--explain_limit', type=int, default=100000, help='Number of test sequences to explain')
    parser.add_argument('--plots_dir', type=str, default='plots', help='Directory to save plots')
    args = parser.parse_args()

    ensure_dir(args.plots_dir)

    # 1) Load & clean per file
    train_dfs, test_dfs = [], []
    drop_cols = [
        'system:index_subbasin_1','system:index_subbasin_2','.geo_subbasin_1','.geo_subbasin_2',
        'system:index_subbasin_3','.geo_subbasin_3',
        'prcp_subbasin_2','srad_subbasin_2','tmax_subbasin_2','tmin_subbasin_2','vp_subbasin_2',
        'prcp_subbasin_3','srad_subbasin_3','tmax_subbasin_3','tmin_subbasin_3','vp_subbasin_3'
    ]

    csv_paths = sorted(glob.glob(os.path.join(args.data_dir, '*.csv')))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in: {args.data_dir}")

    for filepath in csv_paths:
        df = pd.read_csv(filepath, parse_dates=['date'])
        df = df.drop_duplicates(subset=['date']).sort_values('date')
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        df = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')

        n = len(df)
        if n <= 10:
            continue
        split_idx = int(np.floor(0.8 * n))
        train_dfs.append(df.iloc[:split_idx].reset_index(drop=True))
        test_dfs.append(df.iloc[split_idx:].reset_index(drop=True))

    if not train_dfs or not test_dfs:
        raise RuntimeError("Not enough data after splitting. Check input CSVs.")

    df_train = pd.concat(train_dfs, ignore_index=True)
    df_test  = pd.concat(test_dfs, ignore_index=True)

    target_col = args.target
    if target_col not in df_train.columns:
        raise KeyError(f"Target column '{target_col}' not found in training data.")

    feature_cols = [c for c in df_train.columns if c not in ['date', target_col]]
    # 2) Scale
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(df_train[feature_cols])
    X_test_scaled  = scaler_X.transform(df_test[feature_cols])

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(df_train[[target_col]]).ravel()
    y_test_scaled  = scaler_y.transform(df_test[[target_col]]).ravel()

    df_train_norm = pd.DataFrame(X_train_scaled, columns=feature_cols)
    df_train_norm[target_col] = y_train_scaled
    df_test_norm = pd.DataFrame(X_test_scaled, columns=feature_cols)
    df_test_norm[target_col] = y_test_scaled

    # 3) Sequences
    X_train_seq, y_train_seq = create_sequences(df_train_norm, seq_len=args.seq_length, target_col=target_col)
    X_test_seq,  y_test_seq  = create_sequences(df_test_norm,  seq_len=args.seq_length, target_col=target_col)

    # 4) Tensors & loaders
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).view(-1, 1)
    X_test_tensor  = torch.tensor(X_test_seq,  dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test_seq,  dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                              batch_size=args.batch_size, shuffle=True)

    # 5) Model
    input_dim = X_train_tensor.shape[2]
    model = WaterLevelLSTM(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # 6) Train
    model.train()
    for ep in range(1, args.epochs + 1):
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {ep}/{args.epochs} - Loss: {total_loss:.4f}")

    # 7) Predict & align one-step-ahead
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor).numpy().reshape(-1)

    y_test_inv = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).ravel()
    y_pred_inv = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()

    # Shift predictions back by one step
    y_pred_aligned = np.empty_like(y_pred_inv)
    y_pred_aligned[:-1] = y_pred_inv[1:]
    y_pred_aligned[-1] = np.nan

    y_test_aligned = y_test_inv[:-1]
    y_pred_valid = y_pred_aligned[:-1]

    nse = nash_sutcliffe_efficiency(y_test_aligned, y_pred_valid)
    kge = kling_gupta_efficiency(y_test_aligned, y_pred_valid)
    print(f"NSE: {nse:.4f} | KGE: {kge:.4f}")

    # Plot prediction vs actual
    plt.figure(figsize=(12, 4))
    plt.plot(y_test_aligned, label='Actual WL')
    plt.plot(y_pred_valid, label='Predicted WL (shifted back)')
    plt.title('One-Step-Ahead Forecast (Predictions Shifted Back)')
    plt.xlabel('Time Index')
    plt.ylabel('Water Level')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    ensure_dir(args.plots_dir)
    plt.savefig(os.path.join(args.plots_dir, 'pred_vs_actual.png'), dpi=150)
    plt.close()

    # 8) SHAP — DeepExplainer on PyTorch
    model_cpu = model.to('cpu')
    bg_size = min(args.sample_shap, X_train_tensor.shape[0])
    perm = torch.randperm(X_train_tensor.shape[0])[:bg_size]
    background = X_train_tensor[perm]

    explainer = shap.DeepExplainer(model_cpu, background)
    N_explain = min(args.explain_limit, X_test_tensor.shape[0])
    X_explain = X_test_tensor[:N_explain]
    shap_values = explainer.shap_values(X_explain)  # shape: (N, seq_len, n_features+1)

    feature_cols_plus_target = feature_cols + [target_col]
    feature_names = []
    for t in range(args.seq_length):
        for feat in feature_cols_plus_target:
            feature_names.append(f"{feat}@t{t}")

    shap_vals_flat = shap_values.reshape(N_explain, -1)
    X_explain_flat = X_explain.numpy().reshape(N_explain, -1)

    # Original-units version for readability in plots
    N, T, F_plus1 = X_explain.shape
    F = len(feature_cols)
    X_flat = X_explain.numpy().reshape(N * T, F_plus1)
    X_feats_norm = X_flat[:, :F]
    X_wl_norm = X_flat[:, F:]
    X_feats_orig = scaler_X.inverse_transform(X_feats_norm)
    X_wl_orig = scaler_y.inverse_transform(X_wl_norm)
    X_flat_orig = np.hstack([X_feats_orig, X_wl_orig])
    X_orig_for_plot = X_flat_orig.reshape(N, T * (F + 1))
    df_orig = pd.DataFrame(X_orig_for_plot, columns=feature_names)

    # SHAP summary (beeswarm)
    plt.figure()
    shap.summary_plot(shap_vals_flat, df_orig, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.plots_dir, 'shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # SHAP bar plot (mean |SHAP|)
    plt.figure()
    shap.summary_plot(shap_vals_flat, df_orig, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.plots_dir, 'shap_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Time-step heatmap of mean |SHAP| per (timestep, feature)
    heatmap_data = np.mean(np.abs(shap_vals_flat), axis=0).reshape(args.seq_length, F + 1)
    plt.figure(figsize=(10, 4))
    plt.imshow(heatmap_data, aspect='auto')
    plt.title("Mean |SHAP| Over Time Steps")
    plt.xlabel("Feature")
    plt.ylabel("Timestep (0..seq_length-1)")
    plt.xticks(ticks=np.arange(F + 1), labels=feature_cols_plus_target, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(args.seq_length), labels=[f"t{t}" for t in range(args.seq_length)])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(args.plots_dir, 'shap_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
