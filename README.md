# Water Level Forecasting with LSTM + SHAP

This repository contains a complete pipeline to forecast water levels from hydrometeorological time‑series using a PyTorch LSTM, then interpret the model with SHAP. It is designed for datasets exported as multiple CSV files (e.g., one per gauge or sub‑basin), which are merged into train/test splits per file and then concatenated.

## Highlights
- Clean & merge multiple CSV time‑series from a folder
- Train/test split **per file** (80/20) to avoid leakage across stations
- Standardize features & target
- Sequence modeling with a configurable **LSTM**
- One‑step‑ahead evaluation with **NSE** (Nash–Sutcliffe Efficiency) and **KGE** (Kling–Gupta Efficiency)
- **Explainability** with SHAP (summary plot, bar plot, and time‑step heatmap)

## Data expectations
- A folder of CSVs (e.g., `/content/drive/MyDrive/MergedGauge_Daymet/`), each with:
  - a `date` column (parseable as a date)
  - a water level target column named **`wl`**
  - meteorological features (e.g., `prcp_*`, `tmax_*`, `tmin_*`, etc.)
- Some optional columns are dropped if present (subbasin helper fields). See code for the exact list.

## Quickstart (Colab or local)
```bash
# (recommended) create a venv
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# install deps
pip install -r requirements.txt

# run
python waterlevel_lstm_shap.py   --data_dir "/path/to/MergedGauge_Daymet"   --seq_length 3   --batch_size 64   --epochs 5   --sample_shap 10000   --explain_limit 100000
```

The script will:
1. Load/clean all CSVs in `--data_dir`
2. Split each file into 80% train / 20% test (time‑based)
3. Standardize features & target (fit on train, transform test)
4. Build rolling sequences of length `--seq_length`
5. Train an LSTM
6. Produce one‑step‑ahead predictions, compute NSE & KGE
7. Generate SHAP summary/bar plots and a time‑step heatmap

Plots are saved in the working directory by default (`plots/`).

## Outputs
- `plots/pred_vs_actual.png` — quick visual check of alignment
- `plots/shap_summary.png` — SHAP summary (beeswarm)
- `plots/shap_bar.png` — SHAP mean |value| bar plot
- `plots/shap_heatmap.png` — Mean |SHAP| across time‑steps × features

## Configuration
Run `python waterlevel_lstm_shap.py -h` to see all CLI flags. Key ones:
- `--data_dir`: folder of CSVs
- `--seq_length`: number of time‑steps per sequence (default 3)
- `--batch_size`: training batch size (default 64)
- `--epochs`: training epochs (default 5)
- `--sample_shap`: background sample size for SHAP DeepExplainer (default 10000)
- `--explain_limit`: number of test sequences to explain (default 100000)

## Notes
- The target column name is **`wl`**; change `--target` if your file uses a different name.
- The code drops some commonly occurring subbasin helper columns if present.
- SHAP DeepExplainer supports PyTorch models; large backgrounds can be slow—tune `--sample_shap` and `--explain_limit`.
- The script saves plots as PNG without specifying styles/colors so it works in constrained environments.

## License
MIT
