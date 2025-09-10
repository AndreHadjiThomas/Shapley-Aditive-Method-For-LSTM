# Water Level Forecasting with LSTM and SHAP Interpretability

This project uses LSTM-based sequence models to forecast water levels from meteorological and environmental time series data. It also includes model explainability using SHAP (SHapley Additive exPlanations) to interpret feature importances over time.

## Features

- Load and preprocess hydrological and environmental CSV data
- Train/test split with per-file temporal chunking
- Feature scaling using `StandardScaler`
- Sequence creation for LSTM training (multi-step sequence inputs)
- LSTM regression using PyTorch
- Forecasting and alignment for visualization
- Evaluation using NSE (Nash-Sutcliffe Efficiency) and KGE (Kling-Gupta Efficiency)
- SHAP analysis for feature importance over sequence steps
- Interactive and static SHAP plots: summary, force, bar, and heatmap

## Folder Structure

```
📂 MergedGauge_Daymet/         # Folder with input CSVs
├── gauge1.csv
├── gauge2.csv
├── ...
```

## How to Use

1. Upload your gauge data in CSV format to the `MergedGauge_Daymet/` folder in your Google Drive.
2. Mount your drive and run the script in Google Colab.
3. Adjust sequence length, hidden dimensions, and other training parameters as needed.
4. Visualize forecast performance and interpret the model with SHAP.

## Requirements

See `requirements.txt`.

## Output

- Forecast vs actual plot
- NSE and KGE metrics
- SHAP summary plots and heatmaps
- Interactive force plot

---

© 2025 Alexandre Hadji-Thomas