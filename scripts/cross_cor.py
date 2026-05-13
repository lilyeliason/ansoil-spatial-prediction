import pandas as pd

BASE = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/results"
seeds = [7, 42, 73, 123, 256]

# RF
rf_dfs = [
    pd.read_csv(f"{BASE}/rf_seed{s}/ansoil_model_results_rf_{s}.csv").set_index(
        "target"
    )["cv_r2"]
    for s in seeds
]
rf_combined = pd.concat(rf_dfs, axis=1, keys=seeds)
print("RF cross-seed correlations:")
print(rf_combined.corr().round(4))

# XGB
xgb_dfs = [
    pd.read_csv(f"{BASE}/xgb_seed{s}/ansoil_model_results_xgb_{s}.csv").set_index(
        "target"
    )["cv_r2"]
    for s in seeds
]
xgb_combined = pd.concat(xgb_dfs, axis=1, keys=seeds)
print("\nXGB cross-seed correlations:")
print(xgb_combined.corr().round(4))
