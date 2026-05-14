"""
=============================================================================
ANSOIL REGRESSION KRIGING
=============================================================================
Applies Regression Kriging to Random Forest / XGBoost spatial predictions
of Antarctic soil geochemistry.

Method:
  1. RF/XGBoost predicts the environmental trend across 15,769 grid points
  2. Residuals at the 171 sample locations are fitted to a semivariogram
  3. If spatial structure exists, residuals are kriged across the grid
  4. Kriged residuals are added to the RF/XGBoost grid predictions
  5. Leave-One-Out CV compares RF alone vs Regression Kriging

IMPORTANT NOTE ON SAMPLING DENSITY:
  Our 171 sample points span ~3,000 km across Antarctica. The minimum
  distance between the 28 sampling LOCATIONS is small (within-location
  samples cluster tightly), but between-location distances average ~200 km.
  If the fitted variogram range is small relative to inter-location spacing,
  kriging will only correct predictions very close to sample locations and
  most of the 15,769 grid points will receive near-zero correction.
  The LOO cross-validation confirms whether kriging adds real value.

Setup:
    pip install pykrige scikit-learn pandas numpy matplotlib scipy

Usage:
    cd scripts
    python ansoil_kriging.py

Output files (saved to OUTPUT_DIR):
  table1_kriging_{target}.csv       - Kriging results at 15,769 grid points
  table2_arcgis_{target}.csv        - ArcGIS-ready version with lat/lon
  table3_loo_cv_{target}.csv        - LOO cross-validation metrics
  fig1_variogram_{target}.png       - Semivariogram fit
  fig2_loo_cv_{target}.png          - RF vs RK scatter plot
=============================================================================
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_squared_error, r2_score

# =============================================================================
# CONFIGURATION — edit these for each run
# =============================================================================

# ★ CHANGE THIS between runs
# Options: d15n_air_permil, wt_percent_n, digest_mg_kg_k_7664,
#          log_digest_mg_kg_na5895, clr_total_mg_l_po4, log_digest_mg_kg_p_1774
TARGET = "d15n_air_permil"

# ★ File paths — update to match your folder structure
# Run from the scripts/ folder and these paths work as-is
CV_PREDICTIONS_FILE = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/results/rf_seed42/ansoil_cv_predictions_rf_42.csv"
SAMPLE_INDEX_FILE = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/data/ansoil_sample_index.csv"
GRID_PREDICTIONS_FILE = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/results/rf_seed42/ansoil_grid_predictions_rf_42.csv"
GRID_COORDS_FILE = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/data/ansoil_grid_prepared.csv"

# Output folder (created automatically if it doesn't exist)
OUTPUT_DIR = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/results/kriging"

# Variogram model: 'spherical', 'exponential', or 'gaussian'
VARIOGRAM_MODEL = "spherical"

# Nugget:sill ratio above which variogram is considered flat
FLATNESS_NS_THRESHOLD = 0.85

# =============================================================================
# SETUP
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("ANSOIL REGRESSION KRIGING")
print(f"Target:          {TARGET}")
print(f"Variogram model: {VARIOGRAM_MODEL}")
print("=" * 70)

# =============================================================================
# STEP 1: Load files
# =============================================================================

print("\nSTEP 1: Loading files...")

try:
    cv_data = pd.read_csv(CV_PREDICTIONS_FILE)
    sample_idx = pd.read_csv(SAMPLE_INDEX_FILE)
    grid_preds = pd.read_csv(GRID_PREDICTIONS_FILE)
    grid_coords = pd.read_csv(GRID_COORDS_FILE)
except FileNotFoundError as e:
    print(f"\nERROR: {e}")
    print("Check that your file paths in CONFIGURATION are correct.")
    raise SystemExit(1)

print(f"  CV predictions:   {cv_data.shape}")
print(f"  Sample index:     {sample_idx.shape}")
print(f"  Grid predictions: {grid_preds.shape}")
print(f"  Grid coordinates: {grid_coords.shape}")

# =============================================================================
# STEP 2: Filter CV predictions to target
# =============================================================================

print(f"\nSTEP 2: Filtering to target '{TARGET}'...")

available = cv_data["target"].unique()
if TARGET not in available:
    print(f"\nERROR: '{TARGET}' not found in CV predictions.")
    print(f"Available targets: {sorted(available)}")
    raise SystemExit(1)

sample_data = (
    cv_data[cv_data["target"] == TARGET][
        ["target", "sample_id", "actual", "predicted", "residual"]
    ]
    .copy()
    .reset_index(drop=True)
)

print(f"  Samples for this target: {len(sample_data)}")

grid_col = f"pred_{TARGET}"
if grid_col not in grid_preds.columns:
    print(f"\nERROR: Column '{grid_col}' not found in grid predictions.")
    print(f"Available: {[c for c in grid_preds.columns if c.startswith('pred_')][:10]}")
    raise SystemExit(1)

# =============================================================================
# STEP 3: Join sample data with projected coordinates
# =============================================================================

print("\nSTEP 3: Joining sample data with coordinates...")

df = pd.merge(
    sample_data,
    sample_idx[["sample_id", "proj_x_epsg3031", "proj_y_epsg3031"]],
    on="sample_id",
    how="left",
)

missing = df["proj_x_epsg3031"].isna().sum()
if missing > 0:
    print(f"  WARNING: {missing} samples missing coordinates — dropping")
    df = df.dropna(subset=["proj_x_epsg3031", "proj_y_epsg3031"])

print(f"  Joined table: {df.shape}")

x = df["proj_x_epsg3031"].values
y = df["proj_y_epsg3031"].values
z = df["residual"].values

print("\n  Residual statistics:")
print(
    f"    Mean:   {z.mean():.4f}  |  Std: {z.std():.4f}  |  "
    f"Min: {z.min():.4f}  |  Max: {z.max():.4f}"
)

# =============================================================================
# STEP 4: Fit semivariogram to residuals
# =============================================================================

print(f"\nSTEP 4: Fitting {VARIOGRAM_MODEL} semivariogram to residuals...")

ok = OrdinaryKriging(
    x, y, z, variogram_model=VARIOGRAM_MODEL, verbose=False, enable_plotting=False
)

# PyKrige returns [psill, range, nugget]
psill = ok.variogram_model_parameters[0]
vrange = ok.variogram_model_parameters[1]
nugget = ok.variogram_model_parameters[2]
total_sill = psill + nugget
ns_ratio = nugget / total_sill if total_sill > 0 else 1.0

print("\n  Variogram parameters:")
print(f"    Nugget:            {nugget:.6f}")
print(f"    Partial sill:      {psill:.6f}")
print(f"    Total sill:        {total_sill:.6f}")
print(f"    Range:             {vrange:.1f} m  ({vrange / 1000:.1f} km)")
print(f"    Nugget:sill ratio: {ns_ratio:.4f}")

# Spatial structure check: does semivariance increase with distance?
sv = ok.semivariance
lags = ok.lags
slope = np.polyfit(lags, sv, 1)[0]

print("\n  Spatial structure check:")
print(
    f"    Semivariance trend slope: {slope:.3e}  "
    f"({'positive — spatial structure present' if slope > 0 else 'flat or negative — no spatial structure'})"
)
print(
    f"    Range vs dataset extent:  {vrange / 1000:.0f} km vs "
    f"{(x.max() - x.min()) / 1000:.0f} km (x) / {(y.max() - y.min()) / 1000:.0f} km (y)"
)

is_flat = (ns_ratio > FLATNESS_NS_THRESHOLD) or (slope <= 0)

if is_flat:
    reason = (
        "nugget:sill > threshold"
        if ns_ratio > FLATNESS_NS_THRESHOLD
        else "semivariance does not increase with distance"
    )
    print(f"\n  NOTE: Semivariogram is effectively flat ({reason}).")
    print(
        "  This is common with sparse continental-scale datasets (~200km between locations)."
    )
    print(
        "  Kriging will likely not improve RF predictions — the LOO CV will confirm this."
    )
else:
    print("\n  Spatial structure detected. Kriging may improve predictions.")

# Variogram plot
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.scatter(
    lags, sv, color="red", marker="*", s=70, zorder=5, label="Experimental variogram"
)

lags_line = np.linspace(0, lags.max() * 1.1, 300)
fitted_line = ok.variogram_function(ok.variogram_model_parameters, lags_line)
ax1.plot(
    lags_line,
    fitted_line,
    color="black",
    linewidth=1.5,
    label=f"Fitted {VARIOGRAM_MODEL}",
)

ax1.axhline(
    total_sill, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="Total sill"
)
ax1.axhline(nugget, color="blue", linestyle=":", linewidth=1, alpha=0.7, label="Nugget")
ax1.axvline(
    vrange,
    color="green",
    linestyle=":",
    linewidth=1,
    alpha=0.7,
    label=f"Range ({vrange / 1000:.0f}km)",
)

ax1.set_xlabel("Distance")
ax1.set_ylabel("Semivariance")
ax1.set_title(
    f"Semivariogram of RF Residuals — {TARGET}\n"
    f"Nugget={nugget:.3f}  Partial Sill={psill:.3f}  "
    f"Range={vrange / 1000:.0f}km  N:S={ns_ratio:.3f}" + ("  [FLAT]" if is_flat else "")
)
ax1.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val / 1000:.0f}km"))
ax1.legend(fontsize=9)
ax1.grid(True, linestyle="--", alpha=0.5)

fig1_path = os.path.join(OUTPUT_DIR, f"fig1_variogram_{TARGET}.png")
fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
print(f"\n  Variogram plot saved: {fig1_path}")
plt.show()

# =============================================================================
# STEP 5: Leave-One-Out Cross-Validation
# =============================================================================

print("\nSTEP 5: Leave-One-Out Cross-Validation...")
print(f"  Running {len(df)} LOO iterations (a few minutes)...")

n = len(df)
loo_rk_preds = np.zeros(n)

for i in range(n):
    mask = np.ones(n, dtype=bool)
    mask[i] = False

    ok_loo = OrdinaryKriging(
        x[mask],
        y[mask],
        z[mask],
        variogram_model=VARIOGRAM_MODEL,
        verbose=False,
        enable_plotting=False,
    )
    pred_resid, _ = ok_loo.execute("points", np.array([x[i]]), np.array([y[i]]))
    loo_rk_preds[i] = df["predicted"].iloc[i] + pred_resid[0]

    if (i + 1) % 25 == 0 or (i + 1) == n:
        print(f"  {i + 1}/{n} complete...")

actuals = df["actual"].values
rf_preds = df["predicted"].values

r2_rf = r2_score(actuals, rf_preds)
rmse_rf = np.sqrt(mean_squared_error(actuals, rf_preds))
mae_rf = np.mean(np.abs(actuals - rf_preds))

r2_rk = r2_score(actuals, loo_rk_preds)
rmse_rk = np.sqrt(mean_squared_error(actuals, loo_rk_preds))
mae_rk = np.mean(np.abs(actuals - loo_rk_preds))

r2_diff = r2_rk - r2_rf

print(f"\n  {'Metric':<22} {'RF Alone':>10} {'Regr. Kriging':>14} {'Change':>10}")
print(f"  {'-' * 58}")
print(f"  {'R²':<22} {r2_rf:>10.4f} {r2_rk:>14.4f} {r2_diff:>+10.4f}")
print(f"  {'RMSE':<22} {rmse_rf:>10.4f} {rmse_rk:>14.4f} {rmse_rk - rmse_rf:>+10.4f}")
print(f"  {'MAE':<22} {mae_rf:>10.4f} {mae_rk:>14.4f} {mae_rk - mae_rf:>+10.4f}")

print("\n  Interpretation:")
if r2_diff > 0.02:
    print(
        f"  Kriging IMPROVES on RF (R² +{r2_diff:.4f}). Use Regression Kriging predictions."
    )
elif r2_diff > 0:
    print(
        f"  Kriging gives marginal improvement (R² +{r2_diff:.4f}). Either approach is defensible."
    )
elif r2_diff > -0.02:
    print(
        f"  Kriging gives no meaningful improvement (R² {r2_diff:+.4f}). Use RF predictions alone."
    )
else:
    print(f"  Kriging HURTS performance (R² {r2_diff:+.4f}). Use RF predictions alone.")

loo_summary = pd.DataFrame(
    {
        "metric": ["R²", "RMSE", "MAE"],
        "RF_alone": [round(r2_rf, 4), round(rmse_rf, 4), round(mae_rf, 4)],
        "regression_kriging": [round(r2_rk, 4), round(rmse_rk, 4), round(mae_rk, 4)],
        "improvement": [
            round(r2_diff, 4),
            round(rmse_rk - rmse_rf, 4),
            round(mae_rk - mae_rf, 4),
        ],
    }
)
loo_path = os.path.join(OUTPUT_DIR, f"table3_loo_cv_{TARGET}.csv")
loo_summary.to_csv(loo_path, index=False)

# LOO scatter plot
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle(f"LOO CV: RF vs Regression Kriging — {TARGET}", fontsize=12)

for ax, preds, label, r2, rmse, color in [
    (axes[0], rf_preds, "RF Alone", r2_rf, rmse_rf, "steelblue"),
    (axes[1], loo_rk_preds, "Regression Kriging", r2_rk, rmse_rk, "darkorange"),
]:
    mn = min(actuals.min(), preds.min())
    mx = max(actuals.max(), preds.max())
    ax.scatter(
        actuals,
        preds,
        alpha=0.65,
        color=color,
        edgecolors="white",
        linewidths=0.4,
        s=40,
    )
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="1:1 line")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{label}\nR² = {r2:.4f}   RMSE = {rmse:.4f}")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, f"fig2_loo_cv_{TARGET}.png")
fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
print(f"\n  LOO scatter plot saved: {fig2_path}")
plt.show()

# =============================================================================
# STEP 6: Krige residuals across 15,769 grid points
# =============================================================================

print("\nSTEP 6: Kriging residuals across grid points...")
if is_flat:
    print(
        "  Note: variogram is flat — kriged residuals will be near zero "
        "at most grid points."
    )

grid_clean = grid_preds[["grid_id", grid_col]].copy()
pred_full = pd.merge(
    grid_clean,
    grid_coords[["grid_id", "proj_x_epsg3031", "proj_y_epsg3031"]],
    on="grid_id",
)

grid_x = pred_full["proj_x_epsg3031"].values
grid_y = pred_full["proj_y_epsg3031"].values

print(f"  Kriging at {len(grid_x):,} grid points...")
k_residual, k_variance = ok.execute("points", grid_x, grid_y)
print("  Done.")

kriging_data = grid_clean.copy()
kriging_data["kriging_residual"] = k_residual
kriging_data["kriging_variance"] = k_variance
kriging_data["final_regression_kriging"] = (
    kriging_data[grid_col] + kriging_data["kriging_residual"]
)

print("\n  Grid prediction statistics:")
print(
    f"    RF range:            {kriging_data[grid_col].min():.4f} to {kriging_data[grid_col].max():.4f}"
)
print(
    f"    Kriged residuals:    {kriging_data['kriging_residual'].min():.4f} to "
    f"{kriging_data['kriging_residual'].max():.4f}  "
    f"(mean abs: {kriging_data['kriging_residual'].abs().mean():.4f})"
)
print(
    f"    Final RK range:      {kriging_data['final_regression_kriging'].min():.4f} to "
    f"{kriging_data['final_regression_kriging'].max():.4f}"
)
print(f"    Mean kriging variance: {kriging_data['kriging_variance'].mean():.4f}")

# =============================================================================
# STEP 7: Save outputs
# =============================================================================

print("\nSTEP 7: Saving outputs...")

t1_path = os.path.join(OUTPUT_DIR, f"table1_kriging_{TARGET}.csv")
kriging_data.to_csv(t1_path, index=False)
print(f"  Saved: {t1_path}")

arcgis_data = pd.merge(
    kriging_data,
    grid_coords[["grid_id", "lat", "lon", "proj_x_epsg3031", "proj_y_epsg3031"]],
    on="grid_id",
)
t2_path = os.path.join(OUTPUT_DIR, f"table2_arcgis_{TARGET}.csv")
arcgis_data.to_csv(t2_path, index=False)
print(f"  Saved: {t2_path}")
print(f"  Saved: {loo_path}")
print(f"  Saved: {fig1_path}")
print(f"  Saved: {fig2_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n{'=' * 70}")
print(f"COMPLETE — {TARGET}")
print(f"{'=' * 70}")
print(
    f"  Variogram:  nugget={nugget:.4f}  partial sill={psill:.4f}  "
    f"range={vrange / 1000:.1f}km  N:S={ns_ratio:.4f}"
)
print(f"  Structure:  {'[FLAT]' if is_flat else '[Spatial structure present]'}")
print(f"  RF R²:      {r2_rf:.4f}")
print(f"  RK R²:      {r2_rk:.4f}  (change: {r2_diff:+.4f})")
print(f"  Output:     {OUTPUT_DIR}")
