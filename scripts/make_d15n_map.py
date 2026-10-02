"""
Build the delta-15N map CSV for ArcGIS.

Run:  python scripts/make_d15n_map.py

Reads
    data/ansoil_grid_prepared.csv        grid coordinates and covariates
    data/ansoil_predictors.csv           the 171 training samples
    results/grid_predictions_mean_rf.csv  RF mean across the 5 seeds
    results/grid_predictions_sd_rf.csv    RF sd across the 5 seeds

Writes
    results/ansoil_d15n_map.csv          all 15,769 ice-free cells
    results/ansoil_d15n_map_<REGION>.csv one per region, same columns

Does three things the raw files do not:
  1. joins mean, sd and coordinates into one row per cell
  2. corrects dist_coast_scar_km on the grid, which is stored in metres
     despite the name while the training table is in km
  3. flags cells where any model input is outside the range seen in training,
     so the map can grey those out instead of pretending to predict there
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # the ansoil-spatial-prediction folder
DATA = ROOT / "data"
RESULTS = ROOT / "results/aggregated"

TARGET = "pred_d15n_air_permil"

# Every numeric input the RF was fitted on, and the flag column each one sets.
# aspect enters the model as sin/cos of aspect_dem, both bounded to [-1, 1],
# so the raw degree column is what gets range-checked.
ENVELOPE = {
    "wgs84_elev_from_pgc": "elev",
    "temperature_racmo": "temp",
    "precipitation_racmo": "precip",
    "dist_coast_scar_km": "coast",
    "slope_dem": "slope",
    "aspect_dem": "aspect",
}
# A one-hot lithology fitted on a handful of samples carries no information.
# This is a support threshold, not an unseen-class test: every lithology on the
# grid does appear in training at least once. Below 5 samples it gets flagged.
LITHO_MIN_SUPPORT = 5

REGIONS = {
    "region_tm": "TAM",
    "region_svl": "SVL",
    "region_nvl": "NVL",
    "region_nwap": "NWAP",
}

grid = pd.read_csv(DATA / "ansoil_grid_prepared.csv")
train = pd.read_csv(DATA / "ansoil_predictors.csv")
mean = pd.read_csv(
    RESULTS / "grid_predictions_mean_rf.csv", usecols=["grid_id", TARGET]
)
sd = pd.read_csv(RESULTS / "grid_predictions_sd_rf.csv", usecols=["grid_id", TARGET])

d = grid.merge(mean.rename(columns={TARGET: "d15n_mean"}), on="grid_id", validate="1:1")
d = d.merge(sd.rename(columns={TARGET: "d15n_sd"}), on="grid_id", validate="1:1")
assert len(d) == len(grid), f"join lost rows: {len(grid)} -> {len(d)}"

# metres to km, so the grid and the training table are comparable
if d.dist_coast_scar_km.max() > 1e4:
    d["dist_coast_scar_km"] /= 1000.0

flags = {}
print(f"training envelope, from {len(train)} samples:")
for col, flag in ENVELOPE.items():
    lo, hi = train[col].min(), train[col].max()
    flags[flag] = (d[col] < lo) | (d[col] > hi)
    print(
        f"  {col:<22} {lo:>10.2f} to {hi:>10.2f}   {flags[flag].sum():>5} cells outside"
    )

support = train.litho.value_counts()
thin = sorted(support.index[support < LITHO_MIN_SUPPORT])
flags["litho"] = d.litho.isin(thin)
print(
    f"  lithology              under-supported {thin}   {flags['litho'].sum():>5} cells"
)

names = list(flags)
stack = np.column_stack([flags[f] for f in names])
d["outside_envelope"] = stack.any(axis=1).astype(int)
d["mask_reason"] = [", ".join(np.array(names)[row]) or "in range" for row in stack]

d["region"] = ""
for col, name in REGIONS.items():
    d.loc[d[col] == 1, "region"] = name

out = d[
    [
        "grid_id",
        "lat",
        "lon",
        "proj_x_epsg3031",
        "proj_y_epsg3031",
        "region",
        "litho",
        "d15n_mean",
        "d15n_sd",
        "outside_envelope",
        "mask_reason",
    ]
].round(
    {
        "d15n_mean": 3,
        "d15n_sd": 4,
        "lat": 6,
        "lon": 6,
        "proj_x_epsg3031": 1,
        "proj_y_epsg3031": 1,
    }
)
assert out.notna().all().all(), "blank cells in output"

RESULTS.mkdir(exist_ok=True)
out.to_csv(RESULTS / "ansoil_d15n_map.csv", index=False)
print(
    f"\nansoil_d15n_map.csv          {len(out):>6} cells   "
    f"{out.outside_envelope.sum()} masked "
    f"({100 * out.outside_envelope.mean():.1f}%)"
)

for name in REGIONS.values():
    sub = out[out.region == name]
    sub.to_csv(RESULTS / f"ansoil_d15n_map_{name}.csv", index=False)
    print(
        f"ansoil_d15n_map_{name + '.csv':<13} {len(sub):>6} cells   "
        f"{sub.outside_envelope.sum()} masked "
        f"({100 * sub.outside_envelope.mean():.1f}%)"
    )

print(f"\nd15n_mean  {out.d15n_mean.min():.2f} to {out.d15n_mean.max():.2f} permil")
print(
    f"d15n_sd    {out.d15n_sd.min():.3f} to {out.d15n_sd.max():.3f} permil "
    f"(spread across the 5 seeds, not prediction uncertainty)"
)
