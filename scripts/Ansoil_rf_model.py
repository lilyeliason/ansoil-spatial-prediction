"""
=============================================================================
ANTARCTIC SOIL GEOCHEMISTRY — RANDOM FOREST MODEL
=============================================================================

QUICK START GUIDE
=================

  1. INSTALL REQUIRED PACKAGES (run once in terminal):

         pip install numpy pandas scikit-learn

  2. PLACE THESE 5 DATA FILES IN THE SAME FOLDER AS THIS SCRIPT:

     File                          What it is
     ─────────────────────────     ──────────────────────────────────────
     ansoil_targets.csv            Target variables (171 rows × 83 cols).
                                   Each column is a soil property to predict
                                   (pH, metals, isotopes, etc). Rows = samples.

     ansoil_predictors.csv         Environmental features (171 rows × 18 cols).
                                   Contains the predictor variables for each
                                   sample: coordinates, elevation, distance to
                                   coast, climate (RACMO), slope, aspect,
                                   lithology class, and region flags.

     ansoil_sample_index.csv       Sample metadata (171 rows × 10 cols).
                                   Links each sample to its sampling location
                                   (e.g., "Shackleton Glacier"). This is what
                                   defines the cross-validation folds — all
                                   samples from the same location are held out
                                   together.

     ansoil_log_targets.csv        Transform lookup table (24 rows × 5 cols).
                                   Tells the script which targets need log
                                   transforms and which get the "dual test"
                                   (try both raw and log, keep the better one).

     ansoil_grid_prepared.csv      Prediction grid (15,769 rows × 17 cols).
                                   Environmental features for each grid point
                                   where we want spatial predictions. Same
                                   columns as predictors.csv but for unsampled
                                   locations across Antarctica.

     These files are all produced by Ansoil_knn_prep.py (the prep pipeline).
     They live in the "prepared/" folder.

     OPTIONAL — for comparison with other models, also place these nearby:
       ansoil_model_results_v6.csv   (KNN results, in results/knn_v6/)

  3. CONFIGURE THE RUN — edit the SEED values:

     ★ SEED = 42 (Line 174)                  Change for each run (e.g., 42, 123, 73, 7, 256)
                                                - After running, change SEED = 42 to a different number,
                                                - Rename or move the output files to a subfolder
                                                - Re-run with a different seed.
  4. RUN IT:

         cd /path/to/folder/with/data/files
         python Ansoil_rf_model.py

     It prints progress as it goes. Each target shows its R² and a bar chart.
     Total time depends on N_RANDOM_SEARCH and the operating system.

  5. OUTPUT FILES (created in the same folder):

     ansoil_model_results_v7_rf.csv             Main results: R² per target
     ansoil_cv_predictions_v7_rf.csv            Predicted vs actual for every sample
     ansoil_grid_predictions_v7_rf.csv          Predictions at all 15,769 grid points
     ansoil_feature_importance_v7_rf.csv        Which features matter for each target
     ansoil_model_comparison_knn_vs_rf.csv      Side-by-side comparison with KNN
     ansoil_models_v7_rf/                       Saved model files (.pkl)


WHAT THIS SCRIPT DOES
======================

  This script predicts 67 Antarctic soil properties from 22 environmental
  features using a Random Forest regression model.

  CROSS-VALIDATION: Leave-One-Location-Out (LOLO) with 28 folds.
  Each fold holds out ALL samples from one sampling location, so the model
  is always tested on a location it has never seen. This gives honest
  estimates of how well the model would predict at a brand new site.

  FEATURES (22 total):
    - 7 continuous: projected X/Y, elevation, distance to coast, precip, temp, slope
    - 2 cyclic: aspect encoded as sin/cos (so 359° and 1° are close)
    - 9 one-hot: lithology classes (a, b, c, d, g, n, p, s, w)
    - 4 binary: region flags (TM, SVL, NVL, NWAP)

  HYPERPARAMETER TUNING: For each target, the script tries N_RANDOM_SEARCH
  random combinations of (max_depth, min_samples_leaf, max_features,
  max_samples) and picks whichever gives the highest R² in cross-validation.

  DUAL-TEST TRANSFORMS: For 13 dissolved-ion targets, the script tries
  both the raw values and log-transformed values and keeps whichever
  produces a better R². Both results are reported for transparency.

  FEATURE IMPORTANCE: After finding the best model for each target, the
  script reports which features are most important (e.g., "elevation
  drives δ15N", "lithology_a drives Na digest").

  The script also auto-detects and fixes a unit mismatch in the grid file
  where distance-to-coast was stored in meters instead of km.

=============================================================================
"""

import math
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")
os.makedirs("ansoil_models_rf", exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Hyperparameter search space — focused for small-n geochemical data
# With n=171 samples and 22 features, 100 trees gives stable predictions
# (Oshiro et al. 2012 show diminishing returns past ~128 trees for small
# datasets). Key tunable parameters: max_depth and min_samples_leaf
# (regularization), max_features (feature decorrelation per split).
# n_estimators is fixed — more trees only smooth variance, not bias.
PARAM_GRID = {
    "n_estimators": [100],
    "max_depth": [None, 15, 30],
    "min_samples_leaf": [1, 3, 5],
    "max_features": ["sqrt", 0.5, 0.75],
    "max_samples": [0.8, None],
}

# Number of random combos to evaluate per target
# 8 combos × 28 folds = 224 RF fits per target (~5-10s on a MacBook Air)
N_RANDOM_SEARCH = 300

# Minimum R² to produce grid predictions
MIN_R2_FOR_GRID = 0.0

# Targets to always exclude (constant / below detection limit)
EXCLUDE_TARGETS = [
    "digest_mg_kg_cd2288",
    "log_digest_mg_kg_se1960",
]

# For KNN comparison (optional — loads if the file exists)
KNN_RESULTS_FILE = "ansoil_model_results_v6.csv"

SEED = 42

# Threshold (meters) above which grid dist_coast is assumed to be in meters
DIST_COAST_UNIT_THRESHOLD = 1000.0


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================


def build_feature_matrix(df):
    """
    Build the RF feature matrix from a predictor-style DataFrame.

    Returns (X: pd.DataFrame, feature_names: list)

    Design choices:
      - No normalization: RF splits are rank-based, scale doesn't matter
      - Cyclic aspect: sin/cos encoding so 359° ≈ 1° (not 358 apart)
      - One-hot lithology: lets RF learn per-class effects directly,
        more flexible than the manual dissimilarity matrix
      - Region flags: already binary, used as-is

    Features (22 total):
      7 continuous  — proj_x, proj_y, elev, dist_coast, precip, temp, slope
      2 cyclic      — aspect_sin, aspect_cos
      9 litho       — litho_a through litho_w (one-hot)
      4 region      — region_tm, region_svl, region_nvl, region_nwap
    """
    X = pd.DataFrame(index=df.index)

    # Continuous (no scaling)
    for c in [
        "proj_x_epsg3031",
        "proj_y_epsg3031",
        "wgs84_elev_from_pgc",
        "dist_coast_scar_km",
        "precipitation_racmo",
        "temperature_racmo",
        "slope_dem",
    ]:
        X[c] = df[c].astype(float).values

    # Cyclic aspect
    aspect_rad = np.deg2rad(df["aspect_dem"].astype(float).values)
    X["aspect_sin"] = np.sin(aspect_rad)
    X["aspect_cos"] = np.cos(aspect_rad)

    # One-hot lithology
    for lc in ["a", "b", "c", "d", "g", "n", "p", "s", "w"]:
        X[f"litho_{lc}"] = (df["litho"].astype(str) == lc).astype(int).values

    # Region flags
    for rc in ["region_tm", "region_svl", "region_nvl", "region_nwap"]:
        X[rc] = df[rc].astype(int).values if rc in df.columns else 0

    return X, list(X.columns)


# =============================================================================
# LOADING AND VALIDATION
# =============================================================================


def load_inputs():
    print("=" * 70)
    print("STEP 1  Load and validate inputs")
    print("=" * 70)

    targets_df = pd.read_csv("ansoil_targets.csv")
    pred_df = pd.read_csv("ansoil_predictors.csv")
    index_df = pd.read_csv("ansoil_sample_index.csv")

    # Alignment check
    assert list(targets_df["sample_id"]) == list(pred_df["sample_id"]), (
        "Row order mismatch: targets vs predictors"
    )
    assert list(targets_df["sample_id"]) == list(index_df["sample_id"]), (
        "Row order mismatch: targets vs sample index"
    )
    print("  Alignment check: PASSED")

    # Build training feature matrix
    X_train, feature_names = build_feature_matrix(pred_df)
    print(
        f"  Training features: {X_train.shape[1]}  ({', '.join(feature_names[:5])}...)"
    )
    print(f"  Training samples:  {X_train.shape[0]}")

    # Log target lookup
    log_lookup = pd.read_csv("ansoil_log_targets.csv")
    established_log_info = {
        row["log_col"]: {
            "raw_col": row["raw_col"],
            "back_transform": row["back_transform"],
        }
        for _, row in log_lookup.iterrows()
        if not row.get("dual_test", False)
    }
    dual_test_info = {
        row["raw_col"]: {
            "log_col": row["log_col"],
            "back_transform": row["back_transform"],
        }
        for _, row in log_lookup.iterrows()
        if row.get("dual_test", False)
    }
    n_log1p = log_lookup.loc[log_lookup["transform"] == "log1p"].shape[0]
    n_dual = log_lookup.loc[log_lookup["dual_test"] == True].shape[0]
    print(f"  Log target lookup: {len(log_lookup)} entries")
    print(f"    {n_log1p} established log1p  |  {n_dual} dual-test candidates")

    # Grid features
    has_grid = False
    X_grid, grid_ids = None, None
    try:
        grid_df = pd.read_csv("ansoil_grid_prepared.csv")

        # ── FIX UNIT MISMATCH ─────────────────────────────────────────────
        # Grid dist_coast_scar_km is in meters in the raw grid file.
        # Training data is in km. Detect and convert.
        grid_max_dc = grid_df["dist_coast_scar_km"].max()
        train_max_dc = pred_df["dist_coast_scar_km"].max()
        if (
            grid_max_dc > DIST_COAST_UNIT_THRESHOLD
            and train_max_dc < DIST_COAST_UNIT_THRESHOLD
        ):
            print("\n  !! UNIT FIX: grid dist_coast_scar_km appears to be in meters")
            print(f"     Grid max: {grid_max_dc:.1f}  |  Train max: {train_max_dc:.1f}")
            grid_df["dist_coast_scar_km"] = grid_df["dist_coast_scar_km"] / 1000.0
            print(
                f"     Converted to km → new max: {grid_df['dist_coast_scar_km'].max():.1f} km"
            )
            print(
                "     NOTE: This mismatch also affected KNN grid predictions (via scaler)."
            )
            print(
                "     KNN CV results were NOT affected (training data self-consistent).\n"
            )

        grid_ids = grid_df["grid_id"].values
        X_grid, _ = build_feature_matrix(grid_df)
        has_grid = True
        print(f"  Grid features: {X_grid.shape}  ({len(grid_ids)} points)")
    except FileNotFoundError:
        print("  Grid file not found — grid prediction skipped")

    # Target list
    all_tgt = [c for c in targets_df.columns if c != "sample_id"]
    constant = [c for c in all_tgt if targets_df[c].std() == 0]
    excluded = list(set(EXCLUDE_TARGETS + constant))

    dual_raw_cols = list(dual_test_info.keys())
    dual_log_cols = [info["log_col"] for info in dual_test_info.values()]
    dual_related = set(dual_raw_cols + dual_log_cols)

    regular_modelable = [
        c for c in all_tgt if c not in excluded and c not in dual_related
    ]

    print(f"\n  Total target columns:     {len(all_tgt)}")
    print(f"  Excluded (BDL/const):     {len(excluded)}")
    print(f"  Dual-test candidates:     {len(dual_raw_cols)}")
    print(f"  Regular targets:          {len(regular_modelable)}")
    print(f"  CV folds (locations):     {index_df['sample_location'].nunique()}")

    return (
        X_train,
        X_grid,
        targets_df,
        index_df,
        feature_names,
        regular_modelable,
        excluded,
        grid_ids,
        has_grid,
        established_log_info,
        dual_test_info,
    )


# =============================================================================
# HELPERS
# =============================================================================


def back_transform(values, back_fn):
    arr = np.asarray(values, dtype=float)
    if back_fn == "exp":
        return np.clip(np.exp(arr), 0, None)
    elif back_fn == "expm1":
        return np.clip(np.expm1(arr), 0, None)
    else:
        raise ValueError(f"Unknown back_transform: '{back_fn}'")


def assign_tier(r2):
    if r2 > 0.5:
        return "strong"
    if r2 > 0.3:
        return "moderate"
    if r2 > 0.0:
        return "weak"
    return "unusable"


def generate_param_combos(n):
    """
    Generate n random hyperparameter combinations from PARAM_GRID.
    Uses deterministic seeding for reproducibility.
    Ensures all values are native Python types (not numpy).
    """
    rng = np.random.RandomState(SEED)
    combos = []
    keys = sorted(PARAM_GRID.keys())
    for _ in range(n):
        combo = {}
        for k in keys:
            vals = PARAM_GRID[k]
            idx = rng.randint(0, len(vals))
            v = vals[idx]
            # Convert numpy types to native Python
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.str_, np.bytes_)):
                v = str(v)
            combo[k] = v
        # max_features=1.0 means all features — use None for sklearn compat
        if (
            isinstance(combo.get("max_features"), float)
            and combo["max_features"] >= 1.0
        ):
            combo["max_features"] = None
        combos.append(combo)
    return combos


# =============================================================================
# CORE MODELING — ONE TARGET
# =============================================================================


def tune_and_evaluate(X_train, y, blocks, feature_names):
    """
    Randomized hyperparameter search with LOLO-CV.

    For each candidate parameter set:
      1. Run full LOLO-CV (28 folds)
      2. Collect OOF predictions
      3. Compute R² and RMSE

    Returns: (best_params, best_r2, best_rmse, tuning_df, best_oof)
    """
    logo = LeaveOneGroupOut()
    combos = generate_param_combos(N_RANDOM_SEARCH)
    records = []

    for ci, params in enumerate(combos):
        actuals, preds, indices = [], [], []

        for tr_idx, te_idx in logo.split(X_train, y, groups=blocks):
            rf = RandomForestRegressor(
                **params,
                n_jobs=-1,
                random_state=SEED,
                oob_score=False,
            )
            rf.fit(X_train[tr_idx], y[tr_idx])
            p = rf.predict(X_train[te_idx])
            actuals.extend(y[te_idx])
            preds.extend(p)
            indices.extend(te_idx)

        actuals = np.array(actuals)
        preds = np.array(preds)
        r2 = r2_score(actuals, preds)
        rmse = math.sqrt(mean_squared_error(actuals, preds))

        records.append(
            {
                **params,
                "r2": r2,
                "rmse": rmse,
                "_actuals": actuals,
                "_preds": preds,
                "_indices": indices,
            }
        )

        # Progress indicator (prints on same line)
        print(
            f"\r    tuning: {ci + 1}/{len(combos)} combos  "
            f"(best so far: R²={max(r['r2'] for r in records):+.3f})",
            end="",
            flush=True,
        )
    print()  # newline after progress

    tuning_df_full = sorted(records, key=lambda x: -x["r2"])
    best_record = tuning_df_full[0]

    # Build clean tuning log (without cached arrays)
    tuning_df = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in tuning_df_full]
    )

    # Extract best params — go back to the original combo dict to avoid
    # pandas NaN conversion (pandas turns None → NaN in numeric columns)
    best_params = {k: best_record[k] for k in sorted(PARAM_GRID.keys())}

    # Defensive type cleaning for sklearn compatibility
    def _clean(v):
        """Convert any numpy/pandas type to native Python."""
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.str_, np.bytes_)):
            return str(v)
        return v

    for k in best_params:
        best_params[k] = _clean(best_params[k])
    # Ensure int types where sklearn expects them
    for k in ["n_estimators", "min_samples_leaf"]:
        if best_params[k] is not None:
            best_params[k] = int(best_params[k])
    if best_params.get("max_depth") is not None:
        best_params["max_depth"] = int(best_params["max_depth"])

    best_r2 = float(best_record["r2"])
    best_rmse = float(best_record["rmse"])

    # Use cached OOF predictions from the best combo (no re-run needed)
    oof_actuals = best_record["_actuals"]
    oof_preds = best_record["_preds"]
    oof_indices = np.array(best_record["_indices"])

    return (
        best_params,
        best_r2,
        best_rmse,
        tuning_df,
        np.array(oof_actuals),
        np.array(oof_preds),
        np.array(oof_indices),
    )


def model_one_target(
    target,
    X_train_full,
    targets_df,
    blocks_full,
    sample_ids_full,
    X_grid,
    has_grid,
    log_info,
    feature_names,
    target_label=None,
):
    """
    Full pipeline for one target: tune → evaluate → importance → grid predict.
    """
    if target_label is None:
        target_label = target

    y_full = targets_df[target].values
    is_log = target in log_info
    back_fn = log_info[target]["back_transform"] if is_log else None
    raw_col = log_info[target]["raw_col"] if is_log else target

    # Handle NaN (some targets may have missing values)
    valid_mask = ~np.isnan(y_full)
    if valid_mask.sum() < 20:
        print(f"    SKIP {target}: only {valid_mask.sum()} non-NaN values")
        return None, None, None, None
    y = y_full[valid_mask]
    X_train = X_train_full[valid_mask]
    blocks = blocks_full[valid_mask]
    sids = sample_ids_full[valid_mask]

    # Tune + evaluate
    best_params, cv_r2, cv_rmse_log, tuning_df, oof_act, oof_pred, oof_idx = (
        tune_and_evaluate(X_train, y, blocks, feature_names)
    )

    # Build OOF DataFrame
    oof_rows = []
    for i in range(len(oof_act)):
        row = {"sample_id": sids[oof_idx[i]]}
        if is_log:
            row["actual_log"] = float(oof_act[i])
            row["pred_log"] = float(oof_pred[i])
            row["actual"] = float(back_transform([oof_act[i]], back_fn)[0])
            row["predicted"] = float(back_transform([oof_pred[i]], back_fn)[0])
        else:
            row["actual_log"] = np.nan
            row["pred_log"] = np.nan
            row["actual"] = float(oof_act[i])
            row["predicted"] = float(oof_pred[i])
        row["residual"] = row["predicted"] - row["actual"]
        oof_rows.append(row)
    oof_df = pd.DataFrame(oof_rows)
    oof_df.insert(0, "target", target_label)

    # RMSE in original units
    if is_log:
        cv_rmse_orig = math.sqrt(
            mean_squared_error(oof_df["actual"], oof_df["predicted"])
        )
    else:
        cv_rmse_orig = cv_rmse_log

    # Train final model on ALL data
    final_rf = RandomForestRegressor(**best_params, n_jobs=-1, random_state=SEED)
    final_rf.fit(X_train, y)

    # Feature importance — use impurity-based (instant, built-in to RF)
    # Permutation importance is more rigorous but too slow for 80+ model fits.
    # Impurity importance is adequate for ranking features within each target
    # and is the standard in environmental ML (Breiman 2001).
    importance_dict = {
        feature_names[i]: float(final_rf.feature_importances_[i])
        for i in range(len(feature_names))
    }

    # Grid prediction
    grid_col_label, grid_preds = None, None
    if has_grid and X_grid is not None and cv_r2 >= MIN_R2_FOR_GRID:
        raw_preds = final_rf.predict(X_grid)
        if is_log and back_fn:
            grid_preds = back_transform(raw_preds, back_fn)
            grid_col_label = f"pred_{raw_col}"
        else:
            grid_preds = raw_preds
            grid_col_label = f"pred_{target}"

    tier = assign_tier(cv_r2)
    n_folds = len(np.unique(blocks))

    summary = {
        "target": target,
        "raw_col": raw_col,
        "log_transformed": is_log,
        "selected_transform": (
            "log1p" if is_log and back_fn == "expm1" else "log" if is_log else "raw"
        ),
        "back_transform": back_fn if back_fn else "none",
        "best_n_estimators": best_params["n_estimators"],
        "best_max_depth": best_params["max_depth"],
        "best_min_samples_leaf": best_params["min_samples_leaf"],
        "best_max_features": best_params["max_features"],
        "best_max_samples": best_params["max_samples"],
        "cv_r2": cv_r2,
        "cv_rmse_log_space": cv_rmse_log,
        "cv_rmse_orig_units": cv_rmse_orig,
        "tier": tier,
        "n_folds": n_folds,
        "n_samples": int(valid_mask.sum()),
        "grid": (
            "predicted"
            if grid_preds is not None
            else (f"skipped R²={cv_r2:.3f}" if has_grid else "no grid file")
        ),
    }

    # Save model pickle
    with open(f"ansoil_models_rf/{target}.pkl", "wb") as f:
        pickle.dump(
            {
                "model": final_rf,
                "target": target,
                "raw_col": raw_col,
                "back_transform": back_fn,
                "best_params": best_params,
                "cv_r2": cv_r2,
                "cv_rmse_orig": cv_rmse_orig,
                "feature_names": feature_names,
                "importance": importance_dict,
                "tuning_results": tuning_df,
            },
            f,
        )

    return summary, oof_df, (grid_col_label, grid_preds), importance_dict


# =============================================================================
# MAIN MODELING FUNCTION
# =============================================================================


def run_all_models(
    X_train_np,
    X_grid_np,
    targets_df,
    index_df,
    feature_names,
    regular_modelable,
    excluded,
    grid_ids,
    has_grid,
    established_log_info,
    dual_test_info,
):

    print("\n" + "=" * 70)
    print("STEP 2  LOLO-CV + Randomized Hyperparameter Search")
    print("=" * 70)
    print(f"  {N_RANDOM_SEARCH} random combos × 28 LOLO folds per target")
    print(f"  Seed: {SEED}\n")

    blocks_full = index_df["sample_location"].values
    sample_ids_full = index_df["sample_id"].values

    all_oof = []
    model_summary = []
    grid_pred_cols = {}
    dual_comparison = []
    all_importances = []

    total = len(regular_modelable) + len(dual_test_info)
    counter = [0]

    def print_progress(target, cv_r2, tier, is_log, params, extra=""):
        counter[0] += 1
        flag = {"strong": "**", "moderate": "* ", "weak": ". ", "unusable": "x "}[tier]
        log_marker = "[L]" if is_log else "   "
        bar = chr(9608) * int(max(0, cv_r2) * 20)
        depth_str = (
            str(params.get("best_max_depth", ""))
            if params.get("best_max_depth")
            else "None"
        )
        print(
            f"  [{counter[0]:02d}/{total:02d}]  {target:<42} "
            f"R²={cv_r2:>+.3f}  n_est={params.get('best_n_estimators', '?'):<5} "
            f"depth={depth_str:<5} {flag} {log_marker} {extra} {bar}"
        )

    # ======================================================================
    # PASS 1: Regular targets
    # ======================================================================
    print(f"\n  Pass 1: {len(regular_modelable)} regular targets")
    print(f"  {'─' * 90}")

    t0 = time.time()
    for target in regular_modelable:
        result = model_one_target(
            target,
            X_train_np,
            targets_df,
            blocks_full,
            sample_ids_full,
            X_grid_np,
            has_grid,
            established_log_info,
            feature_names,
        )
        if result[0] is None:
            continue
        summary, oof_df, (gcol, gpred), importance = result
        all_oof.append(oof_df)
        model_summary.append(summary)
        if gcol and gpred is not None:
            grid_pred_cols[gcol] = gpred
        all_importances.append({"target": target, **importance})
        print_progress(
            target,
            summary["cv_r2"],
            summary["tier"],
            summary["log_transformed"],
            summary,
        )

    elapsed = time.time() - t0
    print(
        f"\n  Pass 1 complete: {elapsed:.0f}s ({elapsed / len(regular_modelable):.1f}s/target)"
    )

    # ======================================================================
    # PASS 2: Dual-test candidates
    # ======================================================================
    print(f"\n  Pass 2: {len(dual_test_info)} dual-test candidates")
    print("  (each modeled TWICE — raw AND log — better R² selected)")
    print(f"  {'─' * 90}")

    t0 = time.time()
    for raw_col, info in dual_test_info.items():
        log_col = info["log_col"]
        back_fn = info["back_transform"]

        has_raw = raw_col in targets_df.columns
        has_log = log_col in targets_df.columns

        if not has_raw and not has_log:
            print(f"  SKIP: neither {raw_col} nor {log_col} found")
            continue

        # Model raw
        r2_raw = np.nan
        if has_raw:
            res_raw = model_one_target(
                raw_col,
                X_train_np,
                targets_df,
                blocks_full,
                sample_ids_full,
                X_grid_np,
                has_grid,
                {},
                feature_names,  # empty log_info → raw
            )
            if res_raw[0] is not None:
                r2_raw = res_raw[0]["cv_r2"]

        # Model log
        r2_log = np.nan
        if has_log:
            temp_log_info = {log_col: {"raw_col": raw_col, "back_transform": back_fn}}
            res_log = model_one_target(
                log_col,
                X_train_np,
                targets_df,
                blocks_full,
                sample_ids_full,
                X_grid_np,
                has_grid,
                temp_log_info,
                feature_names,
            )
            if res_log[0] is not None:
                r2_log = res_log[0]["cv_r2"]

        # Select winner (ties go to raw — simpler, no back-transform)
        if not np.isnan(r2_raw) and not np.isnan(r2_log):
            winner = "raw" if r2_raw >= r2_log else "log"
        elif not np.isnan(r2_raw):
            winner = "raw"
        else:
            winner = "log"

        if winner == "raw" and res_raw[0] is not None:
            final_sum, final_oof, (final_gcol, final_gpred), final_imp = res_raw
        elif res_log[0] is not None:
            final_sum, final_oof, (final_gcol, final_gpred), final_imp = res_log
        else:
            continue

        delta = (
            (r2_log - r2_raw)
            if not np.isnan(r2_raw) and not np.isnan(r2_log)
            else np.nan
        )

        note = ""
        if raw_col == "digest_mg_kg_na5895":
            note = f"Na: {'raw won as expected' if winner == 'raw' else 'UNEXPECTED — log won'}"

        dual_comparison.append(
            {
                "raw_col": raw_col,
                "r2_raw": round(r2_raw, 6) if not np.isnan(r2_raw) else np.nan,
                "r2_log": round(r2_log, 6) if not np.isnan(r2_log) else np.nan,
                "selected_transform": winner,
                "delta_r2": round(delta, 6) if not np.isnan(delta) else np.nan,
                "winner_note": note,
            }
        )

        all_oof.append(final_oof)
        model_summary.append(final_sum)
        if final_gcol and final_gpred is not None:
            grid_pred_cols[final_gcol] = final_gpred
        all_importances.append({"target": final_sum["target"], **final_imp})

        r2r = f"{r2_raw:+.3f}" if not np.isnan(r2_raw) else "  N/A"
        r2l = f"{r2_log:+.3f}" if not np.isnan(r2_log) else "  N/A"
        d_str = f"{delta:+.3f}" if not np.isnan(delta) else "  N/A"
        counter[0] += 1
        print(
            f"  [{counter[0]:02d}/{total:02d}]  {raw_col:<42} "
            f"raw={r2r}  log={r2l}  >> {winner:<5}  Δ={d_str}  {note}"
        )

    elapsed = time.time() - t0
    n_dual = len(dual_test_info)
    if n_dual > 0:
        print(
            f"\n  Pass 2 complete: {elapsed:.0f}s ({elapsed / n_dual:.1f}s/candidate)"
        )

    # ======================================================================
    # SAVE OUTPUTS
    # ======================================================================

    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    # Model results
    results_df = pd.DataFrame(model_summary).sort_values("cv_r2", ascending=False)
    results_df.to_csv("ansoil_model_results_rf.csv", index=False)
    print(f"  ansoil_model_results_rf.csv          {results_df.shape}")

    # Dual-test comparison
    dual_df = pd.DataFrame(dual_comparison).sort_values("r2_raw", ascending=False)
    dual_df.to_csv("ansoil_transform_comparison_rf.csv", index=False)
    print(f"  ansoil_transform_comparison_rf.csv   {dual_df.shape}")

    # OOF predictions
    oof_all = pd.concat(all_oof, ignore_index=True)
    oof_all.to_csv("ansoil_cv_predictions_rf.csv", index=False)
    print(f"  ansoil_cv_predictions_rf.csv         {oof_all.shape}")

    # Grid predictions
    if has_grid and grid_pred_cols:
        grid_out = pd.DataFrame({"grid_id": grid_ids})
        for col_label, arr in grid_pred_cols.items():
            grid_out[col_label] = arr
        grid_out.to_csv("ansoil_grid_predictions_rf.csv", index=False)
        print(
            f"  ansoil_grid_predictions_rf.csv       "
            f"({len(grid_ids)} x {len(grid_pred_cols) + 1})"
        )

    # Feature importance
    imp_df = pd.DataFrame(all_importances)
    imp_df.to_csv("ansoil_feature_importance_rf.csv", index=False)
    print(f"  ansoil_feature_importance_rf.csv     {imp_df.shape}")
    print(f"  ansoil_models_rf/                    {len(model_summary)} .pkl files")

    # ======================================================================
    # KNN vs RF COMPARISON (if KNN results available)
    # ======================================================================
    try:
        knn_df = pd.read_csv(KNN_RESULTS_FILE)
        knn_r2 = knn_df.set_index("target")["cv_r2"].to_dict()
        comp_rows = []
        for _, row in results_df.iterrows():
            t = row["target"]
            knn_val = knn_r2.get(t, np.nan)
            comp_rows.append(
                {
                    "target": t,
                    "r2_knn_v6": round(knn_val, 6) if not np.isnan(knn_val) else np.nan,
                    "r2_rf_v7": round(row["cv_r2"], 6),
                    "delta_r2": round(row["cv_r2"] - knn_val, 6)
                    if not np.isnan(knn_val)
                    else np.nan,
                    "rf_tier": row["tier"],
                }
            )
        comp_df = pd.DataFrame(comp_rows).sort_values("delta_r2", ascending=False)
        comp_df.to_csv("ansoil_model_comparison_knn_vs_rf.csv", index=False)
        print(f"  ansoil_model_comparison_knn_vs_rf.csv  {comp_df.shape}")

        # Summary stats
        matched = comp_df.dropna(subset=["delta_r2"])
        rf_won = (matched["delta_r2"] > 0).sum()
        knn_won = (matched["delta_r2"] < 0).sum()
        tied = (matched["delta_r2"] == 0).sum()
        mean_d = matched["delta_r2"].mean()
        print(
            f"\n  KNN vs RF:  RF better on {rf_won}/{len(matched)} targets  "
            f"|  KNN better on {knn_won}  |  tied: {tied}"
        )
        print(f"  Mean ΔR² (RF − KNN): {mean_d:+.4f}")
    except FileNotFoundError:
        print(f"\n  {KNN_RESULTS_FILE} not found — skipping KNN comparison")
        comp_df = None

    # ======================================================================
    # RESULTS SUMMARY
    # ======================================================================

    r2v = results_df["cv_r2"]
    n_log = results_df["log_transformed"].sum()
    n_raw = len(results_df) - n_log

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY — RANDOM FOREST")
    print("=" * 70)
    print(f"\n  {n_log} targets in log space  |  {n_raw} targets raw")
    print("  [L] = log-transformed  |  RMSE in original units\n")

    for label, mask in [
        ("STRONG    R² > 0.50", r2v > 0.5),
        ("MODERATE  R² 0.30-0.50", (r2v >= 0.3) & (r2v <= 0.5)),
        ("WEAK      R² 0.00-0.30", (r2v >= 0.0) & (r2v < 0.3)),
        ("UNUSABLE  R² < 0.00", r2v < 0.0),
    ]:
        grp = results_df[mask]
        print(f"  {label}  (n={len(grp)})")
        for _, row in grp.iterrows():
            log_tag = "[L]" if row["log_transformed"] else "   "
            print(
                f"    {log_tag} {row['target']:<44}  R²={row['cv_r2']:+.3f}  "
                f"RMSE={row['cv_rmse_orig_units']:.4g}  "
                f"n_est={int(row['best_n_estimators'])}  "
                f"depth={'None' if pd.isna(row['best_max_depth']) else int(row['best_max_depth'])}"
            )
        print()

    # Top feature importances (averaged across all targets)
    if len(all_importances) > 1:
        imp_means = imp_df.drop(columns=["target"]).mean().sort_values(ascending=False)
        print("  TOP 10 FEATURES (mean permutation importance across all targets):")
        for feat, val in imp_means.head(10).items():
            bar = chr(9608) * int(val * 100)
            print(f"    {feat:<28}  {val:+.4f}  {bar}")
        print()

    if dual_comparison:
        print("  DUAL-TRANSFORM COMPARISON:")
        print(
            f"  {'Raw column':<42}  {'R²(raw)':>8}  {'R²(log)':>8}  "
            f"{'winner':<6}  {'ΔR²':>7}"
        )
        print(f"  {'─' * 76}")
        for row in sorted(dual_comparison, key=lambda x: -(x.get("r2_raw") or -999)):
            r2r = (
                f"{row['r2_raw']:+.3f}" if not pd.isna(row.get("r2_raw")) else "   N/A"
            )
            r2l = (
                f"{row['r2_log']:+.3f}" if not pd.isna(row.get("r2_log")) else "   N/A"
            )
            dr = (
                f"{row['delta_r2']:+.3f}"
                if not pd.isna(row.get("delta_r2"))
                else "   N/A"
            )
            note = f"  ← {row['winner_note']}" if row.get("winner_note") else ""
            print(
                f"  {row['raw_col']:<42}  {r2r:>8}  {r2l:>8}  "
                f"{row['selected_transform']:<6}  {dr:>7}{note}"
            )

    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    return results_df, oof_all, dual_df, comp_df


# =============================================================================
# ENTRY POINT
# =============================================================================


def run():
    (
        X_train,
        X_grid,
        targets_df,
        index_df,
        feature_names,
        regular_modelable,
        excluded,
        grid_ids,
        has_grid,
        established_log_info,
        dual_test_info,
    ) = load_inputs()

    # Convert to numpy for indexing compatibility
    X_train_np = X_train.values
    X_grid_np = X_grid.values if X_grid is not None else None

    results_df, oof_df, dual_df, comp_df = run_all_models(
        X_train_np,
        X_grid_np,
        targets_df,
        index_df,
        feature_names,
        regular_modelable,
        excluded,
        grid_ids,
        has_grid,
        established_log_info,
        dual_test_info,
    )
    return results_df, oof_df, dual_df, comp_df


if __name__ == "__main__":
    results_df, oof_df, dual_df, comp_df = run()
