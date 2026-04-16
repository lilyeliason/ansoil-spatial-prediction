"""
ANTARCTIC SOIL GEOCHEMISTRY - XGBoost Model
============================================

Predicts 67 Antarctic soil properties from 22 environmental features
using XGBoost (gradient boosted trees) with Leave-One-Location-Out
cross-validation. No early stopping is used, so all R-squared estimates
are fully unbiased.

Setup:
    pip install numpy pandas scikit-learn xgboost
    On macOS also: brew install libomp

Input files (reads from ../data/):
    ansoil_targets.csv          - 67 soil properties to predict (171 samples)
    ansoil_predictors.csv       - 22 environmental features per sample
    ansoil_sample_index.csv     - Sample locations (defines 28 CV folds)
    ansoil_log_targets.csv      - Transform lookup table
    ansoil_grid_prepared.csv    - 15,769 prediction grid points

Output files (written to current directory):
    ansoil_model_results_xgb.csv          - R-squared per target (main results)
    ansoil_cv_predictions_xgb.csv         - Predicted vs actual per sample
    ansoil_grid_predictions_xgb.csv       - Spatial predictions at grid points
    ansoil_feature_importance_xgb.csv     - Feature importance per target
    ansoil_transform_comparison_xgb.csv   - Raw vs log transform results
    ansoil_model_comparison_xgb.csv       - KNN vs RF vs XGBoost comparison
    ansoil_models_xgb/                    - Saved model files

How to run:
    cd scripts
    python Ansoil_xgb_model.py

    *** CHANGE THIS FOR EACH RUN: 42, 73, 123, 7, 256 ***
    SEED = 42

    After each run, move output files to a subfolder (e.g., ../results/xgb_seed42/)
    before changing the seed and running again.

    N_RANDOM_SEARCH and n_jobs are already configured for a fast machine
    (500 combos, all CPU cores). Expected runtime: ~2-6 hours per seed.
"""

import math
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
os.makedirs("ansoil_models_xgb", exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

# n_estimators is TUNED (not set via early stopping).
# Coupled with learning_rate: low lr needs more trees, high lr needs fewer.
PARAM_GRID = {
    "n_estimators": [30, 50, 80, 120, 200, 300],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 6, 8],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
}
# This is a subset of the full grid for faster tuning. The full grid has 6×4×4×4×3×3×3 = 7,776 combos.

# Fixed parameters
FIXED_PARAMS = {
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "verbosity": 0,
    "n_jobs": -1,
}

N_RANDOM_SEARCH = 500
MIN_R2_FOR_GRID = 0.0

EXCLUDE_TARGETS = [
    "digest_mg_kg_cd2288",
    "log_digest_mg_kg_se1960",
]

KNN_RESULTS_FILE = "../reference_results/ansoil_model_results_knn.csv"
RF_RESULTS_FILE = "../reference_results/ansoil_model_results_rf_seed42.csv"

SEED = 123
DIST_COAST_UNIT_THRESHOLD = 1000.0


# =============================================================================
# FEATURE ENGINEERING (identical to v7/v8)
# =============================================================================


def build_feature_matrix(df):
    X = pd.DataFrame(index=df.index)
    for col in [
        "proj_x_epsg3031",
        "proj_y_epsg3031",
        "wgs84_elev_from_pgc",
        "dist_coast_scar_km",
        "precipitation_racmo",
        "temperature_racmo",
        "slope_dem",
    ]:
        X[col] = df[col].astype(float).values
    aspect_rad = np.deg2rad(df["aspect_dem"].astype(float).values)
    X["aspect_sin"] = np.sin(aspect_rad)
    X["aspect_cos"] = np.cos(aspect_rad)
    for lc in ["a", "b", "c", "d", "g", "n", "p", "s", "w"]:
        X[f"litho_{lc}"] = (df["litho"].astype(str) == lc).astype(int).values
    for rc in ["region_tm", "region_svl", "region_nvl", "region_nwap"]:
        X[rc] = df[rc].astype(int).values if rc in df.columns else 0
    return X, list(X.columns)


# =============================================================================
# LOADING AND VALIDATION (identical to v8)
# =============================================================================


def load_inputs():
    print("=" * 70)
    print("STEP 1  Load and validate inputs")
    print("=" * 70)

    targets_df = pd.read_csv("../data/ansoil_targets.csv")
    pred_df = pd.read_csv("../data/ansoil_predictors.csv")
    index_df = pd.read_csv("../data/ansoil_sample_index.csv")

    assert list(targets_df["sample_id"]) == list(pred_df["sample_id"])
    assert list(targets_df["sample_id"]) == list(index_df["sample_id"])
    print("  Alignment check: PASSED")

    X_train, feature_names = build_feature_matrix(pred_df)
    print(f"  Training features: {X_train.shape[1]}  samples: {X_train.shape[0]}")

    log_lookup = pd.read_csv("../data/ansoil_log_targets.csv")
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
    print(f"  Log targets: {n_log1p} log1p  |  {n_dual} dual-test")

    has_grid, X_grid, grid_ids = False, None, None
    try:
        grid_df = pd.read_csv("../data/ansoil_grid_prepared.csv")
        gmax = grid_df["dist_coast_scar_km"].max()
        tmax = pred_df["dist_coast_scar_km"].max()
        if gmax > DIST_COAST_UNIT_THRESHOLD and tmax < DIST_COAST_UNIT_THRESHOLD:
            print("  !! UNIT FIX: grid dist_coast m -> km")
            grid_df["dist_coast_scar_km"] /= 1000.0
        grid_ids = grid_df["grid_id"].values
        X_grid, _ = build_feature_matrix(grid_df)
        has_grid = True
        print(f"  Grid: {X_grid.shape}")
    except FileNotFoundError:
        print("  Grid file not found")

    all_tgt = [c for c in targets_df.columns if c != "sample_id"]
    constant = [c for c in all_tgt if targets_df[c].std() == 0]
    excluded = list(set(EXCLUDE_TARGETS + constant))
    dual_raw = list(dual_test_info.keys())
    dual_log = [v["log_col"] for v in dual_test_info.values()]
    dual_related = set(dual_raw + dual_log)
    regular = [c for c in all_tgt if c not in excluded and c not in dual_related]

    print(
        f"  Regular: {len(regular)}  Dual-test: {len(dual_raw)}  Excluded: {len(excluded)}"
    )
    print(f"  LOLO folds: {index_df['sample_location'].nunique()}")
    print("\n  ** NO EARLY STOPPING — fully leak-free evaluation **")

    return (
        X_train,
        X_grid,
        targets_df,
        index_df,
        feature_names,
        regular,
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
    raise ValueError(f"Unknown: {back_fn}")


def assign_tier(r2):
    if r2 > 0.5:
        return "strong"
    if r2 > 0.3:
        return "moderate"
    if r2 > 0.0:
        return "weak"
    return "unusable"


def generate_param_combos(n):
    rng = np.random.RandomState(SEED)
    combos = []
    keys = sorted(PARAM_GRID.keys())
    for _ in range(n):
        combo = {}
        for k in keys:
            vals = PARAM_GRID[k]
            v = vals[rng.randint(0, len(vals))]
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.str_, np.bytes_)):
                v = str(v)
            combo[k] = v
        combos.append(combo)
    return combos


# =============================================================================
# CORE MODELING
# =============================================================================


def tune_and_evaluate(X_train, y, blocks, feature_names):
    """
    Randomized hyperparameter search with LOLO-CV.
    NO EARLY STOPPING — test fold is never seen during training.

    n_estimators is tuned as a regular hyperparameter.
    The model trains on the training fold ONLY and predicts the test fold
    ONLY after training is complete. This is fully unbiased.
    """
    logo = LeaveOneGroupOut()
    combos = generate_param_combos(N_RANDOM_SEARCH)
    records = []

    for ci, tuned_params in enumerate(combos):
        params = {**FIXED_PARAMS, **tuned_params, "random_state": SEED}

        actuals, preds, indices = [], [], []
        for tr_idx, te_idx in logo.split(X_train, y, groups=blocks):
            xgb = XGBRegressor(**params)
            # NO eval_set, NO early_stopping — clean training
            xgb.fit(X_train[tr_idx], y[tr_idx])
            p = xgb.predict(X_train[te_idx])
            actuals.extend(y[te_idx])
            preds.extend(p)
            indices.extend(te_idx)

        actuals = np.array(actuals)
        preds = np.array(preds)
        r2 = r2_score(actuals, preds)
        rmse = math.sqrt(mean_squared_error(actuals, preds))

        records.append(
            {
                **tuned_params,
                "r2": r2,
                "rmse": rmse,
                "_actuals": actuals,
                "_preds": preds,
                "_indices": indices,
            }
        )

        print(
            f"\r    tuning: {ci + 1}/{len(combos)} combos  "
            f"(best: R\u00b2={max(r['r2'] for r in records):+.3f}  "
            f"n={tuned_params['n_estimators']} lr={tuned_params['learning_rate']})",
            end="",
            flush=True,
        )
    print()

    tuning_df_full = sorted(records, key=lambda x: -x["r2"])
    best_record = tuning_df_full[0]
    tuning_df = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in tuning_df_full]
    )

    best_tuned = {k: best_record[k] for k in sorted(PARAM_GRID.keys())}

    def _clean(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.str_, np.bytes_)):
            return str(v)
        return v

    for k in best_tuned:
        best_tuned[k] = _clean(best_tuned[k])

    return (
        best_tuned,
        float(best_record["r2"]),
        float(best_record["rmse"]),
        tuning_df,
        best_record["_actuals"],
        best_record["_preds"],
        np.array(best_record["_indices"]),
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
    if target_label is None:
        target_label = target

    y_full = targets_df[target].values
    is_log = target in log_info
    back_fn = log_info[target]["back_transform"] if is_log else None
    raw_col = log_info[target]["raw_col"] if is_log else target

    valid_mask = ~np.isnan(y_full)
    if valid_mask.sum() < 20:
        print(f"    SKIP {target}: only {valid_mask.sum()} non-NaN")
        return None, None, None, None
    y = y_full[valid_mask]
    X_train = X_train_full[valid_mask]
    blocks = blocks_full[valid_mask]
    sids = sample_ids_full[valid_mask]

    (best_tuned, cv_r2, cv_rmse_log, tuning_df, oof_act, oof_pred, oof_idx) = (
        tune_and_evaluate(X_train, y, blocks, feature_names)
    )

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

    if is_log:
        cv_rmse_orig = math.sqrt(
            mean_squared_error(oof_df["actual"], oof_df["predicted"])
        )
    else:
        cv_rmse_orig = cv_rmse_log

    # Final model on all data (same n_estimators as best combo — no early stopping)
    final_params = {**FIXED_PARAMS, **best_tuned, "random_state": SEED}
    final_xgb = XGBRegressor(**final_params)
    final_xgb.fit(X_train, y)

    importance_dict = {
        feature_names[i]: float(final_xgb.feature_importances_[i])
        for i in range(len(feature_names))
    }

    grid_col_label, grid_preds = None, None
    if has_grid and X_grid is not None and cv_r2 >= MIN_R2_FOR_GRID:
        raw_preds = final_xgb.predict(X_grid)
        if is_log and back_fn:
            grid_preds = back_transform(raw_preds, back_fn)
            grid_col_label = f"pred_{raw_col}"
        else:
            grid_preds = raw_preds
            grid_col_label = f"pred_{target}"

    tier = assign_tier(cv_r2)
    summary = {
        "target": target,
        "raw_col": raw_col,
        "log_transformed": is_log,
        "selected_transform": (
            "log1p" if is_log and back_fn == "expm1" else "log" if is_log else "raw"
        ),
        "back_transform": back_fn if back_fn else "none",
        "best_n_estimators": best_tuned["n_estimators"],
        "best_learning_rate": best_tuned["learning_rate"],
        "best_max_depth": best_tuned["max_depth"],
        "best_min_child_weight": best_tuned["min_child_weight"],
        "best_subsample": best_tuned["subsample"],
        "best_colsample_bytree": best_tuned["colsample_bytree"],
        "best_reg_alpha": best_tuned["reg_alpha"],
        "cv_r2": cv_r2,
        "cv_rmse_log_space": cv_rmse_log,
        "cv_rmse_orig_units": cv_rmse_orig,
        "tier": tier,
        "n_folds": len(np.unique(blocks)),
        "n_samples": int(valid_mask.sum()),
        "grid": (
            "predicted"
            if grid_preds is not None
            else (f"skipped R\u00b2={cv_r2:.3f}" if has_grid else "no grid")
        ),
    }

    with open(f"ansoil_models_xgb/{target}.pkl", "wb") as f:
        pickle.dump(
            {
                "model": final_xgb,
                "target": target,
                "raw_col": raw_col,
                "back_transform": back_fn,
                "best_params": {**FIXED_PARAMS, **best_tuned},
                "cv_r2": cv_r2,
                "feature_names": feature_names,
                "importance": importance_dict,
                "tuning_results": tuning_df,
            },
            f,
        )

    return summary, oof_df, (grid_col_label, grid_preds), importance_dict


# =============================================================================
# MAIN
# =============================================================================


def run_all_models(
    X_train_np,
    X_grid_np,
    targets_df,
    index_df,
    feature_names,
    regular,
    excluded,
    grid_ids,
    has_grid,
    established_log_info,
    dual_test_info,
):

    print("\n" + "=" * 70)
    print("STEP 2  LOLO-CV + XGBoost (NO EARLY STOPPING — LEAK-FREE)")
    print("=" * 70)
    print(f"  {N_RANDOM_SEARCH} random combos x 28 LOLO folds per target")
    print(f"  n_estimators tuned over: {PARAM_GRID['n_estimators']}")
    print(f"  Seed: {SEED}\n")

    blocks_full = index_df["sample_location"].values
    sample_ids_full = index_df["sample_id"].values

    all_oof, model_summary, grid_pred_cols = [], [], {}
    dual_comparison, all_importances = [], []

    total = len(regular) + len(dual_test_info)
    counter = [0]

    def pr(target, cv_r2, tier, is_log, summary, extra=""):
        counter[0] += 1
        flag = {"strong": "**", "moderate": "* ", "weak": ". ", "unusable": "x "}[tier]
        lt = "[L]" if is_log else "   "
        bar = chr(9608) * int(max(0, cv_r2) * 20)
        n = summary.get("best_n_estimators", "?")
        lr = summary.get("best_learning_rate", "?")
        d = summary.get("best_max_depth", "?")
        print(
            f"  [{counter[0]:02d}/{total:02d}]  {target:<42} "
            f"R\u00b2={cv_r2:>+.3f}  n={n:<4} lr={lr:<5} d={d:<2} {flag} {lt} {extra} {bar}"
        )

    # Pass 1: Regular targets
    print(f"\n  Pass 1: {len(regular)} regular targets")
    print(f"  {'─' * 90}")
    t0 = time.time()
    for target in regular:
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
        pr(
            target,
            summary["cv_r2"],
            summary["tier"],
            summary["log_transformed"],
            summary,
        )
    print(f"\n  Pass 1: {time.time() - t0:.0f}s")

    # Pass 2: Dual-test
    print(f"\n  Pass 2: {len(dual_test_info)} dual-test candidates")
    print(f"  {'─' * 90}")
    t0 = time.time()
    for raw_col, info in dual_test_info.items():
        log_col, back_fn = info["log_col"], info["back_transform"]
        has_raw = raw_col in targets_df.columns
        has_log = log_col in targets_df.columns
        if not has_raw and not has_log:
            continue

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
                feature_names,
            )
            if res_raw[0]:
                r2_raw = res_raw[0]["cv_r2"]

        r2_log = np.nan
        if has_log:
            tli = {log_col: {"raw_col": raw_col, "back_transform": back_fn}}
            res_log = model_one_target(
                log_col,
                X_train_np,
                targets_df,
                blocks_full,
                sample_ids_full,
                X_grid_np,
                has_grid,
                tli,
                feature_names,
            )
            if res_log[0]:
                r2_log = res_log[0]["cv_r2"]

        if not np.isnan(r2_raw) and not np.isnan(r2_log):
            winner = "raw" if r2_raw >= r2_log else "log"
        elif not np.isnan(r2_raw):
            winner = "raw"
        else:
            winner = "log"

        if winner == "raw" and res_raw[0]:
            fs, fo, (fg, fp), fi = res_raw
        elif res_log[0]:
            fs, fo, (fg, fp), fi = res_log
        else:
            continue

        delta = (
            (r2_log - r2_raw)
            if not np.isnan(r2_raw) and not np.isnan(r2_log)
            else np.nan
        )
        note = ""
        if raw_col == "digest_mg_kg_na5895":
            note = f"Na: {'raw won' if winner == 'raw' else 'log won'}"

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

        all_oof.append(fo)
        model_summary.append(fs)
        if fg and fp is not None:
            grid_pred_cols[fg] = fp
        all_importances.append({"target": fs["target"], **fi})

        r2r = f"{r2_raw:+.3f}" if not np.isnan(r2_raw) else "  N/A"
        r2l = f"{r2_log:+.3f}" if not np.isnan(r2_log) else "  N/A"
        counter[0] += 1
        print(
            f"  [{counter[0]:02d}/{total:02d}]  {raw_col:<42} raw={r2r} log={r2l} >> {winner}"
        )

    print(f"\n  Pass 2: {time.time() - t0:.0f}s")

    # ======================================================================
    # SAVE OUTPUTS
    # ======================================================================
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    results_df = pd.DataFrame(model_summary).sort_values("cv_r2", ascending=False)
    results_df.to_csv("ansoil_model_results_xgb.csv", index=False)
    print(f"  ansoil_model_results_xgb.csv     {results_df.shape}")

    dual_df = pd.DataFrame(dual_comparison).sort_values("r2_raw", ascending=False)
    dual_df.to_csv("ansoil_transform_comparison_xgb.csv", index=False)

    oof_all = pd.concat(all_oof, ignore_index=True)
    oof_all.to_csv("ansoil_cv_predictions_xgb.csv", index=False)

    if has_grid and grid_pred_cols:
        grid_out = pd.DataFrame({"grid_id": grid_ids})
        for col, arr in grid_pred_cols.items():
            grid_out[col] = arr
        grid_out.to_csv("ansoil_grid_predictions_xgb.csv", index=False)

    imp_df = pd.DataFrame(all_importances)
    imp_df.to_csv("ansoil_feature_importance_xgb.csv", index=False)

    # ======================================================================
    # COMPARISON WITH PREVIOUS MODELS
    # ======================================================================
    knn_r2, rf_r2 = {}, {}
    for fname, d, label in [
        (KNN_RESULTS_FILE, knn_r2, "KNN"),
        (RF_RESULTS_FILE, rf_r2, "RF"),
    ]:
        try:
            df = pd.read_csv(fname)
            d.update(df.set_index("target")["cv_r2"].to_dict())
            print(f"  Loaded {label}: {len(d)} targets")
        except FileNotFoundError:
            print(f"  {fname} not found")

    comp_rows = []
    for _, row in results_df.iterrows():
        t = row["target"]
        kv = knn_r2.get(t, np.nan)
        rv = rf_r2.get(t, np.nan)
        comp_rows.append(
            {
                "target": t,
                "r2_knn": round(kv, 6) if not np.isnan(kv) else np.nan,
                "r2_rf": round(rv, 6) if not np.isnan(rv) else np.nan,
                "r2_xgb": round(row["cv_r2"], 6),
                "xgb_vs_rf": round(row["cv_r2"] - rv, 6)
                if not np.isnan(rv)
                else np.nan,
                "xgb_vs_knn": round(row["cv_r2"] - kv, 6)
                if not np.isnan(kv)
                else np.nan,
                "tier": row["tier"],
            }
        )
    comp_df = pd.DataFrame(comp_rows).sort_values("r2_xgb", ascending=False)
    comp_df.to_csv("ansoil_model_comparison_xgb.csv", index=False)

    # XGBoost vs RF
    rf_matched = comp_df.dropna(subset=["xgb_vs_rf"])
    if len(rf_matched) > 0:
        print("\n  XGBoost vs RF:")
        print(
            f"    XGB better: {(rf_matched['xgb_vs_rf'] > 0).sum()}/{len(rf_matched)}"
        )
        print(f"    Mean delta: {rf_matched['xgb_vs_rf'].mean():+.4f}")

    # ======================================================================
    # RESULTS SUMMARY
    # ======================================================================
    r2v = results_df["cv_r2"]
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY — XGBoost (LEAK-FREE)")
    print("=" * 70)

    for label, mask in [
        ("STRONG    R\u00b2 > 0.50", r2v > 0.5),
        ("MODERATE  R\u00b2 0.30-0.50", (r2v >= 0.3) & (r2v <= 0.5)),
        ("WEAK      R\u00b2 0.00-0.30", (r2v >= 0.0) & (r2v < 0.3)),
        ("UNUSABLE  R\u00b2 < 0.00", r2v < 0.0),
    ]:
        grp = results_df[mask]
        print(f"\n  {label}  (n={len(grp)})")
        for _, row in grp.iterrows():
            lt = "[L]" if row["log_transformed"] else "   "
            print(
                f"    {lt} {row['target']:<44} R\u00b2={row['cv_r2']:+.3f}  "
                f"n={int(row['best_n_estimators'])} lr={row['best_learning_rate']} "
                f"d={int(row['best_max_depth'])}"
            )

    if all_importances:
        imp_means = imp_df.drop(columns=["target"]).mean().sort_values(ascending=False)
        print("\n  TOP 10 FEATURES:")
        for feat, val in imp_means.head(10).items():
            print(f"    {feat:<28} {val:.4f}")

    print(f"\n{'=' * 70}\nCOMPLETE\n{'=' * 70}")
    return results_df, oof_all, dual_df, comp_df


def run():
    (
        X_train,
        X_grid,
        targets_df,
        index_df,
        feature_names,
        regular,
        excluded,
        grid_ids,
        has_grid,
        established_log_info,
        dual_test_info,
    ) = load_inputs()
    return run_all_models(
        X_train.values,
        X_grid.values if X_grid is not None else None,
        targets_df,
        index_df,
        feature_names,
        regular,
        excluded,
        grid_ids,
        has_grid,
        established_log_info,
        dual_test_info,
    )


if __name__ == "__main__":
    results_df, oof_df, dual_df, comp_df = run()
