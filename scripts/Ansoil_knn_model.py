"""
=============================================================================
ANTARCTIC SOIL GEOCHEMISTRY — KNN MODEL (Baseline)
=============================================================================

QUICK START GUIDE
=================

  NOTE: This is the BASELINE model. It uses K-Nearest Neighbors with a
  precomputed distance matrix. It does NOT need to be re-run on the fast
  computer — the KNN results are already final and are used only as a
  reference point to show how much RF and XGBoost improve over it.

  If you do want to re-run it (e.g., to verify results):

  1. INSTALL REQUIRED PACKAGES (run once in terminal):

         pip install numpy pandas scikit-learn

  2. PLACE THESE 5 DATA FILES IN THE SAME FOLDER AS THIS SCRIPT:

     File                          What it is
     ─────────────────────────     ──────────────────────────────────────
     ansoil_distance_matrix.csv    Precomputed 171x171 distance matrix.
                                   This encodes how "similar" each pair of
                                   samples is based on their environmental
                                   features (elevation, lithology, climate,
                                   etc). Produced by Ansoil_knn_prep.py.

     ansoil_grid_distances.csv     Distances from each grid prediction point
                                   to each training sample (15769 x 171).
                                   Also produced by Ansoil_knn_prep.py.

     ansoil_targets.csv            Target variables (171 rows x 83 cols).

     ansoil_sample_index.csv       Sample metadata (171 rows x 10 cols).
                                   Defines which samples belong to which
                                   sampling location (for LOLO-CV folds).

     ansoil_log_targets.csv        Transform lookup table (24 rows x 5 cols).

     The first two files are KNN-specific (distance matrices).
     The last three are shared with RF and XGBoost.

  3. RUN IT:

         python Ansoil_knn_model.py

     Takes about 2-5 minutes. No configuration needed — the KNN
     hyperparameter search is small (14 combinations of k and weighting
     scheme per target) and runs quickly.

  4. OUTPUT FILES (created in the same folder):

     ansoil_model_results_v6.csv          Main results: R² per target
     ansoil_cv_predictions_v6.csv         Predicted vs actual per sample
     ansoil_grid_predictions_v6.csv       Spatial grid predictions
     ansoil_transform_comparison_v6.csv   Dual-test raw vs log results
     ansoil_models_v6/                    Saved model files (.pkl)

  5. NO SEED NEEDED:

     Unlike RF and XGBoost, KNN is deterministic — there is no random
     component, so running it again will always produce the same results.
     You only need one run.


WHAT THIS SCRIPT DOES (for those who want to understand it)
===========================================================

  This is the simplest of the three models. For each soil property:

  1. It tries 14 combinations of K (number of neighbors: 3,5,7,9,11,15,20)
     and weighting scheme (uniform vs distance-weighted).

  2. Each combination is evaluated using Leave-One-Location-Out
     cross-validation (LOLO-CV) with 28 folds — the same CV design
     used by RF and XGBoost for fair comparison.

  3. The combination with the lowest RMSE is selected.

  4. The model predicts by averaging the target values of the K most
     similar training samples (where "similar" is defined by the
     precomputed environmental distance matrix).

  KEY LIMITATION: The distance matrix uses manually assigned feature
  weights (lithology=3.0, elevation=2.0, coordinates=0.15-0.25).
  These weights were set by geochemical reasoning, NOT optimized by
  the data. RF and XGBoost learn feature importance from the data,
  which is one reason they outperform KNN.

  RESULTS: 0 strong, 4 moderate, 30 weak, 33 unusable targets.
  Mean R² = 0.038. Best = 0.362 (Ti digest). This is the baseline
  that RF and XGBoost are compared against.

=============================================================================

"""

import math
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore")
os.makedirs("ansoil_models_v6", exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

K_RANGE = [3, 5, 7, 9, 11, 15, 20]
WEIGHT_OPTIONS = ["uniform", "distance"]
MIN_R2_FOR_GRID = 0.0  # include all positive-R² targets on grid

# Constant or below-detection-limit targets — always excluded
EXCLUDE_TARGETS = [
    "digest_mg_kg_cd2288",
    "log_digest_mg_kg_se1960",
]


# =============================================================================
# STEP 1 — LOAD AND VALIDATE
# =============================================================================


def load_inputs():
    print("=" * 70)
    print("STEP 1  Load and validate inputs")
    print("=" * 70)

    D_train_df = pd.read_csv("ansoil_distance_matrix.csv", index_col=0)
    targets_df = pd.read_csv("ansoil_targets.csv")
    index_df = pd.read_csv("ansoil_sample_index.csv")

    assert list(D_train_df.index) == list(targets_df["sample_id"]), (
        "Row order mismatch: distance matrix vs targets"
    )
    assert list(D_train_df.index) == list(index_df["sample_id"]), (
        "Row order mismatch: distance matrix vs sample index"
    )
    print("  Alignment check: PASSED")

    # ── Load log target lookup ────────────────────────────────────────────────
    try:
        log_lookup = pd.read_csv("ansoil_log_targets.csv")

        # Established log1p/log entries (dual_test=False): final transform known
        established_log_info = {
            row["log_col"]: {
                "raw_col": row["raw_col"],
                "back_transform": row["back_transform"],
            }
            for _, row in log_lookup.iterrows()
            if not row.get("dual_test", False)
        }

        # Dual-test entries: transform to be determined by CV comparison
        dual_test_info = {
            row["raw_col"]: {
                "log_col": row["log_col"],
                "back_transform": row["back_transform"],  # 'exp' — used if log wins
            }
            for _, row in log_lookup.iterrows()
            if row.get("dual_test", False)
        }

        # v6 BUG FIX: use .loc[] filter, not groupby
        n_log1p = log_lookup.loc[log_lookup["transform"] == "log1p"].shape[0]
        n_dual = log_lookup.loc[log_lookup["dual_test"] == True].shape[0]
        print(f"  Log target lookup: {len(log_lookup)} entries")
        print(f"    {n_log1p} established log1p  |  {n_dual} dual-test candidates")

    except FileNotFoundError:
        print("  WARNING: ansoil_log_targets.csv not found — no back-transformation")
        established_log_info, dual_test_info = {}, {}

    # ── Grid distances ────────────────────────────────────────────────────────
    has_grid, D_grid, grid_ids = False, None, None
    try:
        D_grid_df = pd.read_csv("ansoil_grid_distances.csv", index_col=0)
        nan_mask = D_grid_df.isnull().any(axis=1)
        n_nan = nan_mask.sum()
        if n_nan > 0:
            print(f"  WARNING: {n_nan} grid rows with NaN distances dropped")
            D_grid_df = D_grid_df[~nan_mask]
        grid_ids = list(D_grid_df.index)
        D_grid = D_grid_df.values
        has_grid = True
        suffix = f" ({n_nan} rows dropped for NaN)" if n_nan > 0 else ""
        print(f"  Grid distances: {D_grid.shape}{suffix}")
    except FileNotFoundError:
        print("  Grid distances: NOT FOUND — grid prediction skipped")

    # ── Target list ───────────────────────────────────────────────────────────
    all_tgt = [c for c in targets_df.columns if c != "sample_id"]
    constant = [c for c in all_tgt if targets_df[c].std() == 0]
    excluded = list(set(EXCLUDE_TARGETS + constant))

    # For dual-test candidates, we will handle both raw and log_ versions
    # in run_all_models(). Remove them from the regular modelable list here
    # so they don't get modeled twice by accident.
    dual_raw_cols = list(dual_test_info.keys())
    dual_log_cols = [info["log_col"] for info in dual_test_info.values()]
    dual_related = set(dual_raw_cols + dual_log_cols)

    # Regular modelable targets = all targets minus excluded minus dual-test-related
    # (dual-test targets are handled separately in the dual-test loop)
    regular_modelable = [
        c for c in all_tgt if c not in excluded and c not in dual_related
    ]

    print(f"\n  Training samples:         {len(D_train_df)}")
    print(f"  Total target columns:     {len(all_tgt)}")
    print(f"  Excluded (BDL/const):     {len(excluded)}")
    for e in excluded:
        print(f"    {e}")
    print(f"  Dual-test candidates:     {len(dual_raw_cols)}  (raw+log both tested)")
    print(f"  Regular targets to model: {len(regular_modelable)}")
    print(f"  CV folds (locations):     {index_df['sample_location'].nunique()}")

    return (
        D_train_df.values,
        D_grid,
        targets_df,
        index_df,
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
    """
    Reverse a log transformation on predictions.
      back_fn='exp'   → for log()   targets  (dissolved ions/EC)
      back_fn='expm1' → for log1p() targets  (trace metals/CEC)
    Clips at 0 to prevent tiny negatives from floating-point noise.
    """
    arr = np.asarray(values, dtype=float)
    if back_fn == "exp":
        return np.clip(np.exp(arr), 0, None)
    elif back_fn == "expm1":
        return np.clip(np.expm1(arr), 0, None)
    else:
        raise ValueError(f"Unknown back_transform: '{back_fn}'")


def tune_target(D_train, y, blocks):
    """
    Run LOLO-CV for all (k, weighting) combinations.
    Returns DataFrame sorted by RMSE ascending (row 0 = best hyperparameters).
    Always operates in the native space of y (log or raw — caller's choice).
    """
    logo = LeaveOneGroupOut()
    records = []
    for w in WEIGHT_OPTIONS:
        for k in K_RANGE:
            actuals, preds = [], []
            for tr_idx, te_idx in logo.split(D_train, y, groups=blocks):
                if len(tr_idx) < k:
                    continue
                knn = KNeighborsRegressor(
                    n_neighbors=k, metric="precomputed", weights=w
                )
                knn.fit(D_train[np.ix_(tr_idx, tr_idx)], y[tr_idx])
                preds.extend(knn.predict(D_train[np.ix_(te_idx, tr_idx)]))
                actuals.extend(y[te_idx])
            if not actuals:
                continue
            actuals = np.array(actuals)
            preds = np.array(preds)
            rmse = math.sqrt(mean_squared_error(actuals, preds))
            r2 = r2_score(actuals, preds)
            n_folds = logo.get_n_splits(D_train, y, groups=blocks)
            records.append(
                {
                    "k": k,
                    "weighting": w,
                    "rmse": rmse,
                    "r2": r2,
                    "n_folds": n_folds,
                }
            )
    return pd.DataFrame(records).sort_values("rmse").reset_index(drop=True)


def collect_oof(D_train, y, blocks, best_k, best_w, sample_ids, target, log_info):
    """
    Re-run LOLO-CV with the best hyperparameters to collect OOF predictions.
    For log-transformed targets, stores predictions in BOTH log space and
    original units (back-transformed).
    """
    logo = LeaveOneGroupOut()
    rows = []
    is_log = target in log_info
    back_fn = log_info[target]["back_transform"] if is_log else None

    for tr_idx, te_idx in logo.split(D_train, y, groups=blocks):
        if len(tr_idx) < best_k:
            continue
        knn = KNeighborsRegressor(
            n_neighbors=best_k, metric="precomputed", weights=best_w
        )
        knn.fit(D_train[np.ix_(tr_idx, tr_idx)], y[tr_idx])
        preds = knn.predict(D_train[np.ix_(te_idx, tr_idx)])

        for idx, pred_log in zip(te_idx, preds):
            row = {
                "sample_id": sample_ids[idx],
                "actual_log": float(y[idx]) if is_log else np.nan,
                "pred_log": float(pred_log) if is_log else np.nan,
            }
            if is_log:
                row["actual"] = float(back_transform([y[idx]], back_fn)[0])
                row["predicted"] = float(back_transform([pred_log], back_fn)[0])
                row["residual"] = row["predicted"] - row["actual"]
            else:
                row["actual"] = float(y[idx])
                row["predicted"] = float(pred_log)
                row["residual"] = float(pred_log - y[idx])
            rows.append(row)

    return pd.DataFrame(rows)


def train_final(D_train, y, best_k, best_w):
    knn = KNeighborsRegressor(n_neighbors=best_k, metric="precomputed", weights=best_w)
    knn.fit(D_train, y)
    return knn


def assign_tier(r2):
    if r2 > 0.5:
        return "strong"
    if r2 > 0.3:
        return "moderate"
    if r2 > 0.0:
        return "weak"
    return "unusable"


# =============================================================================
# CORE MODELING LOOP — runs one target, returns summary dict + oof DataFrame
# =============================================================================


def model_one_target(
    target,
    D_train,
    targets_df,
    blocks,
    sample_ids,
    D_grid,
    has_grid,
    log_info,
    target_label=None,
):
    """
    Tune, train, and collect OOF predictions for a single target column.
    target      : column name in targets_df (may be log_ prefixed)
    log_info    : dict mapping log_col -> {raw_col, back_transform}
                  (only established log1p/log entries, not dual-test)
    target_label: display name for progress printing (defaults to target)
    Returns (summary_dict, oof_df, grid_col_label, grid_preds_or_None)
    """
    if target_label is None:
        target_label = target

    y = targets_df[target].values
    is_log = target in log_info

    tuning = tune_target(D_train, y, blocks)
    best = tuning.iloc[0]
    best_k = int(best["k"])
    best_w = best["weighting"]
    cv_r2 = float(best["r2"])
    cv_rmse_log = float(best["rmse"])

    oof_df = collect_oof(
        D_train, y, blocks, best_k, best_w, sample_ids, target, log_info
    )
    oof_df.insert(0, "target", target_label)

    # RMSE in original units
    if is_log and len(oof_df) > 0:
        cv_rmse_orig = math.sqrt(
            mean_squared_error(oof_df["actual"], oof_df["predicted"])
        )
        raw_col = log_info[target]["raw_col"]
        back_fn = log_info[target]["back_transform"]
    else:
        cv_rmse_orig = cv_rmse_log
        raw_col = target
        back_fn = None

    # Final model on all data
    final_knn = train_final(D_train, y, best_k, best_w)

    # Grid prediction
    grid_col_label, grid_preds = None, None
    if has_grid and cv_r2 >= MIN_R2_FOR_GRID:
        raw_preds = final_knn.predict(D_grid)
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
        "best_k": best_k,
        "weighting": best_w,
        "cv_r2": cv_r2,
        "cv_rmse_log_space": cv_rmse_log,
        "cv_rmse_orig_units": cv_rmse_orig,
        "tier": tier,
        "n_folds": int(best["n_folds"]),
        "grid": (
            "predicted"
            if grid_preds is not None
            else (f"skipped R²={cv_r2:.3f}" if has_grid else "no grid file")
        ),
    }

    # Save model pickle
    with open(f"ansoil_models_v6/{target}.pkl", "wb") as f:
        pickle.dump(
            {
                "model": final_knn,
                "target": target,
                "raw_col": raw_col,
                "back_transform": back_fn,
                "best_k": best_k,
                "weighting": best_w,
                "cv_r2_log": cv_r2,
                "cv_rmse_log": cv_rmse_log,
                "cv_rmse_orig_units": cv_rmse_orig,
                "sample_ids": sample_ids,
                "full_tuning_results": tuning,
            },
            f,
        )

    return summary, oof_df, grid_col_label, grid_preds


# =============================================================================
# MAIN MODELING FUNCTION
# =============================================================================


def run_all_models(
    D_train,
    D_grid,
    targets_df,
    index_df,
    regular_modelable,
    excluded,
    grid_ids,
    has_grid,
    established_log_info,
    dual_test_info,
):

    print("\n" + "=" * 70)
    print("STEPS 2+3  LOLO-CV + Hyperparameter Tuning")
    print("=" * 70)

    blocks = index_df["sample_location"].values
    sample_ids = index_df["sample_id"].values

    # Lookup table for collect_oof / back-transform in model_one_target
    # Only established log entries (not dual-test — those handled separately)
    y_lookup = {c: targets_df[c] for c in targets_df.columns if c != "sample_id"}

    all_oof = []
    model_summary = []
    grid_pred_cols = {}
    dual_comparison = []

    total = len(regular_modelable) + len(dual_test_info)
    counter = [0]

    def print_progress(target, cv_r2, best_k, best_w, tier, is_log, dual_tag=""):
        counter[0] += 1
        flag = {"strong": "**", "moderate": "* ", "weak": ". ", "unusable": "x "}[tier]
        log_marker = "[L]" if is_log else "   "
        bar = chr(9608) * int(max(0, cv_r2) * 20)
        print(
            f"  [{counter[0]:02d}/{total:02d}]  {target:<42} {cv_r2:>+.3f}  "
            f"k={best_k:<2}  {best_w[:4]:<5}  {flag} {log_marker} {dual_tag}  {bar}"
        )

    # ==========================================================================
    # PASS 1: Regular targets (established log1p + all non-dual-test targets)
    # ==========================================================================
    print(f"\n  Pass 1: {len(regular_modelable)} regular targets")
    print(
        f"  {'#':>7}  {'Target':<42} {'R²':>6}  {'k':<4}  {'wt':<5}  flag\n  {'-' * 72}"
    )

    for target in regular_modelable:
        summary, oof_df, grid_col_label, grid_preds = model_one_target(
            target,
            D_train,
            targets_df,
            blocks,
            sample_ids,
            D_grid,
            has_grid,
            established_log_info,
        )

        all_oof.append(oof_df)
        model_summary.append(summary)
        if grid_col_label and grid_preds is not None:
            grid_pred_cols[grid_col_label] = grid_preds

        print_progress(
            target,
            summary["cv_r2"],
            summary["best_k"],
            summary["weighting"],
            summary["tier"],
            summary["log_transformed"],
        )

    # ==========================================================================
    # PASS 2: Dual-test candidates — run CV on both raw and log_, keep better
    # ==========================================================================
    print(f"\n  Pass 2: {len(dual_test_info)} dual-test candidates")
    print("  (each modeled twice — raw AND log — better R² selected)")
    print(
        f"  {'#':>7}  {'Raw col':<42} {'R²(raw)':>8}  {'R²(log)':>8}  "
        f"{'winner':<6}  {'ΔR²':>6}\n"
        f"  {'-' * 78}"
    )

    for raw_col, info in dual_test_info.items():
        log_col = info["log_col"]
        back_fn = info["back_transform"]  # 'exp'

        # Check both columns are actually present in targets_df
        has_raw = raw_col in y_lookup
        has_log = log_col in y_lookup
        if not has_raw and not has_log:
            print(f"  SKIP: neither {raw_col} nor {log_col} found in targets")
            continue

        # ── Model raw version ─────────────────────────────────────────────────
        r2_raw = np.nan
        if has_raw:
            sum_raw, oof_raw, gcol_raw, gpred_raw = model_one_target(
                raw_col,
                D_train,
                targets_df,
                blocks,
                sample_ids,
                D_grid,
                has_grid,
                {},  # empty log_info → raw only
            )
            r2_raw = sum_raw["cv_r2"]
        else:
            sum_raw, oof_raw, gcol_raw, gpred_raw = None, None, None, None

        # ── Model log version ─────────────────────────────────────────────────
        r2_log = np.nan
        if has_log:
            # Provide a minimal log_info so collect_oof back-transforms correctly
            temp_log_info = {log_col: {"raw_col": raw_col, "back_transform": back_fn}}
            sum_log, oof_log, gcol_log, gpred_log = model_one_target(
                log_col,
                D_train,
                targets_df,
                blocks,
                sample_ids,
                D_grid,
                has_grid,
                temp_log_info,
            )
            r2_log = sum_log["cv_r2"]
        else:
            sum_log, oof_log, gcol_log, gpred_log = None, None, None, None

        # ── Select winner ─────────────────────────────────────────────────────
        # If both valid: choose higher R². If only one available: use it.
        # Ties go to raw (more interpretable, no back-transform complexity).
        if not np.isnan(r2_raw) and not np.isnan(r2_log):
            if r2_raw >= r2_log:
                winner = "raw"
                final_sum, final_oof, final_gcol, final_gpred = (
                    sum_raw,
                    oof_raw,
                    gcol_raw,
                    gpred_raw,
                )
            else:
                winner = "log"
                final_sum, final_oof, final_gcol, final_gpred = (
                    sum_log,
                    oof_log,
                    gcol_log,
                    gpred_log,
                )
        elif not np.isnan(r2_raw):
            winner = "raw"
            final_sum, final_oof, final_gcol, final_gpred = (
                sum_raw,
                oof_raw,
                gcol_raw,
                gpred_raw,
            )
        else:
            winner = "log"
            final_sum, final_oof, final_gcol, final_gpred = (
                sum_log,
                oof_log,
                gcol_log,
                gpred_log,
            )

        delta = (
            (r2_log - r2_raw)
            if not np.isnan(r2_raw) and not np.isnan(r2_log)
            else np.nan
        )

        # Note for Na digest: flag if raw doesn't win as expected
        if raw_col == "digest_mg_kg_na5895":
            if winner == "raw":
                note = "Na: raw won as expected (validates revert from v5)"
            else:
                note = "Na: UNEXPECTED — log won. Investigate before reporting."
        else:
            note = ""

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

        # Add winning version to outputs
        all_oof.append(final_oof)
        model_summary.append(final_sum)
        if final_gcol and final_gpred is not None:
            grid_pred_cols[final_gcol] = final_gpred

        # Print dual-test progress line
        r2_raw_str = f"{r2_raw:+.3f}" if not np.isnan(r2_raw) else "  N/A"
        r2_log_str = f"{r2_log:+.3f}" if not np.isnan(r2_log) else "  N/A"
        delta_str = f"{delta:+.3f}" if not np.isnan(delta) else "  N/A"
        counter[0] += 1
        print(
            f"  [{counter[0]:02d}/{total:02d}]  {raw_col:<42} "
            f"raw={r2_raw_str}  log={r2_log_str}  "
            f"{'>> ' + winner:<8}  Δ={delta_str}  {note}"
        )

    # ==========================================================================
    # SAVE OUTPUTS
    # ==========================================================================

    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    results_df = pd.DataFrame(model_summary).sort_values("cv_r2", ascending=False)
    results_df.to_csv("ansoil_model_results_v6.csv", index=False)
    print(f"  ansoil_model_results_v6.csv     {results_df.shape}")

    dual_df = pd.DataFrame(dual_comparison).sort_values("r2_raw", ascending=False)
    dual_df.to_csv("ansoil_transform_comparison_v6.csv", index=False)
    print(f"  ansoil_transform_comparison_v6.csv  {dual_df.shape}")
    print("    (raw_col, r2_raw, r2_log, selected_transform, delta_r2, winner_note)")

    oof_all = pd.concat(all_oof, ignore_index=True)
    oof_all.to_csv("ansoil_cv_predictions_v6.csv", index=False)
    print(f"  ansoil_cv_predictions_v6.csv    {oof_all.shape}")

    if has_grid and grid_pred_cols:
        grid_out = pd.DataFrame({"grid_id": grid_ids})
        for col_label, arr in grid_pred_cols.items():
            grid_out[col_label] = arr
        grid_out.to_csv("ansoil_grid_predictions_v6.csv", index=False)
        print(
            f"  ansoil_grid_predictions_v6.csv  "
            f"({len(grid_ids)} rows x {len(grid_pred_cols) + 1} cols)"
        )
        print("    All values in ORIGINAL units (back-transformed where applicable)")

    print(f"  ansoil_models_v6/               {len(model_summary)} .pkl files")

    # ==========================================================================
    # RESULTS SUMMARY
    # ==========================================================================

    r2v = results_df["cv_r2"]
    n_log = results_df["log_transformed"].sum()
    n_raw = len(results_df) - n_log

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
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
                f"k={int(row['best_k'])}  {row['weighting']}"
            )
        print()

    print("  EXCLUDED:")
    for e in excluded:
        print(f"    {e}")

    # Dual-test summary
    if len(dual_df) > 0:
        print("\n  DUAL-TRANSFORM COMPARISON SUMMARY")
        print(
            f"  {'Raw column':<42}  {'R²(raw)':>8}  {'R²(log)':>8}  "
            f"{'winner':<6}  {'ΔR²':>7}"
        )
        print(f"  {'-' * 76}")
        for _, row in dual_df.iterrows():
            r2r = f"{row['r2_raw']:+.3f}" if not pd.isna(row["r2_raw"]) else "   N/A"
            r2l = f"{row['r2_log']:+.3f}" if not pd.isna(row["r2_log"]) else "   N/A"
            dr = f"{row['delta_r2']:+.3f}" if not pd.isna(row["delta_r2"]) else "   N/A"
            note = f"  ← {row['winner_note']}" if row["winner_note"] else ""
            print(
                f"  {row['raw_col']:<42}  {r2r:>8}  {r2l:>8}  "
                f"{row['selected_transform']:<6}  {dr:>7}{note}"
            )

        raw_won = dual_df[dual_df["selected_transform"] == "raw"]
        log_won = dual_df[dual_df["selected_transform"] == "log"]
        print(
            f"\n  Raw won: {len(raw_won)} targets  |  Log won: {len(log_won)} targets"
        )

        # Na validation check
        na_row = dual_df[dual_df["raw_col"] == "digest_mg_kg_na5895"]
        if len(na_row) > 0:
            na = na_row.iloc[0]
            print("\n  INTERNAL VALIDATION — digest_mg_kg_na5895 (sodium digest):")
            print("    v4 raw R²:    0.318  (expected baseline)")
            print("    v5 log R²:    0.030  (was artificially harmed)")
            print(f"    v6 raw R²:    {na['r2_raw']:+.3f}")
            print(f"    v6 log R²:    {na['r2_log']:+.3f}")
            print(f"    Selected:     {na['selected_transform'].upper()}")
            if na["selected_transform"] == "raw" and na["r2_raw"] > 0.25:
                print(
                    "    STATUS: PASSED — revert confirmed, sodium restored to moderate tier"
                )
            elif na["selected_transform"] == "raw":
                print(
                    "    STATUS: Raw won but R² lower than expected — check data / pipeline"
                )
            else:
                print(
                    "    STATUS: WARNING — log won unexpectedly. Review before reporting."
                )

    print("""
  TO MAP IN QGIS:
    1. Load ansoil_grid_prepared.csv  (X=lon, Y=lat, CRS=EPSG:4326)
    2. Load ansoil_grid_predictions_v6.csv  (pred_* cols, original units)
    3. Join both on grid_id
    4. Re-project to EPSG:3031 for polar stereographic display
    5. All values are already back-transformed — no conversion needed
""")
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    return results_df, oof_all, dual_df


# =============================================================================
# ENTRY POINT
# =============================================================================


def run():
    (
        D_train,
        D_grid,
        targets_df,
        index_df,
        regular_modelable,
        excluded,
        grid_ids,
        has_grid,
        established_log_info,
        dual_test_info,
    ) = load_inputs()

    results_df, oof_df, dual_df = run_all_models(
        D_train,
        D_grid,
        targets_df,
        index_df,
        regular_modelable,
        excluded,
        grid_ids,
        has_grid,
        established_log_info,
        dual_test_info,
    )

    return results_df, oof_df, dual_df


if __name__ == "__main__":
    results_df, oof_df, dual_df = run()
