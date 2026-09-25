"""
ml_results_clean.py
===================
Builds a single clean, verified results set for ANSOIL from the committed model
outputs. Re-trains nothing. Every number is recomputed from the per-seed
out-of-fold prediction files, so the outputs here are independent of the earlier
aggregation scripts.

What this fixes relative to aggregate_results.py / summary_table.py
-------------------------------------------------------------------
1. Dual-transform flips are RECONCILED, not deleted. When the raw/log winner
   changed between seeds, the old pipeline dropped the minority-seed rows and
   blamed a back-transform error that never happened. Grouping on the physical
   property restores all five seeds and removes the spurious 68- and 69-row
   counts.
2. Every R2 is reported in BOTH the space it was modelled in and, where a
   back-transform exists, the property's native measured units. The old single
   R2 column silently mixed log1p, natural-log, CLR and native spaces.
3. The dual-transform winner is re-selected on a single common scale (native
   units) as well as on the pipeline's own basis, so you can see which choices
   were artifacts of comparing a log-space R2 against a raw-space one.
4. Tier counts, the win/tie/loss tally and the mappable set are computed over
   67 physical properties, once each.
5. Cross-seed tier instability is surfaced instead of being hidden by the mode.
6. Mappability uses mean R2 > 0.30 across all seeds, not R2 >= 0 per seed.
7. The grid outputs are audited for physically impossible values, single-seed
   maps with undefined SD, and prediction outside the training envelope.

Usage
-----
    python ml_results_clean.py                     # canonical repo layout
    python ml_results_clean.py --root /path/to/repo
    python ml_results_clean.py --flat-dir ./all_csvs   # everything in one folder
    python ml_results_clean.py --skip-grid             # faster, tables only

Outputs (written to results/clean/ by default)
----------------------------------------------
    verified_per_seed.csv          one row per target per model per seed
    verified_per_property.csv      67 properties x model, both scorings
    verified_model_comparison.csv  head-to-head, 67 rows, no orphans
    verified_mappable.csv          the defensible map set
    tier_instability.csv           targets whose tier moves between seeds
    transform_audit.csv            dual-transform decisions on both scales
    grid_audit.csv                 per-map anomaly flags
    grid_extrapolation.csv         per-predictor envelope breach counts
    knn_baseline.csv               KNN reconciled onto the same 67 properties
    VERIFIED_RESULTS.md            the clean results document
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = [7, 42, 73, 123, 256]
MODELS = ["rf", "xgb"]

# Tier thresholds, matching assign_tier() in the model scripts exactly.
TIER_STRONG = 0.5
TIER_MODERATE = 0.3
TIER_WEAK = 0.0

# A target is mappable when its mean R2 across seeds clears this in either
# model. The model scripts gate grid prediction at R2 >= 0 per seed, which
# produces maps of targets with no predictive skill.
MAP_MIN_MEAN_R2 = 0.30

# Calling a head-to-head difference a tie below this. Kept at the project's
# existing 0.02 so the numbers stay comparable to earlier reporting.
TIE_THRESHOLD = 0.02

# Quantities that are legitimately negative. Everything else is a
# concentration, a percentage, a ratio, a pH or a conductivity, and a negative
# prediction for one of those is a physical impossibility rather than a
# modelling choice. CLR components are log-ratios centred on zero.
NEGATIVE_IS_VALID_EXACT = {"d13c_vpdb_permil", "d15n_air_permil"}
NEGATIVE_IS_VALID_PREFIX = ("clr_",)

# Predictors to check the grid against for extrapolation beyond training range.
ENVELOPE_PREDICTORS = [
    "wgs84_elev_from_pgc",
    "dist_coast_scar_km",
    "precipitation_racmo",
    "temperature_racmo",
    "slope_dem",
]

# The grid file stores dist_coast in metres while training data is in km. Both
# model scripts detect and divide at runtime; mirror that here so the envelope
# comparison is apples to apples.
DIST_COAST_UNIT_THRESHOLD = 1000.0


# ---------------------------------------------------------------------------
# File resolution
# ---------------------------------------------------------------------------


class Paths:
    """Resolves input files under the canonical repo layout, with a flat-directory
    fallback so this runs against a folder of exported CSVs too."""

    def __init__(self, root: str, flat_dir: str | None, out_dir: str | None):
        self.root = os.path.abspath(root)
        self.flat = os.path.abspath(flat_dir) if flat_dir else None
        self.out = os.path.abspath(
            out_dir or os.path.join(self.root, "results", "clean")
        )
        os.makedirs(self.out, exist_ok=True)

    def _candidates(self, name: str, model: str | None, seed: int | None) -> list[str]:
        c = []
        if model and seed is not None:
            c.append(os.path.join(self.root, "results", f"{model}_seed{seed}", name))
        c.append(os.path.join(self.root, "data", name))
        c.append(os.path.join(self.root, "reference_results", name))
        c.append(os.path.join(self.root, "results", "aggregated", name))
        c.append(os.path.join(self.root, name))
        if self.flat:
            c.append(os.path.join(self.flat, name))
        return c

    def find(
        self, name: str, model: str | None = None, seed: int | None = None,
        required: bool = True,
    ) -> str | None:
        for p in self._candidates(name, model, seed):
            if os.path.exists(p):
                return p
        if required:
            tried = "\n  ".join(self._candidates(name, model, seed))
            raise FileNotFoundError(f"Could not find {name}. Looked in:\n  {tried}")
        return None

    def out_path(self, name: str) -> str:
        return os.path.join(self.out, name)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def r2_score(actual, predicted) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def rmse(actual, predicted) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(actual, predicted) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(a - p)))


def assign_tier(r2: float) -> str:
    if not np.isfinite(r2):
        return "no data"
    if r2 > TIER_STRONG:
        return "strong"
    if r2 > TIER_MODERATE:
        return "moderate"
    if r2 > TIER_WEAK:
        return "weak"
    return "unusable"


def negative_is_valid(target: str) -> bool:
    return target in NEGATIVE_IS_VALID_EXACT or target.startswith(
        NEGATIVE_IS_VALID_PREFIX
    )


# ---------------------------------------------------------------------------
# Transform bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class TransformMap:
    """Maps target column names onto physical properties and measurement spaces."""

    log1p_cols: set = field(default_factory=set)
    dual_log_to_raw: dict = field(default_factory=dict)
    dual_raw_cols: set = field(default_factory=set)

    @classmethod
    def from_lookup(cls, lookup: pd.DataFrame) -> "TransformMap":
        dual = lookup[lookup["dual_test"] == True]  # noqa: E712
        established = lookup[lookup["dual_test"] != True]  # noqa: E712
        return cls(
            log1p_cols=set(established["log_col"]),
            dual_log_to_raw=dict(zip(dual["log_col"], dual["raw_col"])),
            dual_raw_cols=set(dual["raw_col"]),
        )

    def property_of(self, target: str) -> str:
        """Collapse a dual-test log column onto the property it measures."""
        return self.dual_log_to_raw.get(target, target)

    def space_of(self, target: str) -> str:
        if target in self.dual_log_to_raw:
            return "natural log"
        if target in self.log1p_cols:
            return "log1p"
        if target.startswith("clr_"):
            return "CLR"
        return "native units"

    def has_native_backtransform(self, target: str) -> bool:
        """True when a back-transform to the measured concentration exists.

        CLR components do not: a CLR value is a log-ratio against the geometric
        mean of the composition, so it has no single-concentration inverse.
        """
        return target in self.dual_log_to_raw or target in self.log1p_cols


# ---------------------------------------------------------------------------
# Stage 1: recompute every per-seed metric from the raw predictions
# ---------------------------------------------------------------------------


def build_per_seed(paths: Paths, tm: TransformMap) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            cv_path = paths.find(
                f"ansoil_cv_predictions_{model}_{seed}.csv", model, seed
            )
            res_path = paths.find(
                f"ansoil_model_results_{model}_{seed}.csv", model, seed
            )
            cv = pd.read_csv(cv_path)
            reported = pd.read_csv(res_path).set_index("target")

            for target, g in cv.groupby("target", sort=False):
                # The pipeline scored whichever space it modelled in. When
                # pred_log is populated the model worked in log space.
                modelled_in_log = g["pred_log"].notna().all()
                if modelled_in_log:
                    a_model, p_model = g["actual_log"].values, g["pred_log"].values
                else:
                    a_model, p_model = g["actual"].values, g["predicted"].values

                rec = {
                    "model": model,
                    "seed": seed,
                    "target": target,
                    "property": tm.property_of(target),
                    "scored_in": tm.space_of(target),
                    "n_samples": len(g),
                    "n_unique_samples": g["sample_id"].nunique(),
                    "r2_model_space": r2_score(a_model, p_model),
                    "rmse_model_space": rmse(a_model, p_model),
                    "r2_native": (
                        r2_score(g["actual"], g["predicted"])
                        if tm.has_native_backtransform(target)
                        else np.nan
                    ),
                    "rmse_native": rmse(g["actual"], g["predicted"]),
                    "mae_native": mae(g["actual"], g["predicted"]),
                }

                if target in reported.index:
                    r = reported.loc[target]
                    rec["reported_r2"] = float(r["cv_r2"])
                    rec["reported_rmse_orig"] = float(r["cv_rmse_orig_units"])
                    rec["reported_tier"] = r["tier"]
                    rec["reported_n_folds"] = int(r["n_folds"])
                    rec["reproduces"] = (
                        abs(rec["r2_model_space"] - rec["reported_r2"]) < 1e-6
                    )
                else:
                    rec["reported_r2"] = np.nan
                    rec["reproduces"] = False

                rows.append(rec)

    df = pd.DataFrame(rows)

    # For targets already in native units, the two R2 columns are the same
    # quantity. Fill so downstream means are over all 67 properties.
    df["r2_native"] = df["r2_native"].fillna(df["r2_model_space"])
    df["tier_reported_basis"] = df["r2_model_space"].map(assign_tier)
    df["tier_native_basis"] = df["r2_native"].map(assign_tier)
    return df


def verify_reproduction(df: pd.DataFrame) -> dict:
    checked = df["reported_r2"].notna()
    diffs = (df.loc[checked, "r2_model_space"] - df.loc[checked, "reported_r2"]).abs()
    return {
        "pairs_checked": int(checked.sum()),
        "max_abs_diff": float(diffs.max()) if len(diffs) else float("nan"),
        "failures": int((diffs > 1e-6).sum()),
        "samples_per_target_min": int(df["n_samples"].min()),
        "samples_per_target_max": int(df["n_samples"].max()),
        "all_samples_unique": bool((df["n_samples"] == df["n_unique_samples"]).all()),
        "folds": sorted(df["reported_n_folds"].dropna().unique().tolist()),
    }


# ---------------------------------------------------------------------------
# Stage 2: collapse to physical properties, keeping every seed
# ---------------------------------------------------------------------------


def build_per_property(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, prop), g in df.groupby(["model", "property"], sort=False):
        spaces = sorted(g["scored_in"].unique())
        rows.append(
            {
                "model": model,
                "property": prop,
                "n_seeds": g["seed"].nunique(),
                "scored_in": " / ".join(spaces),
                "transform_flipped_across_seeds": len(spaces) > 1,
                "r2_reported_mean": g["r2_model_space"].mean(),
                "r2_reported_sd": g["r2_model_space"].std(),
                "r2_native_mean": g["r2_native"].mean(),
                "r2_native_sd": g["r2_native"].std(),
                "rmse_native_mean": g["rmse_native"].mean(),
                "mae_native_mean": g["mae_native"].mean(),
                "r2_min_seed": g["r2_model_space"].min(),
                "r2_max_seed": g["r2_model_space"].max(),
                "n_tiers_seen": g["tier_reported_basis"].nunique(),
                "tiers_seen": ", ".join(
                    g.sort_values("seed")["tier_reported_basis"].unique()
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["tier_reported_basis"] = out["r2_reported_mean"].map(assign_tier)
    out["tier_native_basis"] = out["r2_native_mean"].map(assign_tier)
    out["scored_in_native_units"] = out["scored_in"] == "native units"
    out["r2_drop_native"] = out["r2_reported_mean"] - out["r2_native_mean"]
    out["positive_reported_negative_native"] = (out["r2_reported_mean"] > 0) & (
        out["r2_native_mean"] <= 0
    )
    return out.sort_values(["model", "r2_reported_mean"], ascending=[True, False])


def tier_summary(per_prop: pd.DataFrame, model: str, basis: str) -> dict:
    col = f"tier_{basis}_basis"
    r2col = "r2_reported_mean" if basis == "reported" else "r2_native_mean"
    a = per_prop[per_prop["model"] == model]
    counts = a[col].value_counts()
    return {
        "n_properties": len(a),
        "mean_r2": float(a[r2col].mean()),
        "median_r2": float(a[r2col].median()),
        "strong": int(counts.get("strong", 0)),
        "moderate": int(counts.get("moderate", 0)),
        "weak": int(counts.get("weak", 0)),
        "unusable": int(counts.get("unusable", 0)),
        "above_0.30": int((a[r2col] > 0.30).sum()),
        "at_or_below_0": int((a[r2col] <= 0).sum()),
        "mean_cross_seed_sd": float(a["r2_reported_sd"].mean()),
        "median_cross_seed_sd": float(a["r2_reported_sd"].median()),
    }


# ---------------------------------------------------------------------------
# Stage 3: head-to-head comparison, 67 rows, no orphans
# ---------------------------------------------------------------------------


def build_comparison(per_prop: pd.DataFrame) -> pd.DataFrame:
    rf = per_prop[per_prop["model"] == "rf"].set_index("property")
    xgb = per_prop[per_prop["model"] == "xgb"].set_index("property")
    props = sorted(set(rf.index) & set(xgb.index))
    missing = (set(rf.index) ^ set(xgb.index))
    if missing:
        print(
            f"  NOTE: {len(missing)} properties present for only one model: "
            f"{sorted(missing)}"
        )

    def winner(a: float, b: float) -> str:
        if not (np.isfinite(a) and np.isfinite(b)):
            return "no data"
        if abs(a - b) < TIE_THRESHOLD:
            return "tie"
        return "rf" if a > b else "xgb"

    rows = []
    for p in props:
        r, x = rf.loc[p], xgb.loc[p]
        rows.append(
            {
                "property": p,
                "scored_in": r["scored_in"],
                "rf_r2": r["r2_reported_mean"],
                "rf_r2_sd": r["r2_reported_sd"],
                "rf_n_seeds": int(r["n_seeds"]),
                "xgb_r2": x["r2_reported_mean"],
                "xgb_r2_sd": x["r2_reported_sd"],
                "xgb_n_seeds": int(x["n_seeds"]),
                "winner_reported_basis": winner(
                    r["r2_reported_mean"], x["r2_reported_mean"]
                ),
                "rf_r2_native": r["r2_native_mean"],
                "xgb_r2_native": x["r2_native_mean"],
                "winner_native_basis": winner(
                    r["r2_native_mean"], x["r2_native_mean"]
                ),
                "rf_tier": r["tier_reported_basis"],
                "xgb_tier": x["tier_reported_basis"],
                "delta_xgb_minus_rf": x["r2_reported_mean"] - r["r2_reported_mean"],
                "rf_sd_lower": bool(r["r2_reported_sd"] < x["r2_reported_sd"]),
                # A win smaller than the cross-seed noise is not a win.
                "win_inside_noise": bool(
                    abs(x["r2_reported_mean"] - r["r2_reported_mean"])
                    < max(r["r2_reported_sd"], x["r2_reported_sd"])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("xgb_r2", ascending=False)


# ---------------------------------------------------------------------------
# Stage 4: dual-transform audit on a common scale
# ---------------------------------------------------------------------------


def build_transform_audit(paths: Paths) -> pd.DataFrame:
    """The pipeline compares r2_raw (native units) against r2_log (log units)
    and keeps the larger. Those are not the same quantity. This records each
    decision and flags the ones driven by the scale change."""
    frames = []
    for model in MODELS:
        for seed in SEEDS:
            p = paths.find(
                f"ansoil_transform_comparison_{model}_{seed}.csv",
                model,
                seed,
                required=False,
            )
            if not p:
                continue
            d = pd.read_csv(p)
            d["model"], d["seed"] = model, seed
            frames.append(d)
    if not frames:
        return pd.DataFrame()

    t = pd.concat(frames, ignore_index=True)
    t["log_scored_higher"] = t["r2_log"] > t["r2_raw"]
    t["gap_log_minus_raw"] = t["r2_log"] - t["r2_raw"]
    t["comparison_crosses_scales"] = True  # always true by construction

    per = (
        t.groupby(["model", "raw_col"])
        .agg(
            n_seeds=("seed", "nunique"),
            n_log_selected=("selected_transform", lambda s: int((s == "log").sum())),
            n_distinct_choices=("selected_transform", "nunique"),
            mean_r2_raw=("r2_raw", "mean"),
            mean_r2_log=("r2_log", "mean"),
            mean_gap=("gap_log_minus_raw", "mean"),
        )
        .reset_index()
    )
    per["flipped_across_seeds"] = per["n_distinct_choices"] > 1
    return per.sort_values(["model", "raw_col"])


# ---------------------------------------------------------------------------
# Stage 5: mappable set
# ---------------------------------------------------------------------------


def build_mappable(comparison: pd.DataFrame) -> pd.DataFrame:
    m = comparison[
        (comparison["rf_r2"] > MAP_MIN_MEAN_R2)
        | (comparison["xgb_r2"] > MAP_MIN_MEAN_R2)
    ].copy()
    m["rf_mappable"] = m["rf_r2"] > MAP_MIN_MEAN_R2
    m["xgb_mappable"] = m["xgb_r2"] > MAP_MIN_MEAN_R2
    m["caution"] = ""
    m.loc[m["scored_in"] != "native units", "caution"] += (
        "R2 is in a transformed space; see native column. "
    )
    m.loc[m["win_inside_noise"], "caution"] += (
        "Model gap smaller than cross-seed SD; report as a tie. "
    )
    m.loc[
        (m["rf_r2"] > 0) & (m["rf_r2_native"] <= 0)
        | (m["xgb_r2"] > 0) & (m["xgb_r2_native"] <= 0),
        "caution",
    ] += "Native-units R2 is not positive; do not present as predictive. "
    return m.sort_values("xgb_r2", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 6: grid audit
# ---------------------------------------------------------------------------


def build_grid_audit(paths: Paths, per_prop: pd.DataFrame) -> tuple:
    grid_meta_path = paths.find("ansoil_grid_prepared.csv", required=False)
    if not grid_meta_path:
        return pd.DataFrame(), pd.DataFrame(), {}

    grid = pd.read_csv(grid_meta_path)
    targets = pd.read_csv(paths.find("ansoil_targets.csv"))
    preds = pd.read_csv(paths.find("ansoil_predictors.csv"))

    # Mirror the runtime unit fix in the model scripts.
    if (
        grid["dist_coast_scar_km"].max() > DIST_COAST_UNIT_THRESHOLD
        and preds["dist_coast_scar_km"].max() < DIST_COAST_UNIT_THRESHOLD
    ):
        grid["dist_coast_scar_km"] = grid["dist_coast_scar_km"] / 1000.0

    # Envelope breaches
    env_rows = []
    for c in ENVELOPE_PREDICTORS:
        if c not in grid.columns or c not in preds.columns:
            continue
        lo, hi = preds[c].min(), preds[c].max()
        outside = int(((grid[c] < lo) | (grid[c] > hi)).sum())
        env_rows.append(
            {
                "predictor": c,
                "train_min": lo,
                "train_max": hi,
                "grid_min": grid[c].min(),
                "grid_max": grid[c].max(),
                "grid_cells_outside": outside,
                "pct_outside": 100.0 * outside / len(grid),
            }
        )
    envelope = pd.DataFrame(env_rows)

    # Lithology representation
    litho_rows = []
    if "litho" in grid.columns and "litho" in preds.columns:
        tl = preds["litho"].astype(str).value_counts()
        gl = grid["litho"].astype(str).value_counts()
        for cls in sorted(set(tl.index) | set(gl.index)):
            litho_rows.append(
                {
                    "predictor": f"litho_{cls}",
                    "train_min": int(tl.get(cls, 0)),
                    "train_max": int(tl.get(cls, 0)),
                    "grid_min": np.nan,
                    "grid_max": np.nan,
                    "grid_cells_outside": int(gl.get(cls, 0))
                    if tl.get(cls, 0) < 5
                    else 0,
                    "pct_outside": 100.0 * gl.get(cls, 0) / len(grid)
                    if tl.get(cls, 0) < 5
                    else 0.0,
                }
            )
    if litho_rows:
        envelope = pd.concat(
            [envelope, pd.DataFrame(litho_rows)], ignore_index=True
        )

    # Per-map flags
    rows = []
    for model in MODELS:
        mean_path = paths.find(
            f"grid_predictions_mean_{model}.csv", required=False
        )
        sd_path = paths.find(f"grid_predictions_sd_{model}.csv", required=False)
        if not mean_path:
            continue
        M = pd.read_csv(mean_path)
        S = pd.read_csv(sd_path) if sd_path else None
        lookup = per_prop[per_prop["model"] == model].set_index("property")

        for col in [c for c in M.columns if c.startswith("pred_")]:
            target = col[len("pred_") :]
            mean_r2 = (
                float(lookup.loc[target, "r2_reported_mean"])
                if target in lookup.index
                else np.nan
            )
            vals = M[col]
            n_neg = int((vals < 0).sum())
            sd_all_nan = bool(S is not None and col in S.columns and S[col].isna().all())

            lo = hi = np.nan
            n_outside = 0
            if target in targets.columns:
                lo, hi = targets[target].min(), targets[target].max()
                n_outside = int(((vals < lo) | (vals > hi)).sum())

            flags = []
            if n_neg and not negative_is_valid(target):
                flags.append("physically impossible negative values")
            if sd_all_nan:
                flags.append("SD undefined: only one seed produced this map")
            if np.isfinite(mean_r2) and mean_r2 <= 0:
                flags.append("mean R2 <= 0")
            elif np.isfinite(mean_r2) and mean_r2 <= MAP_MIN_MEAN_R2:
                flags.append(f"mean R2 below {MAP_MIN_MEAN_R2:.2f}")
            if n_outside:
                flags.append("predicts outside observed training range")

            rows.append(
                {
                    "model": model,
                    "target": target,
                    "mean_r2": mean_r2,
                    "tier": assign_tier(mean_r2),
                    "n_cells": len(M),
                    "n_negative_cells": n_neg,
                    "negative_is_valid": negative_is_valid(target),
                    "sd_all_nan": sd_all_nan,
                    "obs_min": lo,
                    "obs_max": hi,
                    "pred_min": float(vals.min()),
                    "pred_max": float(vals.max()),
                    "n_cells_outside_obs_range": n_outside,
                    "keep_under_new_gate": bool(
                        np.isfinite(mean_r2) and mean_r2 > MAP_MIN_MEAN_R2
                    ),
                    "flags": "; ".join(flags),
                }
            )

    audit = pd.DataFrame(rows)
    summary = {}
    if len(audit):
        for model in MODELS:
            a = audit[audit["model"] == model]
            if not len(a):
                continue
            summary[model] = {
                "maps_present": len(a),
                "maps_kept_under_new_gate": int(a["keep_under_new_gate"].sum()),
                "maps_with_impossible_negatives": int(
                    ((a["n_negative_cells"] > 0) & (~a["negative_is_valid"])).sum()
                ),
                "maps_with_undefined_sd": int(a["sd_all_nan"].sum()),
                "maps_with_nonpositive_r2": int((a["mean_r2"] <= 0).sum()),
                "maps_outside_training_range": int(
                    (a["n_cells_outside_obs_range"] > 0).sum()
                ),
            }
    acbr = (
        grid["acbr"].value_counts().to_dict() if "acbr" in grid.columns else {}
    )
    summary["grid_cells"] = len(grid)
    summary["grid_cells_by_acbr"] = acbr
    return audit, envelope, summary


# ---------------------------------------------------------------------------
# Stage 7: KNN reconciled onto the same properties
# ---------------------------------------------------------------------------


def build_knn(paths: Paths, tm: TransformMap) -> pd.DataFrame:
    p = paths.find("ansoil_model_results_knn.csv", required=False)
    if not p:
        return pd.DataFrame()
    k = pd.read_csv(p)
    k["property"] = k["target"].map(tm.property_of)
    k["scored_in"] = k["target"].map(tm.space_of)
    k["tier_recomputed"] = k["cv_r2"].map(assign_tier)

    cvp = paths.find("ansoil_cv_predictions_knn.csv", required=False)
    if cvp:
        cv = pd.read_csv(cvp)
        native = {}
        for t, g in cv.groupby("target"):
            native[t] = r2_score(g["actual"], g["predicted"])
        k["r2_native"] = k["target"].map(native)
        k.loc[~k["target"].map(tm.has_native_backtransform), "r2_native"] = k["cv_r2"]
    return k[
        [
            "property",
            "target",
            "scored_in",
            "cv_r2",
            "r2_native",
            "cv_rmse_orig_units",
            "tier_recomputed",
            "best_k",
            "weighting",
        ]
    ].sort_values("cv_r2", ascending=False)


# ---------------------------------------------------------------------------
# Stage 8: the clean document
# ---------------------------------------------------------------------------


def fmt(v, nd=3, signed=True) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def md_table(df: pd.DataFrame, cols: list, headers: list, fmts: dict) -> str:
    # Leading blank line: CommonMark requires one before a table.
    lines = ["", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            f = fmts.get(c)
            cells.append(f(r[c]) if f else str(r[c]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_document(
    paths: Paths,
    repro: dict,
    per_prop: pd.DataFrame,
    comparison: pd.DataFrame,
    mappable: pd.DataFrame,
    instability: pd.DataFrame,
    transform_audit: pd.DataFrame,
    grid_audit: pd.DataFrame,
    envelope: pd.DataFrame,
    grid_summary: dict,
    knn: pd.DataFrame,
) -> str:
    T = {m: {b: tier_summary(per_prop, m, b) for b in ("reported", "native")} for m in MODELS}
    tally_rep = comparison["winner_reported_basis"].value_counts().to_dict()
    tally_nat = comparison["winner_native_basis"].value_counts().to_dict()
    n_props = T["rf"]["reported"]["n_properties"]

    prop_fmt = {
        "property": lambda v: f"`{v}`",
        "scored_in": str,
        "r2_reported_mean": lambda v: fmt(v),
        "r2_reported_sd": lambda v: fmt(v, 3, False),
        "r2_native_mean": lambda v: fmt(v),
        "rmse_native_mean": lambda v: f"{v:.4g}",
        "n_seeds": lambda v: str(int(v)),
        "tier_reported_basis": str,
    }
    prop_cols = [
        "property", "scored_in", "r2_reported_mean", "r2_reported_sd",
        "r2_native_mean", "rmse_native_mean", "tier_reported_basis", "n_seeds",
    ]
    prop_head = [
        "Property", "Scored in", "R2 as modelled", "SD", "R2 native units",
        "RMSE native", "Tier", "Seeds",
    ]

    parts = []
    parts.append(
        f"""# ANSOIL Verified Results

Generated by `ml_results_clean.py`. Every metric recomputed from the per-seed
out-of-fold prediction files. No models were re-trained.

Seeds: {", ".join(map(str, SEEDS))}. Models: {", ".join(m.upper() for m in MODELS)}.
Tie threshold {TIE_THRESHOLD}. Mappability gate: mean R2 > {MAP_MIN_MEAN_R2:.2f}.

## Reproduction check

| Check | Value |
| --- | --- |
| Target/seed pairs checked | {repro['pairs_checked']} |
| Max absolute difference, reported vs recomputed R2 | {repro['max_abs_diff']:.2e} |
| Failures above 1e-6 | {repro['failures']} |
| Samples per target | {repro['samples_per_target_min']} to {repro['samples_per_target_max']} |
| All samples unique within a target | {repro['all_samples_unique']} |
| CV folds | {", ".join(map(str, repro['folds']))} |

## How to read the R2 columns

Two R2 columns appear throughout, because the pipeline does not model every
property in its own units.

**R2 as modelled** is what the pipeline reports. For a log1p or natural-log
target this is an R2 about the logarithm, and for a CLR component it is an R2
about a log-ratio.

**R2 native units** is recomputed on back-transformed predictions, in mg/kg,
mg/L, permil, percent or pH units. CLR components have no single-concentration
inverse, so their two columns are identical by construction and remain
CLR-space figures.

Only the rows marked `native units` under **Scored in** have one unambiguous
R2. Quote the native column whenever the claim is about predicting a
concentration.

## Headline numbers

| Model | Basis | Properties | Mean R2 | Strong | Moderate | Weak | Unusable | Above 0.30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |"""
    )
    for m in MODELS:
        for b, label in (("reported", "as modelled"), ("native", "native units")):
            s = T[m][b]
            parts.append(
                f"| {m.upper()} | {label} | {s['n_properties']} | {fmt(s['mean_r2'])} | "
                f"{s['strong']} | {s['moderate']} | {s['weak']} | {s['unusable']} | "
                f"{s['above_0.30']} |"
            )

    parts.append(
        f"""
Counts are over {n_props} physical properties, each appearing once. Dual-transform
pairs whose winner changed between seeds are reconciled rather than dropped, so
every property carries all {len(SEEDS)} seeds.

## Head-to-head

| Basis | XGB wins | Ties | RF wins |
| --- | --- | --- | --- |
| As modelled | {tally_rep.get('xgb', 0)} | {tally_rep.get('tie', 0)} | {tally_rep.get('rf', 0)} |
| Native units | {tally_nat.get('xgb', 0)} | {tally_nat.get('tie', 0)} | {tally_nat.get('rf', 0)} |

Mean paired difference, XGB minus RF: {fmt(comparison['delta_xgb_minus_rf'].mean())} as modelled.

RF has the lower cross-seed SD on {int(comparison['rf_sd_lower'].sum())} of {len(comparison)} properties.
Mean cross-seed SD: RF {T['rf']['reported']['mean_cross_seed_sd']:.4f}, XGB {T['xgb']['reported']['mean_cross_seed_sd']:.4f}.

{int(comparison['win_inside_noise'].sum())} of {len(comparison)} head-to-head gaps are smaller than the larger of the
two cross-seed SDs. Those should be described as ties in prose regardless of
which side of the {TIE_THRESHOLD} threshold they fall on.

**The search budgets differ: RF draws 54 hyperparameter combinations, XGB draws
500. Both select on the same LOLO-CV score they then report, so the model with
the larger budget gains more upward bias. This comparison is not budget-matched
and should not be presented as a clean algorithm comparison.**

## Properties where the transformed and native R2 disagree

Sorted by the size of the gap. A positive gap means the reported figure
flatters the model relative to predicting the concentration itself."""
    )

    gap = per_prop[
        (~per_prop["scored_in_native_units"])
        & (per_prop["scored_in"] != "CLR")
        & (per_prop["r2_drop_native"].abs() > 0.05)
    ].sort_values("r2_drop_native", ascending=False)
    if len(gap):
        parts.append(
            md_table(
                gap,
                ["model", "property", "scored_in", "r2_reported_mean", "r2_native_mean", "r2_drop_native"],
                ["Model", "Property", "Scored in", "R2 as modelled", "R2 native", "Gap"],
                {
                    "model": lambda v: v.upper(),
                    "property": lambda v: f"`{v}`",
                    "scored_in": str,
                    "r2_reported_mean": lambda v: fmt(v),
                    "r2_native_mean": lambda v: fmt(v),
                    "r2_drop_native": lambda v: fmt(v),
                },
            )
        )
    flip = per_prop[per_prop["positive_reported_negative_native"]]
    if len(flip):
        parts.append(
            f"\n**{len(flip)} property/model combinations report a positive R2 and have a "
            "non-positive R2 in native units.** For these the model is no better "
            "than predicting the dataset mean concentration:\n"
        )
        for m in MODELS:
            names = flip[flip["model"] == m]["property"].tolist()
            if names:
                parts.append(
                    f"- {m.upper()}: " + ", ".join(f"`{n}`" for n in names)
                )

    parts.append(
        f"""
## Mappable properties

{len(mappable)} of {n_props} properties clear mean R2 > {MAP_MIN_MEAN_R2:.2f} in at least one model.
That leaves **{n_props - len(mappable)} properties that should be reported as not mappable**."""
    )
    parts.append(
        md_table(
            mappable,
            ["property", "scored_in", "rf_r2", "xgb_r2", "rf_r2_native", "xgb_r2_native", "winner_reported_basis", "caution"],
            ["Property", "Scored in", "RF R2", "XGB R2", "RF native", "XGB native", "Better", "Caution"],
            {
                "property": lambda v: f"`{v}`",
                "scored_in": str,
                "rf_r2": lambda v: fmt(v),
                "xgb_r2": lambda v: fmt(v),
                "rf_r2_native": lambda v: fmt(v),
                "xgb_r2_native": lambda v: fmt(v),
                "winner_reported_basis": str,
                "caution": lambda v: v.strip() or "",
            },
        )
    )

    if len(instability):
        parts.append(
            f"""
## Tier instability across seeds

{len(instability)} model/property combinations change tier between seeds. The `tier`
column in the old aggregated summaries is the mode, which hides these."""
        )
        parts.append(
            md_table(
                instability,
                ["model", "property", "r2_by_seed", "tiers_seen"],
                ["Model", "Property", "R2 by seed", "Tiers seen"],
                {
                    "model": lambda v: v.upper(),
                    "property": lambda v: f"`{v}`",
                    "r2_by_seed": str,
                    "tiers_seen": str,
                },
            )
        )
        strong_always = per_prop[
            (per_prop["tier_reported_basis"] == "strong")
            & (per_prop["n_tiers_seen"] == 1)
        ]["property"].unique()
        parts.append(
            "\nProperties in the strong tier under **every** seed: "
            + (", ".join(f"`{p}`" for p in strong_always) if len(strong_always) else "none")
            + ". Any strong-tier claim outside that list rests on a subset of seeds."
        )

    if len(transform_audit):
        n_flip = int(transform_audit["flipped_across_seeds"].sum())
        tot = int(transform_audit["n_seeds"].sum())
        n_log = int(transform_audit["n_log_selected"].sum())
        parts.append(
            f"""
## Dual-transform audit

The pipeline selects the dual-test transform by comparing `r2_raw`, computed in
the concentration's own units, against `r2_log`, computed in log units. Those
are different quantities, so the comparison is not valid on its face.

Log was selected in **{n_log} of {tot}** seed-target decisions, with a mean
`r2_log - r2_raw` gap of {fmt(transform_audit['mean_gap'].mean())}. {n_flip} properties
changed their selected transform between seeds, which is what produced the
"incomplete target" rows in the old aggregation. No back-transform error
occurred."""
        )
        parts.append(
            md_table(
                transform_audit[transform_audit["flipped_across_seeds"]],
                ["model", "raw_col", "n_log_selected", "n_seeds", "mean_r2_raw", "mean_r2_log"],
                ["Model", "Property", "Seeds choosing log", "Seeds", "Mean R2 raw", "Mean R2 log"],
                {
                    "model": lambda v: v.upper(),
                    "raw_col": lambda v: f"`{v}`",
                    "n_log_selected": lambda v: str(int(v)),
                    "n_seeds": lambda v: str(int(v)),
                    "mean_r2_raw": lambda v: fmt(v),
                    "mean_r2_log": lambda v: fmt(v),
                },
            )
        )

    if len(grid_audit):
        parts.append("\n## Grid audit\n")
        parts.append(
            f"Grid cells: {grid_summary.get('grid_cells', 'n/a')}. "
            + ", ".join(
                f"{k} {v}" for k, v in grid_summary.get("grid_cells_by_acbr", {}).items()
            )
        )
        parts.append(
            "\n\n| Model | Maps present | Kept under the new gate | Impossible negatives | Undefined SD | Non-positive R2 | Outside training range |\n| --- | --- | --- | --- | --- | --- | --- |"
        )
        for m in MODELS:
            s = grid_summary.get(m)
            if s:
                parts.append(
                    f"| {m.upper()} | {s['maps_present']} | {s['maps_kept_under_new_gate']} | "
                    f"{s['maps_with_impossible_negatives']} | {s['maps_with_undefined_sd']} | "
                    f"{s['maps_with_nonpositive_r2']} | {s['maps_outside_training_range']} |"
                )
        bad = grid_audit[grid_audit["flags"].str.contains("impossible", na=False)]
        if len(bad):
            parts.append(
                "\n### Maps containing physically impossible values\n"
            )
            parts.append(
                md_table(
                    bad,
                    ["model", "target", "n_negative_cells", "pred_min"],
                    ["Model", "Map", "Cells below zero", "Minimum"],
                    {
                        "model": lambda v: v.upper(),
                        "target": lambda v: f"`{v}`",
                        "n_negative_cells": lambda v: str(int(v)),
                        "pred_min": lambda v: f"{v:.4g}",
                    },
                )
            )
        if len(envelope):
            parts.append("\n### Prediction outside the training envelope\n")
            parts.append(
                md_table(
                    envelope[envelope["grid_cells_outside"] > 0],
                    ["predictor", "train_min", "train_max", "grid_min", "grid_max", "pct_outside"],
                    ["Predictor", "Train min", "Train max", "Grid min", "Grid max", "% grid outside"],
                    {
                        "predictor": lambda v: f"`{v}`",
                        "train_min": lambda v: f"{v:.4g}",
                        "train_max": lambda v: f"{v:.4g}",
                        "grid_min": lambda v: "n/a" if not np.isfinite(v) else f"{v:.4g}",
                        "grid_max": lambda v: "n/a" if not np.isfinite(v) else f"{v:.4g}",
                        "pct_outside": lambda v: f"{v:.1f}%",
                    },
                )
            )

    if len(knn):
        kt = knn["tier_recomputed"].value_counts()
        parts.append(
            f"""
## KNN baseline

Mean R2 {fmt(knn['cv_r2'].mean())} across {len(knn)} targets, single run, no cross-seed SD.
Tiers: {kt.get('strong', 0)} strong, {kt.get('moderate', 0)} moderate, {kt.get('weak', 0)} weak, {kt.get('unusable', 0)} unusable.

KNN selects hyperparameters on lowest RMSE while RF and XGB select on highest
R2, so the three-model comparison is not strictly like-for-like. KNN grid
predictions were built from a distance matrix predating the coastal-distance
metre-to-kilometre fix and should not be mapped; its cross-validated R2 values
are unaffected."""
        )

    for m in MODELS:
        parts.append(f"\n## All {n_props} properties, {m.upper()}\n")
        parts.append(
            md_table(
                per_prop[per_prop["model"] == m], prop_cols, prop_head, prop_fmt
            )
        )

    parts.append(
        """
## Known limitations of these numbers

1. **Hyperparameter tuning is not nested.** Both scripts select the
   configuration with the highest LOLO-CV R2 and then report that same R2, so
   every figure here is optimistic. Measured on this dataset, searching 54
   configurations rather than one is worth roughly +0.07 R2 on average.
2. **The model comparison is not budget-matched.** RF draws 54 combinations
   with replacement from a 54-combination grid, yielding about 33 distinct
   settings; XGB draws 500 from a 7,776-combination grid.
3. **LOLO folds are unbalanced.** One location holds 53 of 171 samples, and R2
   is computed by pooling all out-of-fold predictions rather than averaging
   per-fold scores, so that location drives about a third of every figure.
4. **One region has a single location.** When it is held out there is no
   training data for it, so its map cells are extrapolative rather than
   validated.
5. **Feature importance is impurity-based**, not permutation-based, and is
   biased toward continuous high-cardinality predictors over binary flags.
"""
    )

    doc = "\n".join(parts) + "\n"
    p = paths.out_path("VERIFIED_RESULTS.md")
    with open(p, "w") as f:
        f.write(doc)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    ap.add_argument("--flat-dir", default=None, help="fallback folder holding every CSV")
    ap.add_argument("--out-dir", default=None, help="output folder (default: <root>/results/clean)")
    ap.add_argument("--skip-grid", action="store_true", help="skip the grid audit")
    args = ap.parse_args(argv)

    paths = Paths(args.root, args.flat_dir, args.out_dir)
    print(f"Repo root: {paths.root}")
    print(f"Output:    {paths.out}\n")

    tm = TransformMap.from_lookup(pd.read_csv(paths.find("ansoil_log_targets.csv")))

    print("Recomputing per-seed metrics from raw predictions...")
    per_seed = build_per_seed(paths, tm)
    per_seed.to_csv(paths.out_path("verified_per_seed.csv"), index=False)
    repro = verify_reproduction(per_seed)
    print(
        f"  {repro['pairs_checked']} pairs, max |diff| {repro['max_abs_diff']:.2e}, "
        f"{repro['failures']} failures"
    )
    if repro["failures"]:
        print("  WARNING: reported metrics do not reproduce. Investigate before using.")

    print("Collapsing to physical properties...")
    per_prop = build_per_property(per_seed)
    per_prop.to_csv(paths.out_path("verified_per_property.csv"), index=False)
    for m in MODELS:
        s = tier_summary(per_prop, m, "reported")
        n = tier_summary(per_prop, m, "native")
        print(
            f"  {m.upper()}: {s['n_properties']} properties | mean R2 "
            f"{s['mean_r2']:+.4f} as modelled, {n['mean_r2']:+.4f} native | "
            f"tiers {s['strong']}/{s['moderate']}/{s['weak']}/{s['unusable']}"
        )

    print("Building head-to-head comparison...")
    comparison = build_comparison(per_prop)
    comparison.to_csv(paths.out_path("verified_model_comparison.csv"), index=False)
    t = comparison["winner_reported_basis"].value_counts().to_dict()
    print(
        f"  {len(comparison)} rows | XGB {t.get('xgb', 0)}, tie {t.get('tie', 0)}, "
        f"RF {t.get('rf', 0)} | {int(comparison['win_inside_noise'].sum())} gaps inside cross-seed noise"
    )

    print("Selecting the mappable set...")
    mappable = build_mappable(comparison)
    mappable.to_csv(paths.out_path("verified_mappable.csv"), index=False)
    print(
        f"  {len(mappable)} mappable, "
        f"{len(comparison) - len(mappable)} not mappable (gate: mean R2 > {MAP_MIN_MEAN_R2})"
    )

    print("Flagging tier instability...")
    inst = per_prop[per_prop["n_tiers_seen"] > 1].copy()
    if len(inst):
        by_seed = (
            per_seed.sort_values("seed")
            .groupby(["model", "property"])["r2_model_space"]
            .apply(lambda s: ", ".join(f"{v:+.3f}" for v in s))
        )
        inst["r2_by_seed"] = inst.set_index(["model", "property"]).index.map(by_seed)
    else:
        inst["r2_by_seed"] = []
    inst.to_csv(paths.out_path("tier_instability.csv"), index=False)
    print(f"  {len(inst)} model/property combinations change tier between seeds")

    print("Auditing dual-transform decisions...")
    ta = build_transform_audit(paths)
    if len(ta):
        ta.to_csv(paths.out_path("transform_audit.csv"), index=False)
        print(
            f"  {int(ta['flipped_across_seeds'].sum())} properties flipped transform "
            f"between seeds; log selected in {int(ta['n_log_selected'].sum())} decisions"
        )
    else:
        print("  transform comparison files not found, skipped")

    grid_audit, envelope, grid_summary = (pd.DataFrame(), pd.DataFrame(), {})
    if not args.skip_grid:
        print("Auditing grid predictions...")
        grid_audit, envelope, grid_summary = build_grid_audit(paths, per_prop)
        if len(grid_audit):
            grid_audit.to_csv(paths.out_path("grid_audit.csv"), index=False)
            envelope.to_csv(paths.out_path("grid_extrapolation.csv"), index=False)
            for m in MODELS:
                s = grid_summary.get(m)
                if s:
                    print(
                        f"  {m.upper()}: {s['maps_present']} maps, "
                        f"{s['maps_kept_under_new_gate']} survive the new gate, "
                        f"{s['maps_with_impossible_negatives']} with impossible negatives, "
                        f"{s['maps_with_undefined_sd']} with undefined SD"
                    )
        else:
            print("  grid files not found, skipped")

    print("Reconciling KNN baseline...")
    knn = build_knn(paths, tm)
    if len(knn):
        knn.to_csv(paths.out_path("knn_baseline.csv"), index=False)
        print(f"  {len(knn)} targets, mean R2 {knn['cv_r2'].mean():+.4f}")
    else:
        print("  KNN results not found, skipped")

    print("Writing document...")
    doc = write_document(
        paths, repro, per_prop, comparison, mappable, inst, ta,
        grid_audit, envelope, grid_summary, knn,
    )
    with open(paths.out_path("run_summary.json"), "w") as f:
        json.dump(
            {
                "reproduction": repro,
                "tiers": {m: {b: tier_summary(per_prop, m, b) for b in ("reported", "native")} for m in MODELS},
                "winner_tally_reported": comparison["winner_reported_basis"].value_counts().to_dict(),
                "winner_tally_native": comparison["winner_native_basis"].value_counts().to_dict(),
                "n_mappable": len(mappable),
                "n_not_mappable": len(comparison) - len(mappable),
                "grid": grid_summary,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nDone. Document: {doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
