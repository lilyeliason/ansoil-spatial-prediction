"""
aggregate_results.py

Aggregates ANSOIL RF and XGB multi-seed results into:
  - Per-target metrics summary (mean, median, SD, min, max of cv_r2, cv_rmse_orig_units)
  - Per-grid-point mean and SD of predictions across seeds
  - Model comparison table (RF vs XGB, with winner column)
  - Unreliable target flags (high SD relative to mean R2)

Outputs (written to results/aggregated/):
  - metrics_summary_rf.csv
  - metrics_summary_xgb.csv
  - grid_predictions_mean_rf.csv
  - grid_predictions_sd_rf.csv
  - grid_predictions_mean_xgb.csv
  - grid_predictions_sd_xgb.csv
  - model_comparison.csv
  - unreliable_targets.csv

Usage:
  Run from the repo root:
    python scripts/aggregate_results.py
"""

import os

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

SEEDS = [7, 42, 73, 123, 256]
MODELS = ["rf", "xgb"]

# Repo root = one level up from the scripts/ folder this file lives in
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "aggregated")
GRID_META = os.path.join(REPO_ROOT, "data", "ansoil_grid_prepared.csv")

# Metrics columns to aggregate across seeds
METRIC_COLS = ["cv_r2", "cv_rmse_orig_units", "cv_rmse_log_space"]

# A target is flagged unreliable if:
#   SD(cv_r2) / mean(cv_r2) > this threshold   (coefficient of variation)
# AND mean(cv_r2) < this floor
CV_THRESHOLD = 0.25  # 25% relative SD on R2
R2_FLOOR = 0.10  # also flag if mean R2 is very low regardless

# Targets excluded due to incomplete seed runs — only seed 7 produced results
# for these, most likely because near-zero values in held-out folds triggered
# a back-transform error that silently skipped writing the result row.
# No cross-seed SD is available so they are dropped from all aggregated outputs.
INCOMPLETE_TARGETS = {
    ("rf", "log_digest_mg_kg_na5895"),
    ("xgb", "log_hr_24_mg_l_so4"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_metrics(model: str) -> pd.DataFrame:
    """Load and concatenate per-seed metrics CSVs for one model."""
    frames = []
    for seed in SEEDS:
        path = os.path.join(
            RESULTS_DIR,
            f"{model}_seed{seed}",
            f"ansoil_model_results_{model}_{seed}.csv",
        )
        if not os.path.exists(path):
            print(
                f"  WARNING: missing {path} — skipping seed {seed} for {model.upper()}"
            )
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No metrics files found for model '{model}'.")
    return pd.concat(frames, ignore_index=True)


def load_grid_predictions(model: str) -> pd.DataFrame:
    """Load and concatenate per-seed grid prediction CSVs for one model."""
    frames = []
    for seed in SEEDS:
        path = os.path.join(
            RESULTS_DIR,
            f"{model}_seed{seed}",
            f"ansoil_grid_predictions_{model}_{seed}.csv",
        )
        if not os.path.exists(path):
            print(
                f"  WARNING: missing {path} — skipping seed {seed} for {model.upper()}"
            )
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No grid prediction files found for model '{model}'.")
    return pd.concat(frames, ignore_index=True)


def summarise_metrics(combined: pd.DataFrame, model: str) -> pd.DataFrame:
    """Compute mean/median/SD/min/max of each metric column, grouped by target."""
    # Drop targets with incomplete seed runs before aggregating
    incomplete = {t for (m, t) in INCOMPLETE_TARGETS if m == model}
    if incomplete:
        dropped = combined[combined["target"].isin(incomplete)]["target"].unique()
        for t in dropped:
            n = combined[combined["target"] == t]["seed"].nunique()
            print(f"  EXCLUDED (incomplete): {t} — only {n}/5 seeds present")
        combined = combined[~combined["target"].isin(incomplete)]

    agg_funcs = {
        col: ["mean", "median", "std", "min", "max"]
        for col in METRIC_COLS
        if col in combined.columns
    }
    summary = combined.groupby("target").agg(agg_funcs)
    # Flatten multi-level columns: cv_r2_mean, cv_r2_std, etc.
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    # Also carry through tier if consistent across seeds
    if "tier" in combined.columns:
        tier_mode = combined.groupby("target")["tier"].agg(lambda x: x.mode().iloc[0])
        summary = summary.merge(tier_mode.rename("tier"), on="target")
    return summary


def load_grid_meta() -> pd.DataFrame:
    """Load grid_id, lat, lon, acbr from the master grid file."""
    return pd.read_csv(GRID_META, usecols=["grid_id", "lat", "lon", "acbr"])


def summarise_grid_predictions(
    combined: pd.DataFrame, model: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-grid-point mean and SD of predicted values across seeds."""
    incomplete = {f"pred_{t}" for (m, t) in INCOMPLETE_TARGETS if m == model}
    pred_cols = [
        c for c in combined.columns if c.startswith("pred_") and c not in incomplete
    ]
    mean_df = combined.groupby("grid_id")[pred_cols].mean().reset_index()
    sd_df = combined.groupby("grid_id")[pred_cols].std().reset_index()

    # Join lat, lon, acbr from master grid file and place after grid_id
    meta = load_grid_meta()
    mean_df = meta.merge(mean_df, on="grid_id", how="right")
    sd_df = meta.merge(sd_df, on="grid_id", how="right")
    return mean_df, sd_df


def build_comparison_table(
    rf_summary: pd.DataFrame, xgb_summary: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge RF and XGB summaries into one comparison table.
    Adds a 'winner' column based on mean cv_r2.
    """
    rf_cols = {
        "target": "target",
        "cv_r2_mean": "rf_r2_mean",
        "cv_r2_std": "rf_r2_sd",
        "cv_rmse_orig_units_mean": "rf_rmse_mean",
    }
    xgb_cols = {
        "target": "target",
        "cv_r2_mean": "xgb_r2_mean",
        "cv_r2_std": "xgb_r2_sd",
        "cv_rmse_orig_units_mean": "xgb_rmse_mean",
    }

    rf_sub = rf_summary[[c for c in rf_cols if c in rf_summary.columns]].rename(
        columns=rf_cols
    )
    xgb_sub = xgb_summary[[c for c in xgb_cols if c in xgb_summary.columns]].rename(
        columns=xgb_cols
    )

    comp = rf_sub.merge(xgb_sub, on="target", how="outer")

    # Winner column
    def pick_winner(row):
        rf_r2 = row.get("rf_r2_mean", np.nan)
        xgb_r2 = row.get("xgb_r2_mean", np.nan)
        if pd.isna(rf_r2) and pd.isna(xgb_r2):
            return "no data"
        if pd.isna(rf_r2):
            return "xgb only"
        if pd.isna(xgb_r2):
            return "rf only"
        diff = rf_r2 - xgb_r2
        if abs(diff) < 0.02:  # within 0.02 R2 = effectively tied
            return "tie"
        return "rf" if diff > 0 else "xgb"

    comp["winner"] = comp.apply(pick_winner, axis=1)
    comp = comp.sort_values("rf_r2_mean", ascending=False).reset_index(drop=True)
    return comp


def flag_unreliable(
    rf_summary: pd.DataFrame, xgb_summary: pd.DataFrame
) -> pd.DataFrame:
    """
    Flag targets where SD/mean(R2) is high or mean R2 is very low.
    Returns a table of flagged targets with reason annotations.
    """
    rows = []
    for model_name, summary in [("rf", rf_summary), ("xgb", xgb_summary)]:
        if "cv_r2_mean" not in summary.columns or "cv_r2_std" not in summary.columns:
            continue
        df = summary[["target", "cv_r2_mean", "cv_r2_std"]].copy()
        df["cv_of_r2"] = df["cv_r2_std"] / df["cv_r2_mean"].abs().clip(lower=1e-6)
        df["model"] = model_name

        flagged = df[
            (df["cv_of_r2"] > CV_THRESHOLD) | (df["cv_r2_mean"] < R2_FLOOR)
        ].copy()

        reasons = []
        for _, row in flagged.iterrows():
            r = []
            if row["cv_r2_mean"] < R2_FLOOR:
                r.append(f"low mean R2 ({row['cv_r2_mean']:.3f})")
            if row["cv_of_r2"] > CV_THRESHOLD:
                r.append(f"high R2 variability (CV={row['cv_of_r2']:.2f})")
            reasons.append("; ".join(r))
        flagged["flag_reason"] = reasons
        rows.append(flagged)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["model", "cv_r2_mean"])


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summaries = {}
    grid_means = {}
    grid_sds = {}

    for model in MODELS:
        print(f"\n── {model.upper()} ──────────────────────────────")

        # Metrics
        print("  Loading metrics...")
        metrics_combined = load_metrics(model)
        summary = summarise_metrics(metrics_combined, model)
        summaries[model] = summary
        out_path = os.path.join(OUTPUT_DIR, f"metrics_summary_{model}.csv")
        summary.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}  ({len(summary)} targets)")

        # Grid predictions
        print("  Loading grid predictions...")
        grid_combined = load_grid_predictions(model)
        mean_df, sd_df = summarise_grid_predictions(grid_combined, model)
        grid_means[model] = mean_df
        grid_sds[model] = sd_df

        mean_path = os.path.join(OUTPUT_DIR, f"grid_predictions_mean_{model}.csv")
        sd_path = os.path.join(OUTPUT_DIR, f"grid_predictions_sd_{model}.csv")
        mean_df.to_csv(mean_path, index=False)
        sd_df.to_csv(sd_path, index=False)
        print(f"  Saved: {mean_path}")
        print(f"  Saved: {sd_path}")

    # Model comparison table
    print("\n── Comparison table ────────────────────────")
    if "rf" in summaries and "xgb" in summaries:
        comp = build_comparison_table(summaries["rf"], summaries["xgb"])
        comp_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
        comp.to_csv(comp_path, index=False)
        print(f"  Saved: {comp_path}  ({len(comp)} targets)")

        # Quick winner tally
        tally = comp["winner"].value_counts()
        print("\n  Winner tally:")
        for label, count in tally.items():
            print(f"    {label}: {count}")

    # Unreliable target flags
    print("\n── Unreliable target flags ─────────────────")
    flagged = flag_unreliable(
        summaries.get("rf", pd.DataFrame()),
        summaries.get("xgb", pd.DataFrame()),
    )
    if not flagged.empty:
        flag_path = os.path.join(OUTPUT_DIR, "unreliable_targets.csv")
        flagged.to_csv(flag_path, index=False)
        print(f"  Saved: {flag_path}  ({len(flagged)} flagged entries)")
        print("\n  Flagged targets (first 10):")
        print(
            flagged[["model", "target", "cv_r2_mean", "cv_of_r2", "flag_reason"]]
            .head(10)
            .to_string(index=False)
        )
    else:
        print("  No targets flagged.")

    print("\n── Done ─────────────────────────────────────")
    print(f"All outputs written to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
