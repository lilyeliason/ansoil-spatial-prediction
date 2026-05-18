"""
paper_figures.py

Generates four paper-ready figures for the ANSOIL project:

  Figure 1: R² dot plot — RF vs XGB mean LOLO CV R², mappable targets only,
            ordered by best R², with ±1 SD error bars.

  Figure 2: Model comparison boxplots — distribution of per-target mean R²
            across all targets, grouped by model. One box per model per tier.

  Figure 3: Cross-seed stability heatmap — R² per target per seed for RF and
            XGB side by side. Filtered to mappable targets only.

  Figure 4: Residual spatial scatter — mean absolute residual per sampling
            location, plotted on lat/lon. Requires ansoil_sample_predictions
            files (see NOTE below).

Run from anywhere inside the repo:
  python3 scripts/paper_figures.py

Outputs saved to: figures/paper/
  - fig1_r2_dotplot.png
  - fig2_model_boxplots.png
  - fig3_seed_heatmap.png
  - fig4_residual_spatial.png  (only if sample predictions files exist)

Requirements: pandas, numpy, matplotlib
NOTE on Figure 4: requires per-sample CV prediction files with columns:
  target, sample_location, LAT, LON, observed, predicted
  Named: ansoil_sample_predictions_{model}_{seed}.csv
  in:    results/{model}_seed{seed}/
  If missing, Fig 4 is skipped with an informative message.
"""

import os
import re
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)

RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
OUT_DIR = os.path.join(_REPO_ROOT, "figures", "paper")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

SEEDS = [7, 42, 73, 123, 256]
MODELS = ["rf", "xgb"]

INCOMPLETE_TARGETS = {
    "log_digest_mg_kg_na5895",
    "log_hr_24_mg_l_so4",
}

MAPPABILITY_THRESHOLD = 0.20

COLOR = {"rf": "#2166ac", "xgb": "#d6604d"}
LABEL = {"rf": "Random Forest", "xgb": "XGBoost"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)

# ── Target label cleanup ──────────────────────────────────────────────────────

LABEL_OVERRIDES = {
    "d15n_air_permil": "delta-15N (permil)",
    "d13c_vpdb_permil": "delta-13C (permil)",
    "wt_percent_n": "Wt% N",
    "wt_percent_c": "Wt% C",
    "c_n_ratio": "C/N ratio",
    "ph_mq": "pH (MQ)",
    "ph_kcl": "pH (KCl)",
    "ph_cacl2": "pH (CaCl2)",
    "ec_us_cm": "EC (uS/cm)",
    "cec_meq_100g": "CEC (meq/100g)",
    "digest_mg_kg_al": "Al digest",
    "digest_mg_kg_as": "As digest",
    "digest_mg_kg_b": "B digest",
    "digest_mg_kg_ba": "Ba digest",
    "digest_mg_kg_be": "Be digest",
    "digest_mg_kg_ca": "Ca digest",
    "digest_mg_kg_co": "Co digest",
    "digest_mg_kg_cr": "Cr digest",
    "digest_mg_kg_cu": "Cu digest",
    "digest_mg_kg_fe": "Fe digest",
    "digest_mg_kg_hg": "Hg digest",
    "digest_mg_kg_k": "K digest",
    "digest_mg_kg_li": "Li digest",
    "digest_mg_kg_mg": "Mg digest",
    "digest_mg_kg_mn": "Mn digest",
    "digest_mg_kg_mo": "Mo digest",
    "digest_mg_kg_na": "Na digest",
    "digest_mg_kg_ni": "Ni digest",
    "digest_mg_kg_p": "P digest",
    "digest_mg_kg_pb": "Pb digest",
    "digest_mg_kg_sb": "Sb digest",
    "digest_mg_kg_si": "Si digest",
    "digest_mg_kg_sn": "Sn digest",
    "digest_mg_kg_sr": "Sr digest",
    "digest_mg_kg_ti": "Ti digest",
    "digest_mg_kg_tl": "Tl digest",
    "digest_mg_kg_v": "V digest",
    "digest_mg_kg_zn": "Zn digest",
    "hr_1_mg_l_f": "F- (1hr)",
    "hr_1_mg_l_cl": "Cl- (1hr)",
    "hr_1_mg_l_no3": "NO3- (1hr)",
    "hr_1_mg_l_po4": "PO4 (1hr)",
    "hr_1_mg_l_so4": "SO4 (1hr)",
    "hr_24_mg_l_f": "F- (24hr)",
    "hr_24_mg_l_cl": "Cl- (24hr)",
    "hr_24_mg_l_no3": "NO3- (24hr)",
    "hr_24_mg_l_po4": "PO4 (24hr)",
    "hr_24_mg_l_so4": "SO4 (24hr)",
    "total_mg_l_f": "F- (total)",
    "total_mg_l_cl": "Cl- (total)",
    "total_mg_l_no3": "NO3- (total)",
    "total_mg_l_po4": "PO4 (total)",
    "total_mg_l_so4": "SO4 (total)",
    "total_mg_l_ca2": "Ca2+ (total)",
    "total_mg_l_k": "K+ (total)",
    "total_mg_l_mg2": "Mg2+ (total)",
    "total_mg_l_na": "Na+ (total)",
    "total_mg_l_sr2": "Sr2+ (total)",
}


def clean_target_label(raw: str) -> str:
    s = re.sub(r"^(log_|clr_)", "", raw)
    s_stripped = re.sub(r"_\d{4,}$", "", s)
    if s_stripped in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[s_stripped]
    if s in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[s]
    return s_stripped.replace("_", " ").title()


# ── Data loading ──────────────────────────────────────────────────────────────


def load_metrics(model: str) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        path = os.path.join(
            RESULTS_DIR,
            f"{model}_seed{seed}",
            f"ansoil_model_results_{model}_{seed}.csv",
        )
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}")
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No results files found for model='{model}'. "
            f"Expected in: {RESULTS_DIR}/{model}_seed*/"
        )
    out = pd.concat(frames, ignore_index=True)
    return out[~out["target"].isin(INCOMPLETE_TARGETS)].copy()


def load_sample_predictions(model: str) -> pd.DataFrame:
    """
    Load per-sample CV predictions. Tries two sources in order:
      1. Per-seed files: results/{model}_seed{seed}/ansoil_sample_predictions_{model}_{seed}.csv
         (future format, written by updated model scripts)
      2. Single-seed files: results/{model}_seed{seed}/ansoil_cv_predictions_{model}.csv
         (current format — same predictions written per seed run)
    Joins LAT/LON/sample_location from ansoil_sample_index.csv.
    Columns used downstream: target, sample_location, LAT, LON, observed, predicted
    """
    # Sample index for lat/lon/location join
    sample_index_paths = [
        os.path.join(_REPO_ROOT, "data", "ansoil_sample_index.csv"),
        os.path.join(_REPO_ROOT, "ansoil_sample_index.csv"),
    ]
    sample_index = None
    for p in sample_index_paths:
        if os.path.exists(p):
            sample_index = pd.read_csv(p)
            break

    frames = []
    for seed in SEEDS:
        seed_dir = os.path.join(RESULTS_DIR, f"{model}_seed{seed}")

        # Prefer new per-seed format
        path = os.path.join(seed_dir, f"ansoil_sample_predictions_{model}_{seed}.csv")
        if not os.path.exists(path):
            # Current format: ansoil_cv_predictions_{model}_{seed}.csv
            path = os.path.join(seed_dir, f"ansoil_cv_predictions_{model}_{seed}.csv")
        if not os.path.exists(path):
            # Fallback: without seed suffix
            path = os.path.join(seed_dir, f"ansoil_cv_predictions_{model}.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        df["seed"] = seed

        # Normalise column names to what Fig 4 expects
        if "actual" in df.columns and "observed" not in df.columns:
            df = df.rename(columns={"actual": "observed"})

        # Join location info if we have the sample index
        if sample_index is not None and "sample_id" in df.columns:
            loc_cols = [
                c
                for c in [
                    "sample_id",
                    "sample_location",
                    "lat",
                    "lon",
                    "LAT",
                    "LON",
                    "acbr",
                ]
                if c in sample_index.columns
            ]
            if "sample_id" in loc_cols:
                df = df.merge(sample_index[loc_cols], on="sample_id", how="left")
            # Normalise to uppercase LAT/LON for Fig 4
            if "lat" in df.columns and "LAT" not in df.columns:
                df = df.rename(columns={"lat": "LAT", "lon": "LON"})

        frames.append(df)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out[~out["target"].isin(INCOMPLETE_TARGETS)].copy()


def mean_r2_per_target(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_df.groupby("target")["cv_r2"]
        .agg(mean_r2="mean", sd_r2="std")
        .reset_index()
    )


# ── Figure 1: R² dot plot ─────────────────────────────────────────────────────


def fig1_r2_dotplot(rf_metrics: pd.DataFrame, xgb_metrics: pd.DataFrame):
    """
    Cleveland-style dot plot of mean LOLO CV R² for RF and XGB.
    Filtered to mappable targets only. Ordered by best R² descending.
    Error bars show +/-1 SD across seeds. Clean chemical display names.
    """
    print("Building Figure 1: R2 dot plot...")

    rf_agg = mean_r2_per_target(rf_metrics).rename(
        columns={"mean_r2": "rf_r2", "sd_r2": "rf_sd"}
    )
    xgb_agg = mean_r2_per_target(xgb_metrics).rename(
        columns={"mean_r2": "xgb_r2", "sd_r2": "xgb_sd"}
    )

    merged = rf_agg.merge(xgb_agg, on="target", how="inner")
    merged = merged[
        (merged["rf_r2"] >= MAPPABILITY_THRESHOLD)
        | (merged["xgb_r2"] >= MAPPABILITY_THRESHOLD)
    ].copy()

    merged["best_r2"] = merged[["rf_r2", "xgb_r2"]].max(axis=1)
    merged = merged.sort_values("best_r2", ascending=True).reset_index(drop=True)
    merged["label"] = merged["target"].apply(clean_target_label)

    n = len(merged)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(6.5, max(4, n * 0.32)))

    # Connecting line between the two model dots
    for i, row in merged.iterrows():
        ax.plot(
            [row["rf_r2"], row["xgb_r2"]], [i, i], color="#cccccc", lw=0.9, zorder=1
        )

    ax.errorbar(
        merged["rf_r2"],
        y,
        xerr=merged["rf_sd"],
        fmt="o",
        color=COLOR["rf"],
        label=LABEL["rf"],
        ms=5,
        capsize=2.5,
        elinewidth=0.8,
        zorder=3,
    )

    ax.errorbar(
        merged["xgb_r2"],
        y,
        xerr=merged["xgb_sd"],
        fmt="o",
        color=COLOR["xgb"],
        label=LABEL["xgb"],
        ms=5,
        capsize=2.5,
        elinewidth=0.8,
        zorder=3,
    )

    ax.axvline(
        MAPPABILITY_THRESHOLD,
        color="black",
        lw=0.9,
        ls="--",
        alpha=0.6,
        zorder=2,
        label=f"Mappability threshold (R2 = {MAPPABILITY_THRESHOLD})",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(merged["label"], fontsize=8)
    ax.set_xlabel("Mean LOLO CV R2", fontsize=9)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower right", fontsize=9, frameon=False, bbox_to_anchor=(1.0, 0.01))
    ax.set_title("Model performance by target variable", fontsize=13, pad=10)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig1_r2_dotplot.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: Model comparison boxplots ───────────────────────────────────────


def fig2_model_boxplots(rf_metrics: pd.DataFrame, xgb_metrics: pd.DataFrame):
    """
    Two panels:
      Left:  per-target mean R2 boxplots split by performance tier (RF vs XGB).
      Right: per-seed R2 distribution for each model (all targets pooled),
             showing within-model seed-to-seed variance.
    """
    print("Building Figure 2: Model comparison boxplots...")

    rf_mean = mean_r2_per_target(rf_metrics)[["target", "mean_r2"]].copy()
    xgb_mean = mean_r2_per_target(xgb_metrics)[["target", "mean_r2"]].copy()
    rf_mean["model"] = "RF"
    xgb_mean["model"] = "XGB"
    combined_mean = pd.concat([rf_mean, xgb_mean], ignore_index=True)

    def tier(r2):
        if r2 >= 0.40:
            return "Strong\n(R2>=0.40)"
        if r2 >= 0.20:
            return "Moderate\n(0.20-0.40)"
        return "Weak\n(R2<0.20)"

    combined_mean["tier"] = combined_mean["mean_r2"].apply(tier)
    tier_order = ["Strong\n(R2>=0.40)", "Moderate\n(0.20-0.40)", "Weak\n(R2<0.20)"]

    rf_seed = rf_metrics[["target", "seed", "cv_r2"]].copy()
    rf_seed["model"] = "RF"
    xgb_seed = xgb_metrics[["target", "seed", "cv_r2"]].copy()
    xgb_seed["model"] = "XGB"
    all_seed = pd.concat([rf_seed, xgb_seed], ignore_index=True)

    rf_patch = mpatches.Patch(color=COLOR["rf"], label=LABEL["rf"])
    xgb_patch = mpatches.Patch(color=COLOR["xgb"], label=LABEL["xgb"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.55})

    # Left: tiered boxplot
    ax = axes[0]
    group_gap = 1.6  # wider gap between tier groups
    model_gap = 0.45  # gap between RF/XGB within a group
    box_width = 0.35
    x = 0
    tick_pos = []

    for tier_name in tier_order:
        tier_start = x
        for model_name, color in [("RF", COLOR["rf"]), ("XGB", COLOR["xgb"])]:
            data = combined_mean[
                (combined_mean["tier"] == tier_name)
                & (combined_mean["model"] == model_name)
            ]["mean_r2"].dropna()

            ax.boxplot(
                data,
                positions=[x],
                widths=box_width,
                patch_artist=True,
                medianprops=dict(color="white", linewidth=1.5),
                boxprops=dict(facecolor=color, alpha=0.85),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(
                    marker="o",
                    markersize=3,
                    markerfacecolor=color,
                    alpha=0.5,
                    linestyle="none",
                ),
            )
            x += model_gap

        tick_pos.append((tier_start + x - model_gap) / 2)
        x += group_gap

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(
        ["Strong\n(R²≥0.40)", "Moderate\n(0.20–0.40)", "Weak\n(R²<0.20)"], fontsize=8.5
    )
    ax.set_ylabel("Mean LOLO CV R²", fontsize=9)
    ax.set_title("Performance by tier", fontsize=10, pad=8)
    ax.axhline(MAPPABILITY_THRESHOLD, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.legend(
        handles=[rf_patch, xgb_patch],
        fontsize=8.5,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )

    # Right: per-seed distribution
    ax2 = axes[1]
    x = 0
    seed_tick_pos = []
    group_gap_s = 1.6
    model_gap_s = 0.45

    for seed in SEEDS:
        seed_start = x
        for model_name, color in [("RF", COLOR["rf"]), ("XGB", COLOR["xgb"])]:
            data = all_seed[
                (all_seed["seed"] == seed) & (all_seed["model"] == model_name)
            ]["cv_r2"].dropna()

            ax2.boxplot(
                data,
                positions=[x],
                widths=box_width,
                patch_artist=True,
                medianprops=dict(color="white", linewidth=1.5),
                boxprops=dict(facecolor=color, alpha=0.85),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(
                    marker="o",
                    markersize=2.5,
                    markerfacecolor=color,
                    alpha=0.4,
                    linestyle="none",
                ),
            )
            x += model_gap_s

        seed_tick_pos.append((seed_start + x - model_gap_s) / 2)
        x += group_gap_s

    ax2.set_xticks(seed_tick_pos)
    ax2.set_xticklabels([str(s) for s in SEEDS], fontsize=9)
    ax2.set_xlabel("Seed", fontsize=9)
    ax2.set_ylabel("LOLO CV R² (all targets)", fontsize=9)
    ax2.set_title("Cross-seed stability (all targets)", fontsize=10, pad=8)
    ax2.axhline(0, color="black", lw=0.6, alpha=0.3)
    ax2.legend(
        handles=[rf_patch, xgb_patch],
        fontsize=8.5,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )

    # Save each panel as its own PNG
    for panel_ax, panel_title, fname in [
        (ax, "Performance by tier", "fig2a_performance_by_tier.png"),
        (ax2, "Cross-seed stability (all targets)", "fig2b_crossseed_stability.png"),
    ]:
        extent = panel_ax.get_tightbbox(fig.canvas.get_renderer())
        # expand bbox slightly to include legend / labels outside the axes
        extent = extent.expanded(1.02, 1.08)
        out_p = os.path.join(OUT_DIR, fname)
        fig.savefig(
            out_p,
            dpi=200,
            bbox_inches=extent.transformed(fig.dpi_scale_trans.inverted()),
        )
        print(f"  Saved: {out_p}")

    # Also save the combined figure for reference
    fig.suptitle("Model comparison: Random Forest vs XGBoost", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    out = os.path.join(OUT_DIR, "fig2_model_boxplots.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3: Cross-seed stability heatmap ────────────────────────────────────


def fig3_seed_heatmap(rf_metrics: pd.DataFrame, xgb_metrics: pd.DataFrame):
    """
    Heatmap of LOLO CV R2 per target per seed for RF and XGB.
    Filtered to mappable targets only. Sorted by RF mean R2 descending.
    Clean chemical labels. NaN cells shown with grey hatching.
    Colorscale capped at 0.75 so delta-15N does not bleach the palette.
    """
    print("Building Figure 3: Cross-seed stability heatmap...")

    def pivot(df):
        return df.pivot_table(
            index="target", columns="seed", values="cv_r2", aggfunc="mean"
        ).reindex(columns=SEEDS)

    rf_pivot = pivot(rf_metrics)
    xgb_pivot = pivot(xgb_metrics)

    rf_mean = rf_pivot.mean(axis=1)
    xgb_mean = xgb_pivot.mean(axis=1)
    shared = rf_pivot.index.intersection(xgb_pivot.index)
    mappable = shared[
        (rf_mean.reindex(shared) >= MAPPABILITY_THRESHOLD)
        | (xgb_mean.reindex(shared) >= MAPPABILITY_THRESHOLD)
    ]

    order = rf_mean.reindex(mappable).sort_values(ascending=False).index
    rf_pivot = rf_pivot.loc[order]
    xgb_pivot = xgb_pivot.loc[order]

    labels = [clean_target_label(t) for t in order]
    n = len(order)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.5, max(4, n * 0.30 + 1.2)),
        sharey=True,
        gridspec_kw={"wspace": 0.06},
    )

    vmin, vmax = 0.0, 0.75  # cap so d15N doesn't bleach everything

    for ax, pivot_df, model in zip(axes, [rf_pivot, xgb_pivot], MODELS):
        data = pivot_df.values.astype(float)

        im = ax.imshow(
            data,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="RdYlGn",
            interpolation="nearest",
        )

        for row_i in range(n):
            for col_j in range(len(SEEDS)):
                val = data[row_i, col_j]
                if np.isnan(val):
                    ax.add_patch(
                        plt.Rectangle(
                            (col_j - 0.5, row_i - 0.5),
                            1,
                            1,
                            fill=True,
                            facecolor="#eeeeee",
                            hatch="///",
                            edgecolor="#bbbbbb",
                            lw=0,
                        )
                    )
                else:
                    text_color = "white" if (val < 0.15 or val > 0.58) else "black"
                    ax.text(
                        col_j,
                        row_i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=text_color,
                    )

        ax.set_xticks(range(len(SEEDS)))
        ax.set_xticklabels([str(s) for s in SEEDS], fontsize=8)
        ax.set_xlabel("Seed", fontsize=8)
        ax.set_title(LABEL[model], fontsize=9, pad=5, color="black", fontweight="bold")

        ax.set_xticks(np.arange(-0.5, len(SEEDS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.4)
        ax.tick_params(which="minor", length=0)

    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(labels, fontsize=7.5)

    cbar = fig.colorbar(
        im, ax=axes, orientation="vertical", fraction=0.025, pad=0.02, shrink=0.95
    )
    cbar.set_label("LOLO CV R²", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0.0, 0.25, 0.50, 0.75])
    cbar.set_ticklabels(["0.00", "0.25", "0.50", "0.75 (max)"])

    fig.suptitle(
        "Cross-seed stability of LOLO CV R2\n(mappable targets only)",
        fontsize=10,
        y=0.94,
    )

    out = os.path.join(OUT_DIR, "fig3_seed_heatmap.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 4: Residual spatial scatter ───────────────────────────────────────


def fig4_acbr_error_table(
    rf_samples: pd.DataFrame,
    xgb_samples: pd.DataFrame,
    rf_metrics: pd.DataFrame,
    xgb_metrics: pd.DataFrame,
):
    """
    Heatmap table: rows = mappable targets (sorted by mean R² descending),
    columns = 4 ACBRs + overall mean R².
    Cell value = normalized MAE (MAE / observed range) for the best model
    per target, averaged across all 5 seeds.
    Color: white (low error) -> red (high error). Overall R² column uses
    a separate blue colorscale for contrast.

    Requires sample predictions with sample_id, joined to ACBR via sample index.
    """
    if rf_samples.empty and xgb_samples.empty:
        print("  SKIPPED Fig 4: no sample prediction files found.")
        print(
            "  Expected: results/{model}_seed{seed}/ansoil_cv_predictions_{model}_{seed}.csv"
        )
        print("  Also needs: data/ansoil_sample_index.csv with acbr column")
        return

    if "acbr" not in rf_samples.columns and "acbr" not in xgb_samples.columns:
        print("  SKIPPED Fig 4: 'acbr' column not found after sample index join.")
        print("  Check that data/ansoil_sample_index.csv has an 'acbr' column.")
        return

    print("Building Figure 4: ACBR error table...")

    # ── Best model per target by mean R² ─────────────────────────────────────
    rf_agg = mean_r2_per_target(rf_metrics).set_index("target")["mean_r2"]
    xgb_agg = mean_r2_per_target(xgb_metrics).set_index("target")["mean_r2"]

    # Collect mappable targets sorted by best R²
    all_targets = rf_agg.index.union(xgb_agg.index)
    best_r2 = pd.Series(
        {t: max(rf_agg.get(t, -np.inf), xgb_agg.get(t, -np.inf)) for t in all_targets}
    )
    mappable = (
        best_r2[best_r2 >= MAPPABILITY_THRESHOLD]
        .sort_values(ascending=False)
        .index.tolist()
    )

    # ── Compute normalized MAE per target per ACBR ───────────────────────────
    def norm_mae_by_acbr(samples_df, targets):
        """For each target, compute MAE/range per ACBR, averaged across seeds."""
        if samples_df.empty or "acbr" not in samples_df.columns:
            return pd.DataFrame()
        rows = []
        for target in targets:
            df_t = samples_df[samples_df["target"] == target].copy()
            if df_t.empty:
                continue
            df_t["abs_resid"] = (df_t["observed"] - df_t["predicted"]).abs()
            obs_range = df_t["observed"].max() - df_t["observed"].min()
            if obs_range == 0:
                continue
            acbr_mae = df_t.groupby("acbr")["abs_resid"].mean() / obs_range
            acbr_mae.name = target
            rows.append(acbr_mae)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # Use best model's predictions per target
    records = []
    for target in mappable:
        rf_r = rf_agg.get(target, -np.inf)
        xgb_r = xgb_agg.get(target, -np.inf)
        samples = rf_samples if rf_r >= xgb_r else xgb_samples
        df_t = samples[samples["target"] == target].copy()
        if df_t.empty or "acbr" not in df_t.columns:
            continue
        df_t["abs_resid"] = (df_t["observed"] - df_t["predicted"]).abs()
        obs_range = df_t["observed"].max() - df_t["observed"].min()
        if obs_range == 0:
            continue
        acbr_mae = df_t.groupby("acbr")["abs_resid"].mean() / obs_range
        acbr_mae["target"] = target
        acbr_mae["mean_r2"] = best_r2[target]
        acbr_mae["best_model"] = "RF" if rf_r >= xgb_r else "XGB"
        records.append(acbr_mae)

    if not records:
        print("  SKIPPED Fig 4: could not compute ACBR MAE (check acbr column).")
        return

    table = pd.DataFrame(records).set_index("target")
    acbr_cols = [c for c in table.columns if c not in ("mean_r2", "best_model")]
    acbr_cols_sorted = sorted(acbr_cols)  # alphabetical ACBR order
    table = table[acbr_cols_sorted + ["mean_r2", "best_model"]]
    table["label"] = [clean_target_label(t) for t in table.index]

    # ── Sample counts per ACBR ────────────────────────────────────────────────
    all_samp = pd.concat(
        [s for s in [rf_samples, xgb_samples] if not s.empty], ignore_index=True
    )
    if "acbr" in all_samp.columns and "sample_id" in all_samp.columns:
        acbr_counts = (
            all_samp.drop_duplicates("sample_id").groupby("acbr")["sample_id"].count()
        )
    else:
        acbr_counts = pd.Series(dtype=int)

    def shorten_acbr(name):
        m = {
            "North Victoria Land": "N. Victoria\nLand",
            "Northwest Antarctic Peninsula": "NW Antarctic\nPeninsula",
            "South Victoria Land": "S. Victoria\nLand",
            "Transantarctic Mountains": "Transantarctic\nMtns",
            "East Antarctica": "East\nAntarctica",
            "West Antarctica": "West\nAntarctica",
        }
        return m.get(name, "\n".join(name.split(" ", 1)))

    table = pd.DataFrame(records).set_index("target")
    acbr_cols = [c for c in table.columns if c not in ("mean_r2", "best_model")]
    acbr_cols_sorted = sorted(acbr_cols)
    table = table[acbr_cols_sorted + ["mean_r2", "best_model"]]
    table["label"] = [clean_target_label(t) for t in table.index]

    n_rows = len(table)
    n_acbr = len(acbr_cols_sorted)
    labels = table["label"].tolist()
    y = np.arange(n_rows)

    err_data = table[acbr_cols_sorted].values.astype(float)
    vmax_err = np.nanpercentile(err_data[~np.isnan(err_data)], 95)

    # ── Figure geometry ───────────────────────────────────────────────────────
    ROW_H = 0.28  # inches per row
    ACBR_W = 1.25  # inches per ACBR column
    R2_W = 0.70
    MDL_W = 0.52
    LABEL_W = 2.05  # left margin for row labels
    PAD_R = 0.25  # right padding

    fig_w = LABEL_W + ACBR_W * n_acbr + R2_W + MDL_W + PAD_R
    fig_h = n_rows * ROW_H + 1.5  # +1.5 for header + colorbar

    fig = plt.figure(figsize=(fig_w, fig_h))

    from matplotlib.gridspec import GridSpec

    gs = GridSpec(
        1,
        n_acbr + 2,
        figure=fig,
        left=LABEL_W / fig_w,
        right=1 - PAD_R / fig_w,
        top=0.86,
        bottom=0.10,
        wspace=0.025,
        width_ratios=[ACBR_W] * n_acbr + [R2_W, MDL_W],
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_acbr + 2)]

    def tc(val, vmin, vmax):
        return "white" if (val - vmin) / max(vmax - vmin, 1e-9) > 0.62 else "#222"

    # ── ACBR columns ──────────────────────────────────────────────────────────
    for col_i, acbr in enumerate(acbr_cols_sorted):
        ax = axes[col_i]
        vals = table[acbr].values.astype(float)

        im_err = ax.imshow(
            vals.reshape(-1, 1),
            aspect="auto",
            vmin=0,
            vmax=vmax_err,
            cmap="RdYlGn_r",
            interpolation="nearest",
        )

        for ri, v in enumerate(vals):
            if np.isnan(v):
                ax.add_patch(
                    plt.Rectangle(
                        (-0.5, ri - 0.5),
                        1,
                        1,
                        facecolor="#f2f2f2",
                        hatch="//",
                        edgecolor="#ccc",
                        lw=0,
                        zorder=2,
                    )
                )
            else:
                ax.text(
                    0,
                    ri,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=tc(v, 0, vmax_err),
                )

        n_s = acbr_counts.get(acbr, "?")
        short = shorten_acbr(acbr)
        ax.set_title(
            f"{short}\nn = {n_s}",
            fontsize=7.5,
            pad=5,
            fontweight="bold",
            linespacing=1.5,
        )

        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.set_yticks(y)
        if col_i == 0:
            ax.set_yticklabels(labels, fontsize=7.5, ha="right")
            ax.tick_params(axis="y", length=0, pad=5)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", length=0)
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)

    # ── Mean R² column ────────────────────────────────────────────────────────
    ax_r2 = axes[n_acbr]
    r2_vals = table["mean_r2"].values.astype(float)
    im_r2 = ax_r2.imshow(
        r2_vals.reshape(-1, 1),
        aspect="auto",
        vmin=MAPPABILITY_THRESHOLD,
        vmax=max(r2_vals.max(), 0.70),
        cmap="Blues",
        interpolation="nearest",
    )
    for ri, v in enumerate(r2_vals):
        ax_r2.text(
            0,
            ri,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color=tc(v, MAPPABILITY_THRESHOLD, 0.70),
        )
    ax_r2.set_title("Mean\nR²", fontsize=7.5, pad=5, fontweight="bold")
    ax_r2.set_xticks([])
    ax_r2.set_yticklabels([])
    ax_r2.set_xlim(-0.5, 0.5)
    ax_r2.set_ylim(n_rows - 0.5, -0.5)
    ax_r2.set_yticks(y)
    ax_r2.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax_r2.grid(which="minor", color="white", linewidth=1.0)
    ax_r2.tick_params(which="minor", length=0)
    for sp in ax_r2.spines.values():
        sp.set_linewidth(0.4)

    # ── Best model column ─────────────────────────────────────────────────────
    ax_m = axes[n_acbr + 1]
    ax_m.set_xlim(-0.5, 0.5)
    ax_m.set_ylim(n_rows - 0.5, -0.5)
    for ri, mdl in enumerate(table["best_model"].tolist()):
        ax_m.add_patch(
            plt.Rectangle(
                (-0.5, ri - 0.5),
                1,
                1,
                facecolor=COLOR.get(mdl.lower(), "#aaa"),
                alpha=0.82,
                zorder=1,
            )
        )
        ax_m.text(
            0,
            ri,
            mdl,
            ha="center",
            va="center",
            fontsize=6.5,
            color="white",
            fontweight="bold",
        )
    ax_m.set_title("Best\nmodel", fontsize=7.5, pad=5, fontweight="bold")
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    ax_m.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax_m.grid(which="minor", color="white", linewidth=1.0)
    ax_m.tick_params(which="minor", length=0)
    for sp in ax_m.spines.values():
        sp.set_linewidth(0.4)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    left_frac = LABEL_W / fig_w
    width_frac = (ACBR_W * n_acbr) / fig_w
    cbar_ax = fig.add_axes([left_frac, 0.04, width_frac, 0.018])
    cbar = fig.colorbar(im_err, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Normalized MAE (lower = better)", fontsize=7.5, labelpad=2)
    cbar.ax.tick_params(labelsize=7)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(
        left_frac + width_frac / 2,
        0.955,
        "Prediction error by ACBR — normalized MAE, best model per target",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )

    out = os.path.join(OUT_DIR, "fig4_acbr_error_table.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 5: Four-model comparison (small multiples grid) ───────────────────


def fig5_four_model_comparison(rf_metrics: pd.DataFrame, xgb_metrics: pd.DataFrame):
    """
    Small-multiples grid: one panel per mappable target, four horizontal bars
    per panel (KNN, RF, XGB, Kriging). Each panel has its own x-axis so
    negative kriging values don't compress the other bars. Panels arranged
    in N_COLS columns, sorted best-to-worst by mean RF/XGB R².

    Kriging bars only appear where data exists. KNN loaded automatically.
    CONFIG: update KNN_PATH candidates and KRIGING_DIR if needed.
    """
    # ── paths ─────────────────────────────────────────────────────────────────
    KRIGING_DIR = os.path.join(RESULTS_DIR, "kriging", "kriging_results")
    KRIGING_R2_COL = "regression_kriging"
    KNN_CANDIDATES = [
        os.path.join(RESULTS_DIR, "knn", "ansoil_model_results_knn.csv"),
        os.path.join(RESULTS_DIR, "ansoil_model_results_knn.csv"),
        os.path.join(_REPO_ROOT, "ansoil_model_results_knn.csv"),
    ]
    KNN_PATH = next((p for p in KNN_CANDIDATES if os.path.exists(p)), None)
    N_COLS = 4  # panels per row — adjust if layout feels too wide/narrow
    # ─────────────────────────────────────────────────────────────────────────

    print("Building Figure 5: Four-model small-multiples grid...")

    # ── Load kriging ──────────────────────────────────────────────────────────
    kriging_rows = {}
    if os.path.isdir(KRIGING_DIR):
        for fname in sorted(os.listdir(KRIGING_DIR)):
            if not fname.startswith("table_loo_cv_") or not fname.endswith(".csv"):
                continue
            remainder = fname.replace("table_loo_cv_", "").replace(".csv", "")
            target = None
            is_rf = False
            for tag in ("rf_", "xgb_"):
                if remainder.startswith(tag):
                    target = remainder[len(tag) :]
                    is_rf = tag == "rf_"
                    break
            if target is None or target in INCOMPLETE_TARGETS:
                continue
            if target in kriging_rows and not is_rf:
                continue
            try:
                df = pd.read_csv(os.path.join(KRIGING_DIR, fname))
                if KRIGING_R2_COL in df.columns:
                    kriging_rows[target] = float(df[KRIGING_R2_COL].iloc[0])
            except Exception:
                continue
    kriging_agg = pd.Series(kriging_rows, name="kriging_r2")
    print(f"  Kriging: {len(kriging_rows)} targets")

    # ── Load KNN ─────────────────────────────────────────────────────────────
    knn_agg = pd.Series(dtype=float, name="knn_r2")
    if KNN_PATH:
        try:
            knn_df = pd.read_csv(KNN_PATH)
            knn_df = knn_df[~knn_df["target"].isin(INCOMPLETE_TARGETS)]
            # KNN uses raw column names; RF/XGB use log_/clr_ prefixed names.
            # Build a lookup from stripped name -> KNN R² so we can match both.
            knn_df["target_stripped"] = knn_df["target"].str.replace(
                r"^(log_|clr_)", "", regex=True
            )
            knn_lookup = knn_df.set_index("target_stripped")["cv_r2"]
            knn_agg = knn_df.set_index("target")["cv_r2"].rename("knn_r2")
            print(f"  KNN: {len(knn_agg)} targets (raw names)")
        except Exception as e:
            print(f"  KNN load failed: {e}")
    else:
        print("  KNN file not found — KNN dots will be omitted")
        knn_lookup = pd.Series(dtype=float)

    # ── Merge — match KNN to RF/XGB targets by stripping log_/clr_ prefix ───
    rf_agg = (
        mean_r2_per_target(rf_metrics).set_index("target")["mean_r2"].rename("rf_r2")
    )
    xgb_agg = (
        mean_r2_per_target(xgb_metrics).set_index("target")["mean_r2"].rename("xgb_r2")
    )

    merged = rf_agg.to_frame().join(xgb_agg, how="outer").join(kriging_agg, how="left")

    # Match KNN by stripping prefix from RF/XGB target names
    def lookup_knn(target_name):
        # Try exact match first
        if target_name in knn_agg.index:
            return knn_agg[target_name]
        # Try stripped match
        stripped = re.sub(r"^(log_|clr_)", "", target_name)
        if stripped in knn_lookup.index:
            return knn_lookup[stripped]
        return np.nan

    merged["knn_r2"] = [lookup_knn(t) for t in merged.index]
    merged = merged.reset_index()

    merged = merged[
        (merged["rf_r2"].fillna(0) >= MAPPABILITY_THRESHOLD)
        | (merged["xgb_r2"].fillna(0) >= MAPPABILITY_THRESHOLD)
    ].copy()

    merged["best_r2"] = merged[["rf_r2", "xgb_r2"]].max(axis=1)
    merged = merged.sort_values("best_r2", ascending=True).reset_index(drop=True)
    merged["label"] = merged["target"].apply(clean_target_label)

    n_knn = int(merged["knn_r2"].notna().sum())
    n_kriging = int(merged["kriging_r2"].notna().sum())
    n = len(merged)
    print(f"  Matched: KNN={n_knn}/{n}, Kriging={n_kriging}/{n} targets")

    # ── Strip chart: one row per target, four markers per row ────────────────
    MODEL_COLS = ["knn_r2", "rf_r2", "xgb_r2", "kriging_r2"]
    MODEL_LABELS = ["KNN", "Random Forest", "XGBoost", "Kriging (RK)"]
    MODEL_COLORS = ["#4dac26", "#2166ac", "#d6604d", "#8856a7"]
    MODEL_MARKS = ["^", "o", "s", "D"]
    MODEL_SIZES = [36, 40, 36, 36]

    y = np.arange(n)
    fig, ax = plt.subplots(figsize=(7.5, max(5, n * 0.33)))

    # Light row bands for readability
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f7f7f7", zorder=0)

    # Connecting line across all present model values per row
    for i, row in merged.iterrows():
        vals = [row[c] for c in MODEL_COLS if pd.notna(row[c]) and row[c] >= -0.1]
        if len(vals) >= 2:
            ax.plot([min(vals), max(vals)], [i, i], color="#cccccc", lw=0.8, zorder=1)

    # Plot each model's dots
    for col, label, color, marker, ms in zip(
        MODEL_COLS, MODEL_LABELS, MODEL_COLORS, MODEL_MARKS, MODEL_SIZES
    ):
        valid = merged[col].notna()
        # Flag extreme negatives (kriging) — plot at floor with open marker
        extreme = valid & (merged[col] < -0.1)
        normal = valid & (merged[col] >= -0.1)

        if normal.any():
            ax.scatter(
                merged.loc[normal, col],
                y[normal],
                label=label,
                color=color,
                marker=marker,
                s=ms,
                zorder=4,
                linewidths=0.4,
                edgecolors="white",
            )
        if extreme.any():
            # Plot at -0.1 with open marker + annotation
            ax.scatter(
                [-0.1] * extreme.sum(),
                y[extreme],
                color=color,
                marker=marker,
                s=ms,
                zorder=4,
                linewidths=1.0,
                edgecolors=color,
                facecolors="none",
                label="_nolegend_",
            )
            for yi, val in zip(y[extreme], merged.loc[extreme, col]):
                ax.annotate(
                    f"{val:.2f}",
                    xy=(-0.1, yi),
                    xytext=(-0.14, yi),
                    ha="right",
                    fontsize=6,
                    color=color,
                    va="center",
                )

    # Mappability threshold
    ax.axvline(
        MAPPABILITY_THRESHOLD,
        color="black",
        lw=0.9,
        ls="--",
        alpha=0.6,
        label=f"Mappability threshold (R² = {MAPPABILITY_THRESHOLD})",
    )
    ax.axvline(0, color="#999", lw=0.5, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(merged["label"], fontsize=8)
    ax.set_xlabel("Mean LOLO CV R²", fontsize=9)
    ax.set_xlim(-0.18, merged[["rf_r2", "xgb_r2", "knn_r2"]].max().max() * 1.06)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(
        loc="lower right", fontsize=8.5, frameon=False, bbox_to_anchor=(1.0, 0.01)
    )
    ax.set_title(
        f"Four-model comparison: KNN, RF, XGBoost, Kriging\n"
        f"(KNN: {n_knn}/{n} matched  |  Kriging: {n_kriging}/{n} targets)",
        fontsize=10,
        pad=10,
    )

    out = os.path.join(OUT_DIR, "fig5_four_model_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading metrics...")
    rf_metrics = load_metrics("rf")
    xgb_metrics = load_metrics("xgb")

    print("Loading sample predictions (needed for Fig 4 only)...")
    rf_samples = load_sample_predictions("rf")
    xgb_samples = load_sample_predictions("xgb")

    fig1_r2_dotplot(rf_metrics, xgb_metrics)
    fig2_model_boxplots(rf_metrics, xgb_metrics)
    fig3_seed_heatmap(rf_metrics, xgb_metrics)
    fig4_acbr_error_table(rf_samples, xgb_samples, rf_metrics, xgb_metrics)
    fig5_four_model_comparison(rf_metrics, xgb_metrics)

    print(f"\nDone. Figures saved to: {OUT_DIR}/")
