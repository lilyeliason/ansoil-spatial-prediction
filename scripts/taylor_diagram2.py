"""
Taylor Diagram — ANSOIL Geochemical ML Project
================================================
Figure 1: Primary — moderate + strong targets only, single panel.
Figure 2: Supplementary — 1×3 grid by domain family, all tiers.

Run from any directory; data paths are absolute.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titleweight": "semibold",
        "figure.dpi": 150,
    }
)

# ── Absolute data paths ───────────────────────────────────────────────────
RF_METRICS = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/results/aggregated/metrics_summary_rf.csv"
XGB_METRICS = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/results/aggregated/metrics_summary_xgb.csv"
TARGETS_CSV = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/data/ansoil_targets.csv"
LOG_CSV = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/data/ansoil_log_targets.csv"

# ── Visual constants ──────────────────────────────────────────────────────
TIER_COLORS = {
    "strong": "#1A5276",
    "moderate": "#1E8449",
    "weak": "#BA4A00",
    "unusable": "#95A5A6",
}
TIER_LABELS = {
    "strong": r"Strong ($R^2 \geq 0.60$)",
    "moderate": r"Moderate ($0.30 \leq R^2 < 0.60$)",
    "weak": r"Weak ($0.00 \leq R^2 < 0.30$)",
    "unusable": r"Unusable ($R^2 < 0$)",
}
MODEL_MARKERS = {"RF": "o", "XGBoost": "D"}
MODEL_LABELS = {"RF": "Random Forest", "XGBoost": "XGBoost"}

CORR_TICKS = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
CRMSD_LEVELS = [0.25, 0.50, 0.75, 1.00]
MAX_STD = 1.05  # all real data falls below 0.85

FAMILY_ORDER = ["Soil Properties", "Digest Metals", "Water Chemistry"]

# Single font size for ALL numeric labels on the diagram (arc + x-axis)
TICK_FS = 8.0

# ── Name mappings ─────────────────────────────────────────────────────────
LABELS = {
    "c_n_ratio": "C:N ratio",
    "clr_hr_1_mg_l_cl": "Cl⁻ HR-1h (CLR)",
    "clr_hr_1_mg_l_f": "F⁻ HR-1h (CLR)",
    "clr_hr_1_mg_l_no3": "NO₃⁻ HR-1h (CLR)",
    "clr_hr_1_mg_l_po4": "PO₄³⁻ HR-1h (CLR)",
    "clr_hr_1_mg_l_so4": "SO₄²⁻ HR-1h (CLR)",
    "clr_hr_24_mg_l_cl": "Cl⁻ HR-24h (CLR)",
    "clr_hr_24_mg_l_f": "F⁻ HR-24h (CLR)",
    "clr_hr_24_mg_l_no3": "NO₃⁻ HR-24h (CLR)",
    "clr_hr_24_mg_l_po4": "PO₄³⁻ HR-24h (CLR)",
    "clr_hr_24_mg_l_so4": "SO₄²⁻ HR-24h (CLR)",
    "clr_total_mg_l_ca2": "Ca²⁺ total (CLR)",
    "clr_total_mg_l_cl": "Cl⁻ total (CLR)",
    "clr_total_mg_l_f": "F⁻ total (CLR)",
    "clr_total_mg_l_k": "K⁺ total (CLR)",
    "clr_total_mg_l_mg2": "Mg²⁺ total (CLR)",
    "clr_total_mg_l_na": "Na⁺ total (CLR)",
    "clr_total_mg_l_no3": "NO₃⁻ total (CLR)",
    "clr_total_mg_l_po4": "PO₄³⁻ total (CLR)",
    "clr_total_mg_l_so4": "SO₄²⁻ total (CLR)",
    "clr_total_mg_l_sr2": "Sr²⁺ total (CLR)",
    "d13c_vpdb_permil": "δ¹³C (VPDB, ‰)",
    "d15n_air_permil": "δ¹⁵N (air, ‰)",
    "digest_mg_kg_al3082": "Al (digest)",
    "digest_mg_kg_as1890": "As (digest)",
    "digest_mg_kg_b_2496": "B (digest)",
    "digest_mg_kg_ba4554": "Ba (digest)",
    "digest_mg_kg_be3130": "Be (digest)",
    "digest_mg_kg_co2286": "Co (digest)",
    "digest_mg_kg_cr2835": "Cr (digest)",
    "digest_mg_kg_cu3247": "Cu (digest)",
    "digest_mg_kg_fe2599": "Fe (digest)",
    "digest_mg_kg_k_7664": "K (digest)",
    "digest_mg_kg_mg2852": "Mg (digest)",
    "digest_mg_kg_mn2576": "Mn (digest)",
    "digest_mg_kg_na5895": "Na (digest)",
    "digest_mg_kg_ni2316": "Ni (digest)",
    "digest_mg_kg_si2516": "Si (digest)",
    "digest_mg_kg_ti3349": "Ti (digest)",
    "digest_mg_kg_v_2924": "V (digest)",
    "digest_mg_kg_zn2138": "Zn (digest)",
    "ec_us_cm": "EC (µS/cm)",
    "hr_24_mg_l_so4": "SO₄²⁻ HR-24h",
    "log_cec_meq_100g": "CEC (log)",
    "log_digest_mg_kg_ca3158": "Ca (digest, log)",
    "log_digest_mg_kg_hg1849": "Hg (digest, log)",
    "log_digest_mg_kg_li6707": "Li (digest, log)",
    "log_digest_mg_kg_mo2020": "Mo (digest, log)",
    "log_digest_mg_kg_na5895": "Na (digest, log)",
    "log_digest_mg_kg_p_1774": "P (digest, log)",
    "log_digest_mg_kg_pb2203": "Pb (digest, log)",
    "log_digest_mg_kg_sb2068": "Sb (digest, log)",
    "log_digest_mg_kg_sn1899": "Sn (digest, log)",
    "log_digest_mg_kg_sr4077": "Sr (digest, log)",
    "log_digest_mg_kg_tl1908": "Tl (digest, log)",
    "log_hr_1_mg_l_cl": "Cl⁻ HR-1h (log)",
    "log_hr_1_mg_l_so4": "SO₄²⁻ HR-1h (log)",
    "log_hr_24_mg_l_cl": "Cl⁻ HR-24h (log)",
    "log_hr_24_mg_l_so4": "SO₄²⁻ HR-24h (log)",
    "log_total_mg_l_ca2": "Ca²⁺ total (log)",
    "log_total_mg_l_cl": "Cl⁻ total (log)",
    "log_total_mg_l_mg2": "Mg²⁺ total (log)",
    "log_total_mg_l_na": "Na⁺ total (log)",
    "log_total_mg_l_so4": "SO₄²⁻ total (log)",
    "ph_cacl2": "pH (CaCl₂)",
    "ph_kcl": "pH (KCl)",
    "ph_mq": "pH (MQ)",
    "wt_percent_c": "wt% C",
    "wt_percent_n": "wt% N",
}
SHORT = {
    "c_n_ratio": "C:N",
    "clr_hr_1_mg_l_cl": "Cl·1h",
    "clr_hr_1_mg_l_f": "F·1h",
    "clr_hr_1_mg_l_no3": "NO₃·1h",
    "clr_hr_1_mg_l_po4": "PO₄·1h",
    "clr_hr_1_mg_l_so4": "SO₄·1h",
    "clr_hr_24_mg_l_cl": "Cl·24h",
    "clr_hr_24_mg_l_f": "F·24h",
    "clr_hr_24_mg_l_no3": "NO₃·24h",
    "clr_hr_24_mg_l_po4": "PO₄·24h",
    "clr_hr_24_mg_l_so4": "SO₄·24h",
    "clr_total_mg_l_ca2": "Ca(t)",
    "clr_total_mg_l_cl": "Cl(t)",
    "clr_total_mg_l_f": "F(t)",
    "clr_total_mg_l_k": "K(t)",
    "clr_total_mg_l_mg2": "Mg(t)",
    "clr_total_mg_l_na": "Na(t)",
    "clr_total_mg_l_no3": "NO₃(t)",
    "clr_total_mg_l_po4": "PO₄(t)",
    "clr_total_mg_l_so4": "SO₄(t)",
    "clr_total_mg_l_sr2": "Sr(t)",
    "d13c_vpdb_permil": "δ¹³C",
    "d15n_air_permil": "δ¹⁵N",
    "digest_mg_kg_al3082": "Al",
    "digest_mg_kg_as1890": "As",
    "digest_mg_kg_b_2496": "B",
    "digest_mg_kg_ba4554": "Ba",
    "digest_mg_kg_be3130": "Be",
    "digest_mg_kg_co2286": "Co",
    "digest_mg_kg_cr2835": "Cr",
    "digest_mg_kg_cu3247": "Cu",
    "digest_mg_kg_fe2599": "Fe",
    "digest_mg_kg_k_7664": "K",
    "digest_mg_kg_mg2852": "Mg",
    "digest_mg_kg_mn2576": "Mn",
    "digest_mg_kg_na5895": "Na",
    "digest_mg_kg_ni2316": "Ni",
    "digest_mg_kg_si2516": "Si",
    "digest_mg_kg_ti3349": "Ti",
    "digest_mg_kg_v_2924": "V",
    "digest_mg_kg_zn2138": "Zn",
    "ec_us_cm": "EC",
    "hr_24_mg_l_so4": "SO₄·24h",
    "log_cec_meq_100g": "CEC†",
    "log_digest_mg_kg_ca3158": "Ca†",
    "log_digest_mg_kg_hg1849": "Hg†",
    "log_digest_mg_kg_li6707": "Li†",
    "log_digest_mg_kg_mo2020": "Mo†",
    "log_digest_mg_kg_na5895": "Na†",
    "log_digest_mg_kg_p_1774": "P†",
    "log_digest_mg_kg_pb2203": "Pb†",
    "log_digest_mg_kg_sb2068": "Sb†",
    "log_digest_mg_kg_sn1899": "Sn†",
    "log_digest_mg_kg_sr4077": "Sr†",
    "log_digest_mg_kg_tl1908": "Tl†",
    "log_hr_1_mg_l_cl": "Cl·1h†",
    "log_hr_1_mg_l_so4": "SO₄·1h†",
    "log_hr_24_mg_l_cl": "Cl·24h†",
    "log_hr_24_mg_l_so4": "SO₄·24h†",
    "log_total_mg_l_ca2": "Ca(t)†",
    "log_total_mg_l_cl": "Cl(t)†",
    "log_total_mg_l_mg2": "Mg(t)†",
    "log_total_mg_l_na": "Na(t)†",
    "log_total_mg_l_so4": "SO₄(t)†",
    "ph_cacl2": "pH-Ca",
    "ph_kcl": "pH-K",
    "ph_mq": "pH-MQ",
    "wt_percent_c": "wt%C",
    "wt_percent_n": "wt%N",
}


# ── Data helpers ──────────────────────────────────────────────────────────


def assign_tier(r2):
    if r2 >= 0.60:
        return "strong"
    if r2 >= 0.30:
        return "moderate"
    if r2 >= 0.00:
        return "weak"
    return "unusable"


def compute_sigma_n(cv_rmse, sigma_obs, r):
    disc = np.maximum(r**2 - 1.0 + (cv_rmse / sigma_obs) ** 2, 0.0)
    return r + np.sqrt(disc)


def family(t):
    if t.startswith("ph_") or t in (
        "d15n_air_permil",
        "d13c_vpdb_permil",
        "wt_percent_n",
        "wt_percent_c",
        "c_n_ratio",
        "log_cec_meq_100g",
    ):
        return "Soil Properties"
    if t.startswith("digest_") or t.startswith("log_digest_"):
        return "Digest Metals"
    return "Water Chemistry"


def load_data():
    rf = pd.read_csv(RF_METRICS)
    xgb = pd.read_csv(XGB_METRICS)
    tgt = pd.read_csv(TARGETS_CSV)
    log = pd.read_csv(LOG_CSV)
    sigma_obs = tgt.drop(columns=["sample_id"]).std(ddof=1)
    all_log_cols = set(log["log_col"])
    rows = []
    for model_label, df in [("RF", rf), ("XGBoost", xgb)]:
        for _, row in df.iterrows():
            t = row["target"]
            r2 = row["cv_r2_mean"]
            # KEY FIX: for r2<0, r is negative (anti-correlated), not zero.
            # We don't have raw Pearson r, so approximate as -sqrt(|r2|) for r2<0.
            r = np.sqrt(abs(r2)) * (1.0 if r2 >= 0 else -1.0)
            if t in all_log_cols:
                cv_rmse = row["cv_rmse_log_space_mean"]
                s_obs = sigma_obs.get(t, np.nan)
            else:
                cv_rmse = row["cv_rmse_orig_units_mean"]
                s_obs = sigma_obs.get(t, np.nan)
            if pd.isna(s_obs) or s_obs <= 0:
                continue
            sigma_n = float(compute_sigma_n(cv_rmse, s_obs, r))
            rows.append(
                {
                    "target": t,
                    "short": SHORT.get(t, t),
                    "label": LABELS.get(t, t),
                    "model": model_label,
                    "r2": round(r2, 4),
                    "r": round(r, 4),
                    "sigma_n": round(sigma_n, 4),
                    "tier": assign_tier(r2),
                    "family": family(t),
                }
            )
    return pd.DataFrame(rows)


# ── Shared polar axes setup ───────────────────────────────────────────────


def _setup_taylor_axes(ax, max_std=MAX_STD):
    """
    Standard Taylor orientation:
      theta=0 (east/right) → r=1.0
      theta=90° (north/top) → r=0.0
    All numeric labels (arc ticks + radial ring labels) use TICK_FS.
    """
    ax.set_theta_direction(1)
    ax.set_theta_zero_location("E")
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_rlim(0, max_std)
    ax.set_facecolor("white")

    # Correlation arc tick labels — TICK_FS
    theta_deg = np.degrees(np.arccos(CORR_TICKS))
    ax.set_thetagrids(
        theta_deg,
        labels=[str(c) for c in CORR_TICKS],
        fontsize=TICK_FS,
        color="#444444",
    )

    # σ_n ring labels along the x-axis (angle=0) — same TICK_FS
    std_vals = np.arange(0.25, max_std + 0.01, 0.25)
    ax.set_rgrids(
        std_vals,
        labels=[f"{v:.2f}" for v in std_vals],
        angle=0,
        fontsize=TICK_FS,
        color="#666666",
    )

    # Add 0 label at the origin end of the x-axis
    ax.text(
        0,
        0,
        "0",
        fontsize=TICK_FS,
        color="#666666",
        ha="center",
        va="top",
        transform=ax.transData,
        zorder=10,
    )

    ax.grid(True, color="#d0d0d0", linewidth=0.35, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["polar"].set_visible(False)

    # Outer boundary arc (scatter avoids matplotlib fill artefact)
    t_bnd = np.linspace(0, np.pi / 2, 600)
    ax.scatter(
        t_bnd, np.full_like(t_bnd, max_std), s=0.5, c="#888888", linewidths=0, zorder=3
    )
    ax.plot([0, 0], [0, max_std], color="#888", lw=0.9, zorder=3)
    ax.plot([np.pi / 2, np.pi / 2], [0, max_std], color="#888", lw=0.9, zorder=3)

    # σ_n = 1.0 reference arc
    t_ref = np.linspace(0, np.pi / 2, 400)
    ax.scatter(
        t_ref,
        np.ones_like(t_ref),
        s=0.7,
        c="#555555",
        linewidths=0,
        alpha=0.7,
        zorder=3,
    )
    return ax


def _draw_crmsd_contours(ax, max_std=MAX_STD, levels=None):
    if levels is None:
        levels = CRMSD_LEVELS
    phi = np.linspace(0, 2 * np.pi, 4000)
    for crmsd in levels:
        cx = 1.0 + crmsd * np.cos(phi)
        cy = 0.0 + crmsd * np.sin(phi)
        r_arc = np.sqrt(cx**2 + cy**2)
        t_arc = np.arctan2(cy, cx)
        mask = (t_arc >= 0) & (t_arc <= np.pi / 2) & (r_arc <= max_std)
        if mask.sum() < 5:
            continue
        idx = np.argsort(t_arc[mask])
        t_s = t_arc[mask][idx]
        r_s = r_arc[mask][idx]
        ax.scatter(t_s, r_s, s=0.3, c="#aaaaaa", linewidths=0, alpha=0.9, zorder=2)
        if r_s[0] < max_std * 0.97 and t_s[0] < np.radians(18):
            ax.text(
                t_s[0] + np.radians(1.5),
                r_s[0] + 0.02,
                f"{crmsd:.2f}",
                fontsize=TICK_FS - 1.5,
                color="#aaaaaa",
                ha="left",
                va="bottom",
                zorder=10,
            )


def _draw_reference_point(ax):
    ax.scatter([0.0], [1.0], marker="*", s=140, c="black", linewidths=0, zorder=8)


def _scatter_model(ax, df_model, model_name, marker_size=55, alpha=0.92):
    marker = MODEL_MARKERS[model_name]
    for _, row in df_model.iterrows():
        theta = float(np.arccos(np.clip(row["r"], -1.0, 1.0)))
        r_val = float(np.clip(row["sigma_n"], 0.0, MAX_STD))
        ax.scatter(
            theta,
            r_val,
            marker=marker,
            s=marker_size,
            c=TIER_COLORS[row["tier"]],
            alpha=alpha,
            edgecolors="white",
            linewidths=0.65,
            zorder=5,
        )


# ── Legend builders ───────────────────────────────────────────────────────


def _make_legend_handles(include_obs=True):
    handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS[m],
            color="w",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=7,
            label=MODEL_LABELS[m],
        )
        for m in ["RF", "XGBoost"]
    ]
    if include_obs:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="black",
                markersize=9,
                label="Observation (ref.)",
            )
        )
    tier_handles = [
        Patch(
            facecolor=TIER_COLORS[t],
            edgecolor="white",
            linewidth=0.5,
            label=TIER_LABELS[t],
        )
        for t in ["strong", "moderate", "weak", "unusable"]
    ]
    return handles, tier_handles


# ── Figure 1 ──────────────────────────────────────────────────────────────


def make_figure1(df, output_path="figure1_taylor.pdf"):
    df_plot = df[df["tier"].isin(["strong", "moderate"])].copy()
    n_strong = int(df_plot[df_plot["tier"] == "strong"]["target"].nunique())
    n_mod = int(df_plot[df_plot["tier"] == "moderate"]["target"].nunique())

    fig = plt.figure(figsize=(7.0, 8.2))
    # Axes box: generous bottom margin for x-axis label + tick labels + legends
    ax = fig.add_axes([0.13, 0.28, 0.76, 0.60], projection="polar")

    _setup_taylor_axes(ax)
    _draw_crmsd_contours(ax)
    _draw_reference_point(ax)

    for model in ["RF", "XGBoost"]:
        _scatter_model(
            ax, df_plot[df_plot["model"] == model], model, marker_size=55, alpha=0.92
        )

    # No point labels

    # ── Axis annotations ──────────────────────────────────────────────────
    # y-axis label (left side, rotated)
    fig.text(
        0.03,
        0.58,
        r"Normalized standard deviation  $\sigma_n$",
        ha="center",
        va="center",
        fontsize=9.0,
        color="#333333",
        rotation=90,
    )

    # x-axis label placed below the x-axis tick numbers
    # In figure coordinates this sits beneath the polar axes bottom edge
    fig.text(
        0.52,
        0.175,
        r"Normalized standard deviation  $\sigma_n$",
        ha="center",
        va="top",
        fontsize=9.0,
        color="#333333",
    )

    # Centered RMSD label inside the diagram
    fig.text(
        0.195,
        0.52,
        "Centered\nRMSD",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#aaaaaa",
        style="italic",
    )

    # ── Title block ───────────────────────────────────────────────────────
    fig.text(
        0.50,
        0.970,
        "Taylor Diagram \u2014 Moderate and Strong Targets",
        ha="center",
        va="top",
        fontsize=11.5,
        fontweight="semibold",
        color="#1a1a1a",
    )

    fig.text(
        0.50,
        0.948,
        f"RF and XGBoost predictions vs. Antarctic soil observations  "
        f"({n_strong} strong, {n_mod} moderate of 68 total)  "
        f"|  \u2020\u2009log-transformed",
        ha="center",
        va="top",
        fontsize=7.8,
        color="#555555",
    )

    # ── Legends ───────────────────────────────────────────────────────────
    model_handles, tier_handles = _make_legend_handles(include_obs=True)

    leg1 = fig.legend(
        handles=model_handles,
        title="Model",
        loc="lower left",
        bbox_to_anchor=(0.04, 0.01),
        fontsize=7.5,
        title_fontsize=8.0,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        ncol=1,
    )

    fig.legend(
        handles=tier_handles,
        title="Performance tier",
        loc="lower right",
        bbox_to_anchor=(0.97, 0.01),
        fontsize=7.5,
        title_fontsize=8.0,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        ncol=2,
    )

    fig.add_artist(leg1)

    fig.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"Figure 1 saved \u2192 {output_path}")
    return fig


# ── Figure 2 ──────────────────────────────────────────────────────────────


def make_figure2(df, output_path="figure2_taylor_supplementary.pdf"):
    fig = plt.figure(figsize=(16.0, 7.0))

    # top=0.78 leaves ~22% of figure above the panels for the 3 header rows.
    # The rows themselves are spaced tightly so they sit close to the panels.
    gs = fig.add_gridspec(
        1, 3, left=0.04, right=0.97, bottom=0.17, top=0.78, wspace=0.30
    )

    for col, fam in enumerate(FAMILY_ORDER):
        ax = fig.add_subplot(gs[0, col], projection="polar")
        df_fam = df[df["family"] == fam].copy()

        _setup_taylor_axes(ax)
        _draw_crmsd_contours(ax, levels=[0.25, 0.50, 0.75])
        _draw_reference_point(ax)

        for model in ["RF", "XGBoost"]:
            _scatter_model(
                ax, df_fam[df_fam["model"] == model], model, marker_size=40, alpha=0.90
            )

        # No point labels

        n_vars = int(df_fam["target"].nunique())
        ax.set_title(
            f"{fam}\n({n_vars} variable{'s' if n_vars != 1 else ''})",
            fontsize=10.0,
            pad=14,
            color="#222222",
            fontweight="semibold",
        )

    # Shared x-axis label
    fig.text(
        0.50,
        0.115,
        r"Normalized standard deviation  $\sigma_n$",
        ha="center",
        va="top",
        fontsize=9.0,
        color="#333333",
    )

    # ── Header rows — snug above the panel titles ─────────────────────────
    # gs top=0.78, so we place rows at 0.990 → 0.960 → 0.930,
    # all above 0.78, giving ~3 lines of breathing room between row 3 and panels.

    # Row 1: main title
    fig.text(
        0.50,
        0.990,
        "Taylor Diagram \u2014 All Targets by Domain Family  (Supplementary)",
        ha="center",
        va="top",
        fontsize=12.0,
        fontweight="semibold",
        color="#1a1a1a",
    )

    # Row 2: study info — tightly below row 1
    fig.text(
        0.50,
        0.958,
        r"RF and XGBoost predictions vs. Antarctic soil observations  |  "
        r"Leave-one-location-out CV  |  $n = 171$  |  "
        r"Normalized $\sigma_n$  |  † log-transformed",
        ha="center",
        va="top",
        fontsize=8.0,
        color="#555555",
    )

    # Row 3: axis key — tightly below row 2
    fig.text(
        0.50,
        0.926,
        r"Arc: correlation $r$   |   Radial distance: $\sigma_n$   |   "
        r"Dashed arcs: centered RMSD   |   Circle = RF,  Diamond = XGBoost",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#888888",
        style="italic",
    )

    # ── Shared legend — no "Observation (ref.)" ───────────────────────────
    model_handles, tier_handles = _make_legend_handles(include_obs=False)

    fig.legend(
        handles=model_handles + [Line2D([0], [0], color="none")] + tier_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.005),
        fontsize=7.5,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        ncol=3,
        columnspacing=1.6,
        handletextpad=0.5,
    )

    fig.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"Figure 2 saved \u2192 {output_path}")
    return fig


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    print("=== Tier distribution ===")
    print(
        df.groupby(["family", "tier"])["target"]
        .nunique()
        .unstack(fill_value=0)
        .to_string()
    )
    print()
    print("=== r range ===")
    print(df["r"].describe().round(4).to_string())
    print()
    print("=== Strong targets ===")
    print(
        df[df["tier"] == "strong"][
            ["target", "label", "model", "r2", "r", "sigma_n"]
        ].to_string(index=False)
    )

    fig1 = make_figure1(df)
    fig2 = make_figure2(df)
    plt.show()
