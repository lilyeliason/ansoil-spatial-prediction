"""
make_taylor_diagrams_v7.py
ANSOIL — Taylor Diagram, two panels split by tier

Panel 1: Strong + Moderate targets  (~15–20 points)
Panel 2: Weak + Unusable targets    (~30–40 points, but lower R so spread out)

Each panel:
  - r_max = 95th percentile of that panel's sigma_n + 0.15
  - Numbers ARE the markers (filled circle = RF, open square = XGB)
  - Numbered lookup table sits directly below its panel
  - No floating text on the polar diagram

Run:  python make_taylor_diagrams_v7.py
Out:  taylor_v7.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ── 0. Paths ──────────────────────────────────────────────────────────────────
TARGETS_FILE = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/data/ansoil_targets.csv"
LOG_TABLE_FILE = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/data/ansoil_log_targets.csv"
RF_METRICS_FILE = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/results/aggregated/metrics_summary_rf.csv"
XGB_METRICS_FILE = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/results/aggregated/metrics_summary_xgb.csv"

OUT = "taylor_v7.png"

# ── 1. Load ───────────────────────────────────────────────────────────────────
targets = pd.read_csv(TARGETS_FILE)
log_tbl = pd.read_csv(LOG_TABLE_FILE)
rf = pd.read_csv(RF_METRICS_FILE)
xgb = pd.read_csv(XGB_METRICS_FILE)

dual_test = set(log_tbl[log_tbl["dual_test"] == True]["log_col"])
obs_std = {
    col: targets[col].dropna().std()
    for col in targets.columns
    if col != "sample_id" and targets[col].dropna().std() > 0
}


# ── 2. Taylor statistics ──────────────────────────────────────────────────────
def taylor_stats(r2, rmse, sigma_obs):
    if sigma_obs <= 0 or np.isnan(rmse):
        return np.nan, np.nan, np.nan
    R = np.sqrt(max(r2, 0)) if r2 >= 0 else -np.sqrt(-r2)
    b = -2 * R * sigma_obs
    c = sigma_obs**2 - rmse**2
    disc = b**2 - 4 * c
    if disc >= 0:
        s1 = (-b + np.sqrt(disc)) / 2
        s2 = (-b - np.sqrt(disc)) / 2
        sigma_pred = s1 if s1 >= 0 else s2
    else:
        sigma_pred = R * sigma_obs
    return R, sigma_pred / sigma_obs, rmse / sigma_obs


rows = []
for model, df in [("RF", rf), ("XGB", xgb)]:
    for _, row in df.iterrows():
        t = row["target"]
        if t not in obs_std:
            continue
        rmse = (
            row["cv_rmse_log_space_mean"]
            if t in dual_test
            else row["cv_rmse_orig_units_mean"]
        )
        R, sn, cRn = taylor_stats(row["cv_r2_mean"], rmse, obs_std[t])
        rows.append(
            dict(
                target=t,
                model=model,
                R=R,
                sigma_n=sn,
                cRMSE_n=cRn,
                r2=row["cv_r2_mean"],
                tier=row["tier"],
            )
        )

df_all = pd.DataFrame(rows).dropna(subset=["R", "sigma_n", "cRMSE_n"])
df_all = df_all[df_all["sigma_n"].between(0, 2.5)]
df_all["label"] = (
    df_all["target"]
    .map(
        {
            "d15n_air_permil": "δ¹⁵N",
            "d13c_vpdb_permil": "δ¹³C",
            "wt_percent_n": "wt% N",
            "wt_percent_c": "wt% C",
            "c_n_ratio": "C:N",
            "log_cec_meq_100g": "CEC",
            "ph_mq": "pH (MQ)",
            "ph_kcl": "pH (KCl)",
            "ph_cacl2": "pH (CaCl₂)",
            "digest_mg_kg_ni2316": "Ni",
            "digest_mg_kg_mg2852": "Mg",
            "digest_mg_kg_ti3349": "Ti",
            "digest_mg_kg_na5895": "Na",
            "digest_mg_kg_k_7664": "K",
            "digest_mg_kg_cr2835": "Cr",
            "digest_mg_kg_al3082": "Al",
            "digest_mg_kg_fe2599": "Fe",
            "digest_mg_kg_co2286": "Co",
            "digest_mg_kg_mn2576": "Mn",
            "digest_mg_kg_ba4554": "Ba",
            "digest_mg_kg_si2516": "Si",
            "digest_mg_kg_v_2924": "V",
            "digest_mg_kg_be3130": "Be",
            "digest_mg_kg_zn2138": "Zn",
            "digest_mg_kg_cu3247": "Cu",
            "digest_mg_kg_as1890": "As",
            "digest_mg_kg_b_2496": "B",
            "log_digest_mg_kg_p_1774": "P",
            "log_digest_mg_kg_sr4077": "Sr",
            "log_digest_mg_kg_mo2020": "Mo",
            "log_digest_mg_kg_ca3158": "Ca",
            "log_digest_mg_kg_na5895": "Na*",
            "log_digest_mg_kg_li6707": "Li",
            "log_digest_mg_kg_pb2203": "Pb",
            "log_digest_mg_kg_sb2068": "Sb",
            "clr_total_mg_l_po4": "PO₄ tot",
            "clr_hr_1_mg_l_po4": "PO₄ 1h",
            "clr_hr_24_mg_l_po4": "PO₄ 24h",
            "clr_total_mg_l_k": "K tot",
            "clr_total_mg_l_na": "Na CLR",
            "clr_total_mg_l_cl": "Cl tot",
            "clr_total_mg_l_no3": "NO₃ tot",
            "clr_total_mg_l_so4": "SO₄ tot",
            "clr_total_mg_l_ca2": "Ca tot",
            "clr_total_mg_l_sr2": "Sr tot",
            "clr_hr_1_mg_l_cl": "Cl 1h",
            "clr_hr_1_mg_l_no3": "NO₃ 1h",
            "clr_hr_1_mg_l_so4": "SO₄ 1h",
            "clr_hr_24_mg_l_cl": "Cl 24h",
            "clr_hr_24_mg_l_no3": "NO₃ 24h",
            "clr_hr_24_mg_l_so4": "SO₄ 24h",
            "log_total_mg_l_na": "Na tot",
            "log_total_mg_l_mg2": "Mg tot",
            "log_total_mg_l_ca2": "Ca tot*",
            "log_total_mg_l_cl": "Cl tot*",
            "log_hr_24_mg_l_cl": "Cl 24h*",
            "log_hr_1_mg_l_cl": "Cl 1h*",
            "ec_us_cm": "EC",
            "hr_24_mg_l_so4": "SO₄ 24h*",
        }
    )
    .fillna(df_all["target"])
)

# ── 3. Style ──────────────────────────────────────────────────────────────────
TIER_COLORS = {
    "strong": "#1565c0",
    "moderate": "#2e7d32",
    "weak": "#e65100",
    "unusable": "#757575",
}
JITTER_RAD = 0.013
JITTER_SN = 0.017


def _jitter(th, sn, model):
    s = -1 if model == "RF" else +1
    return th + s * JITTER_RAD, sn + s * JITTER_SN


def compute_rmax(df_sub, pct=0.95, pad=0.18, minimum=0.75):
    if df_sub.empty:
        return minimum
    return max(float(df_sub["sigma_n"].quantile(pct)) + pad, minimum)


# ── 4. Draw one Taylor panel ──────────────────────────────────────────────────
def draw_taylor(ax, df_pts, title, r_max, rmse_contours=None):
    """
    Numbers-as-markers Taylor diagram.
    RF  = filled circle, white number
    XGB = open square, coloured number
    Returns list of (idx, label, tier) sorted by descending mean R.
    """
    if rmse_contours is None:
        rmse_contours = [0.5, 1.0]

    ax.set_facecolor("#ffffff")
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ths = np.linspace(0, np.pi / 2, 300)

    # Sigma grid arcs
    for rv in np.arange(0.2, r_max + 0.01, 0.2):
        ax.plot(ths, np.full_like(ths, rv), "-", color="#f0f0f0", lw=0.5, zorder=0)
        ax.text(
            np.pi / 2 + 0.03,
            rv,
            f"{rv:.1f}",
            ha="left",
            va="center",
            fontsize=5,
            color="#bbbbbb",
        )

    # Hard boundary arc
    ax.plot(ths, np.full_like(ths, r_max), "-", color="#cccccc", lw=0.7, zorder=1)

    # cRMSE arcs
    for rc in rmse_contours:
        ct = np.cos(ths)
        disc = ct**2 - (1 - rc**2)
        r_arc = np.where(disc >= 0, ct + np.sqrt(np.where(disc >= 0, disc, 0)), np.nan)
        r_arc = np.where((r_arc >= 0) & (r_arc <= r_max), r_arc, np.nan)
        ax.plot(ths, r_arc, ":", color="#cccccc", lw=0.85, zorder=1)
        th_l = np.radians(68)
        d_l = np.cos(th_l) ** 2 - (1 - rc**2)
        if d_l >= 0:
            r_l = min(np.cos(th_l) + np.sqrt(d_l), r_max - 0.05)
            if r_l > 0.1:
                ax.text(
                    th_l,
                    r_l,
                    f"{rc}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="#aaaaaa",
                    style="italic",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.4),
                    zorder=2,
                )

    # Reference std = 1
    ax.plot(ths, np.ones_like(ths), "--", color="#555555", lw=1.0, zorder=2)

    # Correlation radials
    for r_val in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
        th = np.arccos(r_val)
        ax.plot([th, th], [0, r_max], "-", color="#f0f0f0", lw=0.5, zorder=0)
        ax.text(
            th,
            r_max + 0.06,
            f"{r_val}",
            ha="center",
            va="bottom",
            fontsize=5.5,
            color="#888888",
        )

    # Axis labels
    ax.text(
        np.radians(45),
        r_max + 0.20,
        "Correlation (R)",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#333333",
    )
    ax.text(
        np.radians(83),
        r_max * 0.38,
        "Norm.\nStd. Dev.",
        ha="center",
        va="center",
        fontsize=6,
        fontweight="bold",
        color="#333333",
    )

    # OBS
    ax.scatter([0], [1.0], marker="*", s=160, color="#111111", zorder=15, clip_on=False)
    ax.text(
        np.radians(1.5),
        1.07,
        "OBS",
        fontsize=6.5,
        fontweight="bold",
        color="#111111",
        ha="left",
        va="bottom",
    )

    # Data — sort targets by mean R descending so index 1 = best performer
    order = df_pts.groupby("target")["R"].mean().sort_values(ascending=False).index

    key_items = []
    for idx, tgt in enumerate(order, start=1):
        sub = df_pts[df_pts["target"] == tgt]
        if sub.empty:
            continue
        tier = sub.iloc[0]["tier"]
        lbl = sub.iloc[0]["label"]
        color = TIER_COLORS.get(tier, "#bbbbbb")
        fsize = 5.2 if idx >= 10 else 6.2
        msz = 160 if idx >= 10 else 130

        for _, row in sub.iterrows():
            if np.isnan(row["R"]) or np.isnan(row["sigma_n"]):
                continue
            th_raw = np.arccos(np.clip(row["R"], -1, 1))
            sn_raw = row["sigma_n"]
            th_j, sn_j = _jitter(th_raw, sn_raw, row["model"])

            clipped = sn_j > r_max
            sn_j = min(sn_j, r_max - 0.05)
            label = "×" if clipped else str(idx)

            if row["model"] == "RF":
                ax.scatter(
                    th_j,
                    sn_j,
                    marker="o",
                    s=msz,
                    color=color,
                    alpha=0.92,
                    zorder=8,
                    edgecolors="white",
                    linewidths=0.3,
                )
                ax.text(
                    th_j,
                    sn_j,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fsize,
                    fontweight="bold",
                    color="white",
                    zorder=9,
                )
            else:
                ax.scatter(
                    th_j,
                    sn_j,
                    marker="s",
                    s=msz + 20,
                    facecolor="white",
                    alpha=1.0,
                    zorder=8,
                    edgecolors=color,
                    linewidths=1.3,
                )
                ax.text(
                    th_j,
                    sn_j,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fsize,
                    fontweight="bold",
                    color=color,
                    zorder=9,
                )

        key_items.append((idx, lbl, tier))

    ax.set_rlim(0, r_max)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_title(
        title, fontsize=9, pad=22, color="#111111", fontweight="bold", loc="center"
    )

    return key_items


# ── 5. Lookup table ───────────────────────────────────────────────────────────
def draw_lookup_table(
    fig, ax_pos, key_items, fig_w_in, fig_h_in, n_cols=3, fontsize=7.5
):
    """
    Draw lookup table below the given axis.
    ax_pos: Bbox in figure fraction (from ax.get_position()).
    Spacing is computed in inches then converted to figure fraction,
    so it's resolution-independent.
    """
    if not key_items:
        return

    line_h_in = fontsize * 1.55 / 72  # line height in inches
    line_h_frac = line_h_in / fig_h_in  # converted to figure fraction
    swatch_w_frac = 0.25 / fig_w_in  # 0.25" swatch column
    idx_w_frac = 0.18 / fig_w_in  # 0.18" for "N." text
    gap_frac = 0.05 / fig_w_in  # gap between columns

    panel_w = ax_pos.width
    n = len(key_items)
    n_rows = -(-n // n_cols)
    col_w = panel_w / n_cols

    # Start just below the axis
    top_y = ax_pos.y0 - (0.18 / fig_h_in)

    # Header
    fig.text(
        ax_pos.x0,
        top_y + line_h_frac * 0.3,
        "Target index",
        fontsize=fontsize - 0.5,
        fontweight="bold",
        color="#555555",
        va="bottom",
        ha="left",
        transform=fig.transFigure,
    )

    for i, (idx, lbl, tier) in enumerate(key_items):
        col = i // n_rows
        row = i % n_rows
        x = ax_pos.x0 + col * col_w
        y = top_y - row * line_h_frac
        c = TIER_COLORS.get(tier, "#bbbbbb")

        # Swatch
        fig.add_artist(
            mpatches.FancyBboxPatch(
                (x, y - line_h_frac * 0.28),
                swatch_w_frac,
                line_h_frac * 0.55,
                boxstyle="square,pad=0",
                transform=fig.transFigure,
                facecolor=c,
                edgecolor="none",
                zorder=5,
            )
        )
        # Index number
        fig.text(
            x + swatch_w_frac + gap_frac,
            y,
            f"{idx}.",
            fontsize=fontsize,
            fontweight="bold",
            color=c,
            va="center",
            ha="left",
            transform=fig.transFigure,
        )
        # Label
        fig.text(
            x + swatch_w_frac + idx_w_frac + gap_frac * 2,
            y,
            lbl,
            fontsize=fontsize,
            color="#222222",
            va="center",
            ha="left",
            transform=fig.transFigure,
        )


# ── 6. Assemble figure ────────────────────────────────────────────────────────
PANEL_SPLITS = [
    ("Strong & Moderate", ["strong", "moderate"]),
    ("Weak & Unusable", ["weak", "unusable"]),
]

# Work out r_max per panel before drawing so we can size the figure correctly
panel_data = []
for title_suffix, tiers in PANEL_SPLITS:
    df_sub = df_all[df_all["tier"].isin(tiers)].copy()
    rmax = compute_rmax(df_sub)
    n_tgts = df_sub["target"].nunique()
    panel_data.append((title_suffix, tiers, df_sub, rmax, n_tgts))
    print(f"{title_suffix}: {n_tgts} targets, r_max={rmax:.3f}")

# Figure dimensions
# Each panel column is 6.5" wide; total width = 13"
# Height: polar takes 55%, table takes ~30%, shared legend 6%, margins 9%
FIG_W = 13.0
FIG_H = 11.0

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

# Panel geometry (figure fraction)
polar_left = [0.04, 0.52]  # left edge of each polar axes
polar_bottom = 0.32
polar_w = 0.43
polar_h = 0.58

axes = []
all_keys = []

for i, (title_suffix, tiers, df_sub, rmax, n_tgts) in enumerate(panel_data):
    ax = fig.add_axes(
        [polar_left[i], polar_bottom, polar_w, polar_h],
        projection="polar",
    )
    panel_title = f"({'a' if i == 0 else 'b'})  {title_suffix}"
    key_items = draw_taylor(
        ax, df_sub, title=panel_title, r_max=rmax, rmse_contours=[0.5, 1.0]
    )
    axes.append(ax)
    all_keys.append(key_items)
    print(f"  Panel {i + 1} drew {len(key_items)} targets.")

# Force a draw so get_position() returns real values
fig.canvas.draw()

# Draw lookup tables
for i, (ax, key_items) in enumerate(zip(axes, all_keys)):
    ax_pos = ax.get_position()
    n_tgts = len(key_items)
    # 3 cols for ≤18 targets, 4 cols for more
    nc = 3 if n_tgts <= 18 else 4
    draw_lookup_table(
        fig, ax_pos, key_items, fig_w_in=FIG_W, fig_h_in=FIG_H, n_cols=nc, fontsize=7.5
    )

# Shared legend
leg_model = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#555",
        markersize=9,
        label="RF — filled circle",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor="white",
        markeredgecolor="#555",
        markeredgewidth=1.3,
        markersize=9,
        label="XGBoost — open square",
    ),
]
leg_tier = [
    mpatches.Patch(color=TIER_COLORS["strong"], label="Strong   (R² ≥ 0.60)"),
    mpatches.Patch(color=TIER_COLORS["moderate"], label="Moderate (R² 0.30–0.60)"),
    mpatches.Patch(color=TIER_COLORS["weak"], label="Weak     (R² 0–0.30)"),
    mpatches.Patch(color=TIER_COLORS["unusable"], label="Unusable (R² < 0)"),
]
fig.legend(
    handles=leg_model + leg_tier,
    loc="lower center",
    ncol=6,
    fontsize=8.5,
    frameon=True,
    framealpha=0.95,
    edgecolor="#dddddd",
    bbox_to_anchor=(0.5, 0.005),
)

fig.suptitle(
    "Taylor Diagram — RF vs. XGBoost  |  ANSOIL Antarctic Soil Geochemistry",
    fontsize=12,
    fontweight="bold",
    y=0.995,
    color="#111111",
)

fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.25)
print(f"\nSaved: {OUT}")
plt.close(fig)
