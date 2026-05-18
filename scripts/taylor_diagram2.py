"""
make_taylor_diagrams_v8.py
ANSOIL — Taylor Diagrams, 3 separate files, one per variable family

Scope: Strong + Moderate targets only, per family
Each figure: ~6–12 targets — readable density for a Taylor diagram
Numbers are the markers (filled circle = RF, open square = XGB)
Lookup table below the diagram

Outputs:
  taylor_soil.png
  taylor_metals.png
  taylor_water.png

Run:  python make_taylor_diagrams_v8.py
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

# Strong + moderate only
df_sm = df_all[df_all["tier"].isin(["strong", "moderate"])].copy()

# ── 3. Family definitions ─────────────────────────────────────────────────────
FAMILIES = {
    "soil": {
        "title": "Soil Properties",
        "subtitle": "pH, stable isotopes, C/N, CEC",
        "outfile": "taylor_soil.png",
        "targets": [
            "ph_mq",
            "ph_kcl",
            "ph_cacl2",
            "d15n_air_permil",
            "d13c_vpdb_permil",
            "wt_percent_n",
            "wt_percent_c",
            "c_n_ratio",
            "log_cec_meq_100g",
        ],
    },
    "metals": {
        "title": "Digest Metals",
        "subtitle": "mg kg⁻¹",
        "outfile": "taylor_metals.png",
        "targets": [
            "digest_mg_kg_ni2316",
            "digest_mg_kg_mg2852",
            "digest_mg_kg_ti3349",
            "digest_mg_kg_na5895",
            "digest_mg_kg_k_7664",
            "digest_mg_kg_cr2835",
            "digest_mg_kg_al3082",
            "digest_mg_kg_fe2599",
            "digest_mg_kg_co2286",
            "digest_mg_kg_mn2576",
            "digest_mg_kg_ba4554",
            "digest_mg_kg_si2516",
            "digest_mg_kg_v_2924",
            "digest_mg_kg_be3130",
            "digest_mg_kg_zn2138",
            "digest_mg_kg_cu3247",
            "digest_mg_kg_as1890",
            "digest_mg_kg_b_2496",
            "log_digest_mg_kg_p_1774",
            "log_digest_mg_kg_sr4077",
            "log_digest_mg_kg_mo2020",
            "log_digest_mg_kg_ca3158",
            "log_digest_mg_kg_na5895",
            "log_digest_mg_kg_li6707",
            "log_digest_mg_kg_pb2203",
            "log_digest_mg_kg_sb2068",
        ],
    },
    "water": {
        "title": "Water Chemistry",
        "subtitle": "Ions, CLR extracts, leachate",
        "outfile": "taylor_water.png",
        "targets": [
            "clr_total_mg_l_po4",
            "clr_hr_1_mg_l_po4",
            "clr_hr_24_mg_l_po4",
            "clr_total_mg_l_k",
            "clr_total_mg_l_na",
            "clr_total_mg_l_cl",
            "clr_total_mg_l_no3",
            "clr_total_mg_l_so4",
            "clr_total_mg_l_ca2",
            "clr_total_mg_l_sr2",
            "clr_hr_1_mg_l_cl",
            "clr_hr_1_mg_l_no3",
            "clr_hr_1_mg_l_so4",
            "clr_hr_24_mg_l_cl",
            "clr_hr_24_mg_l_no3",
            "clr_hr_24_mg_l_so4",
            "log_total_mg_l_na",
            "log_total_mg_l_mg2",
            "log_total_mg_l_ca2",
            "log_total_mg_l_cl",
            "log_hr_24_mg_l_cl",
            "log_hr_1_mg_l_cl",
            "ec_us_cm",
            "hr_24_mg_l_so4",
        ],
    },
}

# ── 4. Style ──────────────────────────────────────────────────────────────────
TIER_COLORS = {
    "strong": "#1565c0",
    "moderate": "#2e7d32",
}
# Small jitter so RF circle and XGB square don't perfectly overlap
JITTER_RAD = 0.012
JITTER_SN = 0.015


def _jitter(th, sn, model):
    s = -1 if model == "RF" else +1
    return th + s * JITTER_RAD, sn + s * JITTER_SN


# ── 5. Draw Taylor panel ──────────────────────────────────────────────────────
def draw_taylor(ax, df_pts, r_max, rmse_contours=(0.5, 1.0)):
    """
    Clean Taylor diagram. Numbers are the markers.
    RF  = filled circle, white number inside.
    XGB = open square, coloured number inside.
    Returns list of (idx, label, tier) for the lookup table.
    """
    ax.set_facecolor("white")
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ths = np.linspace(0, np.pi / 2, 300)

    # ── Grid: sigma arcs ──
    for rv in np.arange(0.2, r_max + 0.01, 0.2):
        ax.plot(ths, np.full_like(ths, rv), color="#f0f0f0", lw=0.6, zorder=0)
        ax.text(
            np.pi / 2 + 0.03,
            rv,
            f"{rv:.1f}",
            ha="left",
            va="center",
            fontsize=5.5,
            color="#bbbbbb",
        )

    # ── Hard outer boundary ──
    ax.plot(ths, np.full_like(ths, r_max), color="#dddddd", lw=0.8, zorder=1)

    # ── cRMSE arcs ──
    for rc in rmse_contours:
        ct = np.cos(ths)
        disc = ct**2 - (1 - rc**2)
        r_arc = np.where(disc >= 0, ct + np.sqrt(np.where(disc >= 0, disc, 0)), np.nan)
        r_arc = np.where((r_arc >= 0) & (r_arc <= r_max), r_arc, np.nan)
        ax.plot(ths, r_arc, linestyle=":", color="#cccccc", lw=1.0, zorder=1)
        # Label near the arc, at ~68°, inside the boundary
        th_l = np.radians(68)
        d_l = np.cos(th_l) ** 2 - (1 - rc**2)
        if d_l >= 0:
            r_l = min(np.cos(th_l) + np.sqrt(d_l), r_max - 0.06)
            if r_l > 0.15:
                ax.text(
                    th_l,
                    r_l,
                    f"cRMSE = {rc}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#aaaaaa",
                    style="italic",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
                    zorder=2,
                )

    # ── Reference std = 1 arc ──
    ax.plot(ths, np.ones_like(ths), linestyle="--", color="#666666", lw=1.1, zorder=2)

    # ── Correlation radials, clipped to r_max ──
    for r_val in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
        th = np.arccos(r_val)
        ax.plot([th, th], [0, r_max], color="#f0f0f0", lw=0.6, zorder=0)
        ax.text(
            th,
            r_max + 0.07,
            f"{r_val}",
            ha="center",
            va="bottom",
            fontsize=6,
            color="#888888",
        )

    # ── Axis labels ──
    ax.text(
        np.radians(45),
        r_max + 0.22,
        "Correlation (R)",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#333333",
    )
    ax.text(
        np.radians(83),
        r_max * 0.35,
        "Normalised\nStd. Dev.",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="#333333",
    )

    # ── OBS reference point ──
    ax.scatter([0], [1.0], marker="*", s=200, color="#111111", zorder=15, clip_on=False)
    ax.text(
        np.radians(1.5),
        1.08,
        "OBS",
        fontsize=7.5,
        fontweight="bold",
        color="#111111",
        ha="left",
        va="bottom",
    )

    # ── Data points ──
    order = df_pts.groupby("target")["R"].mean().sort_values(ascending=False).index

    key_items = []
    for idx, tgt in enumerate(order, start=1):
        sub = df_pts[df_pts["target"] == tgt]
        if sub.empty:
            continue
        tier = sub.iloc[0]["tier"]
        lbl = sub.iloc[0]["label"]
        color = TIER_COLORS.get(tier, "#999999")

        # Slightly smaller font/marker for two-digit indices
        fsize = 6.0 if idx >= 10 else 7.5
        msz = 170 if idx >= 10 else 140

        for _, row in sub.iterrows():
            if np.isnan(row["R"]) or np.isnan(row["sigma_n"]):
                continue
            th = np.arccos(np.clip(row["R"], -1, 1))
            sn = row["sigma_n"]
            th_j, sn_j = _jitter(th, sn, row["model"])
            # Clip to boundary if needed
            sn_j = min(sn_j, r_max - 0.04)

            if row["model"] == "RF":
                ax.scatter(
                    th_j,
                    sn_j,
                    marker="o",
                    s=msz,
                    color=color,
                    alpha=0.93,
                    zorder=8,
                    edgecolors="white",
                    linewidths=0.4,
                )
                ax.text(
                    th_j,
                    sn_j,
                    str(idx),
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
                    s=msz + 25,
                    facecolor="white",
                    alpha=1.0,
                    zorder=8,
                    edgecolors=color,
                    linewidths=1.5,
                )
                ax.text(
                    th_j,
                    sn_j,
                    str(idx),
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
    return key_items


# ── 6. Make one figure ────────────────────────────────────────────────────────
def make_figure(family_key):
    fam = FAMILIES[family_key]
    df_fam = df_sm[df_sm["target"].isin(fam["targets"])].copy()
    n_tgts = df_fam["target"].nunique()

    if n_tgts == 0:
        print(f"  {family_key}: no strong/moderate targets found — skipping.")
        return

    # r_max: 95th percentile of sigma_n + padding, minimum 0.8
    r_max = max(float(df_fam["sigma_n"].quantile(0.95)) + 0.18, 0.80)
    print(f"  {family_key}: {n_tgts} targets, r_max = {r_max:.3f}")

    # ── Figure layout ──────────────────────────────────────────────────────
    # Single 7×8" figure.
    # Polar panel: top 62% of height, centred in left 80% of width
    # Lookup table: below polar panel
    # Legend strip: very bottom
    FIG_W, FIG_H = 7.5, 8.5

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

    # Polar axes: [left, bottom, width, height] in figure fraction
    ax = fig.add_axes([0.06, 0.30, 0.72, 0.62], projection="polar")

    key_items = draw_taylor(ax, df_fam, r_max=r_max, rmse_contours=(0.5, 1.0))

    # ── Title ──────────────────────────────────────────────────────────────
    ax.set_title(
        f"{fam['title']}\n{fam['subtitle']}  |  Strong & Moderate targets  |  ANSOIL",
        fontsize=10,
        fontweight="bold",
        pad=24,
        color="#111111",
        loc="center",
    )

    # ── Lookup table ───────────────────────────────────────────────────────
    # Force draw so get_position() is accurate
    fig.canvas.draw()
    ax_pos = ax.get_position()  # Bbox in figure fraction

    n = len(key_items)
    n_cols = 2 if n <= 8 else 3
    n_rows = -(-n // n_cols)

    # Spacing in inches → figure fraction
    FONT = 8.0  # pt
    lh = FONT * 1.6 / 72 / FIG_H  # line height in figure fraction
    sw = 0.18 / FIG_W  # swatch width
    gap = 0.06 / FIG_W  # gap
    col_w = ax_pos.width / n_cols

    tbl_top = ax_pos.y0 - 0.18 / FIG_H  # start just below axis

    # "Target index" header
    fig.text(
        ax_pos.x0,
        tbl_top + lh * 0.5,
        "Target index",
        fontsize=FONT - 0.5,
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
        y = tbl_top - row * lh
        c = TIER_COLORS.get(tier, "#999999")

        # Colour swatch
        fig.add_artist(
            mpatches.FancyBboxPatch(
                (x, y - lh * 0.30),
                sw,
                lh * 0.60,
                boxstyle="square,pad=0",
                transform=fig.transFigure,
                facecolor=c,
                edgecolor="none",
                zorder=5,
            )
        )
        # Index
        fig.text(
            x + sw + gap,
            y,
            f"{idx}.",
            fontsize=FONT,
            fontweight="bold",
            color=c,
            va="center",
            ha="left",
            transform=fig.transFigure,
        )
        # Label
        fig.text(
            x + sw + gap + 0.16 / FIG_W,
            y,
            lbl,
            fontsize=FONT,
            color="#222222",
            va="center",
            ha="left",
            transform=fig.transFigure,
        )

    # ── Legend ─────────────────────────────────────────────────────────────
    leg_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#555",
            markersize=9,
            label="Random Forest",
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
            label="XGBoost",
        ),
        mpatches.Patch(color=TIER_COLORS["strong"], label="Strong (R² ≥ 0.60)"),
        mpatches.Patch(color=TIER_COLORS["moderate"], label="Moderate (R² 0.30–0.60)"),
    ]
    fig.legend(
        handles=leg_handles,
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=True,
        framealpha=0.95,
        edgecolor="#dddddd",
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.savefig(
        fam["outfile"], dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.2
    )
    print(f"  Saved: {fam['outfile']}")
    plt.close(fig)


# ── 7. Run ────────────────────────────────────────────────────────────────────
print("Generating Taylor diagrams (strong + moderate, per family)...\n")
for key in ("soil", "metals", "water"):
    print(f"Family: {key}")
    make_figure(key)
print("\nDone.")
