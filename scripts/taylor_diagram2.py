"""
make_taylor_diagrams.py
ANSOIL Geochemical Predictions — Taylor Diagrams
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── 0. Paths ───────────────────────────────────────────────────────────────────
TARGETS_FILE = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/data/ansoil_targets.csv"
LOG_TABLE_FILE = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/data/ansoil_log_targets.csv"
RF_METRICS_FILE = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/results/aggregated/metrics_summary_rf.csv"
XGB_METRICS_FILE = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/results/aggregated/metrics_summary_xgb.csv"

OUT_MAIN = "taylor_main.png"
OUT_FAMILIES = "taylor_families.png"

# ── Load data ────────────────────────────────────────────────────────────────
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


# ── Compute Taylor statistics ────────────────────────────────────────────────
def taylor_stats(r2, rmse, sigma_obs):
    if sigma_obs <= 0 or np.isnan(rmse):
        return np.nan, np.nan, np.nan
    R = np.sqrt(max(r2, 0)) if r2 >= 0 else -np.sqrt(-r2)
    a = 1
    b = -2 * R * sigma_obs
    c = sigma_obs**2 - rmse**2
    disc = b**2 - 4 * a * c
    if disc < 0:
        sigma_pred = R * sigma_obs
    else:
        s1 = (-b + np.sqrt(disc)) / 2
        s2 = (-b - np.sqrt(disc)) / 2
        sigma_pred = s1 if s1 >= 0 else s2
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
            {
                "target": t,
                "model": model,
                "R": R,
                "sigma_n": sn,
                "cRMSE_n": cRn,
                "r2": row["cv_r2_mean"],
                "tier": row["tier"],
            }
        )

df_all = pd.DataFrame(rows).dropna(subset=["R", "sigma_n", "cRMSE_n"])
df_all = df_all[df_all["sigma_n"].between(0, 2.0)]

# ── Short display labels ─────────────────────────────────────────────────────
label_clean = {
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
    "log_digest_mg_kg_na5895": "Na(log)",
    "log_digest_mg_kg_li6707": "Li",
    "log_digest_mg_kg_pb2203": "Pb",
    "log_digest_mg_kg_sb2068": "Sb",
    "clr_total_mg_l_po4": "PO₄(tot)",
    "clr_hr_1_mg_l_po4": "PO₄(1h)",
    "clr_hr_24_mg_l_po4": "PO₄(24h)",
    "clr_total_mg_l_k": "K(tot)",
    "clr_total_mg_l_na": "Na(CLR)",
    "clr_total_mg_l_cl": "Cl(tot)",
    "clr_total_mg_l_no3": "NO₃(tot)",
    "clr_total_mg_l_so4": "SO₄(tot)",
    "clr_total_mg_l_ca2": "Ca(tot)",
    "clr_total_mg_l_sr2": "Sr(tot)",
    "clr_hr_1_mg_l_cl": "Cl(1h)",
    "clr_hr_1_mg_l_no3": "NO₃(1h)",
    "clr_hr_1_mg_l_so4": "SO₄(1h)",
    "clr_hr_24_mg_l_cl": "Cl(24h)",
    "clr_hr_24_mg_l_no3": "NO₃(24h)",
    "clr_hr_24_mg_l_so4": "SO₄(24h)",
    "log_total_mg_l_na": "Na(tot)",
    "log_total_mg_l_mg2": "Mg(tot)",
    "log_total_mg_l_ca2": "Ca(tot)",
    "log_total_mg_l_cl": "Cl(tot)L",
    "log_hr_24_mg_l_cl": "Cl(24h)L",
    "log_hr_1_mg_l_cl": "Cl(1h)L",
    "ec_us_cm": "EC",
    "hr_24_mg_l_so4": "SO₄(24h)R",
}
df_all["label"] = df_all["target"].map(label_clean).fillna(df_all["target"])

TIER_COLORS = {
    "strong": "#1565c0",
    "moderate": "#2e7d32",
    "weak": "#e65100",
    "unusable": "#9e9e9e",
}
MARKER = {"RF": "o", "XGB": "D"}


# ── Core drawing function ────────────────────────────────────────────────────
def draw_taylor(
    ax,
    df_pts,
    min_r_max=1.15,
    rmse_contours=None,
    title="",
):
    if rmse_contours is None:
        rmse_contours = [0.25, 0.5, 0.75, 1.0]

    # Dynamically scale the grid to fit the wildest point
    max_sigma_n = df_pts["sigma_n"].max() if not df_pts.empty else 0
    r_max = max(min_r_max, max_sigma_n + 0.1)

    ax.set_facecolor("#ffffff")
    ax.set_thetamin(0)
    ax.set_thetamax(90)

    # 1. Background Grids
    for rv in np.arange(0.25, r_max + 0.01, 0.25):
        ths = np.linspace(0, np.pi / 2, 200)
        ax.plot(ths, np.full_like(ths, rv), "-", color="#f0f0f0", lw=0.8, zorder=0)

    for rc in rmse_contours:
        ths = np.linspace(0, np.pi / 2, 300)
        ct = np.cos(ths)
        disc = ct**2 - (1 - rc**2)
        r_arc = np.where(disc >= 0, ct + np.sqrt(np.where(disc >= 0, disc, 0)), np.nan)
        r_arc = np.where((r_arc >= 0) & (r_arc <= r_max + 0.1), r_arc, np.nan)
        ax.plot(ths, r_arc, ":", color="#cccccc", lw=1.2, zorder=1)

        th_l = np.radians(80)
        ct_l = np.cos(th_l)
        d_l = ct_l**2 - (1 - rc**2)
        if d_l >= 0:
            r_l = ct_l + np.sqrt(d_l)
            if 0 < r_l <= r_max + 0.05:
                ax.text(
                    th_l,
                    r_l,
                    f"{rc}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#999999",
                    style="italic",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1),
                )

    ths_full = np.linspace(0, np.pi / 2, 300)
    ax.plot(ths_full, np.ones_like(ths_full), "--", color="#555555", lw=1.2, zorder=2)

    for r_val in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
        th = np.arccos(r_val)
        if th <= np.pi / 2 + 0.01:
            ax.plot([th, th], [0, r_max + 0.05], "-", color="#f0f0f0", lw=0.8, zorder=0)
            ax.text(
                th,
                r_max + 0.08,
                f"{r_val}",
                ha="center",
                va="center",
                fontsize=7,
                color="#666666",
            )

    # 2. Reference Point (OBS)
    ax.scatter([0], [1.0], marker="*", s=200, color="#1a1a1a", zorder=15, clip_on=False)
    ax.text(
        np.radians(2),
        1.04,
        "OBS",
        fontsize=8,
        fontweight="bold",
        color="#1a1a1a",
        ha="left",
        va="bottom",
    )

    # 3. Plot Models and Build Numeric Key
    key_items = []

    targets_sorted = (
        df_pts.groupby("target")["R"].mean().sort_values(ascending=False).index
    )

    for idx, tgt in enumerate(targets_sorted, start=1):
        sub = df_pts[df_pts["target"] == tgt]
        if sub.empty:
            continue

        th_mean, sn_mean = 0, 0
        count = 0

        for _, row in sub.iterrows():
            if np.isnan(row["R"]) or np.isnan(row["sigma_n"]):
                continue
            th = np.arccos(np.clip(row["R"], 0, 1))
            sn = row["sigma_n"]
            col = TIER_COLORS.get(row["tier"], "#cccccc")
            mk = MARKER.get(row["model"], "o")
            ms = 60 if row["tier"] == "strong" else 40

            # Dropped alpha to 0.6 for more transparency
            ax.scatter(
                th,
                sn,
                marker=mk,
                s=ms,
                color=col,
                alpha=0.6,
                zorder=8,
                edgecolors="#ffffff",
                linewidths=0.5,
            )

            th_mean += th
            sn_mean += sn
            count += 1

        if count > 0:
            th_mean /= count
            sn_mean /= count
            lbl = sub.iloc[0]["label"]

            # Slightly nudged the tag and made the background transparent
            ax.text(
                th_mean,
                sn_mean + 0.03,
                str(idx),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#1a1a1a",
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.5, pad=0.5),
            )

            key_items.append(f"{idx}. {lbl}")

    ax.set_rlim(0, r_max)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)

    # 4. Vertical Key on the Right
    if key_items:
        key_block = "\n".join(key_items)
        ax.annotate(
            key_block,
            xy=(1.15, 0.95),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize=8,
            color="#333333",
            linespacing=1.6,
        )

    if title:
        ax.set_title(
            title, fontsize=10, pad=20, color="#1a1a1a", loc="center", fontweight="bold"
        )


# ── Shared legend handles ────────────────────────────────────────────────────
leg_model = [
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
        marker="D",
        color="w",
        markerfacecolor="#555",
        markersize=8,
        label="XGBoost",
    ),
]
leg_tier = [
    mpatches.Patch(color=TIER_COLORS["strong"], label="Strong (R² ≥ 0.60)"),
    mpatches.Patch(color=TIER_COLORS["moderate"], label="Moderate (R² 0.30–0.60)"),
    mpatches.Patch(color=TIER_COLORS["weak"], label="Weak (R² 0–0.30)"),
    mpatches.Patch(color=TIER_COLORS["unusable"], label="Unusable (R² < 0)"),
]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1  —  Moderate + Strong targets only
# ══════════════════════════════════════════════════════════════════════════════
df_main = df_all[df_all["tier"].isin(["strong", "moderate"])].copy()

# Increased width to fit the vertical list
fig1, ax1 = plt.subplots(
    1, 1, figsize=(11.5, 8.5), subplot_kw={"projection": "polar"}, facecolor="white"
)

draw_taylor(
    ax1,
    df_main,
    min_r_max=1.05,
    rmse_contours=[0.25, 0.5, 0.75],
    title="Taylor Diagram — Moderate & Strong Targets\nRF (●) vs. XGBoost (◆)  |  ANSOIL",
)

ax1.text(
    np.radians(50),
    ax1.get_rmax() + 0.15,
    "Correlation (R)",
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    color="#333333",
)
ax1.text(
    np.radians(72),
    0.65,
    "Norm. Std. Dev.",
    ha="center",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    color="#333333",
)

ax1.legend(
    handles=leg_model,
    loc="upper left",
    bbox_to_anchor=(-0.1, -0.15),
    fontsize=8.5,
    frameon=False,
    title="Model",
    title_fontproperties={"weight": "bold"},
)
ax1.legend(
    handles=leg_tier[:2],
    loc="upper right",
    bbox_to_anchor=(1.1, -0.15),
    fontsize=8.5,
    frameon=False,
    title="Tier",
    title_fontproperties={"weight": "bold"},
)

plt.tight_layout()
plt.savefig(OUT_MAIN, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_MAIN}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2  —  All tiers, split by variable family
# ══════════════════════════════════════════════════════════════════════════════
families = {
    "Soil Properties": [
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
    "Digest Metals": [
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
    "Water Chemistry": [
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
}

# Massively expanded width so the lists don't crash into the next subplot
fig2, axes = plt.subplots(
    1, 3, figsize=(28, 10), subplot_kw={"projection": "polar"}, facecolor="white"
)
fig2.patch.set_facecolor("white")

for ax, (fam_name, fam_targets) in zip(axes, families.items()):
    df_fam = df_all[df_all["target"].isin(fam_targets)].copy()
    draw_taylor(
        ax,
        df_fam,
        min_r_max=1.2,
        rmse_contours=[0.5, 1.0],
        title=fam_name,
    )

fig2.legend(
    handles=leg_model + leg_tier,
    loc="lower center",
    ncol=6,
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(0.5, 0.02),
)
fig2.suptitle(
    "Taylor Diagram — All Variables by Family  |  RF (●) vs. XGBoost (◆)  |  ANSOIL",
    fontsize=14,
    fontweight="bold",
    y=1.02,
    color="#1a1a1a",
)

# Adjusted rect to make sure nothing is cropped
plt.tight_layout(rect=[0, 0.08, 0.95, 0.95])
plt.savefig(OUT_FAMILIES, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_FAMILIES}")
