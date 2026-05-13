"""
Taylor Diagram — ANSOIL  v7
Changes from v6b:
  - adjustText for non-overlapping label placement
  - Correlation numbers closer to arc (RMAX * 1.04)
  - "Correlation (R)" same size as labels (9pt)
  - Title larger (18pt in compose)
  - Legend bigger height, font matches labels
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import io

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.lines import Line2D
from PIL import Image

# ── 0. Paths ───────────────────────────────────────────────────────────────────
TARGETS_FILE = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/data/ansoil_targets.csv"
METRICS_RF = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/results/aggregated/metrics_summary_rf.csv"
METRICS_XGB = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/results/aggregated/metrics_summary_xgb.csv"
OUT_RF = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/results/plots/taylor_diagram_rf.png"
OUT_XGB = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/results/plots/taylor_diagram_xgb.png"
OUT_COMBINED = "/Users/lilyeliason/Documents/lemonte_lab/lab/ansoil-spatial-prediction/results/plots/taylor_diagram_combined.png"


# ── 1. Families ────────────────────────────────────────────────────────────────
def get_family(target):
    t = target.replace("log_", "")
    if t.startswith("ph_") or "ec_us_cm" in t:
        return "pH / EC"
    if t.startswith("wt_percent") or t == "c_n_ratio":
        return "C / N"
    if t.startswith("d15n") or t.startswith("d13c"):
        return "Isotopes"
    if t.startswith("clr_"):
        return "CLR ions"
    if "mg_l" in t:
        return "Water chemistry"
    if "digest" in t:
        return "Digest metals"
    if "cec" in t:
        return "CEC"
    return "Other"


FAMILY_COLORS = {
    "pH / EC": "#e05c2a",
    "C / N": "#8b4fc8",
    "Isotopes": "#1565c0",
    "CLR ions": "#2e7d32",
    "Water chemistry": "#e6a817",
    "Digest metals": "#c62828",
    "CEC": "#00695c",
}

# ── 2. Data ────────────────────────────────────────────────────────────────────
obs_df = pd.read_csv(TARGETS_FILE)
obs_std = {
    col: obs_df[col].dropna().std(ddof=1)
    for col in obs_df.columns
    if col != "sample_id" and len(obs_df[col].dropna()) > 1
}


def compute_stats(metrics_df):
    records = []
    for _, row in metrics_df.iterrows():
        target, r2, tier = row["target"], row["cv_r2_mean"], row["tier"]
        if target not in obs_std or obs_std[target] == 0:
            continue
        sigma_obs = obs_std[target]
        rmse_log = row.get("cv_rmse_log_space_mean", np.nan)
        rmse = (
            rmse_log
            if (target.startswith("log_") and not np.isnan(rmse_log))
            else row["cv_rmse_orig_units_mean"]
        )
        R = np.sqrt(abs(r2)) * np.sign(r2) if r2 != 0 else 0.0
        cRMSE_n = rmse / sigma_obs
        disc = max(4 * R**2 - 4 * (1 - cRMSE_n**2), 0)
        sigma_n = (2 * R + np.sqrt(disc)) / 2
        records.append(
            dict(
                target=target,
                family=get_family(target),
                R=R,
                sigma_n=sigma_n,
                cRMSE_n=cRMSE_n,
                tier=tier,
                r2=r2,
            )
        )
    return pd.DataFrame(records)


rf_stats = compute_stats(pd.read_csv(METRICS_RF))
xgb_stats = compute_stats(pd.read_csv(METRICS_XGB))


def short_name(t):
    return (
        t.replace("log_digest_mg_kg_", "")
        .replace("digest_mg_kg_", "")
        .replace("clr_total_mg_l_", "clr·")
        .replace("clr_hr_1_mg_l_", "hr1·")
        .replace("clr_hr_24_mg_l_", "hr24·")
        .replace("log_total_mg_l_", "log·tot·")
        .replace("log_hr_1_mg_l_", "log·hr1·")
        .replace("log_hr_24_mg_l_", "log·hr24·")
        .replace("_air_permil", "")
        .replace("_vpdb_permil", "")
        .replace("wt_percent_", "wt%")
        .replace("_", " ")
    )


# ── 3. Render square diagram ───────────────────────────────────────────────────
LABEL_FS = 9  # unified font size for labels, tick numbers, subtitle


def render_diagram_square(
    stats_list, markers, sizes, alphas, jitter_seeds, label_thresh=0.50
):
    RMAX = 1.55
    DPI = 180
    fig, ax = plt.subplots(
        figsize=(11, 11), dpi=DPI, subplot_kw={"projection": "polar"}, facecolor="white"
    )
    fig.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_facecolor("#f5f5f5")

    # Correlation guide lines — numbers pulled in closer to arc
    for r_val in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]:
        theta_r = np.arccos(r_val)
        ax.plot([theta_r, theta_r], [0, RMAX], ":", color="#cccccc", lw=1.0, zorder=0)
        ax.text(
            theta_r,
            RMAX * 1.04,  # ← was 1.09, now much closer
            f"{r_val:.2f}",
            ha="center",
            va="center",
            fontsize=LABEL_FS,
            color="#444444",
            fontfamily="monospace",
        )

    # "Correlation (R)" subtitle — same size as labels
    ax.text(
        np.radians(90),
        RMAX * 1.13,
        "Correlation  (R)",
        ha="center",
        va="center",
        fontsize=LABEL_FS,
        fontweight="bold",
        color="#333333",
    )

    # Std arcs
    th = np.linspace(0, np.pi, 300)
    for std_val in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        lw = 2.4 if std_val == 1.0 else 1.0
        col = "#444444" if std_val == 1.0 else "#cccccc"
        ax.plot(th, np.full_like(th, std_val), "-", color=col, lw=lw, zorder=2)

    ax.set_rticks([0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    ax.set_rlabel_position(162)
    ax.tick_params(axis="y", labelsize=LABEL_FS, colors="#555555", pad=5)
    ax.set_xticks([])
    ax.set_rlim(0, RMAX)

    ax.text(
        np.radians(152),
        RMAX * 0.50,
        "Normalized\nStd. Dev.",
        ha="center",
        va="center",
        fontsize=LABEL_FS,
        fontweight="bold",
        color="#333333",
        rotation=35,
    )

    # cRMSE arcs
    for crmse in [0.25, 0.5, 0.75, 1.0]:
        thetas = np.linspace(0, np.pi, 600)
        cos_t = np.cos(thetas)
        disc = cos_t**2 - (1 - crmse**2)
        r_arc = np.where(disc >= 0, cos_t + np.sqrt(np.maximum(disc, 0)), np.nan)
        r_arc = np.where((r_arc >= 0) & (r_arc <= RMAX), r_arc, np.nan)
        ax.plot(thetas, r_arc, "--", color="#c0c0c0", lw=1.0, zorder=1)
        t_l = np.radians(50)
        d_l = np.cos(t_l) ** 2 - (1 - crmse**2)
        if d_l >= 0:
            r_l = np.cos(t_l) + np.sqrt(d_l)
            if 0.1 < r_l < RMAX * 0.97:
                ax.text(
                    t_l,
                    r_l + 0.08,
                    f"cRMSE = {crmse:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=LABEL_FS - 1,
                    color="#aaaaaa",
                    style="italic",
                )

    # OBS
    ax.scatter([0], [1.0], marker="*", s=450, color="#111111", zorder=12)
    ax.annotate(
        "OBS",
        xy=(0, 1.0),
        xytext=(0.13, 1.18),
        fontsize=LABEL_FS,
        fontweight="bold",
        color="#111111",
        arrowprops=dict(arrowstyle="-", color="#333333", lw=1.0),
    )

    # ── Data points — collect positions for adjustText ─────────────────────
    all_scatter_x = []  # theta values (in polar = x for adjustText)
    all_scatter_y = []  # r values

    point_groups = []  # list of (theta_j, sn_j, color, marker, sz, al)
    label_data = []  # list of (theta_j, sn_j, name, color) for labeled pts

    for stats_df, marker, sz, al, jseed in zip(
        stats_list, markers, sizes, alphas, jitter_seeds
    ):
        rng = np.random.default_rng(jseed)
        labeled_targets = set()
        for _, row in stats_df.sort_values("r2").iterrows():
            R, sn = row["R"], row["sigma_n"]
            if np.isnan(R) or np.isnan(sn):
                continue
            theta_j = np.arccos(np.clip(R, -1, 1)) + rng.uniform(-0.012, 0.012)
            sn_j = max(sn + rng.uniform(-0.018, 0.018), 0.01)
            all_scatter_x.append(theta_j)
            all_scatter_y.append(sn_j)
            point_groups.append(
                (
                    theta_j,
                    sn_j,
                    FAMILY_COLORS.get(row["family"], "#888888"),
                    marker,
                    sz,
                    al,
                )
            )
            r2 = row["r2"]
            if (r2 >= label_thresh or r2 < -0.18) and row[
                "target"
            ] not in labeled_targets:
                labeled_targets.add(row["target"])
                label_data.append(
                    (
                        theta_j,
                        sn_j,
                        short_name(row["target"]),
                        FAMILY_COLORS.get(row["family"], "#888888"),
                    )
                )

    # Draw all points
    for theta_j, sn_j, color, marker, sz, al in point_groups:
        ax.scatter(
            theta_j,
            sn_j,
            marker=marker,
            s=sz,
            color=color,
            alpha=al,
            zorder=5,
            edgecolors="white",
            linewidths=1.2,
        )

    # Draw labels with adjustText — convert polar to display coords first
    # adjustText works on the Axes display space
    texts = []
    ann_points_x = []
    ann_points_y = []
    for theta_j, sn_j, name, color in label_data:
        # Place initial text at a fixed angular offset from the point
        t = ax.text(
            theta_j,
            sn_j * 1.12,
            name,
            fontsize=LABEL_FS,
            color=color,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=10,
        )
        texts.append(t)
        ann_points_x.append(theta_j)
        ann_points_y.append(sn_j)

    # adjustText in polar is tricky — use force params to push labels out radially
    adjust_text(
        texts,
        x=np.array(ann_points_x),
        y=np.array(ann_points_y),
        ax=ax,
        force_text=(0.3, 0.5),
        force_points=(0.4, 0.6),
        expand=(1.4, 1.8),
        arrowprops=dict(arrowstyle="-", lw=0.8, alpha=0.8),
        avoid_self=True,
        min_arrow_len=3,
    )

    buf = io.BytesIO()
    fig.savefig(buf, dpi=DPI, bbox_inches="tight", facecolor="white", format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


# ── 4. Autocrop ────────────────────────────────────────────────────────────────
def autocrop_semicircle(img):
    arr = np.array(img.convert("RGB"))
    non_white = (arr < 248).any(axis=2)
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    top = max(rows[0] - 8, 0)
    bottom = min(rows[-1] + 8, img.height)
    left = max(cols[0] - 8, 0)
    right = min(cols[-1] + 8, img.width)
    return img.crop((left, top, right, bottom))


# ── 5. Compose 16:9 ────────────────────────────────────────────────────────────
def compose(diagram_img, title_text, family_colors, extra_handles, out_path):
    TARGET_W = 3200
    TARGET_H = 1800
    TITLE_H = 110  # slightly taller for bigger title
    LEGEND_H = 130  # taller legend strip
    DIAG_H = TARGET_H - TITLE_H - LEGEND_H

    # Title
    fig_t, ax_t = plt.subplots(
        figsize=(TARGET_W / 200, TITLE_H / 200), dpi=200, facecolor="white"
    )
    ax_t.axis("off")
    ax_t.text(
        0.5,
        0.5,
        title_text,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#111111",  # ← 18pt
        transform=ax_t.transAxes,
    )
    buf = io.BytesIO()
    fig_t.savefig(buf, dpi=200, bbox_inches=None, facecolor="white", format="png")
    plt.close(fig_t)
    buf.seek(0)
    t_img = Image.open(buf).convert("RGB").copy()
    buf.close()
    t_img = t_img.resize((TARGET_W, TITLE_H), Image.LANCZOS)

    # Legend — font size matches LABEL_FS (9pt), rendered at same scale as diagram
    patches = [
        mpatches.Patch(facecolor=c, edgecolor="white", lw=0.5, label=f)
        for f, c in family_colors.items()
    ]
    handles = patches + (extra_handles or [])
    fig_l, ax_l = plt.subplots(
        figsize=(TARGET_W / 200, LEGEND_H / 200), dpi=200, facecolor="white"
    )
    ax_l.axis("off")
    ax_l.legend(
        handles=handles,
        fontsize=LABEL_FS,  # ← matches diagram labels
        loc="center",
        ncol=len(handles),
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        title="Variable family",
        title_fontsize=LABEL_FS,  # ← same size
        bbox_to_anchor=(0.5, 0.5),
    )
    buf = io.BytesIO()
    fig_l.savefig(buf, dpi=200, bbox_inches=None, facecolor="white", format="png")
    plt.close(fig_l)
    buf.seek(0)
    l_img = Image.open(buf).convert("RGB").copy()
    buf.close()
    l_img = l_img.resize((TARGET_W, LEGEND_H), Image.LANCZOS)

    # Diagram
    d_crop = autocrop_semicircle(diagram_img)
    scale = DIAG_H / d_crop.height
    new_w = int(d_crop.width * scale)
    d_scaled = d_crop.convert("RGB").resize((new_w, DIAG_H), Image.LANCZOS)
    d_strip = Image.new("RGB", (TARGET_W, DIAG_H), (255, 255, 255))
    x_off = (TARGET_W - new_w) // 2
    d_strip.paste(d_scaled, (max(x_off, 0), 0))

    # Stack
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), (255, 255, 255))
    canvas.paste(t_img, (0, 0))
    canvas.paste(d_strip, (0, TITLE_H))
    canvas.paste(l_img, (0, TITLE_H + DIAG_H))
    canvas.save(out_path, dpi=(200, 200))
    print(f"Saved → {out_path}  ({TARGET_W}×{TARGET_H})")


# ── 6. Render all three ────────────────────────────────────────────────────────
model_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#555555",
        markersize=11,
        label="Random Forest",
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        markerfacecolor="#555555",
        markersize=10,
        label="XGBoost",
    ),
]

diag_rf = render_diagram_square(
    [rf_stats], ["o"], [180], [0.72], [1], label_thresh=0.50
)
compose(
    diag_rf,
    "Random Forest — ANSOIL Geochemical Predictions",
    FAMILY_COLORS,
    None,
    OUT_RF,
)

diag_xgb = render_diagram_square(
    [xgb_stats], ["D"], [165], [0.72], [2], label_thresh=0.50
)
compose(
    diag_xgb, "XGBoost — ANSOIL Geochemical Predictions", FAMILY_COLORS, None, OUT_XGB
)

diag_comb = render_diagram_square(
    [rf_stats, xgb_stats],
    ["o", "D"],
    [145, 130],
    [0.55, 0.55],
    [3, 4],
    label_thresh=0.52,
)
compose(
    diag_comb,
    "RF vs. XGBoost — ANSOIL Geochemical Predictions (combined)",
    FAMILY_COLORS,
    model_handles,
    OUT_COMBINED,
)
