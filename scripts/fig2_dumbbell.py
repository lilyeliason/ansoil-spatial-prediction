"""
Figure 2: predictability by geochemical class. Grouped dumbbell, two panels.

Run:  python scripts/fig2_dumbbell.py

Reads the per-seed leave-one-location-out predictions. It looks for them in the
usual places and prints which one it found:
    results/{model}_seed{seed}/ansoil_cv_predictions_{model}_{seed}.csv
    results/ansoil_cv_predictions_{model}_{seed}.csv
    data/ansoil_cv_predictions_{model}_{seed}.csv

Writes
    figures/fig2_dumbbell.pdf    vector, this is the one for the poster
    figures/fig2_dumbbell.png    200 dpi preview
    figures/fig2_dumbbell.csv    the plotted numbers, for checking the caption

One row per property, two dots per row (RF and XGB) joined by a grey segment,
with whiskers at plus and minus one SD across the five seeds. R2 is recomputed
here in native measured units. It is NOT the pipeline's reported R2, which for
39 targets sits in log or CLR space.

Properties are grouped into five geochemical bands. Band order is fixed because
the geochemical logic is the story; rows sort within a band by the better model.

The layout is computed in inches so the type really is the size it says.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures"

SEEDS = [7, 42, 73, 123, 256]
# Axis limits are derived from the data so no whisker is ever clipped; the
# script checks and says so. Set XLIM to a tuple to override.
XLIM = None
XTICK_STEP = 0.1
MAPPABLE = 0.30

# Any band longer than this is cut to its top rows plus one summary line.
# None keeps every property. About 12 brings the figure down to roughly 11 in.
MAX_ROWS_PER_BAND = None

# --- geometry, inches ---------------------------------------------------
PANEL_W_IN = 5.5
ROW_PITCH_IN = 0.28
BAND_GAP_ROWS = 1.5
PANEL_GAP_IN = 0.45
MARGIN_IN = 0.60
# --- type, points. Marks carry the figure, type stays out of the way. ---
TITLE_PT = 26  # figure title, top left
SUB_PT = 16  # the metric line under the title
PANEL_PT = 20  # panel headings
LABEL_PT = 16  # row labels and x ticks
BAND_PT = 16  # band headings, right-aligned with the row labels
NOTE_PT = 13  # the single line under the panels
DOT_MS = 12  # dot diameter
LINK_LW = 2.4  # the segment joining the pair

# A dot at or below this has no predictive skill and is drawn in grey.
# Set to MAPPABLE to grey everything that does not clear the mappable line.
GREY_BELOW = 0.0

RF, XGB = "#275E5C", "#D5762C"
RF_OFF, XGB_OFF = "#6E6B65", "#BFB9B0"  # no skill: dark grey RF, light grey XGB
LINK, INK, INK2, RULE = "#DAD6CE", "#231f20", "#605E5A", "#A9A49C"
BG = "white"

BAND_NAME = {
    1: "Isotopes",
    2: "Digest Metals",
    3: "Ion Ratios",
    4: "pH & CEC",
    5: "Soluble Ions",
}
PANELS = [
    ([1, 2], "Rock-Derived and Biological"),
    ([3, 4, 5], "Extractable and Solution-Phase"),
]
TITLE = "Predictability by Geochemical Class"
SUBTITLE = "Mean LOLO CV R\u00b2, Native Measured Units"

ELEM = {
    "al3082": "Al",
    "as1890": "As",
    "b_2496": "B",
    "ba4554": "Ba",
    "be3130": "Be",
    "cd2288": "Cd",
    "co2286": "Co",
    "cr2835": "Cr",
    "cu3247": "Cu",
    "fe2599": "Fe",
    "hg1849": "Hg",
    "k_7664": "K",
    "li6707": "Li",
    "mn2576": "Mn",
    "mo2020": "Mo",
    "ni2316": "Ni",
    "p_1774": "P",
    "pb2203": "Pb",
    "sb2068": "Sb",
    "se1960": "Se",
    "si2516": "Si",
    "sn1899": "Sn",
    "sr4077": "Sr",
    "ti3349": "Ti",
    "tl1908": "Tl",
    "v_2924": "V",
    "zn2138": "Zn",
    "na5895": "Na",
    "mg2852": "Mg",
    "ca3158": "Ca",
}
ION = {
    "f": "F",
    "cl": "Cl",
    "no3": "NO3",
    "po4": "PO4",
    "so4": "SO4",
    "ca2": "Ca",
    "k": "K",
    "mg2": "Mg",
    "na": "Na",
    "sr2": "Sr",
}
FIXED = {
    "d15n_air_permil": "d15N",
    "d13c_vpdb_permil": "d13C",
    "wt_percent_n": "Wt% N",
    "wt_percent_c": "Wt% C",
    "c_n_ratio": "C:N",
    "ph_mq": "pH (MQ)",
    "ph_kcl": "pH (KCl)",
    "ph_cacl2": "pH (CaCl2)",
    "cec_meq_100g": "CEC",
    "ec_us_cm": "EC",
}
# The band heading already says digest, ion ratio or soluble, so the row label
# carries only the species and the extraction time.
LEACH = {"total_mg_l_": "Total", "hr_1_mg_l_": "1 hr", "hr_24_mg_l_": "24 hr"}

MAC_FONTS = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/HelveticaNeue.ttc",
    str(Path.home() / "Library/Fonts/HelveticaNeue.ttc"),
]


def use_helvetica():
    """Register the macOS Helvetica files, then report what will actually be used.

    A .ttc is a font collection. Registering it does not guarantee the bold face
    resolves, so this prints the file matplotlib picks for both regular and bold.
    If either line does not say Helvetica, the figure is not in Helvetica.
    """
    for path in MAC_FONTS:
        if Path(path).exists():
            try:
                fm.fontManager.addfont(path)
            except Exception as e:
                print(f"  could not register {path}: {e}")
    wanted = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    have = {f.name for f in fm.fontManager.ttflist}
    picked = next((f for f in wanted if f in have), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": wanted,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
        }
    )
    reg = fm.findfont(fm.FontProperties(family="sans-serif", weight="normal"))
    bold = fm.findfont(fm.FontProperties(family="sans-serif", weight="bold"))
    print(f"font family: {picked}")
    print(f"  regular -> {reg}")
    print(f"  bold    -> {bold}")
    if "elvetica" not in reg or "elvetica" not in bold:
        print(
            "  NOT Helvetica. Delete ~/.matplotlib/fontlist-*.json and rerun so "
            "the font cache is rebuilt."
        )
    return picked


def find_predictions(model, seed):
    """The CV prediction file, wherever it lives in this repo."""
    name = f"ansoil_cv_predictions_{model}_{seed}.csv"
    for p in (
        ROOT / "results" / f"{model}_seed{seed}" / name,
        ROOT / "results" / name,
        ROOT / "data" / name,
        ROOT / name,
    ):
        if p.exists():
            return p
    raise SystemExit(
        f"Cannot find {name}. Looked in results/{model}_seed{seed}/, results/, "
        f"data/ and the repo root, under {ROOT}"
    )


def band_of(prop):
    if prop in (
        "d15n_air_permil",
        "d13c_vpdb_permil",
        "wt_percent_c",
        "wt_percent_n",
        "c_n_ratio",
    ):
        return 1
    if prop.startswith("clr_"):
        return 3
    if prop.startswith(("ph_", "cec_")):
        return 4
    if prop.startswith("digest_mg_kg_"):
        return 2
    if prop == "ec_us_cm" or prop.startswith(tuple(LEACH)):
        return 5
    raise ValueError(f"unclassified property: {prop}")


def label_of(prop):
    if prop in FIXED:
        return FIXED[prop]
    if prop.startswith("digest_mg_kg_"):
        return ELEM[prop[len("digest_mg_kg_") :]]
    rest = prop.removeprefix("clr_")
    for pre, word in LEACH.items():
        if rest.startswith(pre):
            return f"{ION[rest[len(pre) :]]} ({word})"
    raise ValueError(f"no label for {prop}")


def r2(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    sst = ((a - a.mean()) ** 2).sum()
    return np.nan if sst == 0 or len(a) < 2 else 1.0 - ((a - p) ** 2).sum() / sst


def load():
    """Native-unit R2 per property per model: mean and SD across the seeds.

    Strips the `log_` prefix so a property whose transform choice flipped
    between seeds stays one row instead of splitting into two.
    """
    acc = {}
    for model in ("rf", "xgb"):
        acc[model] = {}
        for seed in SEEDS:
            path = find_predictions(model, seed)
            if model == "rf" and seed == SEEDS[0]:
                print(f"predictions: {path.parent}")
            for target, g in pd.read_csv(path).groupby("target"):
                acc[model].setdefault(re.sub(r"^log_", "", target), []).append(
                    r2(g.actual, g.predicted)
                )
    if set(acc["rf"]) != set(acc["xgb"]):
        raise SystemExit("rf and xgb cover different properties")
    props = sorted(acc["rf"])
    d = pd.DataFrame(
        {
            "property": props,
            "label": [label_of(p) for p in props],
            "band": [band_of(p) for p in props],
            "rf": [np.mean(acc["rf"][p]) for p in props],
            "rf_sd": [np.std(acc["rf"][p], ddof=1) for p in props],
            "xgb": [np.mean(acc["xgb"][p]) for p in props],
            "xgb_sd": [np.std(acc["xgb"][p], ddof=1) for p in props],
            "n_seeds": [len(acc["rf"][p]) for p in props],
        }
    )
    d["best"] = d[["rf", "xgb"]].max(axis=1)
    d["worst"] = d[["rf", "xgb"]].min(axis=1)
    if (d.n_seeds != len(SEEDS)).any():
        print(
            "  warning: properties with fewer than "
            f"{len(SEEDS)} seeds:\n{d[d.n_seeds != len(SEEDS)][['property', 'n_seeds']]}"
        )
    return d.sort_values(["band", "best"], ascending=[True, False]).reset_index(
        drop=True
    )


def build_panel(d, bands):
    rows, heads = [], []
    y = 0.0
    for b in reversed(bands):
        sub = d[d.band == b]
        keep, note = sub, None
        if MAX_ROWS_PER_BAND and len(sub) > MAX_ROWS_PER_BAND:
            keep = sub.head(MAX_ROWS_PER_BAND)
            rest = sub.tail(len(sub) - MAX_ROWS_PER_BAND)
            note = (
                f"{len(rest)} further, all below {MAPPABLE:.2f}"
                if (rest.best < MAPPABLE).all()
                else f"{len(rest)} further, not shown"
            )
        if note:
            rows.append({"label": "", "note": note, "y": y, "band": b})
            y += 1
        for _, r in keep.iloc[::-1].iterrows():
            rows.append(
                {
                    "label": r.label,
                    "note": None,
                    "y": y,
                    "band": b,
                    "rf": r.rf,
                    "rf_sd": r.rf_sd,
                    "xgb": r.xgb,
                    "xgb_sd": r.xgb_sd,
                }
            )
            y += 1
        heads.append({"text": BAND_NAME[b], "count": len(sub), "y": y - 0.15})
        y += BAND_GAP_ROWS
    return pd.DataFrame(rows), heads, y - BAND_GAP_ROWS + 1.0


def draw(ax, rows, heads, panel_title, top, gutter_frac, xlim, xticks):
    ax.axvline(0, color=RULE, lw=1.0, zorder=1)
    ax.axvline(MAPPABLE, color=RULE, lw=1.0, ls=(0, (5, 5)), zorder=1)

    body = rows[rows.note.isna()]
    ax.hlines(
        body.y,
        body[["rf", "xgb"]].min(axis=1),
        body[["rf", "xgb"]].max(axis=1),
        color=LINK,
        lw=LINK_LW,
        zorder=2,
    )
    for col, live, dead in (("rf", RF, RF_OFF), ("xgb", XGB, XGB_OFF)):
        for mask, colour in (
            (body[col] > GREY_BELOW, live),
            (body[col] <= GREY_BELOW, dead),
        ):
            if mask.any():
                sub = body[mask]
                ax.errorbar(
                    sub[col],
                    sub.y,
                    xerr=sub[col + "_sd"],
                    fmt="o",
                    ms=DOT_MS,
                    color=colour,
                    mec=BG,
                    mew=1.2,
                    ecolor=colour,
                    elinewidth=1.4,
                    capsize=3,
                    capthick=1.4,
                    ls="none",
                    zorder=3,
                )

    ax.set_yticks(body.y)
    ax.set_yticklabels(body.label, fontsize=LABEL_PT, color=INK)
    ax.set_ylim(-0.9, top)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.1f}" for t in xticks], fontsize=LABEL_PT, color=INK2)
    ax.tick_params(axis="x", colors=RULE, length=5, width=0.9, pad=7)
    ax.tick_params(axis="y", length=0, pad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.spines["bottom"].set_linewidth(0.9)

    for _, r in rows[rows.note.notna()].iterrows():
        ax.text(
            xlim[0] + 0.012,
            r.y,
            r.note,
            fontsize=LABEL_PT,
            color=INK2,
            style="italic",
            va="center",
            ha="left",
        )

    # band headings sit in the label column, right-aligned with the row labels,
    # so nothing ever crosses into the plot
    for h in heads:
        ax.text(
            -0.012,
            h["y"],
            h["text"],
            transform=ax.get_yaxis_transform(),
            fontsize=BAND_PT,
            fontweight="bold",
            color=INK,
            va="bottom",
            ha="right",
            clip_on=False,
        )

    ax.text(
        -gutter_frac,
        top + 0.5,
        panel_title,
        transform=ax.get_yaxis_transform(),
        fontsize=PANEL_PT,
        fontweight="bold",
        color=INK,
        va="bottom",
        ha="left",
        clip_on=False,
    )


def main():
    use_helvetica()
    FIGS.mkdir(exist_ok=True)
    d = load()
    print(
        f"{len(d)} properties, band counts "
        f"{d.band.value_counts().sort_index().to_dict()}"
    )

    n_clear = int((d.best >= MAPPABLE).sum())
    b5 = d[d.band == 5]
    n_bad5 = int((b5.best <= 0).sum())
    lowest = d.loc[d.worst.idxmin()]
    d15 = d[d.property == "d15n_air_permil"].iloc[0]
    print(f"  d15N   RF {d15.rf:+.3f}  XGB {d15.xgb:+.3f}")
    print(f"  {n_clear} of {len(d)} properties clear {MAPPABLE:.2f} in native units")
    print(f"  {n_bad5} of {len(b5)} soluble salt measures at or below zero")
    print(f"  lowest single value: {lowest.label} at {lowest.worst:+.3f}")

    hi = float(
        (d[["rf", "xgb"]].max(axis=1) + d[["rf_sd", "xgb_sd"]].max(axis=1)).max()
    )
    lo = float(
        (d[["rf", "xgb"]].min(axis=1) - d[["rf_sd", "xgb_sd"]].max(axis=1)).min()
    )
    if XLIM:
        xlim = XLIM
    else:
        xlim = (
            np.floor(lo / XTICK_STEP) * XTICK_STEP,
            np.ceil(hi / XTICK_STEP) * XTICK_STEP,
        )
    xticks = np.arange(xlim[0], xlim[1] + 1e-9, XTICK_STEP)
    clipped = d[
        (d[["rf", "xgb"]].max(axis=1) + d[["rf_sd", "xgb_sd"]].max(axis=1) > xlim[1])
        | (d[["rf", "xgb"]].min(axis=1) - d[["rf_sd", "xgb_sd"]].max(axis=1) < xlim[0])
    ]
    print(
        f"x axis {xlim[0]:.2f} to {xlim[1]:.2f}; widest whisker spans "
        f"{lo:.3f} to {hi:.3f}"
    )
    if len(clipped):
        print("  WARNING, clipped rows:", ", ".join(clipped.label))
    else:
        print("  nothing clipped")

    panels = [build_panel(d, bands) for bands, _ in PANELS]
    top = max(p[2] for p in panels)
    # push the shorter panel up so both start level under their headings and the
    # leftover space lands at the bottom, where the legend sits
    shifted = []
    for rows, heads, own_top in panels:
        lift = top - own_top
        if lift:
            rows = rows.assign(y=rows.y + lift)
            heads = [{**h, "y": h["y"] + lift} for h in heads]
        shifted.append((rows, heads, own_top))
    panels = shifted

    longest = max(len(r.label) for rows, _, _ in panels for _, r in rows.iterrows())
    widest_head = max(len(h["text"]) for _, hs, _ in panels for h in hs)
    gutter = max(longest * 0.50, widest_head * 0.53) * LABEL_PT / 72 + 0.22
    plot_h = top * ROW_PITCH_IN
    head_h = MARGIN_IN + TITLE_PT / 72 * 1.5 + SUB_PT / 72 * 3.0 + PANEL_PT / 72 * 1.9
    foot_h = MARGIN_IN + LABEL_PT / 72 * 2.4 + NOTE_PT / 72 * 2.0
    fig_w = MARGIN_IN + 2 * (gutter + PANEL_W_IN) + PANEL_GAP_IN + MARGIN_IN
    fig_h = head_h + plot_h + foot_h
    print(
        f"figure {fig_w:.2f} x {fig_h:.2f} in, {top:.0f} row slots, "
        f"{ROW_PITCH_IN * 72:.1f} pt per row against {LABEL_PT} pt labels"
    )
    if ROW_PITCH_IN * 72 < LABEL_PT * 1.15:
        print(
            "  warning: rows tighter than the type. Raise ROW_PITCH_IN, "
            "lower LABEL_PT, or set MAX_ROWS_PER_BAND."
        )

    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = [
        fig.add_axes(
            [
                (MARGIN_IN + gutter + i * (gutter + PANEL_W_IN + PANEL_GAP_IN)) / fig_w,
                foot_h / fig_h,
                PANEL_W_IN / fig_w,
                plot_h / fig_h,
            ]
        )
        for i in range(2)
    ]

    gutter_frac = gutter / PANEL_W_IN
    rows_a, heads_a, _ = panels[0]
    rows_b, heads_b, _ = panels[1]
    draw(axes[0], rows_a, heads_a, PANELS[0][1], top, gutter_frac, xlim, xticks)
    draw(axes[1], rows_b, heads_b, PANELS[1][1], top, gutter_frac, xlim, xticks)

    axes[1].legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                ms=DOT_MS,
                color=RF,
                mec=BG,
                mew=1.2,
                label="Random Forest",
            ),
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                ms=DOT_MS,
                color=XGB,
                mec=BG,
                mew=1.2,
                label="XGBoost",
            ),
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                ms=DOT_MS,
                color=RF_OFF,
                mec=BG,
                mew=1.2,
                label="No Better Than\nthe Mean",
            ),
            Line2D(
                [],
                [],
                color=RULE,
                lw=1.0,
                ls=(0, (5, 5)),
                label=f"Mappable,\nR\u00b2 = {MAPPABLE:.2f}",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=((MAPPABLE + 0.03 - xlim[0]) / (xlim[1] - xlim[0]), 0.005),
        frameon=False,
        fontsize=LABEL_PT,
        labelcolor=INK,
        handletextpad=0.6,
        borderaxespad=0.0,
        labelspacing=0.9,
    )

    fig.text(
        MARGIN_IN / fig_w,
        1 - (MARGIN_IN + TITLE_PT / 72 * 0.95) / fig_h,
        TITLE,
        fontsize=TITLE_PT,
        fontweight="bold",
        color=INK,
        ha="left",
        va="baseline",
    )
    fig.text(
        MARGIN_IN / fig_w,
        1 - (MARGIN_IN + TITLE_PT / 72 * 1.5 + SUB_PT / 72 * 1.15) / fig_h,
        SUBTITLE,
        fontsize=SUB_PT,
        color=INK2,
        ha="left",
        va="baseline",
    )

    fig.text(
        MARGIN_IN / fig_w,
        MARGIN_IN / fig_h * 0.55,
        f"Whiskers Show \u00b11 SD Across {len(SEEDS)} Seeds",
        fontsize=NOTE_PT,
        color=INK2,
        ha="left",
        va="bottom",
    )

    fig.savefig(FIGS / "fig2_dumbbell.pdf")
    fig.savefig(FIGS / "fig2_dumbbell.png", dpi=200)
    d.to_csv(FIGS / "fig2_dumbbell.csv", index=False)
    print(f"wrote {FIGS / 'fig2_dumbbell.pdf'} and .png and .csv")


if __name__ == "__main__":
    main()
