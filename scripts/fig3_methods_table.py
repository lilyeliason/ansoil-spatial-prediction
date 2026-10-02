"""
Figure 3: four methods on one property. A table, not a chart.

Run:  python scripts/fig3_methods_table.py

Needs pandas, numpy.

Everything is recomputed here from the per-sample leave-one-location-out
predictions, in native permil, so the numbers match Figure 2:

  Random Forest  mean of the per-seed R2 over 5 seeds
  XGBoost        same
  KNN            single run, no seeds
  MLR            refit here on the same 22 covariates, same folds

Regression kriging is not a row. It was tested separately and did not help,
so it appears only as a note, with the count read from the rerun's own output
file rather than typed in. See the methods text on the poster.

Reads
    results/{model}_seed{seed}/ansoil_cv_predictions_{model}_{seed}.csv
    results/ansoil_cv_predictions_knn.csv
    data/ansoil_predictors.csv, data/ansoil_targets.csv
    kriging_comparison.csv           optional, only for the kriging note

Writes
    figures/fig3_methods_table.pdf   vector, this is the one for the poster
    figures/fig3_methods_table.png   200 dpi preview
    figures/fig3_methods_table.csv   the numbers
"""

import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures"
SEEDS = [7, 42, 73, 123, 256]
TARGET = "d15n_air_permil"
PROPERTY = "d15N"

TITLE = "Model Performance Comparison, δ¹⁵N"
SUBTITLE = "RF, XGB, KNN, and MLR Under Leave-One-Location-Out CV"

TITLE_PT, SUB_PT, HEAD_PT, CELL_PT, NOTE_PT = 26, 16, 17, 17, 13
NOTE_WRAP = 100  # characters per note line, so none overflow
NOTE_GAP_IN = 0.30  # clear space under the closing rule
MARGIN_IN = 0.60
ROW_H_IN = 0.46
COL_W_IN = [4.0, 2.2, 2.3, 1.5]  # Method, R2, RMSE, vs RF
INK, INK2, RULE, BG = "#231f20", "#605E5A", "#A9A49C", "white"
HILITE = "#275E5C"  # the one row that fails

MAC_FONTS = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/HelveticaNeue.ttc",
    str(Path.home() / "Library/Fonts/HelveticaNeue.ttc"),
]


def use_helvetica():
    for path in MAC_FONTS:
        if Path(path).exists():
            try:
                fm.fontManager.addfont(path)
            except Exception as e:
                print(f"  could not register {path}: {e}")
    wanted = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": wanted,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
        }
    )
    reg = fm.findfont(fm.FontProperties(family="sans-serif", weight="normal"))
    bold = fm.findfont(fm.FontProperties(family="sans-serif", weight="bold"))
    print(f"font regular -> {reg}\nfont bold    -> {bold}")
    if "elvetica" not in reg or "elvetica" not in bold:
        print("  NOT Helvetica. Delete ~/.matplotlib/fontlist-*.json and rerun.")


SKIP_DIRS = {
    ".venv",
    "venv",
    ".git",
    "__pycache__",
    "node_modules",
    ".ipynb_checkpoints",
}


def find(name, model=None, seed=None, required=True):
    """The named file, wherever it sits in this repo.

    Checks the usual places first, then searches the whole repo, so a file in
    results/knn/ or results/aggregated/ is found without editing anything.
    required=False returns None instead of stopping, for optional inputs.
    """
    cands = [ROOT / "results" / name, ROOT / "data" / name, ROOT / name]
    if model and seed is not None:
        cands.insert(0, ROOT / "results" / f"{model}_seed{seed}" / name)
    for p in cands:
        if p.exists():
            return p
    hits = [
        p for p in ROOT.rglob(name) if not SKIP_DIRS & set(p.relative_to(ROOT).parts)
    ]
    if hits:
        hits.sort(key=lambda p: len(p.relative_to(ROOT).parts))
        if len(hits) > 1:
            print(
                f"  note: {len(hits)} copies of {name}, using "
                f"{hits[0].relative_to(ROOT)}"
            )
        return hits[0]
    if not required:
        return None
    raise SystemExit(f"Cannot find {name} anywhere under {ROOT}")


def r2(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return 1.0 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum()


def rmse(a, p):
    return float(np.sqrt(((np.asarray(a, float) - np.asarray(p, float)) ** 2).mean()))


def kriging_note():
    """One sentence about regression kriging, counted from the rerun's output.

    Nothing is typed by hand: the fractions come out of kriging_comparison.csv.
    Returns None if that file is not in the repo, and the note is skipped.
    """
    p = find("kriging_comparison.csv", required=False)
    if p is None:
        print("  kriging_comparison.csv not found, skipping the kriging note")
        return
    k = pd.read_csv(p)
    if "delta_location_lolo" not in k.columns:
        print(
            "  kriging_comparison.csv has no delta_location_lolo, "
            "skipping the kriging note"
        )
        return
    worse, n = int((k.delta_location_lolo < 0).sum()), len(k)
    print(
        f"  kriging: lower R² in {worse} of {n} model and property "
        f"combinations ({p.relative_to(ROOT)})"
    )


def compute():
    P = pd.read_csv(find("ansoil_predictors.csv"))
    Y = pd.read_csv(find("ansoil_targets.csv"))[["sample_id", TARGET]]
    d = (
        P.merge(Y, on="sample_id", validate="1:1")
        .dropna(subset=[TARGET])
        .reset_index(drop=True)
    )
    y = d[TARGET].to_numpy(float)
    loc = d.sample_location.to_numpy()
    print(f"{len(d)} samples, {len(np.unique(loc))} locations")

    def seed_preds(model, seed):
        c = pd.read_csv(find(f"ansoil_cv_predictions_{model}_{seed}.csv", model, seed))
        c = c[c.target == TARGET].set_index("sample_id")
        return c.reindex(d.sample_id).predicted.to_numpy(float)

    rows = []
    for model, name in (("rf", "Random Forest"), ("xgb", "XGBoost")):
        R = [r2(y, seed_preds(model, s)) for s in SEEDS]
        E = [rmse(y, seed_preds(model, s)) for s in SEEDS]
        rows.append((name, np.mean(R), np.std(R, ddof=1), np.mean(E)))

    k = pd.read_csv(find("ansoil_cv_predictions_knn.csv"))
    k = k[k.target == TARGET].set_index("sample_id")
    kn = k.reindex(d.sample_id).predicted.to_numpy(float)
    rows.append(("KNN", r2(y, kn), np.nan, rmse(y, kn)))

    X = d[
        [
            "proj_x_epsg3031",
            "proj_y_epsg3031",
            "wgs84_elev_from_pgc",
            "dist_coast_scar_km",
            "precipitation_racmo",
            "temperature_racmo",
            "slope_dem",
        ]
    ].copy()
    rad = np.deg2rad(d.aspect_dem)
    X["aspect_sin"], X["aspect_cos"] = np.sin(rad), np.cos(rad)
    for c in sorted(d.litho.unique()):
        X[f"litho_{c}"] = (d.litho == c).astype(int)
    for c in ("region_tm", "region_svl", "region_nvl", "region_nwap"):
        X[c] = d[c]
    X = X.to_numpy(float)
    ml = np.full(len(y), np.nan)
    for g in np.unique(loc):
        te, tr = loc == g, loc != g
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        B = np.column_stack([np.ones(te.sum()), X[te]])
        beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        ml[te] = B @ beta
    rows.append(("Multiple Linear Regression", r2(y, ml), np.nan, rmse(y, ml)))
    print(
        f"  MLR predictions run from {ml.min():.0f} to {ml.max():.0f} permil "
        f"against a measured range of {y.min():.1f} to {y.max():.1f}"
    )

    t = pd.DataFrame(rows, columns=["method", "r2", "r2_sd", "rmse"])
    t = t.sort_values("r2", ascending=False).reset_index(drop=True)
    t["vs_rf"] = t.r2 - t.loc[t.method == "Random Forest", "r2"].iloc[0]
    t["n_seeds"] = [
        len(SEEDS) if m in ("Random Forest", "XGBoost") else 1 for m in t.method
    ]
    return t, float(y.std(ddof=1)), float(ml.min())


def draw(t, y_sd, ml_min, krig):
    head = ["Method", "R²", "RMSE (permil)", "vs RF"]
    align = ["left", "right", "right", "right"]

    def cells(r):
        r2s = f"{r.r2:.3f}" + (f" ± {r.r2_sd:.3f}" if np.isfinite(r.r2_sd) else "")
        return [
            r.method,
            r2s,
            f"{r.rmse:.2f}",
            "—" if abs(r.vs_rf) < 1e-9 else f"{r.vs_rf:+.3f}",
        ]

    # built before the figure is sized, so the footer always has room for them,
    # and wrapped so a long note cannot run off the right edge
    notes = [
        f"Tree methods are the mean over {len(SEEDS)} seeds, ±1 SD. "
        f"KNN and MLR are single fits.",
        f"Predicting the overall mean would give RMSE {y_sd:.2f} permil.",
    ]
    if krig:
        notes.append(krig)
    notes = [ln for note in notes for ln in textwrap.wrap(note, NOTE_WRAP)]

    line_h = NOTE_PT / 72 * 1.45
    fig_w = MARGIN_IN * 2 + sum(COL_W_IN)
    head_h = MARGIN_IN + TITLE_PT / 72 * 1.5 + SUB_PT / 72 * 3.2
    body_h = ROW_H_IN * (len(t) + 1)
    foot_h = MARGIN_IN + NOTE_GAP_IN + line_h * len(notes)
    fig_h = head_h + body_h + foot_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    print(f"figure {fig_w:.2f} x {fig_h:.2f} in")

    def X(col, side):
        left = MARGIN_IN + sum(COL_W_IN[:col])
        return (left if side == "left" else left + COL_W_IN[col] - 0.12) / fig_w

    def Yr(i):  # row centre, 0 = header
        return 1 - (head_h + ROW_H_IN * (i + 0.5)) / fig_h

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

    for c, (label, a) in enumerate(zip(head, align)):
        fig.text(
            X(c, a),
            Yr(0),
            label,
            fontsize=HEAD_PT,
            fontweight="bold",
            color=INK,
            ha=a,
            va="center",
        )

    def rule(i, lw, colour):
        yy = 1 - (head_h + ROW_H_IN * i) / fig_h
        fig.add_artist(
            plt.Line2D(
                [MARGIN_IN / fig_w, 1 - MARGIN_IN / fig_w],
                [yy, yy],
                color=colour,
                lw=lw,
                transform=fig.transFigure,
            )
        )

    rule(1, 1.2, INK)
    for i, r in t.iterrows():
        failed = r.r2 <= 0
        for c, (txt, a) in enumerate(zip(cells(r), align)):
            fig.text(
                X(c, a),
                Yr(i + 1),
                txt,
                fontsize=CELL_PT,
                color=HILITE if failed else INK,
                fontweight="bold" if c == 0 and failed else "normal",
                ha=a,
                va="center",
            )
        if i < len(t) - 1:
            rule(i + 2, 0.6, "#E5E1DA")
    rule(len(t) + 1, 1.2, INK)

    # anchored from the bottom margin upward, so the last line always clears it
    for j, line in enumerate(notes):
        fig.text(
            MARGIN_IN / fig_w,
            (MARGIN_IN + line_h * (len(notes) - j)) / fig_h,
            line,
            fontsize=NOTE_PT,
            color=INK2,
            ha="left",
            va="top",
        )

    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "fig3_methods_table.pdf")
    fig.savefig(FIGS / "fig3_methods_table.png", dpi=200)
    print(f"wrote {FIGS / 'fig3_methods_table.pdf'} and .png")


def main():
    use_helvetica()
    t, y_sd, ml_min = compute()
    krig = kriging_note()
    print()
    print(t.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    FIGS.mkdir(exist_ok=True)
    t.to_csv(FIGS / "fig3_methods_table.csv", index=False)
    draw(t, y_sd, ml_min, krig)


if __name__ == "__main__":
    main()
