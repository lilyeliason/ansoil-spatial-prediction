"""
kriging_fixed.py
================
Regression kriging on ANSOIL model residuals, with three corrections to the
original `kriging.py`.

WHAT WAS WRONG
--------------
1. SIGN ERROR. The model scripts define

       residual = predicted - actual                  (Ansoil_rf_model.py)

   which is the opposite of the usual convention. Standard regression kriging
   interpolates (observed - predicted) and ADDS it to the trend prediction. The
   original script kept the pipeline's reversed residual and still added it:

       loo_rk_preds[i] = df["predicted"].iloc[i] + pred_resid[0]   # wrong

   so every correction pushed the prediction further from the observation. That
   is why all 13 saved comparisons showed kriging destroying R2 by 0.56 to 3.31
   units. A genuinely flat variogram produces a change near zero.

   Fixed here by flipping the residual once, at the point of definition, so the
   rest of the code reads conventionally:

       resid = actual - predicted                     # observed minus trend
       corrected = predicted + kriged(resid)

2. LEAKY CROSS-VALIDATION. Step 5 held out one SAMPLE at a time. With 53 of 171
   samples at Shackleton Glacier, a held-out sample keeps its near neighbours in
   the kriging training set, so the kriged surface interpolates from a point
   metres away. That is the spatial leakage LOLO-CV exists to prevent, and it
   favours kriging. This script reports leave-one-LOCATION-out as the primary
   result and keeps sample-level LOO alongside it for comparison.

3. HARD-CODED TO ONE TARGET, ONE SEED, WITH BLOCKING PLOTS. `TARGET` and the
   absolute input paths had to be edited by hand between runs, and two
   `plt.show()` calls blocked until the windows were closed. This script loops
   over targets and models, takes paths from arguments, and only ever savefig's.

WHAT THIS SCRIPT REPORTS
------------------------
For every model/seed/target it computes R2 four ways so the effect of each
correction is visible in one table:

    r2_trend_only        the ML model alone, no kriging
    r2_rk_original_bug   reproduces the original script, for comparison
    r2_rk_sample_loo     sign corrected, still sample-level LOO (leaky)
    r2_rk_location_lolo  sign corrected, leave-one-location-out  <-- the answer

Plus variogram diagnostics (nugget, partial sill, nugget:sill, range, and
whether semivariance rises with distance) fitted on the same residuals.

Usage
-----
    python scripts/kriging_fixed.py --root .
    python scripts/kriging_fixed.py --root . --targets d15n_air_permil,wt_percent_n
    python scripts/kriging_fixed.py --root . --variogram all --figures
    python scripts/kriging_fixed.py --root . --with-grid      # write corrected maps

Requires: pandas, numpy, pykrige, scikit-learn, matplotlib (only with --figures)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import pandas as pd
    from pykrige.ok import OrdinaryKriging
except ModuleNotFoundError as exc:
    sys.exit(
        f"\nMissing dependency: {exc.name}\n"
        f"Interpreter in use: {sys.executable}\n\n"
        "  python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  pip install pandas numpy pykrige scikit-learn matplotlib\n"
    )

SEEDS_DEFAULT = [42]
MODELS_DEFAULT = ["rf", "xgb"]
VARIOGRAM_CHOICES = ["spherical", "exponential", "gaussian"]

# Nugget:sill above which the variogram is treated as effectively flat. This is
# the original script's threshold, kept so the verdict is comparable.
FLATNESS_NS_THRESHOLD = 0.85

# Kriging improves on the trend only if it beats it by more than this.
IMPROVEMENT_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def r2_score(actual, predicted) -> float:
    a = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    ss_tot = np.sum((a - a.mean()) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - np.sum((a - p) ** 2) / ss_tot)


def rmse(actual, predicted) -> float:
    a = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


class Paths:
    def __init__(self, root: str, flat_dir: str | None, out_dir: str | None):
        self.root = os.path.abspath(root)
        self.flat = os.path.abspath(flat_dir) if flat_dir else None
        self.out = os.path.abspath(
            out_dir or os.path.join(self.root, "results", "kriging_fixed")
        )
        os.makedirs(self.out, exist_ok=True)

    def find(self, name, model=None, seed=None, required=True):
        cands = []
        if model and seed is not None:
            cands.append(os.path.join(self.root, "results", f"{model}_seed{seed}", name))
        cands += [
            os.path.join(self.root, "data", name),
            os.path.join(self.root, "results", "aggregated", name),
            os.path.join(self.root, name),
        ]
        if self.flat:
            cands.append(os.path.join(self.flat, name))
        for c in cands:
            if os.path.exists(c):
                return c
        if required:
            raise FileNotFoundError(
                f"Could not find {name}. Looked in:\n  " + "\n  ".join(cands)
            )
        return None

    def out_path(self, name):
        return os.path.join(self.out, name)


# ---------------------------------------------------------------------------
# Variogram
# ---------------------------------------------------------------------------


def fit_variogram(x, y, z, model: str):
    """Fit one variogram and return its parameters plus a flatness verdict."""
    ok = OrdinaryKriging(
        x, y, z, variogram_model=model, verbose=False, enable_plotting=False
    )
    # PyKrige returns [partial_sill, range, nugget] for these three models.
    psill, vrange, nugget = ok.variogram_model_parameters
    total_sill = psill + nugget
    ns_ratio = nugget / total_sill if total_sill > 0 else 1.0
    slope = float(np.polyfit(ok.lags, ok.semivariance, 1)[0])
    is_flat = bool(ns_ratio > FLATNESS_NS_THRESHOLD or slope <= 0)
    # A range that rivals the dataset's own extent means the model is fitting a
    # broad trend rather than local structure, and is close to unidentifiable
    # from a few clustered locations.
    extent = float(max(x.max() - x.min(), y.max() - y.min()))
    return ok, {
        "variogram_model": model,
        "nugget": float(nugget),
        "partial_sill": float(psill),
        "total_sill": float(total_sill),
        "nugget_sill_ratio": float(ns_ratio),
        "range_km": float(vrange) / 1000.0,
        "dataset_extent_km": extent / 1000.0,
        "range_over_extent": float(vrange) / extent if extent > 0 else float("nan"),
        "semivariance_slope": slope,
        "verdict": "flat / pure nugget" if is_flat else "spatial structure present",
        "range_exceeds_extent": bool(vrange >= extent),
    }


def best_variogram(x, y, z, models):
    """Fit each candidate model and keep the one with the lowest residual sum of
    squares against the experimental variogram."""
    best = None
    all_fits = []
    for m in models:
        try:
            ok, info = fit_variogram(x, y, z, m)
        except Exception as exc:  # pykrige can fail to converge on flat data
            all_fits.append({"variogram_model": m, "error": str(exc)})
            continue
        fitted = ok.variogram_function(ok.variogram_model_parameters, ok.lags)
        info["fit_rss"] = float(np.sum((ok.semivariance - fitted) ** 2))
        all_fits.append(info)
        if best is None or info["fit_rss"] < best[1]["fit_rss"]:
            best = (ok, info)
    if best is None:
        raise RuntimeError("no variogram model could be fitted")
    return best[0], best[1], all_fits


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def krige_holdout(x, y, resid, train_mask, test_mask, model: str):
    """Fit a variogram on the training residuals only and predict the held-out
    points. Refitting per fold is what makes this a real cross-validation."""
    try:
        ok = OrdinaryKriging(
            x[train_mask],
            y[train_mask],
            resid[train_mask],
            variogram_model=model,
            verbose=False,
            enable_plotting=False,
        )
        pred, _ = ok.execute("points", x[test_mask], y[test_mask])
        return np.asarray(pred, float)
    except Exception:
        # A fold whose residuals will not support a variogram contributes a zero
        # correction, which is the neutral outcome rather than a crash.
        return np.zeros(int(test_mask.sum()))


def sample_level_loo(x, y, resid, model):
    """Leave one SAMPLE out. Retained only for comparison with the original
    script: with tightly clustered samples this leaks and favours kriging."""
    n = len(resid)
    out = np.zeros(n)
    for i in range(n):
        tr = np.ones(n, bool)
        tr[i] = False
        te = ~tr
        out[i] = krige_holdout(x, y, resid, tr, te, model)[0]
    return out


def location_level_lolo(x, y, resid, locations, model):
    """Leave one LOCATION out, matching the LOLO-CV used everywhere else in the
    pipeline. This is the defensible test."""
    out = np.zeros(len(resid))
    for loc in pd.unique(locations):
        te = locations == loc
        tr = ~te
        out[te] = krige_holdout(x, y, resid, tr, te, model)
    return out


# ---------------------------------------------------------------------------
# One target
# ---------------------------------------------------------------------------


def run_one(paths, model, seed, target, variogram_models, want_figures, want_grid):
    cv = pd.read_csv(
        paths.find(f"ansoil_cv_predictions_{model}_{seed}.csv", model, seed)
    )
    if target not in set(cv["target"]):
        return None, f"target not present in {model} seed {seed}"

    idx = pd.read_csv(paths.find("ansoil_sample_index.csv"))
    d = cv[cv["target"] == target].merge(
        idx[["sample_id", "sample_location", "proj_x_epsg3031", "proj_y_epsg3031"]],
        on="sample_id",
        how="left",
    )
    d = d.dropna(subset=["proj_x_epsg3031", "proj_y_epsg3031"])
    if len(d) < 20:
        return None, f"only {len(d)} samples with coordinates"

    x = d["proj_x_epsg3031"].to_numpy(float)
    y = d["proj_y_epsg3031"].to_numpy(float)
    actual = d["actual"].to_numpy(float)
    trend = d["predicted"].to_numpy(float)
    locations = d["sample_location"].to_numpy()

    # ---- THE SIGN FIX -----------------------------------------------------
    # The pipeline's `residual` column is predicted - actual. Kriging needs
    # observed - predicted, so flip it once here and add it everywhere below.
    resid = actual - trend
    pipeline_resid = d["residual"].to_numpy(float)
    assert np.allclose(pipeline_resid, -resid, atol=1e-8), (
        "The pipeline's residual column is not predicted - actual. Check the "
        "convention in the model script before trusting this run."
    )

    ok_global, vinfo, all_fits = best_variogram(x, y, resid, variogram_models)
    vmodel = vinfo["variogram_model"]

    # ---- the three corrections, side by side ------------------------------
    k_sample = sample_level_loo(x, y, resid, vmodel)
    k_location = location_level_lolo(x, y, resid, locations, vmodel)

    # The original script's arithmetic: pipeline residual (predicted - actual),
    # still added. Reproduced so the comparison table shows what changed.
    rk_bug = trend + (-k_sample)
    rk_sample = trend + k_sample
    rk_location = trend + k_location

    row = {
        "model": model,
        "seed": seed,
        "target": target,
        "n_samples": len(d),
        "n_locations": int(pd.Series(locations).nunique()),
        "r2_trend_only": r2_score(actual, trend),
        "r2_rk_original_bug": r2_score(actual, rk_bug),
        "r2_rk_sample_loo": r2_score(actual, rk_sample),
        "r2_rk_location_lolo": r2_score(actual, rk_location),
        "rmse_trend_only": rmse(actual, trend),
        "rmse_rk_location_lolo": rmse(actual, rk_location),
        **vinfo,
    }
    row["delta_original_bug"] = row["r2_rk_original_bug"] - row["r2_trend_only"]
    row["delta_sample_loo"] = row["r2_rk_sample_loo"] - row["r2_trend_only"]
    row["delta_location_lolo"] = row["r2_rk_location_lolo"] - row["r2_trend_only"]
    row["kriging_helps_lolo"] = bool(
        row["delta_location_lolo"] > IMPROVEMENT_THRESHOLD
    )
    row["conclusion"] = (
        "kriging improves the trend"
        if row["delta_location_lolo"] > IMPROVEMENT_THRESHOLD
        else "no meaningful improvement"
        if row["delta_location_lolo"] > -IMPROVEMENT_THRESHOLD
        else "kriging degrades the trend"
    )

    if want_figures:
        write_variogram_figure(paths, ok_global, vinfo, model, seed, target)

    if want_grid and row["kriging_helps_lolo"]:
        write_corrected_grid(paths, ok_global, model, seed, target)

    return row, pd.DataFrame(all_fits)


def write_variogram_figure(paths, ok, vinfo, model, seed, target):
    import matplotlib

    matplotlib.use("Agg")  # no window, no blocking
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(
        ok.lags, ok.semivariance, color="#c1443c", marker="o", s=28, zorder=5,
        label="Experimental",
    )
    lags_line = np.linspace(0, ok.lags.max() * 1.1, 300)
    ax.plot(
        lags_line,
        ok.variogram_function(ok.variogram_model_parameters, lags_line),
        color="#1f2933",
        lw=1.5,
        label=f"Fitted {vinfo['variogram_model']}",
    )
    ax.axhline(vinfo["total_sill"], color="#7b8794", ls=":", lw=1, label="Total sill")
    ax.axhline(vinfo["nugget"], color="#2f6f9f", ls=":", lw=1, label="Nugget")
    ax.axvline(
        vinfo["range_km"] * 1000,
        color="#3f8f5f",
        ls=":",
        lw=1,
        label=f"Range ({vinfo['range_km']:.0f} km)",
    )
    ax.set_xlabel("Separation distance")
    ax.set_ylabel("Semivariance")
    ax.set_title(
        f"{target} residuals, {model.upper()} seed {seed}\n"
        f"nugget:sill = {vinfo['nugget_sill_ratio']:.3f}   "
        f"range/extent = {vinfo['range_over_extent']:.2f}   {vinfo['verdict']}",
        fontsize=10,
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:.0f} km"))
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(True, ls="--", alpha=0.35)
    fig.tight_layout()
    p = paths.out_path(f"variogram_{model}_{seed}_{target}.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)


def write_corrected_grid(paths, ok, model, seed, target):
    """Only called when LOLO says kriging actually helps."""
    gp = paths.find(
        f"ansoil_grid_predictions_{model}_{seed}.csv", model, seed, required=False
    )
    gc = paths.find("ansoil_grid_prepared.csv", required=False)
    if not gp or not gc:
        return
    grid = pd.read_csv(gp)
    col = f"pred_{target}"
    if col not in grid.columns:
        return
    coords = pd.read_csv(
        gc, usecols=["grid_id", "lat", "lon", "proj_x_epsg3031", "proj_y_epsg3031"]
    )
    m = grid[["grid_id", col]].merge(coords, on="grid_id", how="inner")
    kr, var = ok.execute(
        "points", m["proj_x_epsg3031"].to_numpy(float), m["proj_y_epsg3031"].to_numpy(float)
    )
    m["kriged_residual"] = np.asarray(kr, float)
    m["kriging_variance"] = np.asarray(var, float)
    m["regression_kriging"] = m[col] + m["kriged_residual"]
    m.to_csv(
        paths.out_path(f"grid_regression_kriging_{model}_{seed}_{target}.csv"),
        index=False,
    )


# ---------------------------------------------------------------------------
# Target discovery
# ---------------------------------------------------------------------------


def discover_targets(paths, arg: str | None) -> list:
    if arg and arg not in ("existing", "mappable"):
        return [t.strip() for t in arg.split(",") if t.strip()]

    if arg in (None, "existing"):
        # The 13 targets the original script was already run on.
        found = set()
        for pattern in ("table_loo_cv_*.csv",):
            for base in (paths.out, paths.root, paths.flat or paths.root):
                for f in glob.glob(os.path.join(base, "**", pattern), recursive=True):
                    stem = os.path.basename(f)[len("table_loo_cv_") : -len(".csv")]
                    for m in MODELS_DEFAULT:
                        if stem.startswith(m + "_"):
                            found.add(stem[len(m) + 1 :])
        if found:
            return sorted(found)

    # Fall back to the mappable set from ml_results_clean.py.
    for base in (
        os.path.join(paths.root, "results", "clean"),
        paths.root,
        paths.flat or paths.root,
    ):
        p = os.path.join(base, "verified_mappable.csv")
        if os.path.exists(p):
            return sorted(pd.read_csv(p)["property"].tolist())

    sys.exit(
        "Could not work out which targets to run.\n"
        "Pass them explicitly:  --targets d15n_air_permil,wt_percent_n\n"
        "Or run ml_results_clean.py first so verified_mappable.csv exists."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Regression kriging on ANSOIL residuals, sign and CV corrected."
    )
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    ap.add_argument("--flat-dir", default=None, help="fallback folder holding every CSV")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--targets",
        default=None,
        help="comma-separated names, or 'existing' (the 13 already run), "
        "or 'mappable' (from verified_mappable.csv)",
    )
    ap.add_argument(
        "--models", default=",".join(MODELS_DEFAULT), help="comma-separated: rf,xgb"
    )
    ap.add_argument(
        "--seeds", default=",".join(map(str, SEEDS_DEFAULT)), help="comma-separated"
    )
    ap.add_argument(
        "--variogram",
        default="spherical",
        help="spherical | exponential | gaussian | all (fits all three, keeps best fit)",
    )
    ap.add_argument("--figures", action="store_true", help="write variogram PNGs")
    ap.add_argument(
        "--with-grid",
        action="store_true",
        help="write corrected grids, only for targets where LOLO says kriging helps",
    )
    args = ap.parse_args(argv)

    paths = Paths(args.root, args.flat_dir, args.out_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    vmodels = (
        VARIOGRAM_CHOICES if args.variogram == "all" else [args.variogram]
    )
    targets = discover_targets(paths, args.targets)

    print(f"Repo root: {paths.root}")
    print(f"Output:    {paths.out}")
    print(f"Models: {models}   Seeds: {seeds}   Variogram: {vmodels}")
    print(f"Targets ({len(targets)}): {', '.join(targets)}\n")

    rows, fits = [], []
    for model in models:
        for seed in seeds:
            for target in targets:
                try:
                    row, extra = run_one(
                        paths, model, seed, target, vmodels, args.figures,
                        args.with_grid,
                    )
                except FileNotFoundError as exc:
                    print(f"  SKIP {model} {seed} {target}: {exc}")
                    continue
                if row is None:
                    print(f"  SKIP {model} {seed} {target}: {extra}")
                    continue
                rows.append(row)
                if isinstance(extra, pd.DataFrame) and len(extra):
                    extra = extra.copy()
                    extra["model"], extra["seed"], extra["target"] = model, seed, target
                    fits.append(extra)
                print(
                    f"  {model:<4} {seed:<4} {target:<26} "
                    f"trend {row['r2_trend_only']:+.4f} | "
                    f"bug {row['delta_original_bug']:+.4f} | "
                    f"sample-LOO {row['delta_sample_loo']:+.4f} | "
                    f"LOLO {row['delta_location_lolo']:+.4f}  "
                    f"[{row['verdict']}]"
                )

    if not rows:
        sys.exit("\nNothing ran. Check --targets, --models and --seeds.")

    res = pd.DataFrame(rows)
    res.to_csv(paths.out_path("kriging_comparison.csv"), index=False)
    if fits:
        pd.concat(fits, ignore_index=True).to_csv(
            paths.out_path("variogram_fits.csv"), index=False
        )

    # ---- summary ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    n = len(res)
    print(f"\n  {n} model/seed/target combinations\n")
    print("  Effect of each correction on mean delta R2:")
    print(f"    original script (sign error, sample LOO) : {res['delta_original_bug'].mean():+.4f}")
    print(f"    sign corrected, sample-level LOO (leaky) : {res['delta_sample_loo'].mean():+.4f}")
    print(f"    sign corrected, leave-one-location-out   : {res['delta_location_lolo'].mean():+.4f}")

    print("\n  Verdict under LOLO, the defensible test:")
    for label, count in res["conclusion"].value_counts().items():
        print(f"    {label}: {count} of {n}")

    helps = res[res["kriging_helps_lolo"]]
    if len(helps):
        print(
            f"\n  Kriging clears the +{IMPROVEMENT_THRESHOLD} threshold on "
            f"{len(helps)} of {n}:"
        )
        for _, r in helps.sort_values("delta_location_lolo", ascending=False).iterrows():
            print(
                f"    {r['model']} seed {r['seed']} {r['target']}: "
                f"{r['r2_trend_only']:+.4f} -> {r['r2_rk_location_lolo']:+.4f} "
                f"({r['delta_location_lolo']:+.4f})"
            )
    else:
        print("\n  Kriging does not improve any target under LOLO.")

    print("\n  Variogram diagnostics:")
    flat = int((res["verdict"] == "flat / pure nugget").sum())
    print(f"    flat / pure nugget           : {flat} of {n}")
    print(f"    spatial structure present    : {n - flat} of {n}")
    print(f"    median nugget:sill ratio     : {res['nugget_sill_ratio'].median():.3f}")
    print(f"    range >= dataset extent      : {int(res['range_exceeds_extent'].sum())} of {n}")
    print(f"    median range / extent        : {res['range_over_extent'].median():.2f}")
    if res["range_over_extent"].median() > 0.5:
        print(
            "\n    NOTE: fitted ranges rival the dataset's own extent, so the\n"
            "    variogram is describing a broad continental trend rather than\n"
            "    local spatial structure. With 28 clustered locations that fit is\n"
            "    close to unidentifiable. Treat the range as weak evidence."
        )

    print(f"\n  Wrote: {paths.out_path('kriging_comparison.csv')}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

