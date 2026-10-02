"""
Four modelling pipelines, three validation schemes, one dataset.

Separates the effect of the MODEL from the effect of how the score was taken.
Every model sees identical folds, identical features and identical targets, and
every model is scored the same three ways:

  in_sample      fit on all 171 samples, score on the same 171.
                 This is what Willmore (2024) reports: "the data was not split
                 into a training and test set."
  random_10fold  shuffled 10-fold CV, which ignores that samples cluster by
                 location. Near-neighbours of most test samples stay in training.
  LOLO           leave-one-location-out, 28 folds. Every sample from a location
                 is withheld together. This is what the ANSOIL pipeline reports.

and in two spaces:

  log            the log1p space the model was fitted in, which is where both
                 Willmore's R2 and the ANSOIL headline numbers live
  native         back-transformed to the measured units

Hyperparameters are FIXED, not searched, so no model gets an advantage from a
bigger tuning budget. That isolates model family and validation scheme as the
only things varying.

Run from the repo root:  python3 pipeline_comparison.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

from ansoil_paths import require, resolve

_DATA, _OUT = resolve(
    description="Compare MLR, KNN, RF and XGB across validation schemes"
)
DATA, OUT = str(_DATA), str(_OUT)
SEED = 42
NUMERIC = [
    "proj_x_epsg3031",
    "proj_y_epsg3031",
    "wgs84_elev_from_pgc",
    "dist_coast_scar_km",
    "precipitation_racmo",
    "temperature_racmo",
    "slope_dem",
]

# A panel spanning the geochemical classes, with the transform each one gets.
# log=True means the model is fitted on log1p(y) and back-transformed with expm1,
# matching both pipelines' handling of skewed data.
PANEL = [
    ("d15n_air_permil", False, "isotope", "d15N"),
    ("wt_percent_c", True, "organic", "weight % C"),
    ("wt_percent_n", True, "organic", "weight % N"),
    ("c_n_ratio", True, "organic", "C:N ratio"),
    ("ph_mq", False, "bulk", "pH"),
    ("cec_meq_100g", True, "bulk", "CEC"),
    ("ec_us_cm", True, "bulk", "EC"),
    ("total_mg_l_na", True, "soluble ion", "soluble Na"),
    ("total_mg_l_ca2", True, "soluble ion", "soluble Ca"),
    ("total_mg_l_mg2", True, "soluble ion", "soluble Mg"),
    ("total_mg_l_so4", True, "soluble ion", "soluble SO4"),
    ("digest_mg_kg_fe2599", True, "digest metal", "digest Fe"),
    ("digest_mg_kg_ni2316", False, "digest metal", "digest Ni"),
]


def r2(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    if len(a) < 2:
        return np.nan
    sst = ((a - a.mean()) ** 2).sum()
    return np.nan if sst == 0 else 1.0 - ((a - p) ** 2).sum() / sst


def models():
    """Fresh estimators. Fixed settings, no hyperparameter search for anyone."""
    return {
        "MLR": ("ols", None),
        "KNN": ("scaled", KNeighborsRegressor(n_neighbors=7, weights="distance")),
        "RF": (
            "raw",
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        "XGB": (
            "raw",
            XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=SEED,
                n_jobs=-1,
                verbosity=0,
            ),
        ),
    }


def fit_predict(kind, est, Xtr, ytr, Xte):
    if kind == "ols":
        A = np.column_stack([np.ones(len(Xtr)), Xtr])
        B = np.column_stack([np.ones(len(Xte)), Xte])
        beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        return B @ beta
    if kind == "scaled":
        s = StandardScaler().fit(Xtr)
        est.fit(s.transform(Xtr), ytr)
        return est.predict(s.transform(Xte))
    est.fit(Xtr, ytr)
    return est.predict(Xte)


def build_design(df):
    X = df[NUMERIC].copy()
    rad = np.deg2rad(df["aspect_dem"])
    X["aspect_sin"], X["aspect_cos"] = np.sin(rad), np.cos(rad)
    for c in sorted(df.litho.unique()):
        X[f"litho_{c}"] = (df.litho == c).astype(int)
    return X.to_numpy(float), list(X.columns)


def schemes(loc, n):
    """The three ways of splitting, as a dict of name -> list of (train, test)."""
    out = {"in_sample": [(np.arange(n), np.arange(n))]}
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    out["random_10fold"] = list(kf.split(np.arange(n)))
    lolo = []
    for g in np.unique(loc):
        te = np.where(loc == g)[0]
        tr = np.where(loc != g)[0]
        lolo.append((tr, te))
    out["LOLO"] = lolo
    return out


def main():
    require(
        _DATA, ["ansoil_predictors.csv", "ansoil_targets.csv"], "pipeline_comparison.py"
    )
    P = pd.read_csv(f"{DATA}/ansoil_predictors.csv")
    T = pd.read_csv(f"{DATA}/ansoil_targets.csv")
    df = P.merge(T, on="sample_id", how="inner", validate="one_to_one")
    X, feat = build_design(df)
    loc = df.sample_location.to_numpy()
    print(
        "%d samples, %d locations, %d features" % (len(df), len(set(loc)), X.shape[1])
    )
    print("features: %s" % ", ".join(feat))
    print("\nfixed settings: MLR = OLS | KNN k=7 distance-weighted on standardised X")
    print(
        "                RF = 300 trees, sqrt features | XGB = 300 rounds, depth 4, lr 0.05"
    )

    rows = []
    for target, uselog, klass, nice in PANEL:
        if target not in df.columns:
            print("  skipping %s, not in targets file" % target)
            continue
        y_native = df[target].to_numpy(float)
        ok = np.isfinite(y_native) & np.isfinite(X).all(axis=1)
        Xo, yn, lo = X[ok], y_native[ok], loc[ok]
        if uselog and (yn < 0).any():
            uselog = False
        yfit = np.log1p(yn) if uselog else yn
        sch = schemes(lo, len(yn))

        for mname, (kind, _) in models().items():
            for sname, folds in sch.items():
                pred = np.full(len(yfit), np.nan)
                for tr, te in folds:
                    if len(tr) <= 2:
                        continue
                    est = models()[mname][1]
                    pred[te] = fit_predict(kind, est, Xo[tr], yfit[tr], Xo[te])
                back = np.expm1(pred) if uselog else pred
                rows.append(
                    {
                        "class": klass,
                        "property": nice,
                        "target": target,
                        "log_fitted": uselog,
                        "model": mname,
                        "validation": sname,
                        "r2_fit_space": r2(yfit, pred),
                        "r2_native": r2(yn, back),
                        "r2_native_floored": max(r2(yn, back), -1.0),
                        "blowup": float(
                            np.nanmax(np.abs(back)) / max(np.nanmax(np.abs(yn)), 1e-9)
                        ),
                        "n": int(np.isfinite(yfit).sum()),
                    }
                )
        print("  done %-14s (%s)" % (nice, klass))

    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/ansoil_pipeline_comparison.csv", index=False)

    # ---------------------------------------------------------------- summary
    def block(title, value_col, note=""):
        print("\n" + "=" * 92)
        print(title)
        if note:
            print(note)
        print("=" * 92)
        t = R.pivot_table(
            index=["class", "property"],
            columns=["validation", "model"],
            values=value_col,
            aggfunc="mean",
        )
        cols = [
            (v, m)
            for v in ("in_sample", "random_10fold", "LOLO")
            for m in ("MLR", "KNN", "RF", "XGB")
        ]
        t = t.reindex(columns=pd.MultiIndex.from_tuples(cols))
        print(t.round(2).to_string())

    block(
        "R2 in the space each model was fitted in (log1p where the property is skewed)",
        "r2_fit_space",
        "in_sample is how Willmore (2024) reports; LOLO is how ANSOIL reports",
    )
    block(
        "R2 in the measured units, after back-transforming (floored at -1)",
        "r2_native_floored",
        "a floor is needed because expm1 of an extrapolating linear fit explodes;\n"
        "the unfloored values are in the CSV",
    )

    print("\n" + "=" * 92)
    print("mean across the panel, the headline of this whole exercise")
    print("=" * 92)
    piv = R.pivot_table(
        index="model",
        columns="validation",
        values=["r2_fit_space", "r2_native_floored"],
        aggfunc="mean",
    )
    piv = piv.reindex(
        columns=pd.MultiIndex.from_product(
            [
                ["r2_fit_space", "r2_native_floored"],
                ["in_sample", "random_10fold", "LOLO"],
            ]
        )
    )
    print(piv.round(3).to_string())
    print("\ntwo artifacts worth naming before anyone quotes this table:")
    kb = R[(R.model == "KNN") & (R.validation == "in_sample")].r2_fit_space.mean()
    print("  KNN in_sample averages %.3f because a distance-weighted neighbour" % kb)
    print("  search finds the point itself at distance zero. An in-sample score is")
    print("  not a measure of anything for KNN, and barely one for RF and XGB.")
    bl = R[R.validation == "LOLO"].groupby("model").blowup.max()
    print("  largest back-transformed prediction as a multiple of the largest")
    print(
        "  measured value, under LOLO: %s"
        % ", ".join("%s %.0fx" % (k, v) for k, v in bl.items())
    )
    print("  MLR is the one that explodes: a linear fit extrapolated to an unseen")
    print("  location, then run through expm1, produces impossible concentrations.")

    print("\nspread attributable to each choice, averaged over the panel:")
    m_eff = R.groupby("model").r2_fit_space.mean()
    v_eff = R.groupby("validation").r2_fit_space.mean()
    print(
        "  choice of MODEL      spans %.3f  (%s %.3f to %s %.3f)"
        % (
            m_eff.max() - m_eff.min(),
            m_eff.idxmin(),
            m_eff.min(),
            m_eff.idxmax(),
            m_eff.max(),
        )
    )
    print(
        "  choice of VALIDATION spans %.3f  (%s %.3f to %s %.3f)"
        % (
            v_eff.max() - v_eff.min(),
            v_eff.idxmin(),
            v_eff.min(),
            v_eff.idxmax(),
            v_eff.max(),
        )
    )

    print("\n" + "=" * 92)
    print("by geochemical class, under LOLO in measured units")
    print("=" * 92)
    cl = R[(R.validation == "LOLO")].pivot_table(
        index="class", columns="model", values="r2_native_floored", aggfunc="mean"
    )
    print(cl.reindex(columns=["MLR", "KNN", "RF", "XGB"]).round(3).to_string())

    # ------------------------------------------- Willmore's own LOOCV, decoded
    print("\n" + "=" * 92)
    print("Willmore (2024) reports LOOCV RMSE next to the measured SD in five cases.")
    print("R2 = 1 - (RMSE/SD)^2 turns those into the same currency as the fit R2.")
    print("=" * 92)
    wil = [
        ("All Regions", "EC", 0.37, 4752, 4657),
        ("TAM", "pH", 0.21, 0.97, 1.23),
        ("SVL", "C:N", 0.83, 13.27, 19.51),
        ("SVL", "weight % C", 0.76, 0.24, 0.32),
        ("SVL", "pH", 0.64, 0.75, 1.18),
        ("SVL", "CEC", 0.57, 1.87, 2.85),
    ]
    print(
        "  %-12s %-12s %8s %10s %8s %10s %9s"
        % ("region", "property", "fit R2", "LOOCV RMSE", "SD", "implied R2", "drop")
    )
    for reg, prop, fit, rmse, sd in wil:
        implied = 1 - (rmse / sd) ** 2
        print(
            "  %-12s %-12s %8.2f %10.2f %8.2f %10.3f %9.3f"
            % (reg, prop, fit, rmse, sd, implied, implied - fit)
        )
    print("\n  note: this conversion assumes RMSE and SD are in the same space and")
    print("  over the same samples, which the thesis implies but does not state.")
    print("  Their LOOCV also withholds ONE SAMPLE, not one location, and their")
    print("  sites hold samples 100 to 500 m apart, so it still leaks neighbours.")

    pd.DataFrame(
        wil, columns=["region", "property", "fit_r2", "loocv_rmse", "measured_sd"]
    ).assign(implied_loocv_r2=lambda d: 1 - (d.loocv_rmse / d.measured_sd) ** 2).to_csv(
        f"{OUT}/ansoil_willmore_loocv_decoded.csv", index=False
    )


if __name__ == "__main__":
    main()
