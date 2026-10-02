"""
Why the Shackleton MLR poster reports R2 around 0.6-0.7 for the same water-soluble
ions that this pipeline scores near zero.

Fits an ordinary least squares multiple linear regression on THIS dataset, using
the same predictor set and the same log transform that poster describes, and
scores it four ways. If the gap is explained by how the score was taken rather
than by the model, the in-sample log-space number should land near the poster's
and collapse under leave-one-location-out and again in measured units.

Run from the repo root:  python3 mlr_comparison.py
"""
import numpy as np
import pandas as pd

from ansoil_paths import resolve, require

_DATA, _OUT = resolve(description="Reproduce the MLR poster numbers on ANSOIL data")
DATA, OUT = str(_DATA), str(_OUT)
# The soil-forming factors that poster lists as independent variables.
PRED = ["lat", "lon", "wgs84_elev_from_pgc", "dist_coast_scar_km",
        "slope_dem", "aspect_dem", "precipitation_racmo", "temperature_racmo"]
# The four properties that poster maps, as named in this dataset.
PROPS = [("total_mg_l_na", "log_total_mg_l_na", "water-soluble Na (mg/L)", 0.72, 0.6798),
         ("total_mg_l_mg2", "log_total_mg_l_mg2", "water-soluble Mg (mg/L)", 0.62, 0.5355),
         ("total_mg_l_ca2", "log_total_mg_l_ca2", "water-soluble Ca (mg/L)", 0.57, 0.5173),
         ("ec_us_cm", "log_ec_us_cm", "electrical conductivity (uS/cm)", 0.73, 0.6783)]


def r2(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    if len(a) < 2:
        return np.nan
    sst = ((a - a.mean()) ** 2).sum()
    return np.nan if sst == 0 else 1.0 - ((a - p) ** 2).sum() / sst


def design(df):
    X = df[PRED].copy()
    # aspect is circular, so it enters as sin/cos exactly as the RF pipeline does
    rad = np.deg2rad(X.pop("aspect_dem"))
    X["aspect_sin"], X["aspect_cos"] = np.sin(rad), np.cos(rad)
    X = X.to_numpy(float)
    return np.column_stack([np.ones(len(X)), X])


def ols_fit_predict(Xtr, ytr, Xte):
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return Xte @ beta


def run(df, label):
    loc = df.sample_location.to_numpy()
    X = design(df)
    print("\n" + "-" * 78)
    print("%s   n = %d samples, %d locations, %d predictors"
          % (label, len(df), len(set(loc)), X.shape[1] - 1))
    print("-" * 78)
    print("%-32s %9s %9s %9s %9s %9s"
          % ("property", "poster", "in-samp", "LOLO", "LOLO", "RF LOLO"))
    print("%-32s %9s %9s %9s %9s %9s"
          % ("", "scatter", "log", "log", "native", "native"))
    out = []
    for raw, logc, nice, poster_scatter, poster_map in PROPS:
        y_native = df[raw].to_numpy(float)
        y_log = df[logc].to_numpy(float)
        ok = np.isfinite(y_log) & np.isfinite(y_native) & np.isfinite(X).all(axis=1)
        Xo, yl, yn, lo = X[ok], y_log[ok], y_native[ok], loc[ok]

        insample = r2(yl, ols_fit_predict(Xo, yl, Xo))

        pred_log = np.full(len(yl), np.nan)
        for g in np.unique(lo):
            te = lo == g
            tr = ~te
            if tr.sum() <= Xo.shape[1]:
                continue
            pred_log[te] = ols_fit_predict(Xo[tr], yl[tr], Xo[te])
        lolo_log = r2(yl, pred_log)
        lolo_native = r2(yn, np.expm1(pred_log))

        out.append({"property": nice, "poster_scatter_r2": poster_scatter,
                    "poster_map_r2": poster_map, "mlr_insample_log": insample,
                    "mlr_lolo_log": lolo_log, "mlr_lolo_native": lolo_native})
        print("%-32s %9.2f %9.3f %9.3f %9.3f %9s"
              % (nice, poster_scatter, insample, lolo_log, lolo_native, "see below"))
    return pd.DataFrame(out)


def main():
    require(_DATA, ["ansoil_predictors.csv", "ansoil_targets.csv",
                   "model_comparison.csv"], "mlr_comparison.py")
    P = pd.read_csv(f"{DATA}/ansoil_predictors.csv")
    T = pd.read_csv(f"{DATA}/ansoil_targets.csv")
    df = P.merge(T, on="sample_id", how="inner", validate="one_to_one")
    print("merged %d samples" % len(df))

    tables = [run(df, "ALL REGIONS, the dataset this pipeline models")]
    tam = df[df.region_tm == 1]
    tables.append(run(tam, "TRANSANTARCTIC MOUNTAINS ONLY, closer to the poster's area"))

    print("\n" + "=" * 78)
    print("what this pipeline reports for the same four properties (5-seed mean)")
    print("=" * 78)
    mc = pd.read_csv(f"{DATA}/model_comparison.csv").set_index("target")
    for raw, logc, nice, _, _ in PROPS:
        row = []
        for name in (raw, logc):
            if name in mc.index:
                s = mc.loc[name]
                row.append("%s: rf %.3f, xgb %.3f" % (name, s.rf_r2_mean, s.xgb_r2_mean))
        print("  %-32s %s" % (nice, " | ".join(row) if row else "not in model_comparison"))

    print("\n" + "=" * 78)
    print("how many of the 171 samples sit at each location")
    print("=" * 78)
    c = df.sample_location.value_counts()
    print("  %d locations, samples per location: min %d, median %d, max %d"
          % (len(c), c.min(), int(c.median()), c.max()))
    print("  a random 80/20 split would put a near-neighbour of almost every test")
    print("  sample into the training set; leave-one-location-out does not")

    pd.concat(tables, keys=["all_regions", "tam_only"]).to_csv(
        f"{OUT}/ansoil_mlr_comparison.csv")


if __name__ == "__main__":
    main()

