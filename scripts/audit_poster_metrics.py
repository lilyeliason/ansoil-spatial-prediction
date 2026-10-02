"""
Independent audit of every metric behind the GSA poster panels.

Recomputes R2 from the raw per-sample cross-validation predictions and checks
it against every summary file in the chain, without using sklearn or any
metric the pipeline itself computed:

  M1  ansoil_cv_predictions_<model>_<seed>.csv -> cv_r2 in ansoil_model_results_*
  M2  ansoil_model_results_*                   -> metrics_summary_<model>.csv
  M3  metrics_summary_*                         -> model_comparison.csv
  M4  model_comparison.csv                      -> summary_table_mappable.csv
  M5  tier labels reproduce from their own thresholds
  M6  the transform-space problem, quantified target by target
  M7  KNN results reproduce from their own CV predictions
  M8  structural integrity of every summary file

R2 is computed as 1 - SSE/SST over the pooled leave-one-location-out
predictions, which is the definition the pipeline used (established in M1).

Run from the repo root:  python3 audit_poster_metrics.py
"""
import numpy as np
import pandas as pd

from ansoil_paths import resolve, require

_DATA, _OUT = resolve(description="Audit every published ANSOIL metric")
DATA, OUT = str(_DATA), str(_OUT)
SEEDS = [7, 42, 73, 123, 256]
MODELS = ["rf", "xgb"]
require(_DATA,
        [f"ansoil_{k}_{m}_{s}.csv" for k in ("model_results", "cv_predictions")
         for m in MODELS for s in SEEDS]
        + [f"metrics_summary_{m}.csv" for m in MODELS]
        + ["model_comparison.csv", "summary_table_mappable.csv",
           "ansoil_model_results_knn.csv", "ansoil_cv_predictions_knn.csv",
           "ansoil_feature_importance_rf_42.csv"],
        "audit_poster_metrics.py")
TIERS = [("strong", 0.5), ("moderate", 0.3), ("weak", 0.0)]   # below 0 -> "unusable"
MAP_GATE = 0.30
TIE = 0.02
FAILURES = []
DEFECTS = []
NOTES = []


def check(label, ok, detail=""):
    """A reproduction check. FAIL here means a number does not trace to its source."""
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", label, ("  " + detail) if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def probe(label, clean, detail=""):
    """A defect probe. The numbers reproduce either way; this asks whether the
    pipeline's own logic is sound. DEFECT is a finding, not a broken audit."""
    print("  [%s] %s%s" % ("CLEAN" if clean else "DEFECT", label, ("  " + detail) if detail else ""))
    if not clean:
        DEFECTS.append(label)
    return clean


def note(text):
    print("  [NOTE] " + text)
    NOTES.append(text)


def hdr(n, title):
    print("\n" + "=" * 78)
    print("M%d  %s" % (n, title))
    print("=" * 78)


def r2(actual, pred):
    a = np.asarray(actual, float)
    p = np.asarray(pred, float)
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    if len(a) < 2:
        return np.nan
    sst = ((a - a.mean()) ** 2).sum()
    return np.nan if sst == 0 else 1.0 - ((a - p) ** 2).sum() / sst


# ---------------------------------------------------------------------------
hdr(1, "per-sample CV predictions -> cv_r2 in ansoil_model_results_*")
# ---------------------------------------------------------------------------
results = {}
recomputed = {}
native = {}
for model in MODELS:
    for seed in SEEDS:
        mr = pd.read_csv(f"{DATA}/ansoil_model_results_{model}_{seed}.csv")
        cv = pd.read_csv(f"{DATA}/ansoil_cv_predictions_{model}_{seed}.csv")
        results[(model, seed)] = mr
        for t, g in cv.groupby("target"):
            has_log = g.actual_log.notna().any() and g.pred_log.notna().any()
            recomputed[(model, seed, t)] = {
                "selected_space": r2(g.actual_log, g.pred_log) if has_log
                                  else r2(g.actual, g.predicted),
                "native_units": r2(g.actual, g.predicted),
                "n": int(g.actual.notna().sum()),
                "has_log": has_log,
            }
worst = 0.0
bad = 0
nfold_bad = 0
nsamp = set()
for (model, seed), mr in results.items():
    for _, row in mr.iterrows():
        rc = recomputed.get((model, seed, row.target))
        if rc is None:
            bad += 1
            continue
        d = abs(rc["selected_space"] - row.cv_r2)
        worst = max(worst, d)
        bad += d > 1e-9
        nfold_bad += int(row.n_folds) != 28
        nsamp.add(int(row.n_samples) if "n_samples" in mr.columns else rc["n"])
check("every reported cv_r2 == pooled R2 recomputed from the predictions",
      bad == 0, "%d mismatches over %d model/seed/target rows, max abs diff %.3e"
      % (bad, sum(len(m) for m in results.values()), worst))
check("every run really used 28 leave-one-location-out folds", nfold_bad == 0,
      "%d rows disagree" % nfold_bad)
check("every run really used 171 samples", nsamp == {171}, "observed %s" % sorted(nsamp))
n_logspace = sum(1 for k, v in recomputed.items() if v["has_log"] and k[1] == 42 and k[0] == "rf")
n_clr = sum(1 for k, v in recomputed.items()
            if k[0] == "rf" and k[1] == 42 and not v["has_log"] and k[2].startswith("clr_"))
note("for rf seed 42, %d of 67 targets are scored in log space and a further %d in "
     "CLR space, so %d of 67 headline R2 values are not in the measured units"
     % (n_logspace, n_clr, n_logspace + n_clr))


# ---------------------------------------------------------------------------
hdr(2, "ansoil_model_results_* -> metrics_summary_<model>.csv")
# ---------------------------------------------------------------------------
summ = {}
for model in MODELS:
    ms = pd.read_csv(f"{DATA}/metrics_summary_{model}.csv")
    summ[model] = ms
    per = {}
    for seed in SEEDS:
        for _, row in results[(model, seed)].iterrows():
            per.setdefault(row.target, []).append(float(row.cv_r2))
    probe("metrics_summary_%s covers exactly the target names the seeds produced" % model,
          set(ms.target) == set(per), "%d rows vs %d distinct names"
          % (len(ms), len(per)))
    wm = wmed = wsd = wmin = wmax = 0.0
    thin = []
    for _, row in ms.iterrows():
        v = np.array(per[row.target], float)
        wm = max(wm, abs(v.mean() - row.cv_r2_mean))
        wmed = max(wmed, abs(np.median(v) - row.cv_r2_median))
        wmin = max(wmin, abs(v.min() - row.cv_r2_min))
        wmax = max(wmax, abs(v.max() - row.cv_r2_max))
        if len(v) > 1:
            wsd = max(wsd, abs(v.std(ddof=1) - row.cv_r2_std))
        if len(v) < len(SEEDS):
            thin.append((row.target, len(v), row.cv_r2_mean, row.cv_r2_std))
    check("%s cv_r2_mean reproduces" % model, wm < 1e-9, "max abs diff %.3e" % wm)
    check("%s cv_r2_median reproduces" % model, wmed < 1e-9, "max abs diff %.3e" % wmed)
    check("%s cv_r2_std is the sample sd (ddof=1)" % model, wsd < 1e-9, "max abs diff %.3e" % wsd)
    check("%s cv_r2_min and cv_r2_max reproduce" % model, max(wmin, wmax) < 1e-9,
          "max abs diff %.3e" % max(wmin, wmax))
    probe("%s has no row averaging fewer than %d seeds" % (model, len(SEEDS)), not thin,
          "%d such rows" % len(thin))
    if thin:
        print("     target                        seeds  reported_mean  reported_sd")
        for t, n, m, s in sorted(thin):
            print("     %-28s %5d  %13.6f  %11s" % (t, n, m, "nan" if pd.isna(s) else "%.6f" % s))
        note("%s: %d of %d rows are means over 1 to 4 seeds, not 5, because the "
             "log-vs-raw transform choice flipped between seeds and the two choices "
             "were filed under different target names" % (model, len(thin), len(ms)))

# which raw columns split
for model in MODELS:
    split = {}
    for seed in SEEDS:
        for _, row in results[(model, seed)].iterrows():
            split.setdefault(row.raw_col, set()).add(row.target)
    flipped = {k: sorted(v) for k, v in split.items() if len(v) > 1}
    probe("%s: no chemical property is split across two summary rows" % model, not flipped,
          "split: %s" % (list(flipped) or "none"))
    for k, v in flipped.items():
        print("     %-24s appears as %s" % (k, " and ".join(v)))


# ---------------------------------------------------------------------------
hdr(3, "metrics_summary_* -> model_comparison.csv")
# ---------------------------------------------------------------------------
mc = pd.read_csv(f"{DATA}/model_comparison.csv")
rf = summ["rf"].set_index("target")
xg = summ["xgb"].set_index("target")
check("model_comparison covers the union of both models' target names",
      set(mc.target) == set(rf.index) | set(xg.index),
      "%d rows, rf %d, xgb %d, union %d"
      % (len(mc), len(rf), len(xg), len(set(rf.index) | set(xg.index))))
w = {k: 0.0 for k in ("rf_r2_mean", "rf_r2_sd", "xgb_r2_mean", "xgb_r2_sd", "rf_rmse_mean", "xgb_rmse_mean")}
miss_rf = miss_xgb = 0
for _, row in mc.iterrows():
    for src, tbl, pre in ((rf, rf, "rf"), (xg, xg, "xgb")):
        if row.target not in tbl.index:
            if pre == "rf":
                miss_rf += 1
            else:
                miss_xgb += 1
            continue
        s = tbl.loc[row.target]
        for a, b in ((f"{pre}_r2_mean", "cv_r2_mean"), (f"{pre}_r2_sd", "cv_r2_std"),
                     (f"{pre}_rmse_mean", "cv_rmse_orig_units_mean")):
            if pd.notna(row[a]) and pd.notna(s[b]):
                w[a] = max(w[a], abs(float(row[a]) - float(s[b])))
for k, v in w.items():
    check("model_comparison %-14s reproduces from metrics_summary" % k, v < 1e-9,
          "max abs diff %.3e" % v)
probe("every model_comparison row has both models present",
      miss_rf == 0 and miss_xgb == 0,
      "%d rows missing an rf value, %d missing an xgb value" % (miss_rf, miss_xgb))
if miss_rf or miss_xgb:
    for _, row in mc.iterrows():
        if row.target not in rf.index or row.target not in xg.index:
            print("     %-28s rf=%s xgb=%s winner=%s"
                  % (row.target,
                     "yes" if row.target in rf.index else "MISSING",
                     "yes" if row.target in xg.index else "MISSING", row.winner))
    note("model_comparison pairs rf against xgb by target NAME, so the two "
         "transform-split properties are compared against a missing counterpart")


def winner_of(a, b):
    if pd.isna(a) or pd.isna(b):
        return None
    if abs(a - b) <= TIE:
        return "tie"
    return "rf" if a > b else "xgb"


wbad = 0
for _, row in mc.iterrows():
    exp = winner_of(row.rf_r2_mean, row.xgb_r2_mean)
    if exp is not None and exp != row.winner:
        wbad += 1
check("the winner column reproduces from a %.2f tie threshold" % TIE, wbad == 0,
      "%d mismatches" % wbad)
tal = mc.winner.value_counts().to_dict()
print("  winner tally as filed: %s" % tal)


# ---------------------------------------------------------------------------
hdr(4, "model_comparison.csv -> summary_table_mappable.csv")
# ---------------------------------------------------------------------------
st = pd.read_csv(f"{DATA}/summary_table_mappable.csv")
mci = mc.set_index("target")
bad = 0
worst = 0.0
for _, row in st.iterrows():
    if row.target not in mci.index:
        bad += 1
        continue
    s = mci.loc[row.target]
    for a, b in (("rf_r2", "rf_r2_mean"), ("rf_r2_sd", "rf_r2_sd"),
                 ("xgb_r2", "xgb_r2_mean"), ("xgb_r2_sd", "xgb_r2_sd")):
        if pd.notna(row[a]) and pd.notna(s[b]):
            worst = max(worst, abs(float(row[a]) - round(float(s[b]), 3)))
    bad += row.better_model != s.winner
check("every mappable row is a rounded copy of its model_comparison row",
      bad == 0 and worst < 1e-9, "%d mismatches, max abs diff %.3e" % (bad, worst))
both = mci.dropna(subset=["rf_r2_mean", "xgb_r2_mean"])
gate_keep = set(both[both[["rf_r2_mean", "xgb_r2_mean"]].max(axis=1) >= MAP_GATE].index)
check("the mappable list == both models present and the better one >= %.2f" % MAP_GATE,
      gate_keep == set(st.target),
      "%d selected, %d in the file, symmetric difference %s"
      % (len(gate_keep), len(st), sorted(gate_keep ^ set(st.target)) or "none"))
dropped = {t for t, s in mci.iterrows()
           if max(s.rf_r2_mean if pd.notna(s.rf_r2_mean) else -9,
                  s.xgb_r2_mean if pd.notna(s.xgb_r2_mean) else -9) >= MAP_GATE} - gate_keep
probe("no target clears the gate but is silently dropped for a missing counterpart",
      not dropped, "dropped: %s" % (sorted(dropped) or "none"))
if dropped:
    for t in sorted(dropped):
        s = mci.loc[t]
        note("%s clears the %.2f gate at R2 %.3f but is absent from the mappable "
             "list because its counterpart model filed it under the other transform name"
             % (t, MAP_GATE, max(v for v in (s.rf_r2_mean, s.xgb_r2_mean) if pd.notna(v))))
rft = summ["rf"].set_index("target").tier
xgt = summ["xgb"].set_index("target").tier
sti = st.set_index("target")
check("mappable rf_tier and xgb_tier match metrics_summary",
      all(sti.rf_tier[t] == rft.get(t) and sti.xgb_tier[t] == xgt.get(t) for t in sti.index))


# ---------------------------------------------------------------------------
hdr(5, "tier labels reproduce from their own thresholds")
# ---------------------------------------------------------------------------
def tier_of(v):
    for name, lo in TIERS:
        if v >= lo:
            return name
    return "unusable"


bad = 0
for (model, seed), mr in results.items():
    for _, row in mr.iterrows():
        bad += tier_of(float(row.cv_r2)) != row.tier
check("per-seed tier == rule strong>=0.5, moderate>=0.3, weak>=0, unusable<0", bad == 0,
      "%d mismatches over %d rows" % (bad, sum(len(m) for m in results.values())))
for model in MODELS:
    ms = summ[model]
    bad = sum(1 for _, r in ms.iterrows() if tier_of(float(r.cv_r2_mean)) != r.tier)
    check("metrics_summary_%s tier == rule applied to cv_r2_mean" % model, bad == 0,
          "%d mismatches" % bad)
    c = ms.tier.value_counts().to_dict()
    print("  %s tiers: %s   mean R2 across targets %.3f"
          % (model, {k: c.get(k, 0) for k in ("strong", "moderate", "weak", "unusable")},
             ms.cv_r2_mean.mean()))


# ---------------------------------------------------------------------------
hdr(6, "the transform-space problem, quantified")
# ---------------------------------------------------------------------------
rowsout = []
for model in MODELS:
    for t in summ[model].target:
        sel, nat = [], []
        for seed in SEEDS:
            rc = recomputed.get((model, seed, t))
            if rc is None:
                continue
            sel.append(rc["selected_space"])
            nat.append(rc["native_units"])
        if sel and any(recomputed[(model, s, t)]["has_log"]
                       for s in SEEDS if (model, s, t) in recomputed):
            rowsout.append({"model": model, "target": t, "n_seeds": len(sel),
                            "r2_as_reported": np.mean(sel),
                            "r2_native_units": np.mean(nat),
                            "inflation": np.mean(sel) - np.mean(nat)})
tf = pd.DataFrame(rowsout).sort_values("inflation", ascending=False)
print("  targets reported in log space, with the same model scored in measured units:")
print("  %-6s %-28s %7s %9s %9s" % ("model", "target", "seeds", "reported", "native"))
for _, r in tf.iterrows():
    print("  %-6s %-28s %7d %9.3f %9.3f" % (r.model, r.target, r.n_seeds,
                                            r.r2_as_reported, r.r2_native_units))
flip = tf[(tf.r2_as_reported >= 0) & (tf.r2_native_units < 0)]
probe("no target is positive as reported but negative in measured units",
      len(flip) == 0, "%d targets flip sign" % len(flip))
if len(flip):
    note("%d model/target pairs read positive in log space and negative in the units "
         "the soil was actually measured in; the largest gap is %s at %+.3f reported "
         "versus %+.3f native"
         % (len(flip), flip.iloc[0].target, flip.iloc[0].r2_as_reported,
            flip.iloc[0].r2_native_units))
for model in MODELS:
    ms = summ[model].set_index("target")
    nat = []
    for t in ms.index:
        v = [recomputed[(model, s, t)]["native_units"] for s in SEEDS
             if (model, s, t) in recomputed]
        nat.append(np.mean(v) if v else np.nan)
    print("  %s mean R2 as reported %.3f, in measured units %.3f"
          % (model, ms.cv_r2_mean.mean(), np.nanmean(nat)))


# ---------------------------------------------------------------------------
hdr(7, "KNN results reproduce from their own CV predictions")
# ---------------------------------------------------------------------------
kn = pd.read_csv(f"{DATA}/ansoil_model_results_knn.csv")
kcv = pd.read_csv(f"{DATA}/ansoil_cv_predictions_knn.csv")
bad = 0
worst = 0.0
for _, row in kn.iterrows():
    g = kcv[kcv.target == row.target]
    if not len(g):
        bad += 1
        continue
    has_log = g.actual_log.notna().any() and g.pred_log.notna().any()
    got = r2(g.actual_log, g.pred_log) if has_log else r2(g.actual, g.predicted)
    d = abs(got - float(row.cv_r2))
    worst = max(worst, d)
    bad += d > 1e-9
check("every KNN cv_r2 reproduces from its own predictions", bad == 0,
      "%d mismatches, max abs diff %.3e" % (bad, worst))
print("  KNN: %d targets, mean R2 %.3f, tiers %s"
      % (len(kn), kn.cv_r2.mean(), kn.tier.value_counts().to_dict()))
print("  KNN k values chosen: %s" % sorted(kn.best_k.unique()))


# ---------------------------------------------------------------------------
hdr(8, "structural integrity of every summary file")
# ---------------------------------------------------------------------------
for name in ["metrics_summary_rf.csv", "metrics_summary_xgb.csv",
             "model_comparison.csv", "summary_table_mappable.csv",
             "ansoil_model_results_knn.csv"]:
    df = pd.read_csv(f"{DATA}/{name}")
    dup = df.target.duplicated().sum()
    nan = df.isna().sum()
    nan = nan[nan > 0].to_dict()
    check("%-28s no duplicate target rows" % name, dup == 0, "%d duplicates" % dup)
    if name == "model_comparison.csv":
        blankrows = sorted(df.loc[df.isna().any(axis=1), "target"])
        probe("%-28s blanks confined to the transform-split rows" % name,
              blankrows == ["log_digest_mg_kg_na5895", "log_hr_24_mg_l_so4"],
              "rows with blanks: %s" % blankrows)
    else:
        check("%-28s no unexpected blanks" % name, not nan, "blanks: %s" % (nan or "none"))
for model in MODELS:
    for seed in SEEDS:
        mr = results[(model, seed)]
        check("%s seed %-4s has 67 targets, no duplicates" % (model, seed),
              len(mr) == 67 and mr.target.duplicated().sum() == 0, "%d rows" % len(mr))


print("\n" + "=" * 78)
print("RESULT: %d reproduction check(s) failed, %d defect(s) found, %d note(s)"
      % (len(FAILURES), len(DEFECTS), len(NOTES)))
if not FAILURES:
    print("Every published number traces exactly to the per-sample CV predictions.")
    print("The defects below are flaws in the pipeline's logic, not arithmetic errors.")
for f in FAILURES:
    print("  FAIL    " + f)
for d in DEFECTS:
    print("  DEFECT  " + d)
for n in NOTES:
    print("  NOTE    " + n)
print("=" * 78)


