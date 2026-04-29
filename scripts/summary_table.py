"""
Generates a clean summary table of mappable targets (strong + moderate tier)
comparing RF and XGB performance across seeds.

Output (written to results/aggregated/):
  - summary_table_mappable.csv   — full data, sorted by XGB R2 descending
  - summary_table_mappable.md    — formatted markdown version for sharing

Usage:
  Run from the repo root:
    python scripts/make_summary_table.py
"""

import os

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGG_DIR = os.path.join(REPO_ROOT, "results", "aggregated")

MAPPABLE_TIERS = {"strong", "moderate"}

# Targets excluded due to incomplete seed runs (only 1 of 5 seeds produced
# results, so no cross-seed SD is available — see methods note)
INCOMPLETE_TARGETS = {
    "log_digest_mg_kg_na5895",  # missing from RF seeds 42, 73, 123, 256
    "log_hr_24_mg_l_so4",  # missing from XGB seeds 42, 73, 123, 256
}

# ── Load inputs ───────────────────────────────────────────────────────────────

rf_summary = pd.read_csv(os.path.join(AGG_DIR, "metrics_summary_rf.csv"))
xgb_summary = pd.read_csv(os.path.join(AGG_DIR, "metrics_summary_xgb.csv"))
comparison = pd.read_csv(os.path.join(AGG_DIR, "model_comparison.csv"))

# ── Filter to mappable targets ────────────────────────────────────────────────
# A target is mappable if it is strong or moderate in EITHER model

rf_mappable = set(rf_summary.loc[rf_summary["tier"].isin(MAPPABLE_TIERS), "target"])
xgb_mappable = set(xgb_summary.loc[xgb_summary["tier"].isin(MAPPABLE_TIERS), "target"])
mappable_targets = (rf_mappable | xgb_mappable) - INCOMPLETE_TARGETS

comp_mappable = comparison[comparison["target"].isin(mappable_targets)].copy()

# Pull tier from each model summary
rf_tier = rf_summary.set_index("target")["tier"].rename("rf_tier")
xgb_tier = xgb_summary.set_index("target")["tier"].rename("xgb_tier")
comp_mappable = comp_mappable.join(rf_tier, on="target")
comp_mappable = comp_mappable.join(xgb_tier, on="target")

# Sort by XGB R2 descending (best targets first)
comp_mappable = comp_mappable.sort_values("xgb_r2_mean", ascending=False).reset_index(
    drop=True
)

# ── Build clean output table ──────────────────────────────────────────────────

table = pd.DataFrame()
table["target"] = comp_mappable["target"]
table["rf_r2"] = comp_mappable["rf_r2_mean"].round(3)
table["rf_r2_sd"] = comp_mappable["rf_r2_sd"].round(3)
table["xgb_r2"] = comp_mappable["xgb_r2_mean"].round(3)
table["xgb_r2_sd"] = comp_mappable["xgb_r2_sd"].round(3)
table["better_model"] = comp_mappable["winner"]
table["rf_tier"] = comp_mappable["rf_tier"]
table["xgb_tier"] = comp_mappable["xgb_tier"]

# ── Save CSV ──────────────────────────────────────────────────────────────────

csv_path = os.path.join(AGG_DIR, "summary_table_mappable.csv")
table.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")
print(f"  {len(table)} mappable targets\n")

# ── Save markdown ─────────────────────────────────────────────────────────────


def fmt_r2(val, sd):
    if pd.isna(val):
        return "—"
    if pd.isna(sd):
        return f"{val:.3f}"
    return f"{val:.3f} ± {sd:.3f}"


lines = []
lines.append("# ANSOIL Mappable Targets: Model Performance Summary\n")
lines.append(
    "Targets in the **strong** or **moderate** tier for at least one model. "
    "R² values are means across 5 seeds ± SD. "
    "Better model column uses a 0.02 R² threshold — differences smaller than this are called a tie.\n"
)
lines.append("| # | Target | RF R² | XGB R² | Better | RF tier | XGB tier |")
lines.append("|---|--------|-------|--------|--------|---------|----------|")

for i, row in table.iterrows():
    rf_str = fmt_r2(row["rf_r2"], row["rf_r2_sd"])
    xgb_str = fmt_r2(row["xgb_r2"], row["xgb_r2_sd"])
    lines.append(
        f"| {i + 1} | `{row['target']}` | {rf_str} | {xgb_str} "
        f"| {row['better_model']} | {row['rf_tier']} | {row['xgb_tier']} |"
    )

md_path = os.path.join(AGG_DIR, "summary_table_mappable.md")
with open(md_path, "w") as f:
    f.write("\n".join(lines))
print(f"Saved: {md_path}")

# ── Print to terminal ─────────────────────────────────────────────────────────

print("\n── Mappable targets ─────────────────────────────────────────────────────")
print(
    f"{'#':<4} {'Target':<35} {'RF R²':>12} {'XGB R²':>12} {'Better':<8} {'RF tier':<10} {'XGB tier'}"
)
print("─" * 95)
for i, row in table.iterrows():
    rf_str = fmt_r2(row["rf_r2"], row["rf_r2_sd"])
    xgb_str = fmt_r2(row["xgb_r2"], row["xgb_r2_sd"])
    print(
        f"{i + 1:<4} {row['target']:<35} {rf_str:>12} {xgb_str:>12} {row['better_model']:<8} {row['rf_tier']:<10} {row['xgb_tier']}"
    )

print(f"\nTotal mappable targets: {len(table)}")
winner_counts = table["better_model"].value_counts()
print("Winner tally (among mappable targets):")
for label, count in winner_counts.items():
    print(f"  {label}: {count}")
