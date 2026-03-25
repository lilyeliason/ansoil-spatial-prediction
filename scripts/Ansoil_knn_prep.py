"""
ANTARCTIC SOIL GEOCHEMISTRY - Data Preparation
===============================================

This script prepares the raw data for all models. Run it ONCE to produce
the input files that KNN, RF, and XGBoost all depend on.

If the output files already exist, you can skip this script entirely.

Setup:
    pip install numpy pandas scikit-learn

Input files (place in same folder as this script):
    Ansoil_data_cleaned_10162025(complete_cleaned).csv  - Master dataset (171 samples)
    ansoil_full_grid.csv                                - Prediction grid (~15,000 points)

Output files (8 total):
    ansoil_targets.csv          - 67 soil properties to predict
    ansoil_predictors.csv       - 22 environmental features per sample
    ansoil_sample_index.csv     - Sample locations (defines CV folds)
    ansoil_log_targets.csv      - Transform lookup table
    ansoil_grid_prepared.csv    - Environmental features for grid points
    ansoil_distance_matrix.csv  - 171x171 sample distances (KNN only)
    ansoil_grid_distances.csv   - Grid-to-sample distances (KNN only)
    ansoil_scaler_params.csv    - Standardization parameters

Run:
    python Ansoil_knn_prep.py

Takes ~10-20 minutes (most time spent computing distance matrices).

What it does:
    1. Loads raw data, projects coordinates to EPSG:3031 (Antarctic Polar Stereographic)
    2. Creates log and CLR transforms for appropriate target variables
    3. Z-score normalizes continuous predictor features
    4. Builds pairwise distance matrices (for KNN model only)
    5. Saves all output files
"""

import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_CSV = "Ansoil_data_cleaned_10162025(complete_cleaned).csv"
GRID_CSV = "ansoil_full_grid.csv"


# =============================================================================
# CONSTANTS (identical to v4/v5 — do not change)
# =============================================================================

LITHO_LABELS = ["a", "b", "c", "d", "g", "n", "p", "s", "w"]
LITHO_COLS = [
    "lithcode_a",
    "lithcode_b",
    "lithcode_c",
    "lithcode_d",
    "lithcode_g",
    "lithcode_n",
    "lithcode_p",
    "lithcode_s",
    "lithcode_w",
]
REGION_MAP = {
    "Transantarctic Mountains": "region_tm",
    "South Victoria Land": "region_svl",
    "North Victoria Land": "region_nvl",
    "North-west Antarctic Peninsula": "region_nwap",
}
REGION_COLS = ["region_tm", "region_svl", "region_nvl", "region_nwap"]
CONT_COLS = [
    "proj_x_epsg3031",
    "proj_y_epsg3031",
    "wgs84_elev_from_pgc",
    "dist_coast_scar_km",
    "precipitation_racmo",
    "temperature_racmo",
    "slope_dem",
    "aspect_dem",
]
WEIGHTS = {
    "proj_x_epsg3031": 0.25,
    "proj_y_epsg3031": 0.15,
    "wgs84_elev_from_pgc": 2.00,
    "dist_coast_scar_km": 2.50,
    "precipitation_racmo": 1.50,
    "temperature_racmo": 2.00,
    "slope_dem": 1.00,
    "aspect_dem": 0.50,
    "litho": 3.00,
    "region_tm": 0.50,
    "region_svl": 0.50,
    "region_nvl": 0.50,
    "region_nwap": 0.50,
}
LITHO_PAIRS = {
    ("a", "b"): 0.33,
    ("a", "c"): 1.00,
    ("a", "d"): 0.67,
    ("a", "g"): 0.67,
    ("a", "n"): 0.89,
    ("a", "p"): 0.89,
    ("a", "s"): 0.89,
    ("a", "w"): 0.78,
    ("b", "c"): 1.00,
    ("b", "d"): 0.56,
    ("b", "g"): 0.78,
    ("b", "n"): 0.89,
    ("b", "p"): 0.89,
    ("b", "s"): 0.89,
    ("b", "w"): 0.78,
    ("c", "d"): 1.00,
    ("c", "g"): 1.00,
    ("c", "n"): 1.00,
    ("c", "p"): 1.00,
    ("c", "s"): 0.56,
    ("c", "w"): 0.67,
    ("d", "g"): 0.44,
    ("d", "n"): 0.78,
    ("d", "p"): 0.78,
    ("d", "s"): 0.89,
    ("d", "w"): 0.78,
    ("g", "n"): 0.67,
    ("g", "p"): 0.78,
    ("g", "s"): 0.89,
    ("g", "w"): 0.78,
    ("n", "p"): 0.33,
    ("n", "s"): 0.78,
    ("n", "w"): 0.78,
    ("p", "s"): 0.67,
    ("p", "w"): 0.67,
    ("s", "w"): 0.33,
}
COMP_1HR = [
    "hr_1_mg_l_f",
    "hr_1_mg_l_cl",
    "hr_1_mg_l_no3",
    "hr_1_mg_l_po4",
    "hr_1_mg_l_so4",
]
COMP_24HR = [
    "hr_24_mg_l_f",
    "hr_24_mg_l_cl",
    "hr_24_mg_l_no3",
    "hr_24_mg_l_po4",
    "hr_24_mg_l_so4",
]
COMP_TOTAL = [
    "total_mg_l_f",
    "total_mg_l_cl",
    "total_mg_l_no3",
    "total_mg_l_po4",
    "total_mg_l_so4",
    "total_mg_l_ca2",
    "total_mg_l_k",
    "total_mg_l_mg2",
    "total_mg_l_na",
    "total_mg_l_sr2",
]


# =============================================================================
# LOG TARGET DEFINITIONS
# =============================================================================

# Established log1p transforms — kept from v4; NOT dual-tested.
# These are trace metals and CEC where near-zero values are possible,
# so log1p() avoids log(0). The appropriate transform is not in question.
LOG_TARGETS_LOG1P = [
    "cec_meq_100g",
    "digest_mg_kg_hg1849",
    "digest_mg_kg_li6707",
    "digest_mg_kg_mo2020",
    "digest_mg_kg_p_1774",
    "digest_mg_kg_pb2203",
    "digest_mg_kg_sb2068",
    "digest_mg_kg_se1960",
    "digest_mg_kg_sn1899",
    "digest_mg_kg_sr4077",
    "digest_mg_kg_tl1908",
]

# Dual-test candidates — model both raw AND log(), keep the better R².
# All confirmed strictly positive (min > 0 in training data) so log() is safe.
# Back-transform if log wins: exp(prediction).
#
# digest_mg_kg_na5895 is included as an internal validation:
#   - v4 (raw)   R² = 0.318  ← expected winner
#   - v5 (log)   R² = 0.030  ← was artificially harmed
#   If the procedure is correct, raw will win and R² will return to ~0.318.
DUAL_TEST_CANDIDATES = [
    "ec_us_cm",  # skew=2.49  electrical conductivity
    "total_mg_l_cl",  # skew=4.27  dissolved chloride (total)
    "hr_1_mg_l_cl",  # skew=4.56  dissolved chloride (1hr)
    "hr_24_mg_l_cl",  # skew=2.58  dissolved chloride (24hr)
    "total_mg_l_so4",  # skew=2.68  dissolved sulfate (total)
    "hr_1_mg_l_so4",  # skew=3.54  dissolved sulfate (1hr)
    "hr_24_mg_l_so4",  # skew=3.27  dissolved sulfate (24hr)
    "total_mg_l_na",  # skew=8.29  dissolved sodium (highest skew)
    "total_mg_l_mg2",  # skew=6.34  dissolved magnesium
    "total_mg_l_ca2",  # skew=2.43  dissolved calcium
    "digest_mg_kg_na5895",  # skew=2.75  REVERTED — expect raw wins ~0.318
    "digest_mg_kg_mg2852",  # skew=2.97  digest magnesium
    "digest_mg_kg_ca3158",  # skew=2.15  digest calcium
]


# =============================================================================
# HELPERS (identical to v4/v5)
# =============================================================================


def latlon_to_epsg3031(lat_deg, lon_deg):
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f**2
    e = math.sqrt(e2)
    phi_ts = math.radians(-71.0)
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)

    def t(p):
        return math.tan(math.pi / 4 + p / 2) / (
            ((1 - e * math.sin(p)) / (1 + e * math.sin(p))) ** (e / 2)
        )

    def m(p):
        return math.cos(p) / math.sqrt(1 - e2 * math.sin(p) ** 2)

    rho = a * m(phi_ts) * t(phi) / t(phi_ts)
    return rho * math.sin(lam), rho * math.cos(lam)


def verify_projection():
    checks = [
        (-90.0, 0.0, 0.0, 0.0),
        (-71.0, 0.0, 0.0, 2082760.1),
        (-71.0, 90.0, 2082760.1, 0.0),
        (-77.847, 166.668, 305456.0, -1288954.0),
    ]
    for lat, lon, ex, ey in checks:
        x, y = latlon_to_epsg3031(lat, lon)
        assert abs(x - ex) < 2 and abs(y - ey) < 2, (
            f"Projection check failed ({lat},{lon}): got ({x:.1f},{y:.1f})"
        )
    print("  Projection verified against 4 ground truths")


def build_litho_matrix():
    n = len(LITHO_LABELS)
    mat = np.zeros((n, n))
    idx = {l: i for i, l in enumerate(LITHO_LABELS)}
    for (a, b), v in LITHO_PAIRS.items():
        mat[idx[a], idx[b]] = v
        mat[idx[b], idx[a]] = v
    assert np.max(np.abs(mat - mat.T)) == 0, "Litho matrix not symmetric"
    assert np.max(np.diag(mat)) == 0, "Litho matrix diagonal non-zero"
    return mat, idx


def get_litho(row):
    for col, ltr in zip(LITHO_COLS, LITHO_LABELS):
        if row.get(col, 0) == 1:
            return ltr
    return None


def add_coords(df):
    df = df.copy()
    df["proj_x_epsg3031"] = df.apply(
        lambda r: latlon_to_epsg3031(r.lat, r.lon)[0], axis=1
    )
    df["proj_y_epsg3031"] = df.apply(
        lambda r: latlon_to_epsg3031(r.lat, r.lon)[1], axis=1
    )
    return df


def add_regions(df):
    df = df.copy()
    for col in REGION_COLS:
        df[col] = 0
    for region, col in REGION_MAP.items():
        df.loc[df["acbr"] == region, col] = 1
    return df


def calculate_clr(data, cols, eps=1e-6):
    temp = data[cols].copy().replace(0, eps)
    logs = np.log(temp)
    gm = logs.mean(axis=1)
    clr = logs.sub(gm, axis=0)
    clr.columns = ["clr_" + c for c in cols]
    return clr


def pairwise_distances(
    X_a, litho_a, regions_a, X_b, litho_b, regions_b, litho_mat, litho_idx
):
    M = len(litho_a)
    N = len(litho_b)
    same = X_a is X_b
    w = np.array([WEIGHTS[c] for c in CONT_COLS])
    D = np.zeros((M, N))
    for i in range(M):
        j_start = i + 1 if same else 0
        for j in range(j_start, N):
            d = np.sum((w * (X_a[i] - X_b[j])) ** 2)
            d += (
                WEIGHTS["litho"]
                * litho_mat[litho_idx[litho_a[i]], litho_idx[litho_b[j]]]
            ) ** 2
            for rc in REGION_COLS:
                d += (
                    WEIGHTS[rc]
                    * abs(int(regions_a[rc].iloc[i]) - int(regions_b[rc].iloc[j]))
                ) ** 2
            v = math.sqrt(d)
            D[i, j] = v
            if same:
                D[j, i] = v
    return D


# =============================================================================
# STEP 1 — PREPARE TRAINING DATA
# =============================================================================


def prepare_training():
    print("=" * 70)
    print("STEP 1  Prepare training dataset")
    print("=" * 70)

    df = pd.read_csv(TRAIN_CSV)
    n_raw = len(df)
    df = df.dropna(subset=["sample_id"]).reset_index(drop=True)
    df = df[df["abbr_id"] != "HR"].reset_index(drop=True)
    print(f"  Loaded: {n_raw}  |  After filtering HR and null IDs: {len(df)}")

    df = add_coords(df)
    df = add_regions(df)
    df["litho"] = df.apply(get_litho, axis=1)

    n_bad_litho = df["litho"].isna().sum()
    if n_bad_litho > 0:
        raise ValueError(f"{n_bad_litho} training samples have no lithology code")
    assert (df[REGION_COLS].sum(axis=1) == 1).all(), (
        "Some samples belong to 0 or >1 regions"
    )

    # ── Established log1p transforms (unchanged from v4) ──────────────────────
    n_log1p, missing_log1p = 0, []
    for col in LOG_TARGETS_LOG1P:
        if col in df.columns:
            df["log_" + col] = np.log1p(df[col])
            n_log1p += 1
        else:
            missing_log1p.append(col)
    if missing_log1p:
        print(
            f"  WARNING: {len(missing_log1p)} log1p cols not in data: {missing_log1p}"
        )

    # ── Dual-test: create log_ columns alongside raw ───────────────────────────
    # Both versions are written to targets CSV. The model script runs CV on
    # each independently and keeps whichever produces a higher R².
    n_dual, skipped_dual = 0, []
    for col in DUAL_TEST_CANDIDATES:
        if col not in df.columns:
            skipped_dual.append(col)
            continue
        min_val = df[col].min()
        if min_val <= 0:
            raise ValueError(
                f"Dual-test target '{col}' contains non-positive values "
                f"(min={min_val:.4g}). Use log1p() or exclude from "
                f"DUAL_TEST_CANDIDATES."
            )
        df["log_" + col] = np.log(df[col])
        n_dual += 1
    if skipped_dual:
        print(
            f"  WARNING: {len(skipped_dual)} dual-test cols not in data: {skipped_dual}"
        )

    # ── CLR transforms (unchanged) ────────────────────────────────────────────
    df = pd.concat(
        [
            df,
            calculate_clr(df, COMP_1HR),
            calculate_clr(df, COMP_24HR),
            calculate_clr(df, COMP_TOTAL),
        ],
        axis=1,
    ).copy()

    n_clr = len(COMP_1HR) + len(COMP_24HR) + len(COMP_TOTAL)
    print(f"  Established log1p:   {n_log1p} cols")
    print(f"  Dual-test pairs:     {n_dual} raw + {n_dual} log_ cols added")
    print(f"  CLR transforms:      {n_clr} cols")
    print(f"  Regions:  { {c: int(df[c].sum()) for c in REGION_COLS} }")
    print(f"  Litho:    {df['litho'].value_counts().sort_index().to_dict()}")
    return df


# =============================================================================
# STEP 2 — PREPARE PREDICTION GRID (unchanged from v4/v5)
# =============================================================================


def prepare_grid():
    print("\n" + "=" * 70)
    print("STEP 2  Prepare prediction grid")
    print("=" * 70)

    grid = pd.read_csv(GRID_CSV)
    print(f"  Raw grid points: {len(grid)}")

    if "grid_id" not in grid.columns:
        grid.insert(0, "grid_id", [f"GRID_{i + 1:05d}" for i in range(len(grid))])

    grid = add_coords(grid)
    grid = add_regions(grid)
    grid["litho"] = grid.apply(get_litho, axis=1)

    n_miss = grid["litho"].isna().sum()
    if n_miss > 0:
        print(f"  WARNING: {n_miss} grid points have no lithology — dropped")
        grid = grid.dropna(subset=["litho"]).reset_index(drop=True)

    n_noreg = (grid[REGION_COLS].sum(axis=1) == 0).sum()
    if n_noreg > 0:
        print(f"  WARNING: {n_noreg} grid points have no region assignment")

    print(f"  Grid points after prep: {len(grid)}")
    print(f"  Regions:  { {c: int(grid[c].sum()) for c in REGION_COLS} }")
    print(f"  Litho:    {grid['litho'].value_counts().sort_index().to_dict()}")
    return grid


# =============================================================================
# STEP 3 — NORMALIZE (unchanged from v4/v5)
# =============================================================================


def normalize(train_df, grid_df):
    print("\n" + "=" * 70)
    print("STEP 3  Z-score normalization (fit on training data only)")
    print("=" * 70)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[CONT_COLS])
    X_grid = scaler.transform(grid_df[CONT_COLS])

    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X_train.std(axis=0), 1, atol=1e-3)

    scaler_params = pd.DataFrame(
        {
            "feature": CONT_COLS,
            "mean": scaler.mean_,
            "std": scaler.scale_,
        }
    )
    print(f"\n  {'Feature':<28}  {'Train Mean':>14}  {'Train Std':>12}")
    print(f"  {'-' * 58}")
    for _, r in scaler_params.iterrows():
        print(f"  {r['feature']:<28}  {r['mean']:>14.4f}  {r['std']:>12.4f}")

    return X_train, X_grid, scaler_params


# =============================================================================
# STEP 4 — BUILD DISTANCE MATRICES (unchanged from v4/v5)
# =============================================================================


def build_distances(train_df, grid_df, X_train, X_grid, litho_mat, litho_idx):
    print("\n" + "=" * 70)
    print("STEP 4  Build distance matrices")
    print("=" * 70)

    N = len(train_df)
    G = len(grid_df)

    print(f"  Training {N}x{N}  ({N * (N - 1) // 2:,} unique pairs) ...")
    D_train = pairwise_distances(
        X_train,
        train_df["litho"].values,
        train_df[REGION_COLS],
        X_train,
        train_df["litho"].values,
        train_df[REGION_COLS],
        litho_mat,
        litho_idx,
    )
    assert np.allclose(D_train, D_train.T, atol=1e-10)
    assert D_train.diagonal().sum() == 0
    print(
        f"  Symmetric, zero diagonal.  "
        f"Range: {D_train[D_train > 0].min():.3f} – {D_train.max():.3f}"
    )

    print(f"\n  Grid {G}x{N}  ({G * N:,} distances) — may take several minutes ...")
    D_grid = pairwise_distances(
        X_grid,
        grid_df["litho"].values,
        grid_df[REGION_COLS],
        X_train,
        train_df["litho"].values,
        train_df[REGION_COLS],
        litho_mat,
        litho_idx,
    )
    print(f"  Done.  Range: {D_grid.min():.3f} – {D_grid.max():.3f}")

    return D_train, D_grid


# =============================================================================
# STEP 5 — SAVE OUTPUTS
# =============================================================================


def save_outputs(train_df, grid_df, D_train, D_grid, scaler_params):
    print("\n" + "=" * 70)
    print("STEP 5  Save outputs")
    print("=" * 70)

    train_ids = train_df["sample_id"].values
    grid_ids = grid_df["grid_id"].values

    # ── Distance matrices ─────────────────────────────────────────────────────
    pd.DataFrame(D_train, index=train_ids, columns=train_ids).to_csv(
        "ansoil_distance_matrix.csv"
    )
    pd.DataFrame(D_grid, index=grid_ids, columns=train_ids).to_csv(
        "ansoil_grid_distances.csv"
    )
    print(f"  ansoil_distance_matrix.csv      ({len(train_ids)} x {len(train_ids)})")
    print(f"  ansoil_grid_distances.csv       ({len(grid_ids)} x {len(train_ids)})")

    # ── Build target column list ──────────────────────────────────────────────
    base_chem = [
        "ph_mq",
        "ph_kcl",
        "ph_cacl2",
        "d15n_air_permil",
        "d13c_vpdb_permil",
        "wt_percent_n",
        "wt_percent_c",
        "c_n_ratio",
    ]

    # Established log1p — only the transformed col goes in (no raw equivalent needed)
    log1p_cols = [
        "log_" + c for c in LOG_TARGETS_LOG1P if "log_" + c in train_df.columns
    ]

    # Dual-test — include BOTH raw AND log_ so model script can test both
    dual_raw_cols = [c for c in DUAL_TEST_CANDIDATES if c in train_df.columns]
    dual_log_cols = [
        "log_" + c for c in DUAL_TEST_CANDIDATES if "log_" + c in train_df.columns
    ]

    clr_cols = (
        ["clr_" + c for c in COMP_1HR]
        + ["clr_" + c for c in COMP_24HR]
        + ["clr_" + c for c in COMP_TOTAL]
    )

    # Any digest columns not in log1p or dual-test lists — still modeled raw
    handled_as_source = set(LOG_TARGETS_LOG1P) | set(DUAL_TEST_CANDIDATES)
    all_dig = [c for c in train_df.columns if c.startswith("digest_mg_kg_")]
    dig_raw = [c for c in all_dig if c not in handled_as_source]

    # Assemble and deduplicate
    tgt_cols_raw = (
        base_chem + log1p_cols + dual_raw_cols + dual_log_cols + clr_cols + dig_raw
    )
    seen, tgt_cols = set(), []
    for c in tgt_cols_raw:
        if c not in seen and c in train_df.columns:
            seen.add(c)
            tgt_cols.append(c)

    train_df[["sample_id"] + tgt_cols].to_csv("ansoil_targets.csv", index=False)
    print(
        f"\n  ansoil_targets.csv              ({len(train_df)} x {len(tgt_cols) + 1})"
    )
    print(f"    {len(log1p_cols)} established log1p targets (trace metals/CEC)")
    print(
        f"    {len(dual_raw_cols)} dual-test raw  +  {len(dual_log_cols)} dual-test log cols"
    )
    print(f"    {len(clr_cols)} CLR  |  {len(dig_raw)} other raw digest")
    print("    Model script selects best transform per dual-test target via CV")

    # ── Log target lookup ─────────────────────────────────────────────────────
    # Established log1p entries have a final known transform.
    # Dual-test entries use transform='dual_test' — placeholder until model
    # script resolves which version won. Model script writes the final
    # selected_transform to ansoil_transform_comparison_v6.csv.
    log_lookup_rows = []

    for col in LOG_TARGETS_LOG1P:
        if "log_" + col in train_df.columns:
            log_lookup_rows.append(
                {
                    "log_col": "log_" + col,
                    "raw_col": col,
                    "transform": "log1p",
                    "back_transform": "expm1",
                    "dual_test": False,
                }
            )

    for col in DUAL_TEST_CANDIDATES:
        if col in train_df.columns:
            log_lookup_rows.append(
                {
                    "log_col": "log_" + col,
                    "raw_col": col,
                    "transform": "dual_test",
                    "back_transform": "exp",  # applies only if log wins
                    "dual_test": True,
                }
            )

    log_lookup = pd.DataFrame(log_lookup_rows)
    log_lookup.to_csv("ansoil_log_targets.csv", index=False)
    print(f"\n  ansoil_log_targets.csv          ({len(log_lookup)} rows)")

    # v6 BUG FIX: .loc[] filter replaces groupby to avoid pandas compat issues
    n_log1p_entries = log_lookup.loc[log_lookup["transform"] == "log1p"].shape[0]
    n_dual_entries = log_lookup.loc[log_lookup["dual_test"] == True].shape[0]
    print(
        f"    {n_log1p_entries} established log1p  |  {n_dual_entries} dual-test pending"
    )

    # ── Sample index ──────────────────────────────────────────────────────────
    icols = [
        "sample_id",
        "abbr_id",
        "sample_location",
        "acbr",
        "litho",
        "lat",
        "lon",
        "proj_x_epsg3031",
        "proj_y_epsg3031",
    ]
    icols = [c for c in icols if c in train_df.columns]
    idf = train_df[icols].copy()
    idf["matrix_position"] = range(len(train_df))
    idf.to_csv("ansoil_sample_index.csv", index=False)
    print(f"  ansoil_sample_index.csv         ({len(idf)} x {len(idf.columns)})")

    # ── Predictors ────────────────────────────────────────────────────────────
    pred_cols = (
        ["sample_id", "sample_location", "acbr", "litho", "lat", "lon"]
        + CONT_COLS
        + REGION_COLS
    )
    pred_cols = [c for c in pred_cols if c in train_df.columns]
    train_df[pred_cols].to_csv("ansoil_predictors.csv", index=False)
    print(f"  ansoil_predictors.csv           ({len(train_df)} x {len(pred_cols)})")

    # ── Grid prepared ─────────────────────────────────────────────────────────
    gcols = [
        "grid_id",
        "lat",
        "lon",
        "proj_x_epsg3031",
        "proj_y_epsg3031",
        "wgs84_elev_from_pgc",
        "dist_coast_scar_km",
        "precipitation_racmo",
        "temperature_racmo",
        "slope_dem",
        "aspect_dem",
        "litho",
        "acbr",
    ] + REGION_COLS
    gcols = [c for c in dict.fromkeys(gcols) if c in grid_df.columns]
    grid_df[gcols].to_csv("ansoil_grid_prepared.csv", index=False)
    print(f"  ansoil_grid_prepared.csv        ({len(grid_df)} x {len(gcols)})")

    # ── Scaler ────────────────────────────────────────────────────────────────
    scaler_params.to_csv("ansoil_scaler_params.csv", index=False)
    print(f"  ansoil_scaler_params.csv        ({len(scaler_params)} x 3)")


# =============================================================================
# MAIN
# =============================================================================


def run():
    verify_projection()
    train_df = prepare_training()
    grid_df = prepare_grid()
    X_train, X_grid, scaler_params = normalize(train_df, grid_df)
    litho_mat, litho_idx = build_litho_matrix()
    D_train, D_grid = build_distances(
        train_df, grid_df, X_train, X_grid, litho_mat, litho_idx
    )
    save_outputs(train_df, grid_df, D_train, D_grid, scaler_params)

    print("\n" + "=" * 70)
    print("PIPELINE v6 COMPLETE")
    print("=" * 70)
    print(f"  Training samples:       {len(train_df)}")
    print(f"  Grid points:            {len(grid_df)}")
    print(f"  Dual-test candidates:   {len(DUAL_TEST_CANDIDATES)}")
    print("    (Na digest expected: raw wins with R² ≈ 0.318)")
    print()
    print("  Next: run ansoil_knn_model_v6.py")
    print("  It will run CV on both raw and log_ per dual-test target,")
    print("  report both R² values, and select the better transform.")
    print("  Full comparison saved to ansoil_transform_comparison_v6.csv")
    print("=" * 70)


if __name__ == "__main__":
    run()
