# ANSOIL Spatial Prediction

Machine learning pipeline for predicting geochemical properties in Antarctic ice-free regions from spatial environmental maps.

This set up is for our run with all combinations for Random Forest and XGBoost

## Setup

Install dependencies:
```
pip install numpy pandas scikit-learn xgboost
```

On macOS, XGBoost also needs: `brew install libomp`

## How to Run

Both scripts are in `scripts/`. Run them from inside that folder:
```
cd scripts
python Ansoil_rf_model.py
python ansoil_xgb_model_v9_noes.py
```

Output files are written to `results/`.

### Changing the seed

The only thing to change between runs is `SEED = 42` near the top of each script. Run each script 5 times with seeds **42, 73, 123, 7, 256**. Between runs, move the output files from `results/` into a seed-specific subfolder:
```
mkdir ../results/rf_seed42
mv ../results/ansoil_model_results_rf.csv ../results/rf_seed42/
mv ../results/ansoil_cv_predictions_rf.csv ../results/rf_seed42/
mv ../results/ansoil_grid_predictions_rf.csv ../results/rf_seed42/
mv ../results/ansoil_feature_importance_rf.csv ../results/rf_seed42/
mv ../results/ansoil_model_comparison_knn_vs_rf.csv ../results/rf_seed42/
mv ../results/ansoil_transform_comparison_rf.csv ../results/rf_seed42/
```

Then change `SEED = 73`, save, and run again. Repeat for all 5 seeds, then do the same for the XGBoost script.

### Summary of all 10 runs

| Run | Script | SEED | Output folder |
|-----|--------|------|---------------|
| 1 | Ansoil_rf_model.py | 42 | results/rf_seed42/ |
| 2 | Ansoil_rf_model.py | 73 | results/rf_seed73/ |
| 3 | Ansoil_rf_model.py | 123 | results/rf_seed123/ |
| 4 | Ansoil_rf_model.py | 7 | results/rf_seed7/ |
| 5 | Ansoil_rf_model.py | 256 | results/rf_seed256/ |
| 6 | ansoil_xgb_model_v9_noes.py | 42 | results/xgb_seed42/ |
| 7 | ansoil_xgb_model_v9_noes.py | 73 | results/xgb_seed73/ |
| 8 | ansoil_xgb_model_v9_noes.py | 123 | results/xgb_seed123/ |
| 9 | ansoil_xgb_model_v9_noes.py | 7 | results/xgb_seed7/ |
| 10 | ansoil_xgb_model_v9_noes.py | 256 | results/xgb_seed256/ |

## What NOT to run

Do not run `Ansoil_xgb_model.py` (the old v8 script if present). That version has a data leakage issue. Only use `ansoil_xgb_model_v9_noes.py`.

## File descriptions

### Data files (in `data/`)
| File | Description |
|------|-------------|
| ansoil_targets.csv | 67 soil properties to predict (171 samples) |
| ansoil_predictors.csv | 22 environmental features per sample |
| ansoil_sample_index.csv | Sample locations for cross-validation folds |
| ansoil_log_targets.csv | Transform lookup table |
| ansoil_grid_prepared.csv | 15,769 prediction grid points |

### Scripts (in `scripts/`)
| Script | What it does |
|--------|-------------|
| Ansoil_knn_prep.py | Data preparation (already run, not needed) |
| Ansoil_knn_model.py | KNN baseline (already run, not needed) |
| Ansoil_rf_model.py | **Random Forest model** |
| ansoil_xgb_model_v9_noes.py | **XGBoost model (leak-free)** |

### Key output files
| File | Description |
|------|-------------|
| ansoil_model_results_rf.csv | R² per target (RF) |
| ansoil_model_results_xgb.csv | R² per target (XGBoost) |
| ansoil_cv_predictions_*.csv | Predicted vs actual per sample |
| ansoil_feature_importance_*.csv | Feature importance per target |
| ansoil_grid_predictions_*.csv | Spatial predictions at grid points |

## Expected runtime

- RF: ~1-4 hours per seed (300 hyperparameter combos)
- XGBoost: ~2-6 hours per seed (500 hyperparameter combos)
- Total for all 10 runs: ~15-50 hours
```

**Push it all to GitHub:**
```
git add .
git commit -m "Initial commit: RF and XGBoost scripts with data"
git push
```

**One important thing:** If `ansoil_grid_prepared.csv` is large (it's ~16K rows, probably 5-10MB), that's fine for GitHub. But if any file is over 100MB, you'll need Git LFS. Your files should all be well under that limit.

**Add a `.gitignore`** so the pickle model files and large outputs don't get committed accidentally:

Create a file called `.gitignore` in the root of the repo with:
```
results/ansoil_models_rf/
results/ansoil_models_xgb/
__pycache__/
*.pyc
.DS_Store
