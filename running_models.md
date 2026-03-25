# Running the Models
 
Step-by-step guide for running the RF and XGBoost models with expanded hyperparameter search across multiple seeds.
 
**You do NOT need to run the KNN scripts.** KNN results are already complete and stored in `reference_results/`.
 
---
 
## Setup (One Time)
 
1. Make sure all files are in the repo (clone it if you haven't):
   ```bash
   git clone https://github.com/lilyeliason/ansoil-spatial-prediction.git
   cd ansoil-spatial-prediction
   ```
 
2. Install Python dependencies:
   ```bash
   pip install numpy pandas scikit-learn xgboost
   ```
   On macOS, XGBoost also needs:
   ```bash
   brew install libomp
   ```
 
---
 
## Running
 
Both scripts are in `scripts/`. Run them from inside that folder:
 
```bash
cd scripts
```
 
### Step-by-Step for Each Seed
 
1. Open the script in a text editor (e.g., VS Code)
2. Find `SEED = 42` near the top. Leave it as 42 for the first run.
3. Run:
   ```bash
   python Ansoil_rf_model.py
   ```
   It prints progress as it goes. Takes ~1-4 hours.
 
4. When finished, move outputs to a subfolder:
   ```bash
   mkdir -p ../results/rf_seed42
   mv ansoil_model_results_rf.csv ../results/rf_seed42/
   mv ansoil_cv_predictions_rf.csv ../results/rf_seed42/
   mv ansoil_grid_predictions_rf.csv ../results/rf_seed42/
   mv ansoil_feature_importance_rf.csv ../results/rf_seed42/
   mv ansoil_transform_comparison_rf.csv ../results/rf_seed42/
   mv ansoil_model_comparison_knn_vs_rf.csv ../results/rf_seed42/
   ```
 
5. Change `SEED = 42` to `SEED = 73`, save the file, run again.
6. Move outputs to `../results/rf_seed73/`. Repeat for seeds **123, 7, 256**.
 
Then do the exact same thing for `Ansoil_xgb_model.py` with the same 5 seeds. XGBoost outputs are named `_xgb` instead of `_rf`:
 
```bash
mkdir -p ../results/xgb_seed42
mv ansoil_model_results_xgb.csv ../results/xgb_seed42/
mv ansoil_cv_predictions_xgb.csv ../results/xgb_seed42/
mv ansoil_grid_predictions_xgb.csv ../results/xgb_seed42/
mv ansoil_feature_importance_xgb.csv ../results/xgb_seed42/
mv ansoil_transform_comparison_xgb.csv ../results/xgb_seed42/
mv ansoil_model_comparison_xgb.csv ../results/xgb_seed42/
```
 
---
 
## All 10 Runs
 
| Run | Script | SEED | Output folder |
|-----|--------|------|---------------|
| 1 | `Ansoil_rf_model.py` | 42 | `results/rf_seed42/` |
| 2 | `Ansoil_rf_model.py` | 73 | `results/rf_seed73/` |
| 3 | `Ansoil_rf_model.py` | 123 | `results/rf_seed123/` |
| 4 | `Ansoil_rf_model.py` | 7 | `results/rf_seed7/` |
| 5 | `Ansoil_rf_model.py` | 256 | `results/rf_seed256/` |
| 6 | `Ansoil_xgb_model.py` | 42 | `results/xgb_seed42/` |
| 7 | `Ansoil_xgb_model.py` | 73 | `results/xgb_seed73/` |
| 8 | `Ansoil_xgb_model.py` | 123 | `results/xgb_seed123/` |
| 9 | `Ansoil_xgb_model.py` | 7 | `results/xgb_seed7/` |
| 10 | `Ansoil_xgb_model.py` | 256 | `results/xgb_seed256/` |
 
---
 
## Expected Runtime
 
| Model | Combos per target | Time per seed | Total (5 seeds) |
|-------|------------------|---------------|-----------------|
| Random Forest | 300 | ~1-4 hours | ~5-20 hours |
| XGBoost | 500 | ~2-6 hours | ~10-30 hours |
 
---
 
## What We Need Back
 
The **most important file** from each run is:
- `ansoil_model_results_rf.csv` (for RF)
- `ansoil_model_results_xgb.csv` (for XGBoost)
 
These contain the R² per target. The other files are useful for deeper analysis but not essential. Ideally push all 10 output folders back to the repo:
 
```bash
git add results/
git commit -m "Add RF and XGBoost results for all 5 seeds"
git push
```
 
---
 
## Troubleshooting
 
| Problem | Solution |
|---------|----------|
| "File not found" error | Make sure you ran `cd scripts` first |
| XGBoost OpenMP error on Mac | Run `brew install libomp` |
| Script seems stuck | Normal. Each target prints a progress line. 67 targets x 300-500 combos takes time. |
| Output files not where expected | They land in whatever directory you ran the command from. Should be `scripts/`. |