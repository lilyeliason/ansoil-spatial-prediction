# ANSOIL: Antarctic Soil Geochemistry Spatial Prediction
 
Machine learning pipeline for predicting 67 Antarctic soil geochemical properties from environmental features across ice-free regions of Antarctica.
 
## Project Summary
 
This project uses three machine learning algorithms (KNN, Random Forest, XGBoost) to predict soil chemistry from landscape-scale environmental covariates. The models are evaluated using Leave-One-Location-Out spatial cross-validation (LOLO-CV) to produce honest, unbiased estimates of prediction accuracy at unsampled locations.
 
**Dataset:** 171 soil samples from 28 sampling locations across 4 Antarctic regions (Transantarctic Mountains, South Victoria Land, North Victoria Land, NW Antarctic Peninsula).
 
**Targets:** 67 soil properties including pH, stable isotopes (d15N, d13C), total C/N, digest metals, dissolved ions, and CLR-transformed compositional leachate data.
 
**Predictors:** 22 environmental features including projected coordinates (EPSG:3031), elevation, distance to coast, RACMO climate variables (precipitation, temperature), slope, cyclic-encoded aspect, lithology classes, and region flags.
 
---
 
## Models
 
| Model | Script | Description | Status |
|-------|--------|-------------|--------|
| KNN | `Ansoil_knn_model.py` | Weighted environmental distance baseline | Complete |
| Random Forest | `Ansoil_rf_model.py` | Ensemble of independent decision trees | Running expanded search |
| XGBoost | `Ansoil_xgb_model.py` | Sequential gradient boosted trees (leak-free) | Running expanded search |
 
---
 
## Key Methodological Decisions
 
### Cross-Validation: Leave-One-Location-Out (LOLO)
 
Standard k-fold CV leaks spatial information because samples at the same location are autocorrelated. LOLO holds out all samples from each of the 28 locations in turn, giving honest estimates of prediction at genuinely new sites.
 
Meyer et al. (2018) demonstrated this directly with Antarctic data: random k-fold CV produced R2 = 0.9 for air temperature prediction, while Leave-Location-Out CV produced R2 = 0.24 for the same model. The difference reflects spatial overfitting that random CV fails to detect. Our LOLO design follows their recommendation that "target-oriented CV is required for reliable error estimates of space-time models."
 
Note: Siqueira et al. (2024), the closest comparable study (RF for Antarctic soil chemistry), used 70-80% train/test splits rather than spatial CV. Our LOLO approach is more conservative and produces lower but more honest R2 estimates.
 
### Lithology Encoding: Two Approaches
 
We use two different lithology representations across our models:
 
- **KNN:** A hand-built 9x9 lithology dissimilarity matrix encoding geochemical similarity between rock types (e.g., granite-gneiss = 0.33, granite-sedimentary = 0.89). This is embedded in the precomputed distance matrix that KNN uses for neighbor lookups. The weights were set by geochemical reasoning, not optimized by data.
 
- **RF and XGBoost:** One-hot encoding of 9 lithology classes (a, b, c, d, g, n, p, s, w), letting the tree models learn per-class effects directly from the data. This is standard practice for categorical variables in tree-based models and is more flexible than the manual matrix because the model discovers which lithology contrasts matter for each target independently.
 
Both approaches are defensible. The KNN dissimilarity matrix incorporates domain knowledge (Siqueira et al., 2024 used environmental covariates including parent material in a similar SCORPAN framework). The one-hot encoding lets RF/XGBoost discover these relationships from data rather than prescribing them.
 
### Feature Engineering
 
- **Cyclic aspect encoding** (sin/cos) so that 359 degrees and 1 degree are treated as nearby, not 358 apart. This is standard for circular variables in ML (Lv et al., 2024).
- **No normalization** for RF/XGBoost: tree-based models are scale-invariant, so standardization is unnecessary (Breiman, 2001). KNN uses z-score normalization because its distance metric is scale-dependent.
- **SCORPAN framework:** Our predictor set follows McBratney et al. (2003), incorporating soil-forming factors: climate (RACMO temperature, precipitation), organisms (not available), relief (elevation, slope, aspect), parent material (lithology), age (not available), and spatial position (coordinates, region, coast distance). Siqueira et al. (2024) and Siqueira et al. (2023) used the same framework for Antarctic soil mapping.
 
### Transform Strategy
 
- **11 trace metals:** Established log1p transforms (near-zero values make log unsafe, log1p handles this)
- **13 dissolved-ion targets:** Dual-test framework where both raw and log versions are evaluated under identical LOLO-CV. The better-performing transform is kept automatically, and both R2 values are reported. This is transparent pre-processing optimization following Wadoux et al. (2020).
- **20 CLR transforms:** Centered log-ratio for compositional leachate data. Log-ratio transforms are standard for compositional geochemical data (Buccianti & Grunsky, 2014; Zhou et al., 2017).
 
### XGBoost: No Early Stopping
 
An earlier version used early stopping on the test fold, which constituted data leakage. The model was seeing test-fold target values during training to decide when to stop adding trees, inflating R2 by a mean of +0.21. The current version tunes n_estimators as a regular hyperparameter alongside learning rate, tree depth, etc. The test fold is never seen during training.
 
Losing & Ebbing (2021) used the same clean approach for Antarctic ML: they tuned all gradient boosting hyperparameters via grid search without using the test set for early stopping decisions.
 
### Multi-Seed Validation
 
Both RF and XGBoost are run across 5 random seeds. Results are reported as mean +/- std across seeds to establish reproducibility. Siqueira et al. (2023) used 100 model runs for stability assessment in their Antarctic soil texture mapping study. Our 5-seed approach with expanded hyperparameter search (300-500 combos) provides a strong balance of thoroughness and computational feasibility.
 
### Hyperparameter Tuning
 
Randomized search over the hyperparameter grid, evaluated via the same LOLO-CV folds. Bergstra & Bengio (2012) showed that random search finds near-optimal configurations with far fewer evaluations than exhaustive grid search, particularly when some hyperparameters matter more than others (which is the case for both RF and XGBoost).
 
---
 
## Repository Structure
 
```
ansoil-spatial-prediction/
|-- README.md                    <- You are here
|-- RUNNING_MODELS.md            <- How to run the models (step-by-step)
|-- scripts/
|   |-- Ansoil_knn_prep.py       # Data preparation (already run)
|   |-- Ansoil_knn_model.py      # KNN baseline (already run)
|   |-- Ansoil_rf_model.py       # Random Forest
|   |-- Ansoil_xgb_model.py      # XGBoost (leak-free)
|-- data/
|   |-- ansoil_targets.csv
|   |-- ansoil_predictors.csv
|   |-- ansoil_sample_index.csv
|   |-- ansoil_log_targets.csv
|   |-- ansoil_grid_prepared.csv
|-- reference_results/
|   |-- ansoil_model_results_knn.csv
|-- results/                      # Output from model runs
```
 
### Data Files
 
| File | Rows | Cols | Description |
|------|------|------|-------------|
| `ansoil_targets.csv` | 171 | 83 | All target variables (raw, log, CLR) |
| `ansoil_predictors.csv` | 171 | 18 | Environmental features per sample |
| `ansoil_sample_index.csv` | 171 | 10 | Sample IDs, locations, regions |
| `ansoil_log_targets.csv` | 24 | 5 | Transform lookup (log1p vs dual-test) |
| `ansoil_grid_prepared.csv` | 15,769 | 17 | Prediction grid across ice-free areas |
 
### 22 Predictor Features
 
| Type | Count | Features |
|------|-------|----------|
| Continuous | 7 | Projected X/Y, elevation, distance to coast, precipitation, temperature, slope |
| Cyclic | 2 | Aspect as sin and cos |
| Lithology | 9 | One-hot encoded classes: a, b, c, d, g, n, p, s, w |
| Region | 4 | Transantarctic Mtns, South Victoria Land, North Victoria Land, NW Antarctic Peninsula |
 
---
 
## References
 
### Spatial Cross-Validation
 
**Meyer, H., Reudenbach, C., Hengl, T., Katurji, M., & Nauss, T. (2018).** Improving performance of spatio-temporal machine learning models using forward feature selection and target-oriented validation. *Environmental Modelling & Software*, 101, 1-9. https://doi.org/10.1016/j.envsoft.2017.12.001
 
Foundational paper for our LOLO-CV design. Demonstrated with Antarctic air temperature data that random k-fold CV (R2=0.9) drastically overestimates model performance compared to Leave-Location-Out CV (R2=0.24). Established that "target-oriented CV is required for reliable error estimates of space-time models."
 
### Antarctic Soil Machine Learning
 
**Siqueira, R.G., Moquedace, C.M., Fernandes-Filho, E.I., Francelino, M.R., & Schaefer, C.E.G.R. (2024).** Modelling and prediction of major soil chemical properties with Random Forest: Machine learning as tool to understand soil-environment relationships in Antarctica. *Catena*, 235, 107677. https://doi.org/10.1016/j.catena.2023.107677
 
Closest comparable study. Used RF for Antarctic soil chemical properties with environmental covariates in the SCORPAN framework. Demonstrated that RF has "greater efficiency against noise and overfitting, good performance on small datasets." Our study extends their work to 67 analytes with spatial CV and multi-algorithm comparison.
 
**Siqueira, R.G., Moquedace, C.M., Francelino, M.R., Schaefer, C.E.G.R., & Fernandes-Filho, E.I. (2023).** Machine learning applied for Antarctic soil mapping: Spatial prediction of soil texture. *Geoderma*, 432, 116405. https://doi.org/10.1016/j.geoderma.2023.116405
 
Compared RF, GBM, k-NN, and GLM for Antarctic soil texture mapping. RF showed the best performance. Used 100 model runs for stability assessment, supporting our multi-seed validation approach.
 
### Geochemical Mapping with ML
 
**Lv, S., Zhu, Y., Cheng, L., Zhang, J., Shen, W., & Li, X. (2024).** Evaluation of the prediction effectiveness for geochemical mapping using machine learning methods. *Science of the Total Environment*, 927, 172223. https://doi.org/10.1016/j.scitotenv.2024.172223
 
Compared 7 ML models (RF, SVM, Ridge, GBDT, ANN, KNN, GPR) for geochemical mapping. Found RF and Gradient Boosting Decision Trees performed best. Supports our choice of RF and XGBoost as primary algorithms.
 
### Antarctic Geoscience ML
 
**Losing, M. & Ebbing, J. (2021).** Predicting geothermal heat flow in Antarctica with a machine learning approach. *Journal of Geophysical Research: Solid Earth*, 126(6). https://doi.org/10.1029/2020JB021499
 
Used gradient boosted regression trees for Antarctic spatial prediction with fivefold grid-search hyperparameter tuning. All hyperparameters tuned explicitly without early stopping on the test set, consistent with our leak-free XGBoost approach.
 
### Antarctic Soil Geochemistry
 
**Bower, D.M. et al. (2021).** Geochemical zones and environmental gradients for soils from the central Transantarctic Mountains, Antarctica. *Biogeosciences*, 18, 1629-1644. https://doi.org/10.5194/bg-18-1629-2021
 
Establishes that elevation and coastal distance drive soil chemistry in the Transantarctic Mountains. Our feature importance analysis confirms this: d15N is driven by coastal distance, Na by elevation, and PO4 by temperature.
 
### Methodological Foundations
 
**McBratney, A.B., Mendonca Santos, M.L., & Minasny, B. (2003).** On digital soil mapping. *Geoderma*, 117(1-2), 3-52. https://doi.org/10.1016/S0016-7061(03)00223-4
 
SCORPAN framework for digital soil mapping. Our predictor set (climate, relief, parent material, spatial position) follows this framework, as do Siqueira et al. (2023, 2024).
 
**Breiman, L. (2001).** Random Forests. *Machine Learning*, 45, 5-32. https://link.springer.com/article/10.1023/A:1010933404324
 
Algorithm reference. Establishes that RF is scale-invariant (no normalization needed) and provides built-in feature importance via impurity decrease.
 
**Chen, T. & Guestrin, C. (2016).** XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785-794. https://dl.acm.org/doi/10.1145/2939672.2939785
 
Algorithm reference for XGBoost (gradient boosted decision trees).
 
**Bergstra, J. & Bengio, Y. (2012).** Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281-305. https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
 
Demonstrated that randomized hyperparameter search finds near-optimal configurations with far fewer evaluations than grid search. Supports our use of 300-500 random combos rather than exhaustive grid search.
 
### Compositional Data
 
**Buccianti, A. & Grunsky, E. (2014).** Compositional data analysis in geochemistry. *Journal of Geochemical Exploration*, 141, 1-5. https://doi.org/10.1016/j.gexplo.2014.03.022
 
Establishes that log-ratio transforms are necessary for meaningful analysis of compositional geochemical data. Supports our CLR transforms for leachate data.
 
**Zhou, S., Zhou, K., Wang, J., Yang, G., & Wang, S. (2017).** Application of cluster analysis to geochemical compositional data. *Frontiers of Earth Science*, 12(3), 491-505. https://link.springer.com/article/10.1007/s11707-017-0682-8
 
Found that CLR transformation combined with ML algorithms is effective for geochemical compositional data analysis.
 
### Additional References
 
**Lindsay, J.J., Hughes, H.S., Yeomans, C.M., Andersen, J.C., & McDonald, I. (2020).** A machine learning approach for regional geochemical data. *Geoscience Frontiers*, 12(3), 101098. https://doi.org/10.1016/j.gsf.2020.10.005
 
**Iwamori, H. et al. (2017).** Classification of geochemical data based on multivariate statistical analyses. *Geochemistry, Geophysics, Geosystems*, 18(3), 994-1012. https://doi.org/10.1002/2016GC006663
 
**Wadoux, A.M.J.-C., Heuvelink, G.B.M., de Bruin, S., & Brus, D.J. (2020).** Spatial cross-validation is not the right way to evaluate map accuracy. *Ecological Modelling*, 457, 109692. https://alexandrewadoux.github.io/assets/pdf/Wadoux_et_al_2021.pdf
