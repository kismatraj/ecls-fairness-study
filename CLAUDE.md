# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project analyzing **algorithmic fairness and temporal generalization** of ML models predicting cognitive development using ECLS-K:2011 public-use data. Goal is peer-reviewed publication examining whether models trained on K-2nd grade data accurately predict 3rd-5th grade outcomes across demographic groups.

## Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix
pip install -r requirements.txt

# Run full pipeline
python scripts/run_pipeline.py --config config.yaml

# Run individual steps
python scripts/run_pipeline.py --step data_prep
python scripts/run_pipeline.py --step train_models
python scripts/run_pipeline.py --step fairness_analysis
python scripts/run_pipeline.py --step figures

# Run tests
pytest
```

## Architecture

### Core Modules (`src/`)

- **data_loader.py**: Data loading and preprocessing. Key functions:
  - `load_ecls_data()` - Load from parquet/csv/stata
  - `handle_missing_values()` - Replace ECLS codes (-1,-7,-8,-9) with NaN
  - `create_race_variable()`, `create_ses_variable()` - Derive demographic categories
  - `create_at_risk_indicator()` - Binary outcome from percentile threshold
  - `prepare_modeling_data()` - Returns (X, y, groups) tuple for modeling

- **models.py**: ML training via `ModelTrainer` class
  - **Classic models**: Logistic regression, elastic net, random forest, XGBoost
  - **2025 State-of-the-Art**: LightGBM, CatBoost, HistGradientBoosting, TabPFN
  - 5-fold CV with grid search, stratified train/test split
  - `get_feature_importance()` for interpretability

- **explainability.py** (NEW 2025): Model interpretability and explanations
  - `SHAPExplainer`: SHAP values, interactions, fairness-aware SHAP by group
  - `PermutationImportanceAnalyzer`: Bootstrap confidence intervals
  - `PartialDependenceAnalyzer`: PDP and ICE plots
  - `CounterfactualExplainer`: DiCE-based counterfactual explanations
  - `LIMEExplainer`: Local interpretable explanations
  - `compare_explanations()`: Compare SHAP vs permutation importance

- **fairness.py**: Fairness evaluation and mitigation (2025 enhanced)
  - `FairnessEvaluator`: Computes TPR/FPR/PPV/accuracy by demographic group
  - `ThresholdOptimizer`: Post-processing threshold adjustment to equalize TPR
  - **2025 Additions**:
    - `IntersectionalFairnessAnalyzer`: Analyze fairness across intersecting attributes
    - `FairnessConfidenceIntervals`: Bootstrap CI for fairness metrics
    - `CalibrationFairnessAnalyzer`: ECE/MCE calibration error by group
    - `IndividualFairnessAnalyzer`: Consistency and Lipschitz violation metrics
    - `generate_comprehensive_fairness_report()`: Full 2025 fairness analysis

- **visualization.py**: Publication figures (2025 enhanced)
  - ROC curves, calibration plots by group
  - **2025 Additions**:
    - `plot_fairness_with_ci()`: Metrics with confidence intervals
    - `plot_intersectional_heatmap()`: Intersectional fairness visualization
    - `plot_calibration_error_comparison()`: ECE/MCE by group
    - `plot_explanation_comparison()`: SHAP vs permutation importance
    - `plot_shap_importance_by_group()`: Fairness-aware SHAP visualization

### Pipeline Flow

```
data_prep → train_models → fairness_analysis → figures
```

Each step reads from previous outputs. Processed data saved to `data/processed/analytic_sample.parquet`.

## Key ECLS Variables

| Type | Variables |
|------|-----------|
| Outcomes (5th grade) | X9RTHETA (reading), X9MTHETA (math) |
| Protected Attributes | X_RACETH_R (race), X1SESQ5 (SES 1-5), X12LANGST (language) |
| Baseline Cognition | X1RTHETK, X2RTHETK, X1MTHETK, X2MTHETK |
| Missing Codes | -1 (N/A), -7 (refused), -8 (don't know), -9 (not ascertained), -2 (suppressed) |

## Configuration

All settings in `config.yaml`:
- Paths for raw/processed data, results
- Variable lists and missing value codes
- Model hyperparameter grids
- Fairness thresholds (disparate impact < 0.80)

## Data Requirements

ECLS-K:2011 public-use file (free, no approval needed):
- Download from https://nces.ed.gov/ecls/dataproducts.asp
- Place `childK5p.dat` in `data/raw/`

## Project Conventions

- `random_state=42` for all reproducibility
- Stratify train/test by race AND SES
- Use `W9C29P_20` weight for longitudinal analyses
- At-risk threshold: bottom 25th percentile
- Flag disparate impact if TPR ratio < 0.80
- Figures: PNG at 300 DPI

## Output Locations

- `results/models/` - Saved .joblib models and JSON model cards
- `results/tables/` - CSV and LaTeX tables
- `results/figures/` - Publication figures
- `logs/pipeline.log` - Pipeline execution logs

## 2025 State-of-the-Art Enhancements

### New ML Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| LightGBM | Fast gradient boosting with leaf-wise growth | Large datasets, speed |
| CatBoost | Gradient boosting with native categorical support | Categorical features |
| HistGradientBoosting | Sklearn's histogram-based boosting | Balanced performance |
| TabPFN | Transformer pre-trained on synthetic tabular data | Small datasets (<10k) |

### Explainability Methods

| Method | Type | Output |
|--------|------|--------|
| SHAP | Global/Local | Feature importance, dependence plots, interactions |
| Permutation Importance | Global | Bootstrap CI for feature importance |
| PDP/ICE | Global | Marginal effect of features |
| LIME | Local | Per-prediction explanations |
| DiCE | Local | Counterfactual explanations |
| Fairness-aware SHAP | Group-level | SHAP importance stratified by protected groups |

### Advanced Fairness Metrics

| Metric | Category | Description |
|--------|----------|-------------|
| Bootstrap CI | Uncertainty | 95% confidence intervals for TPR/FPR/PPV |
| Intersectional Fairness | Group | Metrics for race×SES subgroups |
| ECE/MCE | Calibration | Expected/Maximum Calibration Error by group |
| Consistency Score | Individual | Similar individuals → similar predictions |
| Lipschitz Violation | Individual | Check continuity of predictions |

### Pipeline Flow (2025)

```
data_prep → train_models → explainability_analysis → fairness_analysis → figures
```

### New Configuration Options

```yaml
# Enable/disable 2025 models
model:
  algorithms:
    lightgbm: {enabled: true}
    catboost: {enabled: true}
    hist_gradient_boosting: {enabled: true}
    tabpfn: {enabled: false}  # For small datasets

# Explainability settings
explainability:
  shap: {enabled: true, explainer_type: "auto"}
  permutation: {enabled: true, n_bootstrap: 50}
  counterfactuals: {enabled: false}

# Fairness settings
fairness:
  bootstrap_iterations: 500
  confidence_level: 0.95
  min_intersectional_group_size: 20
```

### New Output Files

| File | Description |
|------|-------------|
| `shap_importance.csv` | SHAP-based feature importance |
| `shap_importance_{group}.csv` | Per-group SHAP importance |
| `permutation_importance_bootstrap.csv` | Permutation importance with CI |
| `explanation_comparison.csv` | SHAP vs permutation comparison |
| `fairness_metrics_with_ci.csv` | Metrics with 95% CI |
| `calibration_fairness.csv` | ECE/MCE by group |
| `intersectional_fairness.csv` | Intersectional subgroup metrics |

### New Figures

- `shap_summary.png` - SHAP beeswarm plot
- `fairness_tpr_with_ci.png` - TPR by group with error bars
- `calibration_error_comparison.png` - ECE/MCE comparison
- `explanation_comparison.png` - SHAP vs permutation
- `intersectional_fairness_heatmap.png` - Intersectional analysis
- `model_comparison.png` - All models performance comparison
