# ECLS-K:2011 Algorithmic Fairness Study

**A multi-dimensional fairness audit of machine learning models predicting cognitive development in children.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project conducts a comprehensive, multi-dimensional fairness audit of predictive models trained on early childhood data (kindergarten through 2nd grade) to predict 5th-grade academic risk. Using the ECLS-K:2011 nationally representative longitudinal study (N = 9,104), we examine:

1. **Model performance** across 7 ML algorithms, including 3 state-of-the-art gradient boosting methods
2. **Fairness disparities** with bootstrap confidence intervals, calibration error analysis, and intersectional (race x SES) assessment
3. **Explainability** via SHAP values and permutation importance with fairness-aware group-level analysis
4. **Temporal generalization** across 4 developmental windows (K fall through 3rd grade)
5. **Sensitivity analysis** of fairness findings to at-risk threshold choice
6. **Bias mitigation** via post-hoc threshold optimization

## Key Findings

- **Classical models match SOTA**: Elastic Net (AUC=0.848) matched or exceeded LightGBM, CatBoost, and HistGradientBoosting
- **Significant fairness disparities**: Hispanic students identified at 2.5x the rate of White students (non-overlapping bootstrap CIs)
- **Calibration unfairness**: Black students experienced 3.35x higher calibration error than White students
- **Intersectional invisibility**: High-SES Black students had 0% TPR -- the model operates as a "poverty detector"
- **Temporal paradox**: More longitudinal data improved accuracy but did not resolve fairness disparities
- **Threshold fragility**: Only 1 of 4 at-risk thresholds passed equal opportunity criteria

## Data Source

**ECLS-K:2011 (Early Childhood Longitudinal Study, Kindergarten Class of 2010-11)**
- Nationally representative sample of 18,174 U.S. children
- Followed from kindergarten (2010-11) through 5th grade (2015-16)
- Free public-use data available from [NCES](https://nces.ed.gov/ecls/dataproducts.asp)

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/kismatraj/ecls-fairness-study.git
cd ecls-fairness-study

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

Download the ECLS-K:2011 Kindergarten-Fifth Grade Public-Use Data File from:
https://nces.ed.gov/ecls/dataproducts.asp

Place the data file in `data/raw/`

### 3. Run Analysis

```bash
# Run full pipeline
python scripts/run_pipeline.py --config config.yaml

# Or run individual steps
python scripts/run_pipeline.py --step data_prep
python scripts/run_pipeline.py --step train_models
python scripts/run_pipeline.py --step fairness_analysis
python scripts/run_pipeline.py --step figures
python scripts/run_pipeline.py --step descriptives
python scripts/run_pipeline.py --step latex_tables
python scripts/run_pipeline.py --step sensitivity
python scripts/run_pipeline.py --step math_outcome
```

### 4. Run Tests

```bash
pytest
```

## Project Structure

```
ecls-fairness-study/
├── CLAUDE.md              # Instructions for Claude Code
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── config.yaml            # Configuration settings
│
├── data/
│   ├── raw/               # Original ECLS data (gitignored)
│   └── processed/         # Cleaned datasets
│
├── src/
│   ├── __init__.py         # Package init
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── models.py           # ML model training (7 algorithms)
│   ├── fairness.py         # Fairness metrics, CI, calibration, intersectional
│   ├── explainability.py   # SHAP, permutation importance, PDP, LIME
│   ├── visualization.py    # Publication-quality figures
│   ├── temporal.py         # Temporal generalization analysis
│   ├── descriptives.py     # Table 1 generation
│   ├── latex_tables.py     # LaTeX table generation
│   └── sensitivity.py      # Threshold sensitivity & outcome comparison
│
├── scripts/
│   └── run_pipeline.py     # Main execution script
│
├── paper/
│   └── main.tex            # Manuscript (LaTeX)
│
├── tests/                  # Test suite (71 tests)
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_models.py
│   ├── test_fairness.py
│   ├── test_temporal.py
│   ├── test_descriptives.py
│   ├── test_visualization.py
│   ├── test_latex_tables.py
│   ├── test_sensitivity.py
│   └── test_explainability.py
│
└── results/
    ├── models/             # Saved .joblib models
    ├── figures/            # Publication figures (PNG, 300 DPI)
    └── tables/             # Results tables (CSV + LaTeX)
```

## ML Algorithms

| Algorithm | Type | AUC |
|-----------|------|-----|
| Elastic Net | Classical (regularized linear) | 0.848 |
| Logistic Regression | Classical (linear) | 0.847 |
| CatBoost | SOTA gradient boosting | 0.846 |
| Random Forest | Classical (ensemble) | 0.841 |
| XGBoost | Gradient boosting | 0.840 |
| HistGradientBoosting | SOTA gradient boosting | 0.839 |
| LightGBM | SOTA gradient boosting | 0.837 |

## Fairness Analysis

### Metrics
- **Group fairness**: TPR, FPR, PPV parity with bootstrap 95% CIs (500 iterations)
- **Calibration fairness**: ECE and MCE by demographic group
- **Intersectional fairness**: Race x SES quintile subgroup analysis
- **Criteria**: Equal opportunity, equalized odds, statistical parity (0.80 threshold)

### Explainability
- SHAP values (global and local feature attribution)
- Permutation importance with bootstrap CIs
- Fairness-aware SHAP (importance stratified by demographic group)
- SHAP vs. permutation importance agreement analysis

## Key Variables

### Outcomes
- `X9RTHETA` - 5th grade reading score (IRT theta)
- `X9MTHETA` - 5th grade math score (IRT theta)

### Protected Attributes
- `X_RACETH_R` - Race/ethnicity (White, Black, Hispanic, Asian, Other)
- `X1SESQ5` - SES quintile (1-5)
- `X12LANGST` - Home language (English, Non-English)

### Predictors
- Baseline cognitive scores (K-2nd grade reading and math theta)
- Executive function measures
- Teacher-rated approaches to learning
- Demographic variables

## Output Files

### Figures (9 main text + supplementary)
- `model_comparison.png` - Performance across all 7 models
- `shap_summary.png` - SHAP beeswarm plot
- `fairness_tpr_with_ci.png` - TPR by group with confidence intervals
- `calibration_error_comparison.png` - ECE/MCE by group
- `intersectional_fairness_heatmap.png` - Race x SES TPR heatmap
- `temporal_performance_trend.png` - AUC across developmental windows
- `temporal_disparity_heatmap.png` - TPR ratios across scenarios
- `roc_curves_by_group.png` - ROC curves by demographic group
- `calibration_curves_by_group.png` - Calibration curves by group

### Tables (CSV + LaTeX)
- `model_performance` - 7-model comparison
- `fairness_metrics_with_ci` - Group fairness with bootstrap CIs
- `fairness_disparities` - Disparity ratios vs. White reference
- `calibration_fairness` - ECE/MCE by group
- `temporal_best_model_summary` - Performance across temporal scenarios
- `sensitivity_criteria` - Fairness pass/fail by threshold
- `table1_descriptives` - Sample characteristics

## Configuration

All settings in `config.yaml`:
- Data paths and variable lists
- Model hyperparameter grids (7 algorithms)
- Fairness thresholds and bootstrap settings
- Temporal scenario definitions
- Explainability method settings

## Citation

If you use this code, please cite:

```
[Citation pending publication]
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Data: National Center for Education Statistics (NCES), ECLS-K:2011
- Fairness metrics: [Fairlearn](https://fairlearn.org/)
- Explainability: [SHAP](https://github.com/slundberg/shap)
