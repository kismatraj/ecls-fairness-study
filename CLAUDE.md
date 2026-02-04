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
  - Supports logistic regression, elastic net, random forest, XGBoost
  - 5-fold CV with grid search, stratified train/test split
  - `get_feature_importance()` for interpretability

- **fairness.py**: Fairness evaluation and mitigation
  - `FairnessEvaluator`: Computes TPR/FPR/PPV/accuracy by demographic group
  - `ThresholdOptimizer`: Post-processing threshold adjustment to equalize TPR
  - `compare_before_after_mitigation()` for pre/post comparison

- **visualization.py**: Publication figures (ROC curves, calibration plots by group)

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
