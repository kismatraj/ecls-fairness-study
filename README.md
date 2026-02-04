# ECLS-K:2011 Algorithmic Fairness Study

**Analyzing temporal generalization and algorithmic fairness of machine learning models for predicting cognitive development in children.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project examines whether machine learning models for predicting children's cognitive outcomes:
1. **Generalize temporally** - Do models trained on early elementary data (K-2nd) accurately predict later outcomes (3rd-5th grade)?
2. **Perform equitably** - Do models show similar accuracy across race/ethnicity, socioeconomic status, and language groups?
3. **Can be improved** - Does post-hoc threshold adjustment reduce demographic disparities?

## Data Source

**ECLS-K:2011 (Early Childhood Longitudinal Study, Kindergarten Class of 2010-11)**
- Nationally representative sample of 18,174 U.S. children
- Followed from kindergarten (2010-11) through 5th grade (2015-16)
- Free public-use data available from [NCES](https://nces.ed.gov/ecls/dataproducts.asp)

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/ecls-fairness-study.git
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
```

## Project Structure

```
ecls-fairness-study/
├── CLAUDE.md              # Instructions for Claude Code
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── config.yaml           # Configuration settings
│
├── data/
│   ├── raw/              # Original ECLS data (gitignored)
│   └── processed/        # Cleaned datasets
│
├── src/
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── models.py         # ML model training
│   ├── fairness.py       # Fairness metrics and mitigation
│   └── visualization.py  # Plotting functions
│
├── scripts/
│   └── run_pipeline.py   # Main execution script
│
├── notebooks/            # Jupyter notebooks for exploration
│
└── results/
    ├── models/           # Saved model files
    ├── figures/          # Publication figures
    └── tables/           # Results tables
```

## Key Variables

### Outcomes
- `X9RTHETA` - 5th grade reading score (IRT theta)
- `X9MTHETA` - 5th grade math score (IRT theta)

### Protected Attributes
- `X_RACETH_R` - Race/ethnicity (White, Black, Hispanic, Asian, Other)
- `X1SESQ5` - SES quintile (1-5)
- `X12LANGST` - Home language (English, Non-English)

### Predictors
- Baseline cognitive scores (K-2nd grade)
- Executive function measures
- Teacher-rated approaches to learning
- Demographic variables

## Methods

### Models
- Logistic Regression
- Elastic Net
- Random Forest
- XGBoost

### Fairness Metrics
- True Positive Rate (TPR) parity
- False Positive Rate (FPR) parity
- Positive Predictive Value (PPV) parity
- Disparate Impact Ratio

### Bias Mitigation
- Post-processing threshold adjustment to equalize TPR across groups

## Expected Outputs

### Tables
1. Sample characteristics by demographic group
2. Model performance metrics (AUC, accuracy, sensitivity)
3. Fairness metrics by group
4. Pre/post mitigation comparison

### Figures
1. ROC curves by demographic group
2. Calibration curves by group
3. Feature importance
4. Fairness-accuracy tradeoff

## Citation

If you use this code, please cite:

```
[Your citation here]
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Data: National Center for Education Statistics (NCES)
- Fairness metrics: [Fairlearn](https://fairlearn.org/)
