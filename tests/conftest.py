"""Shared test fixtures."""

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_data():
    """Create a small synthetic dataset mimicking ECLS structure."""
    rng = np.random.RandomState(42)
    n = 500

    df = pd.DataFrame(
        {
            # Cognitive measures
            "X1RTHETK": rng.normal(50, 10, n),
            "X2RTHETK": rng.normal(55, 10, n),
            "X1MTHETK": rng.normal(50, 10, n),
            "X2MTHETK": rng.normal(55, 10, n),
            "X6DCCSSCR": rng.normal(60, 12, n),
            "X1TCHAPP": rng.uniform(1, 4, n),
            "X2TCHAPP": rng.uniform(1, 4, n),
            "X4TCHAPP": rng.uniform(1, 4, n),
            # Outcomes
            "X9RTHETA": rng.normal(100, 20, n),
            "X9MTHETA": rng.normal(100, 20, n),
            # Demographics
            "X_CHSEX_R": rng.choice([1, 2], n),
            "X_RACETH_R": rng.choice(
                [1, 2, 3, 4, 5], n, p=[0.5, 0.12, 0.24, 0.05, 0.09]
            ),
            "X1SESQ5": rng.choice([1, 2, 3, 4, 5], n),
            "X12LANGST": rng.choice([1, 2], n, p=[0.8, 0.2]),
        }
    )

    # Derived variables
    race_map = {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other"}
    df["race_ethnicity"] = df["X_RACETH_R"].map(race_map)

    # At-risk indicators
    for var in ["X9RTHETA", "X9MTHETA"]:
        threshold = df[var].quantile(0.25)
        df[f"{var}_at_risk"] = (df[var] < threshold).astype(int)

    return df


@pytest.fixture
def sample_config():
    """Minimal config for testing."""
    return {
        "paths": {
            "raw_data": "data/raw/",
            "processed_data": "data/processed/",
            "results": "results/",
            "figures": "results/figures/",
            "tables": "results/tables/",
            "models": "results/models/",
        },
        "data": {
            "filename": "childK5p.dat",
            "missing_codes": [-1, -7, -8, -9],
        },
        "variables": {
            "outcomes": {"reading": "X9RTHETA", "math": "X9MTHETA"},
            "at_risk_percentile": 25,
            "demographics": {
                "race": "X_RACETH_R",
                "sex": "X_CHSEX_R",
                "ses": "X1SESQ5",
                "language": "X12LANGST",
            },
            "predictors": {
                "baseline_cognitive": ["X1RTHETK", "X2RTHETK", "X1MTHETK", "X2MTHETK"],
                "executive_function": ["X6DCCSSCR"],
                "approaches_to_learning": ["X1TCHAPP", "X2TCHAPP", "X4TCHAPP"],
                "child_demographics": [
                    "X_CHSEX_R",
                    "X_RACETH_R",
                    "X1SESQ5",
                    "X12LANGST",
                ],
            },
        },
        "model": {
            "random_state": 42,
            "test_size": 0.30,
            "cv_folds": 3,
            "algorithms": {
                "logistic_regression": {"enabled": True},
            },
        },
        "fairness": {
            "reference_groups": {"race": "White"},
            "bootstrap_iterations": 20,
            "confidence_level": 0.95,
        },
        "temporal": {
            "enabled": True,
            "demographics": ["X_CHSEX_R", "X_RACETH_R", "X1SESQ5", "X12LANGST"],
            "scenarios": [
                {
                    "name": "k_only",
                    "label": "K Fall Only",
                    "features": ["X1RTHETK", "X1MTHETK", "X1TCHAPP"],
                },
                {
                    "name": "k_complete",
                    "label": "K Fall + Spring",
                    "features": [
                        "X1RTHETK",
                        "X1MTHETK",
                        "X1TCHAPP",
                        "X2RTHETK",
                        "X2MTHETK",
                        "X2TCHAPP",
                    ],
                },
            ],
        },
    }


@pytest.fixture
def binary_predictions():
    """Pre-computed predictions for fairness tests."""
    rng = np.random.RandomState(42)
    n = 300

    y_true = rng.choice([0, 1], n, p=[0.75, 0.25])
    y_prob = np.clip(y_true * 0.6 + rng.normal(0.3, 0.15, n), 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)
    groups = pd.Series(rng.choice(["White", "Black", "Hispanic"], n, p=[0.5, 0.2, 0.3]))

    return y_true, y_pred, y_prob, groups
