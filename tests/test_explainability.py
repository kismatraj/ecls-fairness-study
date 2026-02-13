"""Tests for explainability module."""

import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.explainability import (
    PermutationImportanceAnalyzer,
    compare_explanations,
    HAS_SHAP,
)


@pytest.fixture
def trained_model(sample_data):
    """Train a simple logistic regression for explainability tests."""
    features = ["X1RTHETK", "X1MTHETK", "X2RTHETK", "X2MTHETK"]
    X = sample_data[features]
    y = sample_data["X9RTHETA_at_risk"]
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    return model, X, y


def test_permutation_importance(trained_model):
    model, X, y = trained_model
    analyzer = PermutationImportanceAnalyzer(model, X, y.values)
    result = analyzer.compute_importance(n_repeats=5)
    assert len(result) == X.shape[1]
    assert "importance_mean" in result.columns
    assert "ci_lower" in result.columns
    assert "ci_upper" in result.columns


def test_permutation_importance_sorted(trained_model):
    model, X, y = trained_model
    analyzer = PermutationImportanceAnalyzer(model, X, y.values)
    result = analyzer.compute_importance(n_repeats=5)
    # Should be sorted descending
    assert result["importance_mean"].iloc[0] >= result["importance_mean"].iloc[-1]


def test_bootstrap_importance(trained_model):
    model, X, y = trained_model
    analyzer = PermutationImportanceAnalyzer(model, X, y.values)
    result = analyzer.bootstrap_importance(n_bootstrap=5, sample_fraction=0.5)
    assert len(result) == X.shape[1]
    assert "ci_2.5" in result.columns
    assert "ci_97.5" in result.columns


def test_compare_explanations():
    shap_df = pd.DataFrame({
        "feature": ["A", "B", "C"],
        "mean_abs_shap": [0.5, 0.3, 0.1],
    })
    perm_df = pd.DataFrame({
        "feature": ["A", "B", "C"],
        "importance_mean": [0.4, 0.35, 0.05],
    })
    result = compare_explanations(shap_df, perm_df, top_n=3)
    assert "shap_normalized" in result.columns
    assert "perm_normalized" in result.columns
    assert "agreement" in result.columns
    assert len(result) == 3
    # Top feature should have shap_normalized = 1.0
    assert result.iloc[0]["shap_normalized"] == 1.0


@pytest.mark.skipif(not HAS_SHAP, reason="shap not installed")
def test_shap_explainer(trained_model):
    from src.explainability import SHAPExplainer
    model, X, y = trained_model

    explainer = SHAPExplainer(model, X, explainer_type="linear")
    shap_values = explainer.compute_shap_values(X.iloc[:50])
    assert shap_values.shape == (50, X.shape[1])

    importance = explainer.get_feature_importance()
    assert len(importance) <= 20
    assert "mean_abs_shap" in importance.columns


@pytest.mark.skipif(not HAS_SHAP, reason="shap not installed")
def test_fairness_aware_shap(trained_model, sample_data):
    from src.explainability import SHAPExplainer
    model, X, y = trained_model
    groups = sample_data["race_ethnicity"].iloc[:len(X)]

    explainer = SHAPExplainer(model, X, explainer_type="linear")
    explainer.compute_shap_values(X)

    result = explainer.fairness_aware_shap(X, groups)
    assert "White" in result
    assert "differential_importance" in result
