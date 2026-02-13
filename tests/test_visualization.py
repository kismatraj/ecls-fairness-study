"""Tests for visualization module."""

import pandas as pd
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests
import matplotlib.pyplot as plt

from src.visualization import (
    plot_roc_curves_by_group,
    plot_calibration_curves_by_group,
    plot_feature_importance,
    plot_fairness_with_ci,
    plot_calibration_error_comparison,
    plot_explanation_comparison,
    plot_model_comparison,
    plot_intersectional_heatmap,
    plot_temporal_performance_trend,
    plot_temporal_disparity_heatmap,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test to free memory."""
    yield
    plt.close("all")


def test_plot_roc_curves(binary_predictions):
    y_true, _, y_prob, groups = binary_predictions
    fig = plot_roc_curves_by_group(y_true, y_prob, groups)
    assert isinstance(fig, plt.Figure)


def test_plot_roc_curves_save(binary_predictions, tmp_path):
    y_true, _, y_prob, groups = binary_predictions
    path = str(tmp_path / "roc.png")
    fig = plot_roc_curves_by_group(y_true, y_prob, groups, save_path=path)
    assert (tmp_path / "roc.png").exists()


def test_plot_calibration_curves(binary_predictions):
    y_true, _, y_prob, groups = binary_predictions
    fig = plot_calibration_curves_by_group(y_true, y_prob, groups)
    assert isinstance(fig, plt.Figure)


def test_plot_feature_importance():
    df = pd.DataFrame({
        "feature": ["A", "B", "C", "D"],
        "importance": [0.5, 0.3, 0.15, 0.05],
    })
    fig = plot_feature_importance(df, top_n=4)
    assert isinstance(fig, plt.Figure)


def test_plot_fairness_with_ci():
    df = pd.DataFrame({
        "Group": ["White", "Black", "Hispanic"],
        "TPR": [0.16, 0.30, 0.39],
        "TPR_CI_lower": [0.11, 0.21, 0.33],
        "TPR_CI_upper": [0.21, 0.39, 0.46],
    })
    fig = plot_fairness_with_ci(df, metric="TPR")
    assert isinstance(fig, plt.Figure)


def test_plot_calibration_error_comparison():
    df = pd.DataFrame({
        "Group": ["White", "Black", "Hispanic"],
        "ECE": [0.022, 0.074, 0.036],
        "MCE": [0.112, 0.456, 0.115],
    })
    fig = plot_calibration_error_comparison(df)
    assert isinstance(fig, plt.Figure)


def test_plot_explanation_comparison():
    df = pd.DataFrame({
        "feature": ["X2MTHETK", "X2RTHETK", "X1SESQ5"],
        "shap_normalized": [1.0, 0.62, 0.62],
        "perm_normalized": [1.0, 0.50, 0.28],
    })
    fig = plot_explanation_comparison(df)
    assert isinstance(fig, plt.Figure)


def test_plot_model_comparison():
    df = pd.DataFrame({
        "auc_roc": [0.848, 0.847, 0.841],
        "accuracy": [0.851, 0.849, 0.848],
        "f1": [0.399, 0.394, 0.395],
    }, index=["elastic_net", "logistic_regression", "random_forest"])
    fig = plot_model_comparison(df)
    assert isinstance(fig, plt.Figure)


def test_plot_temporal_performance_trend():
    df = pd.DataFrame({
        "scenario_label": ["K Fall Only", "K Fall + Spring"],
        "auc_roc": [0.799, 0.822],
        "accuracy": [0.837, 0.844],
        "f1": [0.343, 0.372],
    })
    fig = plot_temporal_performance_trend(df)
    assert isinstance(fig, plt.Figure)


def test_plot_temporal_disparity_heatmap():
    df = pd.DataFrame({
        "scenario_label": ["K Fall Only", "K Fall Only", "K Fall + Spring", "K Fall + Spring"],
        "Group": ["Black", "Hispanic", "Black", "Hispanic"],
        "TPR Ratio": [1.5, 2.0, 1.6, 2.1],
    })
    fig = plot_temporal_disparity_heatmap(df)
    assert isinstance(fig, plt.Figure)
