"""Tests for descriptives and LaTeX modules."""

import pandas as pd
import pytest
import tempfile
from pathlib import Path

from src.descriptives import generate_table1, generate_table1_latex, save_table1
from src.latex_tables import (
    model_performance_to_latex,
    fairness_metrics_to_latex,
    temporal_summary_to_latex,
)


def test_generate_table1(sample_data):
    table = generate_table1(sample_data)
    assert isinstance(table, pd.DataFrame)
    assert "Variable" in table.columns
    assert "Overall" in table.columns
    assert "White" in table.columns
    # Should have N row
    assert any(table["Variable"] == "N")


def test_table1_latex(sample_data):
    table = generate_table1(sample_data)
    latex = generate_table1_latex(table)
    assert "\\begin{table}" in latex
    assert "\\end{table}" in latex
    assert "\\toprule" in latex


def test_save_table1(sample_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        saved = save_table1(sample_data, tmpdir)
        assert len(saved) == 2
        assert any(s.endswith(".csv") for s in saved)
        assert any(s.endswith(".tex") for s in saved)


def test_model_performance_latex():
    df = pd.DataFrame({
        "auc_roc": [0.85, 0.83],
        "accuracy": [0.80, 0.78],
        "precision": [0.70, 0.68],
        "recall": [0.60, 0.58],
        "f1": [0.65, 0.63],
        "brier_score": [0.12, 0.14],
    }, index=["logistic_regression", "random_forest"])

    latex = model_performance_to_latex(df)
    assert "Logistic Regression" in latex
    assert "0.850" in latex
    assert "\\begin{tabular}" in latex


def test_fairness_metrics_latex():
    df = pd.DataFrame({
        "Group": ["White", "Black"],
        "N": [100, 50],
        "TPR": [0.5, 0.4],
        "TPR_CI_lower": [0.4, 0.3],
        "TPR_CI_upper": [0.6, 0.5],
        "FPR": [0.1, 0.15],
        "FPR_CI_lower": [0.05, 0.1],
        "FPR_CI_upper": [0.15, 0.2],
        "PPV": [0.7, 0.6],
        "PPV_CI_lower": [0.6, 0.5],
        "PPV_CI_upper": [0.8, 0.7],
    })

    latex = fairness_metrics_to_latex(df)
    assert "White" in latex
    assert "95\\% CI" in latex


def test_temporal_summary_latex():
    df = pd.DataFrame({
        "scenario": ["k_only"],
        "scenario_label": ["K Fall Only"],
        "n_features": [7],
        "best_model": ["logistic_regression"],
        "auc_roc": [0.80],
        "accuracy": [0.83],
        "f1": [0.35],
        "brier_score": [0.12],
    })

    latex = temporal_summary_to_latex(df)
    assert "K Fall Only" in latex
    assert "Logistic Regression" in latex
