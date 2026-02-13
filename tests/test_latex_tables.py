"""Tests for latex_tables module."""

import pandas as pd
import numpy as np
import pytest

from src.latex_tables import (
    _escape_latex,
    model_performance_to_latex,
    fairness_metrics_to_latex,
    disparity_table_to_latex,
    temporal_summary_to_latex,
    calibration_to_latex,
    generate_all_latex_tables,
)


def test_escape_latex():
    assert _escape_latex("a & b") == "a \\& b"
    assert _escape_latex("100%") == "100\\%"
    assert _escape_latex("$x$") == "\\$x\\$"
    assert _escape_latex("no_special") == "no\\_special"


def test_model_performance_to_latex():
    df = pd.DataFrame({
        "auc_roc": [0.848, 0.847],
        "accuracy": [0.851, 0.849],
        "precision": [0.675, 0.657],
        "recall": [0.283, 0.281],
        "f1": [0.399, 0.394],
        "brier_score": [0.108, 0.108],
    }, index=["elastic_net", "logistic_regression"])

    latex = model_performance_to_latex(df)
    assert "\\begin{table}" in latex
    assert "\\end{table}" in latex
    assert "0.848" in latex
    assert "Elastic Net" in latex


def test_fairness_metrics_to_latex():
    df = pd.DataFrame({
        "Group": ["White", "Black"],
        "N": [1462, 300],
        "TPR": [0.160, 0.296],
        "TPR_CI_lower": [0.113, 0.207],
        "TPR_CI_upper": [0.206, 0.388],
        "FPR": [0.011, 0.095],
        "FPR_CI_lower": [0.006, 0.052],
        "FPR_CI_upper": [0.017, 0.133],
        "PPV": [0.660, 0.508],
        "PPV_CI_lower": [0.500, 0.341],
        "PPV_CI_upper": [0.794, 0.667],
    })

    latex = fairness_metrics_to_latex(df)
    assert "\\begin{table}" in latex
    assert "White" in latex
    assert "1,462" in latex


def test_disparity_table_to_latex():
    df = pd.DataFrame({
        "Group": ["Hispanic", "Black"],
        "vs": ["White", "White"],
        "TPR Ratio": ["2.458", "1.851"],
        "TPR Diff": ["+0.233", "+0.136"],
        "FPR Ratio": ["5.662", "8.494"],
        "FPR Diff": ["+0.052", "+0.084"],
        "Disparate Impact": ["No", "No"],
    })

    latex = disparity_table_to_latex(df)
    assert "\\begin{table}" in latex
    assert "2.458" in latex


def test_temporal_summary_to_latex():
    df = pd.DataFrame({
        "scenario_label": ["K Fall Only", "K Fall + Spring"],
        "best_model": ["logistic_regression", "logistic_regression"],
        "n_features": [7, 10],
        "auc_roc": [0.799, 0.822],
        "accuracy": [0.837, 0.844],
        "f1": [0.343, 0.372],
        "brier_score": [0.119, 0.113],
    })

    latex = temporal_summary_to_latex(df)
    assert "\\begin{table}" in latex
    assert "K Fall Only" in latex
    assert "0.799" in latex


def test_calibration_to_latex():
    df = pd.DataFrame({
        "Group": ["White", "Black"],
        "N": [1462, 300],
        "ECE": [0.022, 0.074],
        "MCE": [0.112, 0.456],
        "Brier": [0.082, 0.162],
        "ECE_ratio": [1.00, 3.35],
    })

    latex = calibration_to_latex(df)
    assert "\\begin{table}" in latex
    assert "3.35" in latex


def test_generate_all_latex_tables(tmp_path):
    # Create sample CSV files
    perf = pd.DataFrame({
        "auc_roc": [0.848],
        "accuracy": [0.851],
        "precision": [0.675],
        "recall": [0.283],
        "f1": [0.399],
        "brier_score": [0.108],
    }, index=["elastic_net"])
    perf.index.name = "model"
    perf.to_csv(tmp_path / "model_performance.csv")

    saved = generate_all_latex_tables(str(tmp_path))
    assert len(saved) >= 1
    assert any("model_performance.tex" in s for s in saved)
