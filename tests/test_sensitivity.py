"""Tests for sensitivity analysis module."""

import pandas as pd
import numpy as np
import pytest

from src.sensitivity import SensitivityAnalyzer, OutcomeComparisonAnalyzer


@pytest.fixture
def sensitivity_config(sample_config):
    """Config with sensitivity-relevant settings."""
    config = sample_config.copy()
    config["model"]["algorithms"] = {
        "logistic_regression": {"enabled": True},
    }
    return config


def test_sensitivity_analyzer_init(sensitivity_config):
    analyzer = SensitivityAnalyzer(sensitivity_config, percentiles=[10, 25])
    assert analyzer.percentiles == [10, 25]
    assert len(analyzer.results) == 0


def test_sensitivity_run_threshold(sample_data, sensitivity_config):
    analyzer = SensitivityAnalyzer(
        sensitivity_config, percentiles=[20, 25]
    )
    results = analyzer.run_threshold_sensitivity(
        sample_data,
        outcome_var="X9RTHETA",
        group_col="race_ethnicity",
        model_names=["logistic_regression"],
    )
    assert 20 in results
    assert 25 in results
    assert "best_model" in results[20]
    assert "fairness_criteria" in results[25]


def test_sensitivity_compare_performance(sample_data, sensitivity_config):
    analyzer = SensitivityAnalyzer(
        sensitivity_config, percentiles=[20, 25]
    )
    analyzer.run_threshold_sensitivity(
        sample_data,
        outcome_var="X9RTHETA",
        group_col="race_ethnicity",
        model_names=["logistic_regression"],
    )
    perf_df = analyzer.compare_performance()
    assert len(perf_df) == 2
    assert "auc_roc" in perf_df.columns
    assert "percentile" in perf_df.columns


def test_sensitivity_compare_criteria(sample_data, sensitivity_config):
    analyzer = SensitivityAnalyzer(
        sensitivity_config, percentiles=[20, 25]
    )
    analyzer.run_threshold_sensitivity(
        sample_data,
        outcome_var="X9RTHETA",
        group_col="race_ethnicity",
        model_names=["logistic_regression"],
    )
    criteria_df = analyzer.compare_criteria()
    assert len(criteria_df) == 2
    assert "prevalence" in criteria_df.columns


def test_sensitivity_save_results(sample_data, sensitivity_config, tmp_path):
    analyzer = SensitivityAnalyzer(
        sensitivity_config, percentiles=[25]
    )
    analyzer.run_threshold_sensitivity(
        sample_data,
        outcome_var="X9RTHETA",
        group_col="race_ethnicity",
        model_names=["logistic_regression"],
    )
    saved = analyzer.save_results(str(tmp_path))
    assert len(saved) == 4
    assert all(str(tmp_path) in s for s in saved)


def test_outcome_comparison_init(sensitivity_config):
    analyzer = OutcomeComparisonAnalyzer(sensitivity_config)
    assert len(analyzer.results) == 0


def test_outcome_comparison_run(sample_data, sensitivity_config):
    analyzer = OutcomeComparisonAnalyzer(sensitivity_config)
    result = analyzer.run_outcome(
        sample_data,
        outcome_name="reading",
        outcome_var="X9RTHETA",
        group_col="race_ethnicity",
        model_names=["logistic_regression"],
    )
    assert "best_model" in result
    assert "best_auc" in result
    assert 0 < result["best_auc"] < 1


def test_outcome_comparison_compare(sample_data, sensitivity_config):
    analyzer = OutcomeComparisonAnalyzer(sensitivity_config)
    analyzer.run_outcome(
        sample_data, "reading", "X9RTHETA",
        model_names=["logistic_regression"],
    )
    analyzer.run_outcome(
        sample_data, "math", "X9MTHETA",
        model_names=["logistic_regression"],
    )
    perf = analyzer.compare_performance()
    assert len(perf) == 2
    assert set(perf["outcome"]) == {"reading", "math"}
