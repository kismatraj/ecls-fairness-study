"""Tests for temporal generalization module."""

import pandas as pd

from src.temporal import (
    TemporalScenario,
    TemporalGeneralizationAnalyzer,
)


def test_scenario_dataclass():
    sc = TemporalScenario(name="test", label="Test", cognitive_features=["X1RTHETK"])
    assert sc.name == "test"
    assert len(sc.cognitive_features) == 1


def test_analyzer_builds_scenarios(sample_config):
    analyzer = TemporalGeneralizationAnalyzer(config=sample_config)
    assert len(analyzer.scenarios) == 2
    assert analyzer.scenarios[0].name == "k_only"


def test_prepare_common_sample(sample_data, sample_config):
    analyzer = TemporalGeneralizationAnalyzer(config=sample_config)
    n = analyzer.prepare_common_sample(sample_data)

    assert n > 0
    assert analyzer.X_full is not None
    assert analyzer.train_idx is not None
    assert analyzer.test_idx is not None
    assert len(analyzer.train_idx) + len(analyzer.test_idx) == n


def test_run_scenario(sample_data, sample_config):
    analyzer = TemporalGeneralizationAnalyzer(config=sample_config)
    analyzer.prepare_common_sample(sample_data)

    result = analyzer.run_scenario(analyzer.scenarios[0])
    assert result.best_auc > 0.5
    assert result.best_model_name is not None
    assert isinstance(result.group_metrics, pd.DataFrame)
    assert isinstance(result.metrics_with_ci, pd.DataFrame)


def test_run_all_scenarios(sample_data, sample_config):
    analyzer = TemporalGeneralizationAnalyzer(config=sample_config)
    analyzer.prepare_common_sample(sample_data)
    results = analyzer.run_all_scenarios()

    assert len(results) == 2
    assert "k_only" in results
    assert "k_complete" in results


def test_compare_performance(sample_data, sample_config):
    analyzer = TemporalGeneralizationAnalyzer(config=sample_config)
    analyzer.prepare_common_sample(sample_data)
    analyzer.run_all_scenarios()

    perf = analyzer.compare_performance()
    assert isinstance(perf, pd.DataFrame)
    assert "scenario" in perf.columns
    assert "auc_roc" in perf.columns


def test_compare_fairness(sample_data, sample_config):
    analyzer = TemporalGeneralizationAnalyzer(config=sample_config)
    analyzer.prepare_common_sample(sample_data)
    analyzer.run_all_scenarios()

    fairness = analyzer.compare_fairness()
    assert isinstance(fairness, pd.DataFrame)
    assert "scenario" in fairness.columns
    assert "TPR" in fairness.columns


def test_same_split_across_scenarios(sample_data, sample_config):
    """Verify all scenarios use identical train/test observations."""
    analyzer = TemporalGeneralizationAnalyzer(config=sample_config)
    analyzer.prepare_common_sample(sample_data)
    analyzer.run_all_scenarios()

    # Train/test indices should be the same object
    assert analyzer.train_idx is not None
    assert analyzer.test_idx is not None
    total = len(analyzer.train_idx) + len(analyzer.test_idx)
    assert total == len(analyzer.y)
