"""Tests for fairness module."""

import pandas as pd
import numpy as np

from src.fairness import (
    FairnessEvaluator,
    ThresholdOptimizer,
    FairnessConfidenceIntervals,
    CalibrationFairnessAnalyzer,
    IntersectionalFairnessAnalyzer,
)


class TestFairnessEvaluator:
    def test_compute_group_metrics(self, binary_predictions):
        y_true, y_pred, y_prob, groups = binary_predictions
        evaluator = FairnessEvaluator(y_true, y_pred, y_prob, groups)
        metrics = evaluator.compute_group_metrics("White")

        assert metrics is not None
        assert metrics.group == "White"
        assert metrics.n > 0
        assert 0 <= metrics.tpr <= 1
        assert 0 <= metrics.fpr <= 1

    def test_compute_all_groups(self, binary_predictions):
        y_true, y_pred, y_prob, groups = binary_predictions
        evaluator = FairnessEvaluator(y_true, y_pred, y_prob, groups)
        all_metrics = evaluator.compute_all_group_metrics()

        assert len(all_metrics) == 3  # White, Black, Hispanic
        for group, m in all_metrics.items():
            assert m.n > 0

    def test_summary_table(self, binary_predictions):
        y_true, y_pred, y_prob, groups = binary_predictions
        evaluator = FairnessEvaluator(y_true, y_pred, y_prob, groups)
        table = evaluator.get_summary_table()

        assert isinstance(table, pd.DataFrame)
        assert "Group" in table.columns
        assert "TPR" in table.columns
        assert len(table) == 3

    def test_disparity_table(self, binary_predictions):
        y_true, y_pred, y_prob, groups = binary_predictions
        evaluator = FairnessEvaluator(
            y_true, y_pred, y_prob, groups, reference_group="White"
        )
        evaluator.compute_all_group_metrics()
        disparity = evaluator.get_disparity_table()

        assert isinstance(disparity, pd.DataFrame)
        assert len(disparity) == 2  # Black and Hispanic vs White
        assert "TPR Ratio" in disparity.columns

    def test_fairness_criteria(self, binary_predictions):
        y_true, y_pred, y_prob, groups = binary_predictions
        evaluator = FairnessEvaluator(
            y_true, y_pred, y_prob, groups, reference_group="White"
        )
        evaluator.compute_all_group_metrics()
        evaluator.compute_fairness_metrics()
        criteria = evaluator.check_fairness_criteria()

        assert "equal_opportunity" in criteria
        assert "equalized_odds" in criteria
        assert "statistical_parity" in criteria

    def test_nan_groups_filtered(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8])
        groups = pd.Series(["A", "A", "B", "B", np.nan, np.nan])

        evaluator = FairnessEvaluator(y_true, y_pred, y_prob, groups)
        assert len(evaluator.unique_groups) == 2


class TestThresholdOptimizer:
    def test_fit_transform(self, binary_predictions):
        y_true, _, y_prob, groups = binary_predictions
        optimizer = ThresholdOptimizer(constraint="equal_opportunity")
        y_adjusted = optimizer.fit_transform(y_true, y_prob, groups)

        assert len(y_adjusted) == len(y_true)
        assert set(np.unique(y_adjusted)).issubset({0, 1})
        assert len(optimizer.thresholds) == 3


class TestFairnessConfidenceIntervals:
    def test_bootstrap_metrics(self, binary_predictions):
        y_true, _, y_prob, groups = binary_predictions
        ci = FairnessConfidenceIntervals(y_true, y_prob, groups, n_bootstrap=20)
        result = ci.bootstrap_group_metrics()

        assert isinstance(result, pd.DataFrame)
        assert "TPR" in result.columns
        assert "TPR_CI_lower" in result.columns
        assert "TPR_CI_upper" in result.columns
        # CI lower should be <= mean <= CI upper
        for _, row in result.iterrows():
            assert row["TPR_CI_lower"] <= row["TPR"] + 0.01  # small tolerance
            assert row["TPR"] <= row["TPR_CI_upper"] + 0.01


class TestCalibrationFairness:
    def test_analyze_calibration(self, binary_predictions):
        y_true, _, y_prob, groups = binary_predictions
        analyzer = CalibrationFairnessAnalyzer(y_true, y_prob, groups, n_bins=5)
        result = analyzer.analyze_calibration_by_group()

        assert isinstance(result, pd.DataFrame)
        assert "ECE" in result.columns
        assert "MCE" in result.columns

    def test_check_sufficiency(self, binary_predictions):
        y_true, _, y_prob, groups = binary_predictions
        analyzer = CalibrationFairnessAnalyzer(y_true, y_prob, groups, n_bins=5)
        result = analyzer.check_sufficiency()

        assert "satisfies_sufficiency" in result
        assert "max_ece_ratio" in result


class TestIntersectionalFairness:
    def test_compute_metrics(self, binary_predictions):
        y_true, y_pred, y_prob, groups = binary_predictions
        rng = np.random.RandomState(42)
        protected = pd.DataFrame(
            {
                "race": groups.values,
                "ses": rng.choice(["Low", "High"], len(groups)),
            }
        )

        analyzer = IntersectionalFairnessAnalyzer(
            y_true, y_pred, y_prob, protected, min_group_size=10
        )
        metrics = analyzer.compute_intersectional_metrics()
        assert len(metrics) > 0

    def test_summary_table(self, binary_predictions):
        y_true, y_pred, y_prob, groups = binary_predictions
        rng = np.random.RandomState(42)
        protected = pd.DataFrame(
            {
                "race": groups.values,
                "ses": rng.choice(["Low", "High"], len(groups)),
            }
        )

        analyzer = IntersectionalFairnessAnalyzer(
            y_true, y_pred, y_prob, protected, min_group_size=10
        )
        table = analyzer.get_summary_table()
        assert isinstance(table, pd.DataFrame)
        assert "TPR" in table.columns
        # Values should be numeric, not strings
        assert table["TPR"].dtype in [np.float64, float]
