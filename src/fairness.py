"""
Fairness Module (2025 State-of-the-Art)
========================================

Algorithmic fairness metrics, evaluation, and bias mitigation.

Includes:
- Group fairness metrics (TPR/FPR/PPV parity)
- Intersectional fairness analysis
- Individual fairness metrics
- Bootstrap confidence intervals for fairness metrics
- Calibration-based fairness (sufficiency)
- AIF360 integration for additional metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from itertools import combinations

from sklearn.metrics import confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from scipy import stats

try:
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        equalized_odds_ratio,
        false_positive_rate,
        false_negative_rate,
        selection_rate
    )
    HAS_FAIRLEARN = True
except ImportError:
    HAS_FAIRLEARN = False
    logging.warning("fairlearn not installed. Some features unavailable.")

try:
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from aif360.datasets import BinaryLabelDataset
    HAS_AIF360 = True
except ImportError:
    HAS_AIF360 = False
    logging.warning("aif360 not installed. Some advanced metrics unavailable.")

logger = logging.getLogger(__name__)


@dataclass
class GroupMetrics:
    """Metrics for a single demographic group."""
    group: str
    n: int
    prevalence: float
    tpr: float  # True Positive Rate (Sensitivity)
    fpr: float  # False Positive Rate
    ppv: float  # Positive Predictive Value (Precision)
    npv: float  # Negative Predictive Value
    accuracy: float
    positive_rate: float  # Statistical parity


class FairnessEvaluator:
    """
    Evaluate algorithmic fairness across demographic groups.
    
    Attributes:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        groups: Demographic group labels
        reference_group: Reference group for comparisons
    """
    
    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        groups: pd.Series,
        reference_group: Optional[str] = None
    ):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob)
        self.groups = pd.Series(groups).reset_index(drop=True)

        # Filter out NaN groups and get unique sorted values
        valid_groups = self.groups.dropna().unique()
        self.unique_groups = sorted([g for g in valid_groups if pd.notna(g)])
        self.reference_group = reference_group or self.unique_groups[0]

        self.group_metrics = {}
        self.fairness_metrics = {}
    
    def compute_group_metrics(self, group: str) -> GroupMetrics:
        """
        Compute metrics for a single group.
        
        Args:
            group: Group label
        
        Returns:
            GroupMetrics dataclass
        """
        mask = self.groups == group
        
        y_true_g = self.y_true[mask]
        y_pred_g = self.y_pred[mask]
        
        n = len(y_true_g)
        
        if n == 0:
            return None
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            y_true_g, y_pred_g, labels=[0, 1]
        ).ravel() if len(np.unique(y_true_g)) > 1 else (0, 0, 0, 0)
        
        # Metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / n if n > 0 else 0
        positive_rate = y_pred_g.mean()
        prevalence = y_true_g.mean()
        
        return GroupMetrics(
            group=group,
            n=n,
            prevalence=prevalence,
            tpr=tpr,
            fpr=fpr,
            ppv=ppv,
            npv=npv,
            accuracy=accuracy,
            positive_rate=positive_rate
        )
    
    def compute_all_group_metrics(self) -> Dict[str, GroupMetrics]:
        """Compute metrics for all groups."""
        self.group_metrics = {}
        
        for group in self.unique_groups:
            metrics = self.compute_group_metrics(group)
            if metrics:
                self.group_metrics[group] = metrics
        
        return self.group_metrics
    
    def compute_fairness_metrics(self) -> Dict[str, Dict]:
        """
        Compute fairness metrics comparing groups.
        
        Returns:
            Dictionary with fairness metrics
        """
        if not self.group_metrics:
            self.compute_all_group_metrics()
        
        ref_metrics = self.group_metrics.get(self.reference_group)
        if not ref_metrics:
            logger.warning(f"Reference group {self.reference_group} not found")
            return {}
        
        fairness = {}
        
        for group, metrics in self.group_metrics.items():
            if group == self.reference_group:
                continue
            
            # Compute disparities relative to reference
            fairness[group] = {
                "tpr_ratio": metrics.tpr / ref_metrics.tpr if ref_metrics.tpr > 0 else np.nan,
                "tpr_diff": metrics.tpr - ref_metrics.tpr,
                "fpr_ratio": metrics.fpr / ref_metrics.fpr if ref_metrics.fpr > 0 else np.nan,
                "fpr_diff": metrics.fpr - ref_metrics.fpr,
                "ppv_ratio": metrics.ppv / ref_metrics.ppv if ref_metrics.ppv > 0 else np.nan,
                "ppv_diff": metrics.ppv - ref_metrics.ppv,
                "stat_parity_ratio": metrics.positive_rate / ref_metrics.positive_rate if ref_metrics.positive_rate > 0 else np.nan,
                "stat_parity_diff": metrics.positive_rate - ref_metrics.positive_rate
            }
            
            # Flag disparate impact
            fairness[group]["disparate_impact"] = fairness[group]["tpr_ratio"] < 0.8
        
        self.fairness_metrics = fairness
        return fairness
    
    def get_summary_table(self) -> pd.DataFrame:
        """
        Create summary table of group metrics.
        
        Returns:
            DataFrame with metrics by group
        """
        if not self.group_metrics:
            self.compute_all_group_metrics()
        
        rows = []
        for group, m in self.group_metrics.items():
            rows.append({
                "Group": group,
                "N": m.n,
                "Prevalence": f"{m.prevalence:.1%}",
                "TPR": f"{m.tpr:.3f}",
                "FPR": f"{m.fpr:.3f}",
                "PPV": f"{m.ppv:.3f}",
                "Accuracy": f"{m.accuracy:.3f}",
                "Pos. Rate": f"{m.positive_rate:.1%}"
            })
        
        return pd.DataFrame(rows)
    
    def get_disparity_table(self) -> pd.DataFrame:
        """
        Create table of fairness disparities.
        
        Returns:
            DataFrame with disparity metrics
        """
        if not self.fairness_metrics:
            self.compute_fairness_metrics()
        
        rows = []
        for group, metrics in self.fairness_metrics.items():
            rows.append({
                "Group": group,
                "vs": self.reference_group,
                "TPR Ratio": f"{metrics['tpr_ratio']:.3f}",
                "TPR Diff": f"{metrics['tpr_diff']:+.3f}",
                "FPR Ratio": f"{metrics['fpr_ratio']:.3f}",
                "FPR Diff": f"{metrics['fpr_diff']:+.3f}",
                "Disparate Impact": "Yes" if metrics['disparate_impact'] else "No"
            })
        
        return pd.DataFrame(rows)
    
    def check_fairness_criteria(
        self,
        tpr_threshold: float = 0.8,
        fpr_threshold: float = 1.25
    ) -> Dict[str, bool]:
        """
        Check if model meets fairness criteria.
        
        Args:
            tpr_threshold: Minimum TPR ratio (80% rule)
            fpr_threshold: Maximum FPR ratio
        
        Returns:
            Dictionary with pass/fail for each criterion
        """
        if not self.fairness_metrics:
            self.compute_fairness_metrics()
        
        results = {
            "equal_opportunity": True,  # TPR parity
            "equalized_odds": True,     # TPR + FPR parity
            "statistical_parity": True  # Positive rate parity
        }
        
        for group, metrics in self.fairness_metrics.items():
            if metrics["tpr_ratio"] < tpr_threshold:
                results["equal_opportunity"] = False
                results["equalized_odds"] = False
            
            if metrics["fpr_ratio"] > fpr_threshold or metrics["fpr_ratio"] < (1/fpr_threshold):
                results["equalized_odds"] = False
            
            if metrics["stat_parity_ratio"] < tpr_threshold:
                results["statistical_parity"] = False
        
        return results


class ThresholdOptimizer:
    """
    Optimize classification thresholds to improve fairness.
    
    This implements post-processing threshold adjustment to equalize
    metrics across demographic groups.
    """
    
    def __init__(
        self,
        constraint: str = "equal_opportunity",
        target_metric: Optional[float] = None
    ):
        """
        Args:
            constraint: Fairness constraint type
                - 'equal_opportunity': Equalize TPR
                - 'equalized_odds': Equalize TPR and FPR
                - 'statistical_parity': Equalize positive rate
            target_metric: Target metric value (None = overall rate)
        """
        self.constraint = constraint
        self.target_metric = target_metric
        self.thresholds = {}
    
    def find_threshold_for_tpr(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target_tpr: float
    ) -> float:
        """
        Find threshold to achieve target TPR.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            target_tpr: Desired true positive rate
        
        Returns:
            Optimal threshold
        """
        # Get positive cases
        pos_mask = y_true == 1
        pos_probs = y_prob[pos_mask]
        
        if len(pos_probs) == 0:
            return 0.5
        
        # Threshold is percentile of positive class probabilities
        # To get TPR of X%, threshold should be at (1-X)th percentile
        threshold = np.percentile(pos_probs, (1 - target_tpr) * 100)
        
        return threshold
    
    def fit(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        groups: pd.Series
    ) -> Dict[str, float]:
        """
        Fit group-specific thresholds.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            groups: Group labels
        
        Returns:
            Dictionary of thresholds by group
        """
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        groups = pd.Series(groups).reset_index(drop=True)
        
        unique_groups = groups.unique()
        
        # Determine target metric
        if self.target_metric is None:
            # Use overall TPR at default threshold
            y_pred_default = (y_prob >= 0.5).astype(int)
            pos_mask = y_true == 1
            target = y_pred_default[pos_mask].mean()
        else:
            target = self.target_metric
        
        logger.info(f"Target TPR: {target:.3f}")
        
        # Find threshold for each group
        self.thresholds = {}
        
        for group in unique_groups:
            mask = groups == group
            
            threshold = self.find_threshold_for_tpr(
                y_true[mask],
                y_prob[mask],
                target
            )
            
            self.thresholds[group] = threshold
            logger.info(f"  {group}: threshold = {threshold:.4f}")
        
        return self.thresholds
    
    def transform(
        self,
        y_prob: np.ndarray,
        groups: pd.Series
    ) -> np.ndarray:
        """
        Apply group-specific thresholds.
        
        Args:
            y_prob: Predicted probabilities
            groups: Group labels
        
        Returns:
            Adjusted predictions
        """
        if not self.thresholds:
            raise ValueError("Must call fit() first")
        
        y_prob = np.array(y_prob)
        groups = pd.Series(groups).reset_index(drop=True)
        
        y_pred = np.zeros(len(y_prob), dtype=int)
        
        for group, threshold in self.thresholds.items():
            mask = groups == group
            y_pred[mask] = (y_prob[mask] >= threshold).astype(int)
        
        return y_pred
    
    def fit_transform(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        groups: pd.Series
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y_true, y_prob, groups)
        return self.transform(y_prob, groups)


def compute_calibration_by_group(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
    n_bins: int = 10
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute calibration curves by group.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        groups: Group labels
        n_bins: Number of bins for calibration
    
    Returns:
        Dictionary with calibration data per group
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    groups = pd.Series(groups).reset_index(drop=True)
    
    calibration = {}
    
    for group in groups.unique():
        mask = groups == group
        
        if mask.sum() < n_bins * 5:
            logger.warning(f"Group {group} has few samples for calibration")
            continue
        
        prob_true, prob_pred = calibration_curve(
            y_true[mask],
            y_prob[mask],
            n_bins=n_bins,
            strategy='uniform'
        )
        
        calibration[group] = (prob_true, prob_pred)
    
    return calibration


def compare_before_after_mitigation(
    y_true: np.ndarray,
    y_pred_before: np.ndarray,
    y_pred_after: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series
) -> pd.DataFrame:
    """
    Compare metrics before and after bias mitigation.
    
    Args:
        y_true: True labels
        y_pred_before: Predictions before mitigation
        y_pred_after: Predictions after mitigation
        y_prob: Predicted probabilities
        groups: Group labels
    
    Returns:
        DataFrame comparing metrics
    """
    # Evaluate before
    eval_before = FairnessEvaluator(y_true, y_pred_before, y_prob, groups)
    eval_before.compute_all_group_metrics()
    
    # Evaluate after
    eval_after = FairnessEvaluator(y_true, y_pred_after, y_prob, groups)
    eval_after.compute_all_group_metrics()
    
    # Compare
    rows = []
    for group in groups.unique():
        before = eval_before.group_metrics.get(group)
        after = eval_after.group_metrics.get(group)
        
        if before and after:
            rows.append({
                "Group": group,
                "TPR Before": f"{before.tpr:.3f}",
                "TPR After": f"{after.tpr:.3f}",
                "TPR Change": f"{after.tpr - before.tpr:+.3f}",
                "Accuracy Before": f"{before.accuracy:.3f}",
                "Accuracy After": f"{after.accuracy:.3f}",
                "Accuracy Change": f"{after.accuracy - before.accuracy:+.3f}"
            })
    
    return pd.DataFrame(rows)


def generate_fairness_report(
    evaluator: FairnessEvaluator,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a text report of fairness analysis.
    
    Args:
        evaluator: FairnessEvaluator with computed metrics
        output_path: Optional path to save report
    
    Returns:
        Report as string
    """
    report = []
    report.append("=" * 60)
    report.append("ALGORITHMIC FAIRNESS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Group metrics
    report.append("GROUP-LEVEL METRICS")
    report.append("-" * 40)
    
    summary = evaluator.get_summary_table()
    report.append(summary.to_string(index=False))
    report.append("")
    
    # Disparity metrics
    report.append("DISPARITY ANALYSIS")
    report.append("-" * 40)
    report.append(f"Reference group: {evaluator.reference_group}")
    report.append("")
    
    disparity = evaluator.get_disparity_table()
    report.append(disparity.to_string(index=False))
    report.append("")
    
    # Fairness criteria
    report.append("FAIRNESS CRITERIA CHECK")
    report.append("-" * 40)
    
    criteria = evaluator.check_fairness_criteria()
    for criterion, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        report.append(f"  {criterion}: {status}")
    
    report.append("")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Report saved to {output_path}")
    
    return report_text


# =============================================================================
# 2025 State-of-the-Art Fairness Metrics
# =============================================================================


@dataclass
class IntersectionalMetrics:
    """Metrics for an intersectional subgroup (e.g., Black + Low SES)."""
    subgroup: str
    n: int
    prevalence: float
    tpr: float
    fpr: float
    ppv: float
    accuracy: float
    positive_rate: float
    attributes: Dict[str, Any] = field(default_factory=dict)


class IntersectionalFairnessAnalyzer:
    """
    Analyze fairness across intersectional subgroups.

    Examines fairness at the intersection of multiple protected attributes
    (e.g., race × SES × gender) to identify subgroups that may experience
    compounded disadvantage.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        protected_attributes: pd.DataFrame,
        min_group_size: int = 30
    ):
        """
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            protected_attributes: DataFrame with protected attribute columns
            min_group_size: Minimum subgroup size for analysis
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob)
        self.protected_df = protected_attributes.reset_index(drop=True)
        self.min_group_size = min_group_size
        self.intersectional_metrics = {}

    def create_intersectional_groups(
        self,
        attributes: Optional[List[str]] = None
    ) -> pd.Series:
        """Create intersectional group labels from multiple attributes."""
        if attributes is None:
            attributes = list(self.protected_df.columns)

        # Combine attributes into single group label
        groups = self.protected_df[attributes].astype(str).agg("_".join, axis=1)
        return groups

    def compute_intersectional_metrics(
        self,
        attributes: Optional[List[str]] = None
    ) -> Dict[str, IntersectionalMetrics]:
        """
        Compute fairness metrics for all intersectional subgroups.

        Args:
            attributes: List of attributes to intersect

        Returns:
            Dictionary mapping subgroup name to metrics
        """
        groups = self.create_intersectional_groups(attributes)

        for group in groups.unique():
            mask = groups == group

            if mask.sum() < self.min_group_size:
                logger.debug(f"Skipping {group}: n={mask.sum()} < {self.min_group_size}")
                continue

            y_true_g = self.y_true[mask]
            y_pred_g = self.y_pred[mask]

            n = len(y_true_g)

            # Confusion matrix
            if len(np.unique(y_true_g)) < 2 or len(np.unique(y_pred_g)) < 2:
                continue

            tn, fp, fn, tp = confusion_matrix(
                y_true_g, y_pred_g, labels=[0, 1]
            ).ravel()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            accuracy = (tp + tn) / n
            positive_rate = y_pred_g.mean()
            prevalence = y_true_g.mean()

            # Parse attribute values
            attr_values = dict(zip(
                attributes or self.protected_df.columns,
                group.split("_")
            ))

            self.intersectional_metrics[group] = IntersectionalMetrics(
                subgroup=group,
                n=n,
                prevalence=prevalence,
                tpr=tpr,
                fpr=fpr,
                ppv=ppv,
                accuracy=accuracy,
                positive_rate=positive_rate,
                attributes=attr_values
            )

        return self.intersectional_metrics

    def get_worst_subgroups(
        self,
        metric: str = "tpr",
        n: int = 5
    ) -> List[IntersectionalMetrics]:
        """
        Identify subgroups with worst performance on a metric.

        Args:
            metric: Metric to rank by ('tpr', 'fpr', 'ppv', 'accuracy')
            n: Number of worst groups to return

        Returns:
            List of IntersectionalMetrics for worst subgroups
        """
        if not self.intersectional_metrics:
            self.compute_intersectional_metrics()

        metrics_list = list(self.intersectional_metrics.values())

        # Sort by metric (ascending for tpr/ppv/accuracy, descending for fpr)
        reverse = metric == "fpr"
        sorted_metrics = sorted(
            metrics_list,
            key=lambda x: getattr(x, metric),
            reverse=reverse
        )

        return sorted_metrics[:n]

    def get_disparity_matrix(self) -> pd.DataFrame:
        """
        Create matrix of pairwise TPR disparities between subgroups.

        Returns:
            DataFrame with pairwise disparity ratios
        """
        if not self.intersectional_metrics:
            self.compute_intersectional_metrics()

        subgroups = list(self.intersectional_metrics.keys())
        n = len(subgroups)
        matrix = np.zeros((n, n))

        for i, g1 in enumerate(subgroups):
            for j, g2 in enumerate(subgroups):
                tpr1 = self.intersectional_metrics[g1].tpr
                tpr2 = self.intersectional_metrics[g2].tpr
                matrix[i, j] = tpr1 / tpr2 if tpr2 > 0 else np.nan

        return pd.DataFrame(matrix, index=subgroups, columns=subgroups)

    def get_summary_table(self) -> pd.DataFrame:
        """Get summary table of all intersectional subgroups."""
        if not self.intersectional_metrics:
            self.compute_intersectional_metrics()

        rows = []
        for group, m in self.intersectional_metrics.items():
            rows.append({
                "Subgroup": group,
                "N": m.n,
                "Prevalence": f"{m.prevalence:.1%}",
                "TPR": round(m.tpr, 3),
                "FPR": round(m.fpr, 3),
                "PPV": round(m.ppv, 3),
                "Accuracy": round(m.accuracy, 3)
            })

        return pd.DataFrame(rows).sort_values("TPR")


class FairnessConfidenceIntervals:
    """
    Bootstrap confidence intervals for fairness metrics.

    Provides uncertainty quantification for fairness measurements,
    essential for determining statistical significance of disparities.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        groups: pd.Series,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ):
        self.y_true = np.array(y_true)
        self.y_prob = np.array(y_prob)
        self.groups = pd.Series(groups).reset_index(drop=True)
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def _compute_tpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute TPR from true and predicted labels."""
        mask = y_true == 1
        if mask.sum() == 0:
            return np.nan
        return y_pred[mask].mean()

    def bootstrap_group_metrics(
        self,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Compute bootstrap CI for group-level metrics.

        Args:
            threshold: Classification threshold

        Returns:
            DataFrame with metrics and confidence intervals
        """
        y_pred = (self.y_prob >= threshold).astype(int)
        unique_groups = [g for g in self.groups.unique() if pd.notna(g)]

        results = []

        for group in unique_groups:
            mask = self.groups == group
            y_true_g = self.y_true[mask]
            y_prob_g = self.y_prob[mask]

            # Bootstrap
            tpr_samples = []
            fpr_samples = []
            ppv_samples = []

            for i in range(self.n_bootstrap):
                idx = resample(range(len(y_true_g)), random_state=42 + i)
                y_true_b = y_true_g[idx]
                y_pred_b = (y_prob_g[idx] >= threshold).astype(int)

                # Compute metrics
                pos_mask = y_true_b == 1
                neg_mask = y_true_b == 0
                pred_pos = y_pred_b == 1

                tpr = y_pred_b[pos_mask].mean() if pos_mask.sum() > 0 else np.nan
                fpr = y_pred_b[neg_mask].mean() if neg_mask.sum() > 0 else np.nan
                ppv = y_true_b[pred_pos].mean() if pred_pos.sum() > 0 else np.nan

                tpr_samples.append(tpr)
                fpr_samples.append(fpr)
                ppv_samples.append(ppv)

            tpr_samples = np.array(tpr_samples)
            fpr_samples = np.array(fpr_samples)
            ppv_samples = np.array(ppv_samples)

            results.append({
                "Group": group,
                "N": mask.sum(),
                "TPR": np.nanmean(tpr_samples),
                "TPR_CI_lower": np.nanpercentile(tpr_samples, self.alpha / 2 * 100),
                "TPR_CI_upper": np.nanpercentile(tpr_samples, (1 - self.alpha / 2) * 100),
                "FPR": np.nanmean(fpr_samples),
                "FPR_CI_lower": np.nanpercentile(fpr_samples, self.alpha / 2 * 100),
                "FPR_CI_upper": np.nanpercentile(fpr_samples, (1 - self.alpha / 2) * 100),
                "PPV": np.nanmean(ppv_samples),
                "PPV_CI_lower": np.nanpercentile(ppv_samples, self.alpha / 2 * 100),
                "PPV_CI_upper": np.nanpercentile(ppv_samples, (1 - self.alpha / 2) * 100)
            })

        return pd.DataFrame(results)

    def bootstrap_disparity_test(
        self,
        group1: str,
        group2: str,
        metric: str = "tpr",
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Bootstrap hypothesis test for disparity between groups.

        Tests H0: metric(group1) = metric(group2)

        Args:
            group1: First group name
            group2: Second group name
            metric: Metric to compare ('tpr', 'fpr', 'ppv')
            threshold: Classification threshold

        Returns:
            Dictionary with test results including p-value
        """
        mask1 = self.groups == group1
        mask2 = self.groups == group2

        y_true_1 = self.y_true[mask1]
        y_prob_1 = self.y_prob[mask1]
        y_true_2 = self.y_true[mask2]
        y_prob_2 = self.y_prob[mask2]

        # Observed difference
        y_pred_1 = (y_prob_1 >= threshold).astype(int)
        y_pred_2 = (y_prob_2 >= threshold).astype(int)

        if metric == "tpr":
            obs_1 = self._compute_tpr(y_true_1, y_pred_1)
            obs_2 = self._compute_tpr(y_true_2, y_pred_2)
        elif metric == "fpr":
            neg_1 = y_true_1 == 0
            neg_2 = y_true_2 == 0
            obs_1 = y_pred_1[neg_1].mean() if neg_1.sum() > 0 else np.nan
            obs_2 = y_pred_2[neg_2].mean() if neg_2.sum() > 0 else np.nan
        else:
            raise ValueError(f"Unknown metric: {metric}")

        observed_diff = obs_1 - obs_2

        # Bootstrap under null (pooled data)
        pooled_y_true = np.concatenate([y_true_1, y_true_2])
        pooled_y_prob = np.concatenate([y_prob_1, y_prob_2])
        n1, n2 = len(y_true_1), len(y_true_2)

        null_diffs = []
        for i in range(self.n_bootstrap):
            # Permute group labels
            idx = np.random.RandomState(42 + i).permutation(len(pooled_y_true))
            y_true_perm = pooled_y_true[idx]
            y_prob_perm = pooled_y_prob[idx]

            y_true_b1 = y_true_perm[:n1]
            y_prob_b1 = y_prob_perm[:n1]
            y_true_b2 = y_true_perm[n1:]
            y_prob_b2 = y_prob_perm[n1:]

            y_pred_b1 = (y_prob_b1 >= threshold).astype(int)
            y_pred_b2 = (y_prob_b2 >= threshold).astype(int)

            if metric == "tpr":
                val_1 = self._compute_tpr(y_true_b1, y_pred_b1)
                val_2 = self._compute_tpr(y_true_b2, y_pred_b2)
            elif metric == "fpr":
                neg_1 = y_true_b1 == 0
                neg_2 = y_true_b2 == 0
                val_1 = y_pred_b1[neg_1].mean() if neg_1.sum() > 0 else np.nan
                val_2 = y_pred_b2[neg_2].mean() if neg_2.sum() > 0 else np.nan

            null_diffs.append(val_1 - val_2)

        null_diffs = np.array(null_diffs)

        # Two-sided p-value
        p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

        return {
            "group_1": group1,
            "group_2": group2,
            "metric": metric,
            f"{metric}_{group1}": obs_1,
            f"{metric}_{group2}": obs_2,
            "observed_difference": observed_diff,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "ci_lower": np.percentile(null_diffs, self.alpha / 2 * 100),
            "ci_upper": np.percentile(null_diffs, (1 - self.alpha / 2) * 100)
        }


class CalibrationFairnessAnalyzer:
    """
    Calibration-based fairness analysis (Sufficiency criterion).

    Examines whether predicted probabilities are equally well-calibrated
    across demographic groups. Poor calibration in certain groups indicates
    the model's confidence is systematically off for those groups.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        groups: pd.Series,
        n_bins: int = 10
    ):
        self.y_true = np.array(y_true)
        self.y_prob = np.array(y_prob)
        self.groups = pd.Series(groups).reset_index(drop=True)
        self.n_bins = n_bins

    def compute_expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE measures the average gap between predicted probability and
        actual frequency of positive class across probability bins.
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_prob[mask].mean()
                ece += mask.sum() * np.abs(bin_accuracy - bin_confidence)

        return ece / len(y_true)

    def compute_maximum_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """Compute Maximum Calibration Error (MCE)."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        mce = 0.0

        for i in range(self.n_bins):
            mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_prob[mask].mean()
                mce = max(mce, np.abs(bin_accuracy - bin_confidence))

        return mce

    def analyze_calibration_by_group(self) -> pd.DataFrame:
        """
        Analyze calibration metrics for each demographic group.

        Returns:
            DataFrame with ECE, MCE, and Brier score per group
        """
        results = []

        for group in self.groups.unique():
            mask = self.groups == group
            y_true_g = self.y_true[mask]
            y_prob_g = self.y_prob[mask]

            if mask.sum() < self.n_bins * 5:
                logger.warning(f"Group {group} has few samples for calibration")
                continue

            ece = self.compute_expected_calibration_error(y_true_g, y_prob_g)
            mce = self.compute_maximum_calibration_error(y_true_g, y_prob_g)
            brier = brier_score_loss(y_true_g, y_prob_g)

            results.append({
                "Group": group,
                "N": mask.sum(),
                "ECE": ece,
                "MCE": mce,
                "Brier": brier
            })

        df = pd.DataFrame(results)

        # Add calibration disparity
        if len(df) > 0:
            min_ece = df["ECE"].min()
            df["ECE_ratio"] = df["ECE"] / min_ece if min_ece > 0 else np.nan

        return df

    def check_sufficiency(self, threshold: float = 1.5) -> Dict[str, Any]:
        """
        Check if model satisfies sufficiency (calibration fairness).

        A model satisfies sufficiency if calibration is similar across groups.

        Args:
            threshold: Maximum allowed ECE ratio between groups

        Returns:
            Dictionary with sufficiency test results
        """
        calibration_df = self.analyze_calibration_by_group()

        if len(calibration_df) < 2:
            return {"satisfies_sufficiency": None, "reason": "Not enough groups"}

        max_ece_ratio = calibration_df["ECE_ratio"].max()
        satisfies = max_ece_ratio <= threshold

        worst_group = calibration_df.loc[calibration_df["ECE"].idxmax(), "Group"]
        best_group = calibration_df.loc[calibration_df["ECE"].idxmin(), "Group"]

        return {
            "satisfies_sufficiency": satisfies,
            "max_ece_ratio": max_ece_ratio,
            "threshold": threshold,
            "worst_calibrated_group": worst_group,
            "best_calibrated_group": best_group,
            "calibration_by_group": calibration_df
        }


class IndividualFairnessAnalyzer:
    """
    Individual fairness analysis.

    Individual fairness requires that similar individuals receive similar
    predictions. This analyzer measures consistency of predictions for
    similar individuals.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y_prob: np.ndarray,
        distance_metric: str = "euclidean"
    ):
        from sklearn.metrics import pairwise_distances

        self.X = X
        self.y_prob = np.array(y_prob)
        self.distance_metric = distance_metric

        # Compute pairwise distances (for small datasets)
        if len(X) <= 1000:
            self.distances = pairwise_distances(X, metric=distance_metric)
        else:
            self.distances = None
            logger.warning("Dataset too large for full distance matrix. Using sampling.")

    def compute_consistency(
        self,
        n_neighbors: int = 5
    ) -> float:
        """
        Compute consistency score (individual fairness metric).

        Measures how often individuals with similar features get similar predictions.

        Args:
            n_neighbors: Number of neighbors to consider

        Returns:
            Consistency score (0-1, higher is better)
        """
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=self.distance_metric)
        nn.fit(self.X)

        _, indices = nn.kneighbors(self.X)

        # For each point, compare prediction to neighbors (excluding self)
        consistencies = []
        for i, neighbors in enumerate(indices):
            neighbors = neighbors[1:]  # Exclude self
            neighbor_probs = self.y_prob[neighbors]
            consistency = 1 - np.abs(self.y_prob[i] - neighbor_probs.mean())
            consistencies.append(consistency)

        return np.mean(consistencies)

    def compute_lipschitz_violation(
        self,
        sample_size: int = 1000,
        lipschitz_constant: float = 1.0
    ) -> Dict[str, float]:
        """
        Check Lipschitz continuity violation (individual fairness).

        A model is Lipschitz-fair if |f(x) - f(x')| <= L * d(x, x')

        Args:
            sample_size: Number of pairs to sample
            lipschitz_constant: Maximum allowed Lipschitz constant

        Returns:
            Dictionary with violation statistics
        """
        n = len(self.X)

        if n > 1000:
            # Sample pairs
            rng = np.random.RandomState(42)
            idx1 = rng.choice(n, size=sample_size, replace=True)
            idx2 = rng.choice(n, size=sample_size, replace=True)
        else:
            # Use all pairs (for small datasets)
            from itertools import combinations
            pairs = list(combinations(range(n), 2))
            sample_size = min(sample_size, len(pairs))
            rng = np.random.RandomState(42)
            selected = rng.choice(len(pairs), size=sample_size, replace=False)
            idx1 = [pairs[i][0] for i in selected]
            idx2 = [pairs[i][1] for i in selected]

        violations = []
        violation_magnitudes = []

        for i, j in zip(idx1, idx2):
            if i == j:
                continue

            # Feature distance
            if self.distances is not None:
                feat_dist = self.distances[i, j]
            else:
                from sklearn.metrics import pairwise_distances
                feat_dist = pairwise_distances(
                    self.X.iloc[[i]], self.X.iloc[[j]], metric=self.distance_metric
                )[0, 0]

            # Prediction difference
            pred_diff = np.abs(self.y_prob[i] - self.y_prob[j])

            # Check violation
            max_allowed = lipschitz_constant * feat_dist
            if pred_diff > max_allowed and feat_dist > 0:
                violations.append(1)
                violation_magnitudes.append(pred_diff - max_allowed)
            else:
                violations.append(0)

        return {
            "violation_rate": np.mean(violations),
            "mean_violation_magnitude": np.mean(violation_magnitudes) if violation_magnitudes else 0,
            "max_violation_magnitude": np.max(violation_magnitudes) if violation_magnitudes else 0,
            "n_pairs_checked": len(violations),
            "lipschitz_constant": lipschitz_constant
        }


def compute_fairlearn_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series
) -> pd.DataFrame:
    """
    Compute comprehensive fairness metrics using Fairlearn.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        groups: Group labels

    Returns:
        DataFrame with Fairlearn metrics
    """
    if not HAS_FAIRLEARN:
        logger.warning("Fairlearn not available")
        return pd.DataFrame()

    from sklearn.metrics import accuracy_score, precision_score, recall_score

    metrics = {
        "accuracy": accuracy_score,
        "precision": lambda y, p: precision_score(y, p, zero_division=0),
        "recall": lambda y, p: recall_score(y, p, zero_division=0),
        "selection_rate": selection_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups
    )

    results = mf.by_group.copy()
    results["disparity_ratio"] = mf.ratio()
    results["disparity_diff"] = mf.difference()

    return results


def generate_comprehensive_fairness_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
    X: Optional[pd.DataFrame] = None,
    protected_attributes: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive fairness analysis report (2025 state-of-the-art).

    Includes:
    - Group fairness metrics with confidence intervals
    - Intersectional fairness analysis
    - Calibration fairness (sufficiency)
    - Individual fairness (if X provided)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        groups: Primary protected attribute
        X: Feature matrix (for individual fairness)
        protected_attributes: DataFrame of all protected attributes (for intersectional)
        output_dir: Directory to save results

    Returns:
        Dictionary with all fairness analysis results
    """
    from pathlib import Path

    results = {}

    # 1. Basic group fairness
    logger.info("Computing group fairness metrics...")
    evaluator = FairnessEvaluator(y_true, y_pred, y_prob, groups)
    evaluator.compute_all_group_metrics()
    evaluator.compute_fairness_metrics()

    results["group_metrics"] = evaluator.get_summary_table()
    results["disparities"] = evaluator.get_disparity_table()
    results["fairness_criteria"] = evaluator.check_fairness_criteria()

    # 2. Confidence intervals
    logger.info("Computing bootstrap confidence intervals...")
    ci_analyzer = FairnessConfidenceIntervals(y_true, y_prob, groups, n_bootstrap=500)
    results["metrics_with_ci"] = ci_analyzer.bootstrap_group_metrics()

    # 3. Intersectional fairness
    if protected_attributes is not None:
        logger.info("Computing intersectional fairness...")
        intersect_analyzer = IntersectionalFairnessAnalyzer(
            y_true, y_pred, y_prob, protected_attributes
        )
        intersect_analyzer.compute_intersectional_metrics()
        results["intersectional_metrics"] = intersect_analyzer.get_summary_table()
        results["worst_subgroups"] = intersect_analyzer.get_worst_subgroups(metric="tpr", n=5)

    # 4. Calibration fairness
    logger.info("Computing calibration fairness...")
    calib_analyzer = CalibrationFairnessAnalyzer(y_true, y_prob, groups)
    results["calibration_fairness"] = calib_analyzer.check_sufficiency()

    # 5. Individual fairness
    if X is not None:
        logger.info("Computing individual fairness...")
        indiv_analyzer = IndividualFairnessAnalyzer(X, y_prob)
        results["consistency_score"] = indiv_analyzer.compute_consistency()
        results["lipschitz_violation"] = indiv_analyzer.compute_lipschitz_violation()

    # 6. Fairlearn metrics
    if HAS_FAIRLEARN:
        results["fairlearn_metrics"] = compute_fairlearn_metrics(y_true, y_pred, groups)

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results["group_metrics"].to_csv(output_dir / "group_fairness_metrics.csv", index=False)
        results["metrics_with_ci"].to_csv(output_dir / "metrics_with_confidence_intervals.csv", index=False)

        if "intersectional_metrics" in results:
            results["intersectional_metrics"].to_csv(
                output_dir / "intersectional_fairness.csv", index=False
            )

        if "calibration_fairness" in results:
            results["calibration_fairness"]["calibration_by_group"].to_csv(
                output_dir / "calibration_by_group.csv", index=False
            )

        logger.info(f"Fairness report saved to {output_dir}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Fairness module ready (2025 state-of-the-art)")
    print(f"  Fairlearn available: {HAS_FAIRLEARN}")
    print(f"  AIF360 available: {HAS_AIF360}")
