"""
Fairness Module
===============

Algorithmic fairness metrics, evaluation, and bias mitigation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass

from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

try:
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        equalized_odds_ratio
    )
    HAS_FAIRLEARN = True
except ImportError:
    HAS_FAIRLEARN = False
    logging.warning("fairlearn not installed. Some features unavailable.")

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Fairness module ready")
