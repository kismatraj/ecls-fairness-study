"""
Visualization Module
====================

Publication-quality figures for the fairness study.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Color palette for demographic groups
GROUP_COLORS = {
    "White": "#1f77b4",
    "Black": "#ff7f0e", 
    "Hispanic": "#2ca02c",
    "Asian": "#d62728",
    "Other": "#9467bd"
}


def set_publication_style():
    """Set matplotlib parameters for publication figures."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 13,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def plot_roc_curves_by_group(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
    title: str = "ROC Curves by Demographic Group",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curves for each demographic group.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        groups: Group labels
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC=0.50)')
    
    # Plot ROC for each group (filter out NaN)
    unique_groups = sorted([g for g in groups.unique() if pd.notna(g)])
    
    for group in unique_groups:
        mask = groups == group
        
        if mask.sum() < 10:
            continue
        
        fpr, tpr, _ = roc_curve(y_true[mask], y_prob[mask])
        roc_auc = auc(fpr, tpr)
        
        color = GROUP_COLORS.get(group, None)
        ax.plot(
            fpr, tpr,
            color=color,
            lw=2,
            label=f'{group} (AUC={roc_auc:.3f})'
        )
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_calibration_curves_by_group(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
    n_bins: int = 10,
    title: str = "Calibration Curves by Demographic Group",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot calibration curves for each demographic group.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        groups: Group labels
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')
    
    unique_groups = sorted([g for g in groups.unique() if pd.notna(g)])
    
    for group in unique_groups:
        mask = groups == group
        
        if mask.sum() < n_bins * 5:
            continue
        
        prob_true, prob_pred = calibration_curve(
            y_true[mask], y_prob[mask],
            n_bins=n_bins, strategy='uniform'
        )
        
        color = GROUP_COLORS.get(group, None)
        ax.plot(
            prob_pred, prob_true,
            marker='o', color=color,
            lw=2, markersize=6,
            label=group
        )
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_fairness_metrics_comparison(
    metrics_df: pd.DataFrame,
    metric_cols: List[str] = ["tpr", "fpr", "ppv", "accuracy"],
    title: str = "Performance Metrics by Demographic Group",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot bar chart comparing metrics across groups.
    
    Args:
        metrics_df: DataFrame with metrics by group
        metric_cols: Columns to plot
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    n_metrics = len(metric_cols)
    n_groups = len(metrics_df)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, col in zip(axes, metric_cols):
        groups = metrics_df['Group'].values
        values = metrics_df[col].values
        
        colors = [GROUP_COLORS.get(g, '#808080') for g in groups]
        
        bars = ax.bar(groups, values, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_ylabel(col.upper())
        ax.set_ylim([0, 1])
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=9
            )
    
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_disparity_ratios(
    disparity_df: pd.DataFrame,
    reference_group: str,
    threshold: float = 0.8,
    title: str = "Disparity Ratios Relative to Reference Group",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot disparity ratios with 80% threshold.
    
    Args:
        disparity_df: DataFrame with disparity metrics
        reference_group: Name of reference group
        threshold: Threshold for flagging disparity
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract ratio columns
    ratio_cols = [c for c in disparity_df.columns if 'Ratio' in c]
    groups = disparity_df['Group'].values
    
    x = np.arange(len(groups))
    width = 0.2
    
    for i, col in enumerate(ratio_cols):
        values = disparity_df[col].str.replace(',', '').astype(float).values
        offset = (i - len(ratio_cols)/2 + 0.5) * width
        
        bars = ax.bar(x + offset, values, width, label=col, edgecolor='black', linewidth=0.5)
    
    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', lw=2, label=f'{threshold:.0%} threshold')
    ax.axhline(y=1.0, color='gray', linestyle='-', lw=1)
    
    ax.set_ylabel('Ratio (vs. Reference)')
    ax.set_title(f"{title}\n(Reference: {reference_group})")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.5])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Feature Importance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot horizontal bar chart of feature importance.
    
    Args:
        importance_df: DataFrame with feature, importance columns
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    # Get top N
    df = importance_df.head(top_n).copy()
    df = df.sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(df['feature'], df['importance'], color='steelblue', edgecolor='black')
    
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_before_after_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "TPR",
    title: str = "Before vs. After Bias Mitigation",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot before/after comparison of metrics.
    
    Args:
        comparison_df: DataFrame with before/after metrics
        metric: Metric to compare
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    groups = comparison_df['Group'].values
    before = comparison_df[f'{metric} Before'].str.replace(',', '').astype(float).values
    after = comparison_df[f'{metric} After'].str.replace(',', '').astype(float).values
    
    x = np.arange(len(groups))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, before, width, label='Before', color='#ff7f7f', edgecolor='black')
    bars2 = ax.bar(x + width/2, after, width, label='After', color='#7fbf7f', edgecolor='black')
    
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.set_ylim([0, 1])
    
    # Add value labels
    for bar, val in zip(bars1, before):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, after):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def create_all_figures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
    output_dir: str,
    feature_importance: Optional[pd.DataFrame] = None
) -> List[str]:
    """
    Generate all publication figures.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        groups: Group labels
        output_dir: Directory to save figures
        feature_importance: Optional feature importance DataFrame
    
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # ROC curves
    path = str(output_dir / "roc_curves_by_group.png")
    plot_roc_curves_by_group(y_true, y_prob, groups, save_path=path)
    saved_files.append(path)
    
    # Calibration curves
    path = str(output_dir / "calibration_curves_by_group.png")
    plot_calibration_curves_by_group(y_true, y_prob, groups, save_path=path)
    saved_files.append(path)
    
    # Feature importance
    if feature_importance is not None:
        path = str(output_dir / "feature_importance.png")
        plot_feature_importance(feature_importance, save_path=path)
        saved_files.append(path)
    
    logger.info(f"Created {len(saved_files)} figures in {output_dir}")
    
    return saved_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Visualization module ready")
