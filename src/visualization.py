"""
Visualization Module
====================

Publication-quality figures for the fairness study.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

# ============================================================================
# Nature/Science publication style constants
# ============================================================================

# Nature column widths (inches)
FIG_SINGLE = 3.5  # 89mm single column
FIG_ONE_HALF = 4.72  # 120mm 1.5 column
FIG_DOUBLE = 7.2  # 183mm double column

# Colorblind-safe palette (Wong 2011, Nature standard)
GROUP_COLORS = {
    "White": "#0072B2",
    "Black": "#D55E00",
    "Hispanic": "#009E73",
    "Asian": "#E69F00",
    "Other": "#CC79A7",
}

METRIC_COLORS = {
    "auc_roc": "#0072B2",
    "accuracy": "#009E73",
    "f1": "#D55E00",
}

TEMPORAL_COLORS = ["#BFD3E6", "#6BAED6", "#2171B5", "#08306B"]

# ECLS variable codes → human-readable labels
FEATURE_LABELS = {
    "X1RTHETK": "Reading (K fall)",
    "X2RTHETK": "Reading (K spring)",
    "X1MTHETK": "Math (K fall)",
    "X2MTHETK": "Math (K spring)",
    "X9RTHETA": "Reading (5th grade)",
    "X9MTHETA": "Math (5th grade)",
    "X1TCHAPP": "Approaches to learning (K fall)",
    "X2TCHAPP": "Approaches to learning (K spring)",
    "X4TCHAPP": "Approaches to learning (1st grade)",
    "X6DCCSSCR": "Executive function (3rd grade)",
    "X_RACETH_R": "Race/ethnicity",
    "X_CHSEX_R": "Child sex",
    "X1SESQ5": "SES quintile",
    "X12LANGST": "Home language",
}

MODEL_LABELS = {
    "logistic_regression": "Logistic Reg.",
    "elastic_net": "Elastic Net",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "hist_gradient_boosting": "Hist. Grad. Boost.",
}

SES_LABELS = {1: "Q1 (lowest)", 2: "Q2", 3: "Q3", 4: "Q4 (highest)"}

METRIC_DISPLAY = {
    "auc_roc": "AUC-ROC",
    "accuracy": "Accuracy",
    "f1": "F1",
    "AUC_ROC": "AUC-ROC",
    "ACCURACY": "Accuracy",
    "F1": "F1",
}

FAIRNESS_METRIC_LABELS = {
    "tpr": "True positive rate",
    "fpr": "False positive rate",
    "ppv": "Positive predictive value",
    "accuracy": "Accuracy",
    "TPR": "True positive rate",
    "FPR": "False positive rate",
    "PPV": "Positive predictive value",
}


def _map_feature_name(name: str) -> str:
    """Map ECLS variable code to readable label."""
    return FEATURE_LABELS.get(name, name)


def _map_feature_names(names) -> list:
    """Map a list of feature names to readable labels."""
    return [_map_feature_name(n) for n in names]


def _map_model_name(name: str) -> str:
    """Map internal model name to formatted label."""
    return MODEL_LABELS.get(name, name.replace("_", " ").title())


def _save_figure(fig, save_path: str, dpi: int = 600):
    """Save figure as PDF (publication) and PNG (preview)."""
    path = Path(save_path)
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    logger.info(f"Figure saved: {pdf_path}, {png_path}")


def set_publication_style():
    """Set matplotlib parameters for Nature/Science publication figures."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            # Font: Nature requires sans-serif, 5-8pt
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.title_fontsize": 7,
            "figure.titlesize": 8,
            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            # No grid by default
            "axes.grid": False,
            # Line weights
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            # Output
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.format": "pdf",
            # Math
            "mathtext.default": "regular",
            # Legend
            "legend.frameon": False,
            "legend.borderpad": 0.3,
            "legend.handlelength": 1.5,
        }
    )


# Apply publication style at import time
set_publication_style()


def plot_roc_curves_by_group(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
    title: str = "ROC Curves by Demographic Group",
    save_path: Optional[str] = None,
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

    fig, ax = plt.subplots(figsize=(FIG_SINGLE, 3.2))

    # Plot diagonal
    ax.plot([0, 1], [0, 1], ls="--", lw=0.5, color="#AAAAAA")

    # Plot ROC for each group (filter out NaN)
    unique_groups = sorted([g for g in groups.unique() if pd.notna(g)])

    for group in unique_groups:
        mask = groups == group

        if mask.sum() < 10:
            continue

        fpr, tpr, _ = roc_curve(y_true[mask], y_prob[mask])
        roc_auc = auc(fpr, tpr)

        color = GROUP_COLORS.get(group, None)
        ax.plot(fpr, tpr, color=color, lw=1.0, label=f"{group} ({roc_auc:.2f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=6)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_calibration_curves_by_group(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
    n_bins: int = 10,
    title: str = "Calibration Curves by Demographic Group",
    save_path: Optional[str] = None,
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

    fig, ax = plt.subplots(figsize=(FIG_SINGLE, 3.2))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], ls="--", lw=0.5, color="#AAAAAA")

    unique_groups = sorted([g for g in groups.unique() if pd.notna(g)])

    for group in unique_groups:
        mask = groups == group

        if mask.sum() < n_bins * 5:
            continue

        prob_true, prob_pred = calibration_curve(
            y_true[mask], y_prob[mask], n_bins=n_bins, strategy="uniform"
        )

        color = GROUP_COLORS.get(group, None)
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            color=color,
            lw=1.0,
            markersize=3,
            label=group,
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend(loc="upper left", fontsize=6)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_fairness_metrics_comparison(
    metrics_df: pd.DataFrame,
    metric_cols: List[str] = ["tpr", "fpr", "ppv", "accuracy"],
    title: str = "Performance Metrics by Demographic Group",
    save_path: Optional[str] = None,
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
    n_groups = len(metrics_df)  # noqa: F841

    w = min(FIG_SINGLE * n_metrics, FIG_DOUBLE)
    fig, axes = plt.subplots(1, n_metrics, figsize=(w, 2.8))

    if n_metrics == 1:
        axes = [axes]

    for ax, col in zip(axes, metric_cols):
        groups = metrics_df["Group"].values
        values = metrics_df[col].values

        colors = [GROUP_COLORS.get(g, "#808080") for g in groups]

        bars = ax.bar(groups, values, color=colors, edgecolor="none")

        label = FAIRNESS_METRIC_LABELS.get(col, col.upper())
        ax.set_ylabel(label)
        ax.set_ylim([0, 1])
        ax.tick_params(axis="x", rotation=45)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
            )

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_disparity_ratios(
    disparity_df: pd.DataFrame,
    reference_group: str,
    threshold: float = 0.8,
    title: str = "Disparity Ratios Relative to Reference Group",
    save_path: Optional[str] = None,
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

    fig, ax = plt.subplots(figsize=(FIG_ONE_HALF, 2.8))

    # Extract ratio columns
    ratio_cols = [c for c in disparity_df.columns if "Ratio" in c]
    groups = disparity_df["Group"].values

    x = np.arange(len(groups))
    width = 0.18

    for i, col in enumerate(ratio_cols):
        values = disparity_df[col].str.replace(",", "").astype(float).values
        offset = (i - len(ratio_cols) / 2 + 0.5) * width

        ax.bar(x + offset, values, width, label=col, edgecolor="none")

    ax.axhline(y=threshold, color="#D55E00", ls="--", lw=0.8,
               label=f"{threshold:.0%} threshold")
    ax.axhline(y=1.0, color="#AAAAAA", ls="-", lw=0.5)

    ax.set_ylabel("Ratio (vs. reference)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45)
    ax.legend(loc="lower right", fontsize=5)
    ax.set_ylim([0, 1.5])

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
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

    # Get top N and filter zero-importance features
    df = importance_df.head(top_n).copy()
    df = df[df["importance"] > 0]
    df = df.sort_values("importance", ascending=True)
    df["feature"] = df["feature"].map(_map_feature_name)

    fig, ax = plt.subplots(figsize=(FIG_SINGLE, 3.5))

    ax.barh(df["feature"], df["importance"], color="#2171B5", edgecolor="none",
            height=0.6)

    ax.set_xlabel("Feature importance")

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_before_after_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "TPR",
    title: str = "Before vs. After Bias Mitigation",
    save_path: Optional[str] = None,
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

    fig, ax = plt.subplots(figsize=(FIG_ONE_HALF, 2.8))

    groups = comparison_df["Group"].values
    before = comparison_df[f"{metric} Before"].str.replace(",", "").astype(float).values
    after = comparison_df[f"{metric} After"].str.replace(",", "").astype(float).values

    x = np.arange(len(groups))
    width = 0.3

    bars1 = ax.bar(
        x - width / 2, before, width, label="Before", color="#D55E00",
        edgecolor="none", alpha=0.7
    )
    bars2 = ax.bar(
        x + width / 2, after, width, label="After", color="#009E73",
        edgecolor="none", alpha=0.7
    )

    label = FAIRNESS_METRIC_LABELS.get(metric, metric)
    ax.set_ylabel(label)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend(fontsize=6)
    ax.set_ylim([0, 1])

    for bar, val in zip(bars1, before):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=5.5,
        )
    for bar, val in zip(bars2, after):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=5.5,
        )

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def create_all_figures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
    output_dir: str,
    feature_importance: Optional[pd.DataFrame] = None,
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


# =============================================================================
# 2025 State-of-the-Art Visualizations
# =============================================================================


def plot_fairness_with_ci(
    metrics_df: pd.DataFrame,
    metric: str = "TPR",
    title: str = "Group Fairness Metrics with 95% Confidence Intervals",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot fairness metrics with bootstrap confidence intervals.

    Args:
        metrics_df: DataFrame with columns like 'Group', 'TPR', 'TPR_CI_lower', 'TPR_CI_upper'
        metric: Metric to plot ('TPR', 'FPR', 'PPV')
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(FIG_SINGLE, 2.8))

    groups = metrics_df["Group"].values
    values = metrics_df[metric].values
    ci_lower = metrics_df[f"{metric}_CI_lower"].values
    ci_upper = metrics_df[f"{metric}_CI_upper"].values

    x = np.arange(len(groups))
    colors = [GROUP_COLORS.get(g, "#808080") for g in groups]

    bars = ax.bar(x, values, color=colors, edgecolor="none")

    ax.errorbar(
        x, values,
        yerr=[values - ci_lower, ci_upper - values],
        fmt="none", ecolor="#333333",
        capsize=3, capthick=0.8, elinewidth=0.8,
    )

    label = FAIRNESS_METRIC_LABELS.get(metric, metric)
    ax.set_ylabel(label)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)

    # Auto-scale y-axis to data with padding
    y_max = max(ci_upper) * 1.15
    ax.set_ylim([0, min(y_max, 1.0)])

    for bar, val, ci_u in zip(bars, values, ci_upper):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ci_u + 0.01,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=5.5,
        )

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_intersectional_heatmap(
    intersect_df: pd.DataFrame,
    metric: str = "TPR",
    title: str = "Intersectional Fairness Analysis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot heatmap of metrics across intersectional subgroups.

    Args:
        intersect_df: DataFrame with intersectional metrics
        metric: Metric column to visualize
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    set_publication_style()

    # Parse subgroup names to extract attributes
    # Assumes format like "White_High_SES" or "Black_Low_SES"
    df = intersect_df.copy()

    # Try to pivot if we have two attributes
    if "Subgroup" in df.columns:
        # Split subgroup into components
        parts = df["Subgroup"].str.split("_", expand=True)
        if parts.shape[1] >= 2:
            df["Attr1"] = parts[0]
            df["Attr2"] = parts[1] if parts.shape[1] > 1 else "All"

            # Filter out "None" race category
            df = df[df["Attr1"] != "None"]

            pivot = df.pivot_table(
                values=metric, index="Attr1", columns="Attr2", aggfunc="first"
            )
        else:
            return plot_fairness_metrics_comparison(
                df, metric_cols=[metric], title=title, save_path=save_path
            )
    else:
        logger.warning("No Subgroup column found")
        return None

    # Map SES column labels
    new_cols = []
    for c in pivot.columns:
        try:
            new_cols.append(SES_LABELS.get(int(c), str(c)))
        except (ValueError, TypeError):
            new_cols.append(str(c))
    pivot.columns = new_cols

    fig, ax = plt.subplots(figsize=(FIG_ONE_HALF, 3.0))

    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=0.55)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label(FAIRNESS_METRIC_LABELS.get(metric, metric), fontsize=6)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("SES quintile")
    ax.set_ylabel("Race/ethnicity")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if val < 0.25 else "black",
                )

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_calibration_error_comparison(
    calibration_df: pd.DataFrame,
    title: str = "Calibration Error by Demographic Group",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Expected Calibration Error (ECE) comparison across groups.

    Args:
        calibration_df: DataFrame with Group, ECE, MCE columns
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(FIG_SINGLE, 2.8))

    groups = calibration_df["Group"].values
    ece = calibration_df["ECE"].values
    mce = calibration_df["MCE"].values

    x = np.arange(len(groups))
    width = 0.3

    bars1 = ax.bar(
        x - width / 2, ece, width,
        label="ECE", color="#0072B2", edgecolor="none",
    )
    bars2 = ax.bar(
        x + width / 2, mce, width,
        label="MCE", color="#D55E00", edgecolor="none",
    )

    ax.set_ylabel("Calibration error")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend(fontsize=6)

    for bar, val in zip(bars1, ece):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.2f}", ha="center", va="bottom", fontsize=5.5,
        )
    for bar, val in zip(bars2, mce):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.2f}", ha="center", va="bottom", fontsize=5.5,
        )

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_explanation_comparison(
    comparison_df: pd.DataFrame,
    title: str = "Feature Importance: SHAP vs Permutation",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot comparison of feature importance from different explainability methods.

    Args:
        comparison_df: DataFrame with feature, shap_normalized, perm_normalized columns
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    set_publication_style()

    df = comparison_df.head(15).copy()
    # Filter zero-importance features
    df = df[(df["shap_normalized"] > 0) | (df["perm_normalized"] > 0)]
    df["feature"] = df["feature"].map(_map_feature_name)

    fig, ax = plt.subplots(figsize=(FIG_ONE_HALF, 3.5))

    x = np.arange(len(df))
    width = 0.3

    ax.barh(
        x - width / 2, df["shap_normalized"], width,
        label="SHAP", color="#009E73", edgecolor="none",
    )
    ax.barh(
        x + width / 2, df["perm_normalized"], width,
        label="Permutation", color="#0072B2", edgecolor="none",
    )

    ax.set_xlabel("Normalized importance")
    ax.set_yticks(x)
    ax.set_yticklabels(df["feature"])
    ax.legend(loc="lower right", fontsize=6)
    ax.set_xlim([0, 1.05])

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_shap_importance_by_group(
    fairness_shap: Dict[str, pd.DataFrame],
    top_n: int = 10,
    title: str = "SHAP Feature Importance by Demographic Group",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot SHAP feature importance comparison across demographic groups.

    Args:
        fairness_shap: Dictionary mapping group name to importance DataFrame
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    set_publication_style()

    # Get groups (exclude 'differential_importance' and NaN-like)
    groups = [g for g in fairness_shap.keys()
              if g != "differential_importance" and str(g).lower() not in ("nan", "none")]

    if len(groups) == 0:
        logger.warning("No groups found in fairness_shap")
        return None

    # Get top features from first group
    first_group = groups[0]
    top_features = fairness_shap[first_group].head(top_n)["feature"].tolist()
    # Filter out zero-importance features
    top_features = [f for f in top_features
                    if fairness_shap[first_group].set_index("feature").loc[f, "mean_abs_shap"] > 0]

    # Prepare data
    data = []
    for group in groups:
        df = fairness_shap[group].set_index("feature")
        for feat in top_features:
            if feat in df.index:
                data.append(
                    {
                        "feature": _map_feature_name(feat),
                        "group": group,
                        "importance": df.loc[feat, "mean_abs_shap"],
                    }
                )

    plot_df = pd.DataFrame(data)
    mapped_features = [_map_feature_name(f) for f in top_features]

    fig, ax = plt.subplots(figsize=(FIG_DOUBLE, 4.0))

    pivot = plot_df.pivot(index="feature", columns="group", values="importance")
    pivot = pivot.reindex(mapped_features)

    # Use GROUP_COLORS for each group column
    bar_colors = [GROUP_COLORS.get(g, "#808080") for g in pivot.columns]
    pivot.plot(kind="barh", ax=ax, width=0.7, edgecolor="none", color=bar_colors)

    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("")
    ax.legend(title="Race/ethnicity", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=6, title_fontsize=6)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metrics: List[str] = ["auc_roc", "accuracy", "precision", "recall", "f1", "brier_score"],
    title: str = "Model Performance Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Multi-panel dot plot comparing models across metrics with zoomed axes.

    Each metric gets its own panel with an axis range that reveals
    inter-model differences, solving the problem of bar charts where
    similar values (e.g., AUC 0.837--0.848) appear identical.

    Args:
        results_df: DataFrame with model as index and metrics as columns
        metrics: Metrics to compare
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    set_publication_style()

    # Filter to metrics actually present in the data
    metrics = [m for m in metrics if m in results_df.columns]
    n_metrics = len(metrics)

    # Display names and colors for the extended metric set
    display_names = {
        "auc_roc": "AUC-ROC", "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1", "brier_score": "Brier Score",
    }
    panel_colors = {
        "auc_roc": "#0072B2", "accuracy": "#009E73", "precision": "#56B4E9",
        "recall": "#E69F00", "f1": "#D55E00", "brier_score": "#CC79A7",
    }

    # Sort models by AUC descending for consistent ordering
    if "auc_roc" in results_df.columns:
        results_df = results_df.sort_values("auc_roc", ascending=True)
    models = results_df.index.tolist()
    model_labels = [_map_model_name(m) for m in models]

    # Classify models: classical (open marker) vs boosting (filled marker)
    classical = {"logistic_regression", "elastic_net", "random_forest"}
    markers = ["o" if m in classical else "D" for m in models]
    y = np.arange(len(models))

    # Layout: 2 rows x 3 cols (or adjust for fewer metrics)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIG_DOUBLE, 1.6 * n_rows),
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    for idx, metric in enumerate(metrics):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        values = results_df[metric].values
        color = panel_colors.get(metric, "#333333")

        # Plot each model as a dot
        for j, (val, mk) in enumerate(zip(values, markers)):
            fc = color if mk == "D" else "white"
            ax.plot(
                val, y[j], marker=mk, color=color, markerfacecolor=fc,
                markeredgewidth=0.8, markersize=5, zorder=3,
            )

        # Thin horizontal reference lines connecting dots to axis
        for j, val in enumerate(values):
            ax.hlines(y[j], ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else values.min(), val,
                       color="#CCCCCC", linewidth=0.4, zorder=1)

        # Zoom the x-axis: pad 30% of range on each side (minimum 0.005)
        v_min, v_max = values.min(), values.max()
        v_range = max(v_max - v_min, 0.005)
        pad = v_range * 0.5
        ax.set_xlim(v_min - pad, v_max + pad)

        # Light vertical grid for readability
        ax.xaxis.grid(True, linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(True)

        # Labels
        label = display_names.get(metric, metric.upper())
        ax.set_title(label, fontsize=7, fontweight="bold", color=color)
        ax.set_yticks(y)
        if col == 0:
            ax.set_yticklabels(model_labels, fontsize=6)
        ax.tick_params(axis="x", labelsize=5.5)

        # Annotate values next to dots
        for j, val in enumerate(values):
            fmt = f"{val:.3f}" if metric != "brier_score" else f"{val:.4f}"
            ax.annotate(
                fmt, (val, y[j]), textcoords="offset points",
                xytext=(6, 0), fontsize=4.5, va="center", color="#444444",
            )

    # Hide unused panels
    for idx in range(n_metrics, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    # Add legend for marker shapes
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white",
               markeredgewidth=0.8, markersize=5, linestyle="None", label="Classical"),
        Line2D([0], [0], marker="D", color="#555555", markerfacecolor="#555555",
               markeredgewidth=0.8, markersize=4, linestyle="None", label="Gradient Boosting"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=2, fontsize=6, frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        _save_figure(fig, save_path)

    return fig


# =============================================================================
# Temporal Generalization Visualizations
# =============================================================================


def plot_temporal_performance_trend(
    best_model_df: pd.DataFrame,
    title: str = "Model Performance Across Temporal Scenarios",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Line plot of AUC (and optionally accuracy/F1) across temporal scenarios.

    Args:
        best_model_df: DataFrame with columns scenario_label, auc_roc, accuracy, f1
        title: Plot title
        save_path: Path to save figure
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(FIG_ONE_HALF, 2.8))

    x = np.arange(len(best_model_df))
    labels = best_model_df["scenario_label"].values

    for metric, marker, color_key in [
        ("auc_roc", "o", "auc_roc"),
        ("accuracy", "s", "accuracy"),
        ("f1", "^", "f1"),
    ]:
        if metric in best_model_df.columns:
            vals = best_model_df[metric].values
            label = METRIC_DISPLAY.get(metric, metric.upper())
            ax.plot(
                x, vals, marker=marker, color=METRIC_COLORS.get(color_key, f"C0"),
                lw=1.0, markersize=4, label=label,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", fontsize=6)
    ax.set_ylim([0.3, 0.9])
    ax.yaxis.grid(True, alpha=0.15, linewidth=0.3)

    plt.tight_layout()
    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_temporal_fairness_trend(
    fairness_df: pd.DataFrame,
    metric: str = "TPR",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot per-group metric across temporal scenarios with CI ribbons.

    Args:
        fairness_df: Output of TemporalGeneralizationAnalyzer.compare_fairness()
                     with columns: scenario_label, Group, TPR, TPR_CI_lower, TPR_CI_upper, ...
        metric: 'TPR' or 'FPR'
        title: Plot title
        save_path: Path to save figure
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(FIG_ONE_HALF, 3.0))

    scenario_labels = fairness_df["scenario_label"].unique()
    x = np.arange(len(scenario_labels))
    label_map = {lab: i for i, lab in enumerate(scenario_labels)}

    groups = sorted(fairness_df["Group"].unique())
    for group in groups:
        gdf = fairness_df[fairness_df["Group"] == group].copy()
        gdf["x"] = gdf["scenario_label"].map(label_map)
        gdf = gdf.sort_values("x")

        color = GROUP_COLORS.get(group, None)
        ax.plot(gdf["x"], gdf[metric], marker="o", lw=1.0, markersize=3,
                color=color, label=group)

        ci_lo = f"{metric}_CI_lower"
        ci_hi = f"{metric}_CI_upper"
        if ci_lo in gdf.columns and ci_hi in gdf.columns:
            ax.fill_between(gdf["x"], gdf[ci_lo], gdf[ci_hi], alpha=0.12, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=15, ha="right")
    label = FAIRNESS_METRIC_LABELS.get(metric, metric)
    ax.set_ylabel(label)
    ax.legend(title="Race/ethnicity", loc="best", fontsize=5, title_fontsize=6)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, alpha=0.15, linewidth=0.3)

    plt.tight_layout()
    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_temporal_disparity_heatmap(
    disparity_df: pd.DataFrame,
    title: str = "TPR Disparity Ratio Across Temporal Scenarios",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of TPR disparity ratios (scenario x group).

    Args:
        disparity_df: Output of compare_disparities() with
                      scenario_label, Group, TPR Ratio columns
    """
    set_publication_style()

    df = disparity_df.copy()
    # Convert TPR Ratio to numeric, coercing any non-numeric values to NaN
    df["TPR Ratio"] = pd.to_numeric(df["TPR Ratio"], errors="coerce")

    pivot = df.pivot_table(
        values="TPR Ratio", index="Group", columns="scenario_label", aggfunc="first"
    )
    # Ensure float dtype for imshow
    pivot = pivot.astype(float)
    # Reorder columns by scenario order
    ordered = disparity_df["scenario_label"].unique()
    pivot = pivot[[c for c in ordered if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(FIG_ONE_HALF, 2.5))

    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vcenter=1.0, vmin=0.4, vmax=3.5)
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", norm=norm)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("TPR ratio (vs. White)", fontsize=6)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=15, ha="right")
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if val < 0.7 else "black",
                )

    plt.tight_layout()
    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_temporal_fairness_gap(
    fairness_df: pd.DataFrame,
    metric: str = "TPR",
    title: str = "Maximum Inter-Group TPR Gap Across Scenarios",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of the maximum inter-group metric gap per scenario.

    Args:
        fairness_df: Output of compare_fairness()
        metric: Metric column to compute gap on
        title: Plot title
        save_path: Path to save figure
    """
    set_publication_style()

    gaps = []
    for label in fairness_df["scenario_label"].unique():
        sdf = fairness_df[fairness_df["scenario_label"] == label]
        gap = sdf[metric].max() - sdf[metric].min()
        gaps.append({"scenario_label": label, "gap": gap})
    gap_df = pd.DataFrame(gaps)

    fig, ax = plt.subplots(figsize=(FIG_SINGLE, 2.8))
    x = np.arange(len(gap_df))
    n = len(gap_df)
    colors = TEMPORAL_COLORS[:n] if n <= len(TEMPORAL_COLORS) else \
        [TEMPORAL_COLORS[int(i * (len(TEMPORAL_COLORS) - 1) / (n - 1))] for i in range(n)]

    bars = ax.bar(x, gap_df["gap"], color=colors, edgecolor="none")
    for bar, val in zip(bars, gap_df["gap"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=5.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(gap_df["scenario_label"], rotation=20, ha="right")
    label = FAIRNESS_METRIC_LABELS.get(metric, metric)
    ax.set_ylabel(f"Maximum {label.lower()} gap")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    if save_path:
        _save_figure(fig, save_path)
    return fig


def create_all_figures_2025(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
    output_dir: str,
    feature_importance: Optional[pd.DataFrame] = None,
    metrics_with_ci: Optional[pd.DataFrame] = None,
    intersectional_df: Optional[pd.DataFrame] = None,
    calibration_df: Optional[pd.DataFrame] = None,
    explanation_comparison: Optional[pd.DataFrame] = None,
    fairness_shap: Optional[Dict] = None,
    model_results: Optional[pd.DataFrame] = None,
    temporal_results: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Generate all publication figures including 2025 state-of-the-art visualizations.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        groups: Group labels
        output_dir: Directory to save figures
        feature_importance: Feature importance DataFrame
        metrics_with_ci: Metrics with confidence intervals
        intersectional_df: Intersectional fairness metrics
        calibration_df: Calibration metrics by group
        explanation_comparison: SHAP vs permutation comparison
        fairness_shap: SHAP importance by group
        model_results: Model comparison results
        temporal_results: Dict with 'best_model_summary', 'fairness_comparison',
                          'disparity_comparison' DataFrames from temporal analysis

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Basic figures
    path = str(output_dir / "roc_curves_by_group.png")
    plot_roc_curves_by_group(y_true, y_prob, groups, save_path=path)
    saved_files.append(path)
    plt.close()

    path = str(output_dir / "calibration_curves_by_group.png")
    plot_calibration_curves_by_group(y_true, y_prob, groups, save_path=path)
    saved_files.append(path)
    plt.close()

    if feature_importance is not None:
        path = str(output_dir / "feature_importance.png")
        plot_feature_importance(feature_importance, save_path=path)
        saved_files.append(path)
        plt.close()

    # 2025 figures
    if metrics_with_ci is not None:
        for metric in ["TPR", "FPR", "PPV"]:
            if f"{metric}_CI_lower" in metrics_with_ci.columns:
                path = str(output_dir / f"fairness_{metric.lower()}_with_ci.png")
                plot_fairness_with_ci(metrics_with_ci, metric=metric, save_path=path)
                saved_files.append(path)
                plt.close()

    if intersectional_df is not None and len(intersectional_df) > 0:
        path = str(output_dir / "intersectional_fairness_heatmap.png")
        plot_intersectional_heatmap(intersectional_df, save_path=path)
        saved_files.append(path)
        plt.close()

    if calibration_df is not None:
        path = str(output_dir / "calibration_error_comparison.png")
        plot_calibration_error_comparison(calibration_df, save_path=path)
        saved_files.append(path)
        plt.close()

    if explanation_comparison is not None:
        path = str(output_dir / "explanation_comparison.png")
        plot_explanation_comparison(explanation_comparison, save_path=path)
        saved_files.append(path)
        plt.close()

    if fairness_shap is not None:
        path = str(output_dir / "shap_importance_by_group.png")
        plot_shap_importance_by_group(fairness_shap, save_path=path)
        saved_files.append(path)
        plt.close()

    if model_results is not None:
        path = str(output_dir / "model_comparison.png")
        plot_model_comparison(model_results, save_path=path)
        saved_files.append(path)
        plt.close()

    # Temporal generalization figures
    if temporal_results is not None:
        try:
            bm = temporal_results.get("best_model_summary")
            if bm is not None:
                path = str(output_dir / "temporal_performance_trend.png")
                plot_temporal_performance_trend(bm, save_path=path)
                saved_files.append(path)
                plt.close()

            fc = temporal_results.get("fairness_comparison")
            if fc is not None:
                for metric in ["TPR", "FPR"]:
                    path = str(
                        output_dir / f"temporal_fairness_{metric.lower()}_trend.png"
                    )
                    plot_temporal_fairness_trend(fc, metric=metric, save_path=path)
                    saved_files.append(path)
                    plt.close()

            dc = temporal_results.get("disparity_comparison")
            if dc is not None:
                path = str(output_dir / "temporal_disparity_heatmap.png")
                plot_temporal_disparity_heatmap(dc, save_path=path)
                saved_files.append(path)
                plt.close()

            if fc is not None:
                path = str(output_dir / "temporal_fairness_gap.png")
                plot_temporal_fairness_gap(fc, save_path=path)
                saved_files.append(path)
                plt.close()
        except Exception as e:
            logger.warning(f"Temporal figures failed: {e}")

    logger.info(f"Created {len(saved_files)} figures in {output_dir}")

    return saved_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Visualization module ready (2025 state-of-the-art)")
