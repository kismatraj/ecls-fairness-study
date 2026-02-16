"""
LaTeX Table Generation Module
==============================

Convert analysis results to publication-ready LaTeX tables.
"""

import pandas as pd
from typing import List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    for char in ["&", "%", "$", "#", "_", "{", "}"]:
        s = s.replace(char, f"\\{char}")
    return s


def model_performance_to_latex(
    df: pd.DataFrame,
    caption: str = "Model Performance on Test Set",
    label: str = "tab:model_performance",
) -> str:
    """Convert model performance DataFrame to LaTeX."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Model & AUC-ROC & Accuracy & Precision & Recall & F1 & Brier \\\\")
    lines.append("\\midrule")

    for model_name in df.index:
        row = df.loc[model_name]
        name = model_name.replace("_", " ").title()
        line = (
            f"{name} & {row['auc_roc']:.3f} & {row['accuracy']:.3f} & "
            f"{row['precision']:.3f} & {row['recall']:.3f} & "
            f"{row['f1']:.3f} & {row['brier_score']:.3f} \\\\"
        )
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def fairness_metrics_to_latex(
    df: pd.DataFrame,
    caption: str = "Group Fairness Metrics with 95\\% Confidence Intervals",
    label: str = "tab:fairness_ci",
) -> str:
    """Convert fairness metrics with CI to LaTeX."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lrccc}")
    lines.append("\\toprule")
    lines.append("Group & N & TPR [95\\% CI] & FPR [95\\% CI] & PPV [95\\% CI] \\\\")
    lines.append("\\midrule")

    for _, row in df.iterrows():
        group = row["Group"]
        n = int(row["N"])
        tpr = f"{row['TPR']:.3f} [{row['TPR_CI_lower']:.3f}, {row['TPR_CI_upper']:.3f}]"
        fpr = f"{row['FPR']:.3f} [{row['FPR_CI_lower']:.3f}, {row['FPR_CI_upper']:.3f}]"
        ppv = f"{row['PPV']:.3f} [{row['PPV_CI_lower']:.3f}, {row['PPV_CI_upper']:.3f}]"
        lines.append(f"{group} & {n:,} & {tpr} & {fpr} & {ppv} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def disparity_table_to_latex(
    df: pd.DataFrame,
    caption: str = "Disparity Ratios Relative to White Reference Group",
    label: str = "tab:disparities",
) -> str:
    """Convert disparity table to LaTeX."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{llccccl}")
    lines.append("\\toprule")
    lines.append(
        "Group & vs & TPR Ratio & TPR Diff & FPR Ratio & FPR Diff & Disp. Impact \\\\"
    )
    lines.append("\\midrule")

    for _, row in df.iterrows():
        line = (
            f"{row['Group']} & {row['vs']} & {row['TPR Ratio']} & "
            f"{row['TPR Diff']} & {row['FPR Ratio']} & {row['FPR Diff']} & "
            f"{row['Disparate Impact']} \\\\"
        )
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def temporal_summary_to_latex(
    df: pd.DataFrame,
    caption: str = "Best Model Performance Across Temporal Scenarios",
    label: str = "tab:temporal_summary",
) -> str:
    """Convert temporal best model summary to LaTeX."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\toprule")
    lines.append(
        "Scenario & Best Model & Features & AUC-ROC & Accuracy & F1 & Brier \\\\"
    )
    lines.append("\\midrule")

    for _, row in df.iterrows():
        model = row["best_model"].replace("_", " ").title()
        line = (
            f"{row['scenario_label']} & {model} & {int(row['n_features'])} & "
            f"{row['auc_roc']:.3f} & {row['accuracy']:.3f} & "
            f"{row['f1']:.3f} & {row['brier_score']:.3f} \\\\"
        )
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def calibration_to_latex(
    df: pd.DataFrame,
    caption: str = "Calibration Error by Demographic Group",
    label: str = "tab:calibration",
) -> str:
    """Convert calibration fairness table to LaTeX."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lrcccc}")
    lines.append("\\toprule")
    lines.append("Group & N & ECE & MCE & Brier & ECE Ratio \\\\")
    lines.append("\\midrule")

    for _, row in df.iterrows():
        line = (
            f"{row['Group']} & {int(row['N']):,} & {row['ECE']:.4f} & "
            f"{row['MCE']:.4f} & {row['Brier']:.4f} & {row['ECE_ratio']:.2f} \\\\"
        )
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def attrition_comparison_to_latex(
    df: pd.DataFrame,
    caption: str = "Baseline Characteristics: Completers vs.\\ Dropouts",
    label: str = "tab:attrition",
) -> str:
    """Convert attrition comparison table to LaTeX."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lrrrrrc}")
    lines.append("\\toprule")
    lines.append(
        "Variable & Completer $M$ & Completer $SD$ & "
        "Dropout $M$ & Dropout $SD$ & Cohen's $d$ / Cram\\'er's $V$ & $p$ \\\\"
    )
    lines.append("\\midrule")

    for _, row in df.iterrows():
        var = row["Variable"].replace("_", "\\_")
        if row["Type"] == "continuous":
            line = (
                f"{var} & {row['Completer_Mean']:.2f} & {row['Completer_SD']:.2f} & "
                f"{row['Dropout_Mean']:.2f} & {row['Dropout_SD']:.2f} & "
                f"{row['Cohens_d']:.3f} & {row['p_value']:.4f} \\\\"
            )
        else:
            line = (
                f"{var} & \\multicolumn{{2}}{{c}}{{N={int(row['Completer_N']):,}}} & "
                f"\\multicolumn{{2}}{{c}}{{N={int(row['Dropout_N']):,}}} & "
                f"{row['Cohens_d']:.3f} & {row['p_value']:.4f} \\\\"
            )
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\begin{tablenotes}\\footnotesize "
        "\\item Effect size is Cohen's $d$ for continuous variables and "
        "Cram\\'er's $V$ for categorical variables."
        "\\end{tablenotes}"
    )
    lines.append("\\end{table}")
    return "\n".join(lines)


def mice_comparison_to_latex(
    df: pd.DataFrame,
    caption: str = "Comparison of Complete-Case and MICE-Imputed Results",
    label: str = "tab:mice_comparison",
) -> str:
    """Convert MICE vs complete-case comparison to LaTeX."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append(
        "Metric & Group & Complete-Case & MICE Pooled & SE & Difference \\\\"
    )
    lines.append("\\midrule")

    for _, row in df.iterrows():
        line = (
            f"{row['Metric']} & {row['Group']} & "
            f"{row['Complete_Case']:.3f} & {row['MICE_Pooled']:.3f} & "
            f"{row['MICE_SE']:.3f} & {row['Difference']:+.3f} \\\\"
        )
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_all_latex_tables(tables_dir: str) -> List[str]:
    """
    Read existing CSV results and generate LaTeX versions.

    Args:
        tables_dir: Directory containing CSV result tables

    Returns:
        List of saved .tex file paths
    """
    tables_dir = Path(tables_dir)
    saved = []

    converters = {
        "model_performance.csv": (
            model_performance_to_latex,
            "model_performance.tex",
            {"index_col": "model"},
        ),
        "fairness_metrics_with_ci.csv": (
            fairness_metrics_to_latex,
            "fairness_metrics_with_ci.tex",
            {},
        ),
        "fairness_disparities.csv": (
            disparity_table_to_latex,
            "fairness_disparities.tex",
            {},
        ),
        "temporal_best_model_summary.csv": (
            temporal_summary_to_latex,
            "temporal_best_model_summary.tex",
            {},
        ),
        "calibration_fairness.csv": (
            calibration_to_latex,
            "calibration_fairness.tex",
            {},
        ),
        "missing_data_attrition_comparison.csv": (
            attrition_comparison_to_latex,
            "missing_data_attrition_comparison.tex",
            {},
        ),
        "missing_data_mice_vs_complete_case.csv": (
            mice_comparison_to_latex,
            "missing_data_mice_comparison.tex",
            {},
        ),
    }

    for csv_name, (converter_fn, tex_name, read_kwargs) in converters.items():
        csv_path = tables_dir / csv_name
        if not csv_path.exists():
            logger.warning(f"CSV not found, skipping: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path, **read_kwargs)
            latex = converter_fn(df)
            tex_path = tables_dir / tex_name
            with open(tex_path, "w") as f:
                f.write(latex)
            saved.append(str(tex_path))
            logger.info(f"LaTeX table saved: {tex_path}")
        except Exception as e:
            logger.warning(f"Failed to generate {tex_name}: {e}")

    return saved
