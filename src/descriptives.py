"""
Descriptive Statistics Module
=============================

Generate publication-quality descriptive statistics tables (Table 1).
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate_table1(
    df: pd.DataFrame,
    group_col: str = "race_ethnicity",
    outcome_col: str = "X9RTHETA_at_risk",
    continuous_vars: Optional[List[str]] = None,
    categorical_vars: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate Table 1: Sample characteristics by demographic group.

    Args:
        df: Analytic sample DataFrame
        group_col: Grouping variable
        outcome_col: Outcome variable
        continuous_vars: List of continuous variables to summarize
        categorical_vars: List of categorical variables to summarize

    Returns:
        DataFrame formatted as Table 1
    """
    if continuous_vars is None:
        continuous_vars = [
            "X1RTHETK", "X2RTHETK", "X1MTHETK", "X2MTHETK",
            "X6DCCSSCR", "X1TCHAPP", "X2TCHAPP", "X4TCHAPP",
            "X9RTHETA", "X9MTHETA",
        ]
    if categorical_vars is None:
        categorical_vars = ["X_CHSEX_R", "X1SESQ5", "X12LANGST"]

    # Filter to available columns
    continuous_vars = [c for c in continuous_vars if c in df.columns]
    categorical_vars = [c for c in categorical_vars if c in df.columns]

    groups = sorted([g for g in df[group_col].dropna().unique()])
    rows = []

    # --- N per group ---
    row = {"Variable": "N", "Category": ""}
    for g in groups:
        n = (df[group_col] == g).sum()
        row[g] = f"{n:,}"
    row["Overall"] = f"{len(df):,}"
    rows.append(row)

    # --- Outcome prevalence ---
    if outcome_col in df.columns:
        row = {"Variable": f"{outcome_col} prevalence", "Category": ""}
        for g in groups:
            mask = df[group_col] == g
            prev = df.loc[mask, outcome_col].mean()
            row[g] = f"{prev:.1%}"
        row["Overall"] = f"{df[outcome_col].mean():.1%}"
        rows.append(row)

    # --- Continuous variables: Mean (SD) ---
    for var in continuous_vars:
        row = {"Variable": var, "Category": "Mean (SD)"}
        for g in groups:
            mask = df[group_col] == g
            vals = df.loc[mask, var].dropna()
            row[g] = f"{vals.mean():.2f} ({vals.std():.2f})" if len(vals) > 0 else "—"
        overall = df[var].dropna()
        row["Overall"] = f"{overall.mean():.2f} ({overall.std():.2f})" if len(overall) > 0 else "—"
        rows.append(row)

        # Missing rate
        row_miss = {"Variable": var, "Category": "Missing N (%)"}
        for g in groups:
            mask = df[group_col] == g
            n_miss = df.loc[mask, var].isna().sum()
            n_tot = mask.sum()
            pct = n_miss / n_tot * 100 if n_tot > 0 else 0
            row_miss[g] = f"{n_miss} ({pct:.1f}%)" if n_miss > 0 else "0"
        n_miss_all = df[var].isna().sum()
        pct_all = n_miss_all / len(df) * 100
        row_miss["Overall"] = f"{n_miss_all} ({pct_all:.1f}%)" if n_miss_all > 0 else "0"
        rows.append(row_miss)

    # --- Categorical variables ---
    sex_labels = {1: "Male", 2: "Female"}
    ses_labels = {1: "Q1 (Lowest)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (Highest)"}
    lang_labels = {1: "English", 2: "Non-English"}

    label_maps = {
        "X_CHSEX_R": sex_labels,
        "X1SESQ5": ses_labels,
        "X12LANGST": lang_labels,
    }
    var_names = {
        "X_CHSEX_R": "Sex",
        "X1SESQ5": "SES Quintile",
        "X12LANGST": "Home Language",
    }

    for var in categorical_vars:
        labels = label_maps.get(var, {})
        vname = var_names.get(var, var)
        categories = sorted(df[var].dropna().unique())

        for cat in categories:
            cat_label = labels.get(cat, str(cat))
            row = {"Variable": vname, "Category": cat_label}
            for g in groups:
                mask = (df[group_col] == g)
                n_cat = ((df[var] == cat) & mask).sum()
                n_tot = mask.sum()
                pct = n_cat / n_tot * 100 if n_tot > 0 else 0
                row[g] = f"{n_cat:,} ({pct:.1f}%)"
            n_cat_all = (df[var] == cat).sum()
            pct_all = n_cat_all / len(df) * 100
            row["Overall"] = f"{n_cat_all:,} ({pct_all:.1f}%)"
            rows.append(row)

    return pd.DataFrame(rows)


def generate_table1_latex(
    table1_df: pd.DataFrame,
    caption: str = "Sample Characteristics by Race/Ethnicity",
    label: str = "tab:table1",
) -> str:
    """Convert Table 1 DataFrame to publication LaTeX."""
    groups = [c for c in table1_df.columns if c not in ("Variable", "Category", "Overall")]

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    ncols = len(groups) + 3  # Variable, Category, groups..., Overall
    col_spec = "ll" + "r" * (len(groups) + 1)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header = "Variable & Category & " + " & ".join(groups) + " & Overall \\\\"
    lines.append(header)
    lines.append("\\midrule")

    prev_var = None
    for _, row in table1_df.iterrows():
        var = row["Variable"]
        cat = row["Category"]
        vals = [str(row.get(g, "")) for g in groups]
        overall = str(row.get("Overall", ""))

        # Add separator between variable blocks
        if prev_var is not None and var != prev_var:
            lines.append("\\addlinespace")

        var_display = var if var != prev_var else ""
        line = f"{var_display} & {cat} & " + " & ".join(vals) + f" & {overall} \\\\"
        lines.append(line)
        prev_var = var

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def save_table1(
    df: pd.DataFrame,
    output_dir: str,
    group_col: str = "race_ethnicity",
    outcome_col: str = "X9RTHETA_at_risk",
) -> List[str]:
    """Generate and save Table 1 in CSV and LaTeX formats."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    table1 = generate_table1(df, group_col=group_col, outcome_col=outcome_col)

    csv_path = out / "table1_descriptives.csv"
    table1.to_csv(csv_path, index=False)
    saved.append(str(csv_path))

    latex_path = out / "table1_descriptives.tex"
    latex_str = generate_table1_latex(table1)
    with open(latex_path, "w") as f:
        f.write(latex_str)
    saved.append(str(latex_path))

    logger.info(f"Table 1 saved: {saved}")
    return saved
