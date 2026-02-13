"""
Data Loader Module
==================

Load and preprocess ECLS-K:2011 public-use data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import yaml

logger = logging.getLogger(__name__)

# ECLS Missing Value Codes
MISSING_CODES = [-1, -7, -8, -9]
SUPPRESSED_CODE = -2


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_ecls_data(
    filepath: str, columns: Optional[List[str]] = None, format: str = "auto"
) -> pd.DataFrame:
    """
    Load ECLS-K:2011 data file.

    Args:
        filepath: Path to data file
        columns: Specific columns to load
        format: File format (auto, csv, stata, parquet, dat)

    Returns:
        DataFrame with ECLS data
    """
    path = Path(filepath)

    if format == "auto":
        format = path.suffix.lstrip(".")

    logger.info(f"Loading data from {filepath}")

    if format in ["csv", "txt"]:
        df = pd.read_csv(path, usecols=columns, low_memory=False)
    elif format in ["dta", "stata"]:
        df = pd.read_stata(path, columns=columns)
    elif format == "parquet":
        df = pd.read_parquet(path, columns=columns)
    elif format in ["sav", "spss"]:
        df = pd.read_spss(path, usecols=columns)
    elif format == "dat":
        # For ASCII .dat files, check for pre-extracted parquet
        extracted_path = path.parent.parent / "processed" / "ecls_extracted.parquet"
        if extracted_path.exists():
            logger.info(f"Loading pre-extracted data from {extracted_path}")
            df = pd.read_parquet(extracted_path, columns=columns)
        else:
            raise ValueError(
                "ASCII .dat file requires pre-extraction. "
                "Run scripts/parse_ascii_data.py first."
            )
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def handle_missing_values(
    df: pd.DataFrame, missing_codes: List[int] = MISSING_CODES, strategy: str = "to_nan"
) -> pd.DataFrame:
    """
    Handle ECLS missing value codes.

    Args:
        df: Input DataFrame
        missing_codes: Codes to treat as missing
        strategy: 'to_nan' or 'flag'

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()

    # Replace missing codes with NaN
    for code in missing_codes:
        df = df.replace(code, np.nan)

    # Log missing rates
    missing_pct = df.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 20]
    if len(high_missing) > 0:
        logger.warning(f"Variables with >20% missing:\n{high_missing}")

    if strategy == "flag":
        # Create missing indicators
        for col in df.columns:
            if df[col].isnull().any():
                df[f"{col}_missing"] = df[col].isnull().astype(int)

    return df


def create_race_variable(
    df: pd.DataFrame, race_col: str = "X_RACETH_R", simplify: bool = True
) -> pd.DataFrame:
    """
    Create cleaned race/ethnicity variable.

    Args:
        df: Input DataFrame
        race_col: Name of race column
        simplify: Combine small groups into 'Other'

    Returns:
        DataFrame with race_ethnicity column
    """
    df = df.copy()

    race_map = {
        1: "White",
        2: "Black",
        3: "Hispanic",
        4: "Asian",
        5: "Other",  # NHPI
        6: "Other",  # AIAN
        7: "Other",  # Multiracial
    }

    df["race_ethnicity"] = df[race_col].map(race_map)

    return df


def create_ses_variable(df: pd.DataFrame, ses_col: str = "X1SESQ5") -> pd.DataFrame:
    """
    Create SES category variable.

    Args:
        df: Input DataFrame
        ses_col: Name of SES quintile column

    Returns:
        DataFrame with ses_category column
    """
    df = df.copy()

    ses_map = {1: "Q1 (Lowest)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (Highest)"}

    df["ses_category"] = df[ses_col].map(ses_map)
    # Use nullable Int64 to handle NaN values
    df["ses_low"] = (df[ses_col] <= 2).astype("Int64")

    return df


def create_at_risk_indicator(
    df: pd.DataFrame, outcome_col: str, percentile: int = 25, name: Optional[str] = None
) -> pd.DataFrame:
    """
    Create binary at-risk indicator based on percentile.

    Args:
        df: Input DataFrame
        outcome_col: Column with outcome scores
        percentile: Percentile threshold (at-risk = below)
        name: Name for indicator column

    Returns:
        DataFrame with at_risk indicator
    """
    df = df.copy()

    threshold = df[outcome_col].quantile(percentile / 100)

    col_name = name or f"{outcome_col}_at_risk"
    df[col_name] = (df[outcome_col] < threshold).astype(int)

    prevalence = df[col_name].mean()
    logger.info(
        f"Created {col_name}: threshold={threshold:.3f}, prevalence={prevalence:.1%}"
    )

    return df


def create_analytic_sample(
    df: pd.DataFrame,
    predictor_cols: List[str],
    outcome_col: str,
    require_baseline: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Create analytic sample with inclusion criteria.

    Args:
        df: Input DataFrame
        predictor_cols: Required predictor columns
        outcome_col: Required outcome column
        require_baseline: Require baseline (K) data

    Returns:
        Tuple of (analytic DataFrame, sample stats dict)
    """
    n_start = len(df)
    stats = {"n_total": n_start}

    # Require outcome
    df = df[df[outcome_col].notna()].copy()
    stats["n_with_outcome"] = len(df)

    # Require baseline predictors
    if require_baseline:
        baseline_cols = [c for c in predictor_cols if c.startswith(("X1", "X2"))]
        df = df.dropna(subset=baseline_cols, how="all")
        stats["n_with_baseline"] = len(df)

    stats["n_analytic"] = len(df)
    stats["retention"] = len(df) / n_start

    logger.info(f"Analytic sample: {len(df):,} ({stats['retention']:.1%} of original)")

    return df, stats


def get_variable_lists(config: dict) -> Dict[str, List[str]]:
    """
    Extract variable lists from config.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with categorized variable lists
    """
    var_cfg = config["variables"]

    # Flatten predictors
    predictors = []
    for category, vars in var_cfg["predictors"].items():
        predictors.extend(vars)

    return {
        "outcomes": list(var_cfg["outcomes"].values()),
        "demographics": list(var_cfg["demographics"].values()),
        "predictors": predictors,
        "all": list(var_cfg["outcomes"].values())
        + list(var_cfg["demographics"].values())
        + predictors,
    }


def prepare_modeling_data(
    df: pd.DataFrame,
    predictor_cols: List[str],
    outcome_col: str,
    group_col: str = "race_ethnicity",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare X, y, groups for modeling.

    Args:
        df: Input DataFrame
        predictor_cols: Feature columns
        outcome_col: Target column
        group_col: Protected attribute column

    Returns:
        Tuple of (X, y, groups)
    """
    # Select available predictors
    available = [c for c in predictor_cols if c in df.columns]
    missing = set(predictor_cols) - set(available)
    if missing:
        logger.warning(f"Predictors not found: {missing}")

    X = df[available].copy()
    y = df[outcome_col].copy()
    groups = df[group_col].copy() if group_col in df.columns else None

    # Drop rows with missing values
    valid_mask = X.notna().all(axis=1) & y.notna()

    logger.info(f"Modeling data: {valid_mask.sum():,} complete cases")

    return (
        X[valid_mask],
        y[valid_mask],
        groups[valid_mask] if groups is not None else None,
    )


def get_sample_characteristics(
    df: pd.DataFrame, by_group: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate sample characteristics table.

    Args:
        df: Input DataFrame
        by_group: Optional grouping variable

    Returns:
        DataFrame with sample characteristics
    """
    results = []

    # Overall N
    results.append({"Variable": "N", "Overall": f"{len(df):,}"})

    # Demographics
    for var in ["race_ethnicity", "ses_category", "X_CHSEX_R"]:
        if var in df.columns:
            counts = df[var].value_counts()
            for cat, n in counts.items():
                pct = n / len(df) * 100
                results.append(
                    {
                        "Variable": var,
                        "Category": str(cat),
                        "Overall": f"{n:,} ({pct:.1f}%)",
                    }
                )

    return pd.DataFrame(results)


# Convenience function for complete pipeline
def load_and_prepare(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, dict]:
    """
    Complete data loading and preparation pipeline.

    Args:
        config_path: Path to config file

    Returns:
        Tuple of (prepared DataFrame, metadata)
    """
    config = load_config(config_path)
    vars = get_variable_lists(config)

    # Load data
    data_path = Path(config["paths"]["processed_data"]) / "analytic_sample.parquet"

    if data_path.exists():
        df = load_ecls_data(str(data_path))
    else:
        raw_path = Path(config["paths"]["raw_data"]) / config["data"]["filename"]
        df = load_ecls_data(str(raw_path), columns=vars["all"])

    # Preprocess
    df = handle_missing_values(df)
    df = create_race_variable(df)
    df = create_ses_variable(df)

    # Create outcomes
    for name, col in config["variables"]["outcomes"].items():
        percentile = config["variables"]["at_risk_percentile"]
        df = create_at_risk_indicator(df, col, percentile)

    # Create analytic sample
    outcome_col = f"{config['variables']['outcomes']['reading']}_at_risk"
    df, stats = create_analytic_sample(df, vars["predictors"], outcome_col)

    metadata = {"config": config, "variables": vars, "sample_stats": stats}

    return df, metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Data loader module ready")
