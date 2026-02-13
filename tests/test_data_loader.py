"""Tests for data_loader module."""

import pandas as pd
import numpy as np
import pytest

from src.data_loader import (
    handle_missing_values,
    create_race_variable,
    create_ses_variable,
    create_at_risk_indicator,
    prepare_modeling_data,
    get_variable_lists,
)


def test_handle_missing_values():
    df = pd.DataFrame({"A": [1, -1, 3, -9], "B": [10, 20, -7, 40]})
    result = handle_missing_values(df, missing_codes=[-1, -7, -9])
    assert result["A"].isna().sum() == 2
    assert result["B"].isna().sum() == 1
    assert result.loc[0, "A"] == 1


def test_create_race_variable():
    df = pd.DataFrame({"X_RACETH_R": [1, 2, 3, 4, 5, 6, 7]})
    result = create_race_variable(df)
    assert "race_ethnicity" in result.columns
    assert result["race_ethnicity"].iloc[0] == "White"
    assert result["race_ethnicity"].iloc[1] == "Black"
    assert result["race_ethnicity"].iloc[4] == "Other"


def test_create_ses_variable():
    df = pd.DataFrame({"X1SESQ5": [1, 2, 3, 4, 5]})
    result = create_ses_variable(df)
    assert "ses_category" in result.columns
    assert "ses_low" in result.columns
    assert result["ses_low"].iloc[0] == 1  # Q1 is low
    assert result["ses_low"].iloc[4] == 0  # Q5 is not low


def test_create_at_risk_indicator():
    df = pd.DataFrame({"score": np.arange(100)})
    result = create_at_risk_indicator(df, "score", percentile=25)
    assert "score_at_risk" in result.columns
    # Roughly 25% should be at-risk
    assert 0.20 <= result["score_at_risk"].mean() <= 0.30


def test_prepare_modeling_data(sample_data):
    X, y, groups = prepare_modeling_data(
        sample_data,
        ["X1RTHETK", "X1MTHETK"],
        "X9RTHETA_at_risk",
        "race_ethnicity",
    )
    assert len(X) == len(y) == len(groups)
    assert X.shape[1] == 2
    assert set(y.unique()).issubset({0, 1})


def test_prepare_modeling_data_drops_missing(sample_data):
    df = sample_data.copy()
    df.loc[0:9, "X1RTHETK"] = np.nan
    X, y, groups = prepare_modeling_data(
        df, ["X1RTHETK", "X1MTHETK"], "X9RTHETA_at_risk", "race_ethnicity"
    )
    assert len(X) == len(sample_data) - 10


def test_get_variable_lists(sample_config):
    result = get_variable_lists(sample_config)
    assert "predictors" in result
    assert "outcomes" in result
    assert "X1RTHETK" in result["predictors"]
