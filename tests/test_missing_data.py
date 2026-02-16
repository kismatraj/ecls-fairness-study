"""Tests for missing data sensitivity analysis module."""

import numpy as np
import pandas as pd
import pytest

from src.missing_data import AttritionAnalyzer, MICEAnalyzer, IPWAnalyzer


@pytest.fixture
def full_data_with_missing():
    """Dataset with realistic missing values simulating attrition."""
    rng = np.random.RandomState(42)
    n = 600

    df = pd.DataFrame({
        "X1RTHETK": rng.normal(50, 10, n),
        "X2RTHETK": rng.normal(55, 10, n),
        "X1MTHETK": rng.normal(50, 10, n),
        "X2MTHETK": rng.normal(55, 10, n),
        "X6DCCSSCR": rng.normal(60, 12, n),
        "X1TCHAPP": rng.uniform(1, 4, n),
        "X2TCHAPP": rng.uniform(1, 4, n),
        "X4TCHAPP": rng.uniform(1, 4, n),
        "X9RTHETA": rng.normal(100, 20, n),
        "X_CHSEX_R": rng.choice([1, 2], n),
        "X_RACETH_R": rng.choice(
            [1, 2, 3, 4, 5], n, p=[0.5, 0.12, 0.24, 0.05, 0.09]
        ),
        "X1SESQ5": rng.choice([1, 2, 3, 4, 5], n),
        "X12LANGST": rng.choice([1, 2], n, p=[0.8, 0.2]),
    })

    # Derived variables
    race_map = {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other"}
    df["race_ethnicity"] = df["X_RACETH_R"].map(race_map)

    # At-risk indicator
    threshold = df["X9RTHETA"].quantile(0.25)
    df["X9RTHETA_at_risk"] = (df["X9RTHETA"] < threshold).astype(int)

    # Introduce missing values (30% outcome, 20% exec function, 10% baseline)
    missing_outcome = rng.choice(n, size=int(0.30 * n), replace=False)
    df.loc[missing_outcome, "X9RTHETA"] = np.nan
    df.loc[missing_outcome, "X9RTHETA_at_risk"] = np.nan

    missing_exec = rng.choice(n, size=int(0.20 * n), replace=False)
    df.loc[missing_exec, "X6DCCSSCR"] = np.nan

    missing_base = rng.choice(n, size=int(0.10 * n), replace=False)
    df.loc[missing_base, "X1RTHETK"] = np.nan

    return df


@pytest.fixture
def predictor_cols():
    return [
        "X1RTHETK", "X2RTHETK", "X1MTHETK", "X2MTHETK",
        "X6DCCSSCR", "X1TCHAPP", "X2TCHAPP", "X4TCHAPP",
        "X_CHSEX_R", "X_RACETH_R", "X1SESQ5", "X12LANGST",
    ]


@pytest.fixture
def missing_config():
    """Config with missing data settings."""
    return {
        "variables": {
            "outcomes": {"reading": "X9RTHETA"},
            "at_risk_percentile": 25,
            "demographics": {
                "race": "X_RACETH_R",
                "sex": "X_CHSEX_R",
                "ses": "X1SESQ5",
                "language": "X12LANGST",
            },
            "predictors": {
                "baseline_cognitive": ["X1RTHETK", "X2RTHETK", "X1MTHETK", "X2MTHETK"],
                "executive_function": ["X6DCCSSCR"],
                "approaches_to_learning": ["X1TCHAPP", "X2TCHAPP", "X4TCHAPP"],
                "child_demographics": [
                    "X_CHSEX_R", "X_RACETH_R", "X1SESQ5", "X12LANGST",
                ],
            },
        },
        "model": {
            "random_state": 42,
            "test_size": 0.30,
            "cv_folds": 3,
            "algorithms": {
                "logistic_regression": {"enabled": True},
            },
        },
        "fairness": {
            "reference_groups": {"race": "White"},
            "bootstrap_iterations": 20,
        },
        "missing_data": {
            "enabled": True,
            "mice": {
                "enabled": True,
                "n_imputations": 2,
                "max_iter": 3,
                "model_for_comparison": "logistic_regression",
            },
            "ipw": {
                "enabled": True,
                "weight_cap_percentile": 99,
                "stabilize": True,
                "model_for_comparison": "logistic_regression",
            },
            "ipw_predictors": ["X_RACETH_R", "X_CHSEX_R", "X1RTHETK", "X1MTHETK"],
        },
    }


# ----------------------------------------------------------------
# AttritionAnalyzer Tests
# ----------------------------------------------------------------

class TestAttritionAnalyzer:

    def test_identifies_completers(self, full_data_with_missing, predictor_cols):
        analyzer = AttritionAnalyzer(
            full_data_with_missing, predictor_cols, "X9RTHETA_at_risk"
        )
        assert analyzer.n_complete > 0
        assert analyzer.n_dropout > 0
        assert analyzer.n_complete + analyzer.n_dropout == len(full_data_with_missing)

    def test_compare_baseline(self, full_data_with_missing, predictor_cols):
        analyzer = AttritionAnalyzer(
            full_data_with_missing, predictor_cols, "X9RTHETA_at_risk"
        )
        comparison = analyzer.compare_baseline_characteristics()
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) > 0
        assert "Variable" in comparison.columns
        assert "Cohens_d" in comparison.columns
        assert "p_value" in comparison.columns

    def test_compare_demographics(self, full_data_with_missing, predictor_cols):
        analyzer = AttritionAnalyzer(
            full_data_with_missing, predictor_cols, "X9RTHETA_at_risk"
        )
        demo = analyzer.compare_demographics()
        assert isinstance(demo, pd.DataFrame)
        assert "Category" in demo.columns
        assert "Completer_Pct" in demo.columns

    def test_get_summary(self, full_data_with_missing, predictor_cols):
        analyzer = AttritionAnalyzer(
            full_data_with_missing, predictor_cols, "X9RTHETA_at_risk"
        )
        summary = analyzer.get_summary()
        assert "n_total" in summary
        assert "n_completers" in summary
        assert "completion_rate" in summary
        assert 0 < summary["completion_rate"] < 1


# ----------------------------------------------------------------
# MICEAnalyzer Tests
# ----------------------------------------------------------------

class TestMICEAnalyzer:

    def test_create_imputed_datasets(
        self, full_data_with_missing, predictor_cols, missing_config
    ):
        mice = MICEAnalyzer(missing_config, n_imputations=2, max_iter=3)
        datasets = mice.create_imputed_datasets(
            full_data_with_missing, predictor_cols, "X9RTHETA"
        )
        assert len(datasets) == 2
        # Imputed datasets should have no NaN in imputed columns
        for ds in datasets:
            for col in predictor_cols:
                if col in ds.columns:
                    assert ds[col].notna().all(), f"{col} still has NaN"

    def test_train_on_imputed(
        self, full_data_with_missing, predictor_cols, missing_config
    ):
        mice = MICEAnalyzer(missing_config, n_imputations=2, max_iter=3)
        datasets = mice.create_imputed_datasets(
            full_data_with_missing, predictor_cols, "X9RTHETA"
        )
        # Create at-risk indicator
        for i, ds in enumerate(datasets):
            threshold = ds["X9RTHETA"].quantile(0.25)
            datasets[i]["X9RTHETA_at_risk"] = (
                ds["X9RTHETA"] < threshold
            ).astype(int)

        results = mice.train_on_imputed(
            datasets, predictor_cols, "X9RTHETA_at_risk",
            model_name="logistic_regression",
        )
        assert len(results) == 2
        assert all("auc_roc" in r for r in results)
        assert all(0 <= r["auc_roc"] <= 1 for r in results)

    def test_pool_results(
        self, full_data_with_missing, predictor_cols, missing_config
    ):
        mice = MICEAnalyzer(missing_config, n_imputations=2, max_iter=3)
        datasets = mice.create_imputed_datasets(
            full_data_with_missing, predictor_cols, "X9RTHETA"
        )
        for i, ds in enumerate(datasets):
            threshold = ds["X9RTHETA"].quantile(0.25)
            datasets[i]["X9RTHETA_at_risk"] = (
                ds["X9RTHETA"] < threshold
            ).astype(int)

        results = mice.train_on_imputed(
            datasets, predictor_cols, "X9RTHETA_at_risk",
            model_name="logistic_regression",
        )
        pooled = mice.pool_results(results)
        assert "performance" in pooled
        assert "fairness" in pooled
        assert pooled["performance"]["auc_roc"] > 0


# ----------------------------------------------------------------
# IPWAnalyzer Tests
# ----------------------------------------------------------------

class TestIPWAnalyzer:

    def test_compute_weights(
        self, full_data_with_missing, predictor_cols, missing_config
    ):
        ipw = IPWAnalyzer(missing_config)
        weights, diagnostics, summary = ipw.compute_weights(
            full_data_with_missing, predictor_cols, "X9RTHETA_at_risk",
            ipw_predictors=["X_RACETH_R", "X_CHSEX_R", "X1MTHETK"],
        )
        assert len(weights) > 0
        assert (weights > 0).all(), "All weights should be positive"
        assert summary["n_complete"] > 0

    def test_weights_stabilized(
        self, full_data_with_missing, predictor_cols, missing_config
    ):
        ipw = IPWAnalyzer(missing_config, stabilize=True)
        weights, _, _ = ipw.compute_weights(
            full_data_with_missing, predictor_cols, "X9RTHETA_at_risk",
            ipw_predictors=["X_RACETH_R", "X_CHSEX_R", "X1MTHETK"],
        )
        # Stabilized weights should average close to the completion rate
        assert weights.mean() < 2.0, "Stabilized weights should not be extreme"

    def test_train_with_weights(
        self, full_data_with_missing, predictor_cols, missing_config
    ):
        ipw = IPWAnalyzer(missing_config)
        weights, _, _ = ipw.compute_weights(
            full_data_with_missing, predictor_cols, "X9RTHETA_at_risk",
            ipw_predictors=["X_RACETH_R", "X_CHSEX_R", "X1MTHETK"],
        )
        # Get the complete-case subset
        from src.data_loader import prepare_modeling_data
        X, y, groups = prepare_modeling_data(
            full_data_with_missing, predictor_cols, "X9RTHETA_at_risk"
        )
        # Build complete df for training
        cc_df = full_data_with_missing.loc[X.index].copy()

        result = ipw.train_with_weights(
            cc_df, weights, predictor_cols, "X9RTHETA_at_risk",
            model_name="logistic_regression",
        )
        assert "auc_roc" in result
        assert 0 <= result["auc_roc"] <= 1
