"""
Missing Data Sensitivity Analysis Module
==========================================

Assess robustness of findings to missing data handling strategy.
Implements attrition analysis, multiple imputation (MICE via
IterativeImputer), and inverse probability weighting (IPW).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from scipy import stats

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression

from .models import ModelTrainer
from .fairness import FairnessEvaluator, FairnessConfidenceIntervals
from .data_loader import (
    get_variable_lists,
    prepare_modeling_data,
    create_at_risk_indicator,
    create_race_variable,
)

logger = logging.getLogger(__name__)


class AttritionAnalyzer:
    """Compare baseline characteristics of completers vs. dropouts."""

    def __init__(
        self,
        df_full: pd.DataFrame,
        predictor_cols: List[str],
        outcome_col: str,
        group_col: str = "race_ethnicity",
    ):
        self.df_full = df_full
        self.predictor_cols = predictor_cols
        self.outcome_col = outcome_col
        self.group_col = group_col

        # Identify completers using same logic as prepare_modeling_data
        available = [c for c in predictor_cols if c in df_full.columns]
        X = df_full[available]
        y = df_full[outcome_col] if outcome_col in df_full.columns else None

        if y is not None:
            self.is_complete = X.notna().all(axis=1) & y.notna()
        else:
            self.is_complete = X.notna().all(axis=1)

        self.n_complete = self.is_complete.sum()
        self.n_dropout = (~self.is_complete).sum()
        logger.info(
            f"Attrition analysis: {self.n_complete:,} completers, "
            f"{self.n_dropout:,} dropouts"
        )

    def compare_baseline_characteristics(self) -> pd.DataFrame:
        """
        Compare continuous and categorical baseline variables between
        completers and dropouts.

        Returns:
            DataFrame with columns: Variable, Completer_Mean, Completer_SD,
            Dropout_Mean, Dropout_SD, Cohens_d, p_value, Type
        """
        completers = self.df_full[self.is_complete]
        dropouts = self.df_full[~self.is_complete]

        rows = []
        # Continuous variables: cognitive and approaches-to-learning scores
        continuous_vars = [
            c for c in self.predictor_cols
            if c in self.df_full.columns and c not in (
                "X_CHSEX_R", "X_RACETH_R", "X1SESQ5", "X12LANGST"
            )
        ]
        for var in continuous_vars:
            c_vals = completers[var].dropna()
            d_vals = dropouts[var].dropna()
            if len(c_vals) < 2 or len(d_vals) < 2:
                continue

            c_mean, c_std = c_vals.mean(), c_vals.std()
            d_mean, d_std = d_vals.mean(), d_vals.std()

            # Cohen's d (pooled SD)
            pooled_sd = np.sqrt(
                ((len(c_vals) - 1) * c_std**2 + (len(d_vals) - 1) * d_std**2)
                / (len(c_vals) + len(d_vals) - 2)
            )
            cohens_d = (c_mean - d_mean) / pooled_sd if pooled_sd > 0 else 0.0

            # Welch's t-test
            t_stat, p_val = stats.ttest_ind(c_vals, d_vals, equal_var=False)

            rows.append({
                "Variable": var,
                "Completer_N": len(c_vals),
                "Completer_Mean": round(c_mean, 3),
                "Completer_SD": round(c_std, 3),
                "Dropout_N": len(d_vals),
                "Dropout_Mean": round(d_mean, 3),
                "Dropout_SD": round(d_std, 3),
                "Cohens_d": round(cohens_d, 3),
                "p_value": round(p_val, 4),
                "Type": "continuous",
            })

        # Categorical variables
        categorical_vars = ["X_CHSEX_R", "X_RACETH_R", "X1SESQ5", "X12LANGST"]
        for var in categorical_vars:
            if var not in self.df_full.columns:
                continue
            c_vals = completers[var].dropna()
            d_vals = dropouts[var].dropna()
            if len(c_vals) < 2 or len(d_vals) < 2:
                continue

            # Chi-squared test
            c_counts = c_vals.value_counts().sort_index()
            d_counts = d_vals.value_counts().sort_index()
            # Align indices
            all_cats = sorted(set(c_counts.index) | set(d_counts.index))
            contingency = pd.DataFrame({
                "completer": [c_counts.get(cat, 0) for cat in all_cats],
                "dropout": [d_counts.get(cat, 0) for cat in all_cats],
            }, index=all_cats)

            chi2, p_val, dof, expected = stats.chi2_contingency(contingency.values)

            # Cramér's V as effect size
            n_total = contingency.values.sum()
            k = min(contingency.shape)
            cramers_v = np.sqrt(chi2 / (n_total * (k - 1))) if k > 1 else 0.0

            rows.append({
                "Variable": var,
                "Completer_N": len(c_vals),
                "Completer_Mean": np.nan,
                "Completer_SD": np.nan,
                "Dropout_N": len(d_vals),
                "Dropout_Mean": np.nan,
                "Dropout_SD": np.nan,
                "Cohens_d": round(cramers_v, 3),
                "p_value": round(p_val, 4),
                "Type": "categorical",
            })

        return pd.DataFrame(rows)

    def compare_demographics(self) -> pd.DataFrame:
        """
        Compare race/ethnicity and SES distributions between
        completers and dropouts.

        Returns:
            DataFrame with Variable, Category, Completer_N, Completer_Pct,
            Dropout_N, Dropout_Pct
        """
        completers = self.df_full[self.is_complete]
        dropouts = self.df_full[~self.is_complete]

        rows = []
        for var in [self.group_col, "ses_category", "X_CHSEX_R", "X12LANGST"]:
            if var not in self.df_full.columns:
                continue
            c_counts = completers[var].value_counts()
            d_counts = dropouts[var].value_counts()

            all_cats = sorted(
                set(c_counts.index) | set(d_counts.index), key=str
            )
            for cat in all_cats:
                c_n = c_counts.get(cat, 0)
                d_n = d_counts.get(cat, 0)
                rows.append({
                    "Variable": var,
                    "Category": str(cat),
                    "Completer_N": c_n,
                    "Completer_Pct": round(100 * c_n / len(completers), 1)
                    if len(completers) > 0 else 0,
                    "Dropout_N": d_n,
                    "Dropout_Pct": round(100 * d_n / len(dropouts), 1)
                    if len(dropouts) > 0 else 0,
                })

        return pd.DataFrame(rows)

    def get_summary(self) -> dict:
        """Return summary statistics."""
        comparison = self.compare_baseline_characteristics()
        sig_vars = comparison[comparison["p_value"] < 0.05]["Variable"].tolist()
        large_smd = comparison[
            comparison["Cohens_d"].abs() >= 0.20
        ]["Variable"].tolist()

        return {
            "n_total": len(self.df_full),
            "n_completers": int(self.n_complete),
            "n_dropouts": int(self.n_dropout),
            "completion_rate": round(self.n_complete / len(self.df_full), 3),
            "significant_differences": sig_vars,
            "large_effect_sizes": large_smd,
        }


class MICEAnalyzer:
    """
    Multiple imputation via sklearn IterativeImputer.

    Creates m imputed datasets, trains models on each, and pools
    results using Rubin's rules.
    """

    def __init__(
        self,
        config: dict,
        n_imputations: int = 10,
        max_iter: int = 10,
        random_state: int = 42,
    ):
        self.config = config
        self.n_imputations = n_imputations
        self.max_iter = max_iter
        self.random_state = random_state

    def create_imputed_datasets(
        self,
        df: pd.DataFrame,
        predictor_cols: List[str],
        outcome_col: str,
    ) -> List[pd.DataFrame]:
        """
        Generate multiply-imputed datasets from the full sample.

        Uses IterativeImputer with BayesianRidge and sample_posterior=True
        to introduce proper between-imputation variance.

        Args:
            df: Full dataset (with missing values)
            predictor_cols: Predictor column names
            outcome_col: Outcome column name

        Returns:
            List of m complete DataFrames
        """
        available = [c for c in predictor_cols if c in df.columns]
        impute_cols = available + [outcome_col]

        # Keep only columns that exist and are numeric
        df_numeric = df[impute_cols].copy()
        for col in df_numeric.columns:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")

        datasets = []
        for m in range(self.n_imputations):
            logger.info(
                f"  Creating imputed dataset {m + 1}/{self.n_imputations}..."
            )
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=self.max_iter,
                random_state=self.random_state + m,
                sample_posterior=True,
            )
            imputed_values = imputer.fit_transform(df_numeric)
            df_imputed = pd.DataFrame(
                imputed_values, columns=impute_cols, index=df.index
            )

            # Round categorical variables to nearest valid integer
            cat_vars = {
                "X_RACETH_R": (1, 7),
                "X_CHSEX_R": (1, 2),
                "X1SESQ5": (1, 5),
                "X12LANGST": (1, 2),
            }
            for var, (lo, hi) in cat_vars.items():
                if var in df_imputed.columns:
                    df_imputed[var] = (
                        df_imputed[var].round().clip(lo, hi).astype(int)
                    )

            # Rebuild full dataframe: start with original, overwrite imputed cols
            df_complete = df.copy()
            for col in impute_cols:
                df_complete[col] = df_imputed[col]

            # Reconstruct race_ethnicity from imputed X_RACETH_R
            if "X_RACETH_R" in df_complete.columns:
                df_complete = create_race_variable(
                    df_complete, race_col="X_RACETH_R"
                )

            datasets.append(df_complete)

        logger.info(f"Created {len(datasets)} imputed datasets")
        return datasets

    def train_on_imputed(
        self,
        imputed_datasets: List[pd.DataFrame],
        predictor_cols: List[str],
        outcome_col: str,
        group_col: str = "race_ethnicity",
        model_name: str = "elastic_net",
    ) -> List[dict]:
        """
        Train model on each imputed dataset and collect results.

        Args:
            imputed_datasets: List of complete DataFrames
            predictor_cols: Feature columns
            outcome_col: Binary outcome column (e.g. X9RTHETA_at_risk)
            group_col: Protected attribute
            model_name: Which model to train

        Returns:
            List of result dicts per imputation
        """
        ref_group = self.config.get("fairness", {}).get(
            "reference_groups", {}
        ).get("race", "White")
        results = []

        for i, df_imp in enumerate(imputed_datasets):
            logger.info(f"  Training on imputed dataset {i + 1}...")

            X, y, groups = prepare_modeling_data(
                df_imp, predictor_cols, outcome_col, group_col
            )

            if len(X) == 0:
                logger.warning(f"  Imputed dataset {i + 1} has no valid rows")
                continue

            trainer = ModelTrainer(
                random_state=self.config["model"]["random_state"],
                test_size=self.config["model"]["test_size"],
                cv_folds=self.config["model"]["cv_folds"],
            )
            X_train, X_test, y_train, y_test = trainer.split_data(X, y)
            groups_test = groups.loc[X_test.index].reset_index(drop=True)

            trainer.train_model(model_name, X_train, y_train)
            perf_df = trainer.evaluate_all_models(X_test, y_test)

            model = trainer.models[model_name]
            y_pred, y_prob = trainer.get_predictions(model, X_test)

            evaluator = FairnessEvaluator(
                y_test.values,
                y_pred,
                y_prob,
                groups_test,
                reference_group=ref_group,
            )
            evaluator.compute_all_group_metrics()

            result = {
                "imputation": i + 1,
                "n_samples": len(X),
                "auc_roc": perf_df.loc[model_name, "auc_roc"],
                "accuracy": perf_df.loc[model_name, "accuracy"],
                "group_metrics": evaluator.get_summary_table(),
            }
            results.append(result)

        return results

    def pool_results(self, imputation_results: List[dict]) -> dict:
        """
        Pool results across imputations using Rubin's rules.

        For each metric Q:
          Q_bar = mean(Q_m)
          B = var(Q_m)  (between-imputation variance)
          T = B + B/m   (simplified total variance when within-imp var unavailable)
          SE = sqrt(T)
          95% CI: Q_bar +/- 1.96 * SE

        Returns:
            Dict with pooled performance and fairness metrics
        """
        m = len(imputation_results)
        if m == 0:
            return {}

        # Pool performance
        aucs = [r["auc_roc"] for r in imputation_results]
        accs = [r["accuracy"] for r in imputation_results]

        pooled_perf = {
            "auc_roc": np.mean(aucs),
            "auc_roc_se": np.sqrt(np.var(aucs, ddof=1) * (1 + 1 / m)),
            "accuracy": np.mean(accs),
            "accuracy_se": np.sqrt(np.var(accs, ddof=1) * (1 + 1 / m)),
        }

        # Pool fairness by group
        group_metrics_list = [r["group_metrics"] for r in imputation_results]
        all_groups = set()
        for gm in group_metrics_list:
            all_groups.update(gm["Group"].tolist())

        pooled_fairness_rows = []
        for group in sorted(all_groups):
            tprs, fprs, ppvs = [], [], []
            for gm in group_metrics_list:
                row = gm[gm["Group"] == group]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                for metric, lst in [("TPR", tprs), ("FPR", fprs), ("PPV", ppvs)]:
                    if metric in row:
                        val = pd.to_numeric(row[metric], errors="coerce")
                        if pd.notna(val):
                            lst.append(float(val))

            pooled_row = {"Group": group}
            for metric_name, values in [
                ("TPR", tprs), ("FPR", fprs), ("PPV", ppvs)
            ]:
                if len(values) >= 2:
                    q_bar = np.mean(values)
                    b = np.var(values, ddof=1)
                    t_var = b * (1 + 1 / m)
                    se = np.sqrt(t_var)
                    pooled_row[metric_name] = round(q_bar, 4)
                    pooled_row[f"{metric_name}_SE"] = round(se, 4)
                    pooled_row[f"{metric_name}_CI_Lower"] = round(
                        q_bar - 1.96 * se, 4
                    )
                    pooled_row[f"{metric_name}_CI_Upper"] = round(
                        q_bar + 1.96 * se, 4
                    )
                elif len(values) == 1:
                    pooled_row[metric_name] = round(values[0], 4)
                    pooled_row[f"{metric_name}_SE"] = np.nan
                    pooled_row[f"{metric_name}_CI_Lower"] = np.nan
                    pooled_row[f"{metric_name}_CI_Upper"] = np.nan

            pooled_fairness_rows.append(pooled_row)

        return {
            "performance": pooled_perf,
            "fairness": pd.DataFrame(pooled_fairness_rows),
            "n_imputations": m,
            "n_samples_per_imputation": [
                r["n_samples"] for r in imputation_results
            ],
        }

    def compare_with_complete_case(
        self,
        pooled: dict,
        cc_auc: float,
        cc_fairness: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compare MICE pooled results with complete-case results.

        Args:
            pooled: Output from pool_results()
            cc_auc: Complete-case AUC
            cc_fairness: Complete-case fairness table (Group, TPR, FPR, PPV)

        Returns:
            Comparison DataFrame
        """
        rows = []

        # Performance comparison
        mice_auc = pooled["performance"]["auc_roc"]
        mice_auc_se = pooled["performance"]["auc_roc_se"]
        rows.append({
            "Metric": "AUC-ROC",
            "Group": "Overall",
            "Complete_Case": round(cc_auc, 4),
            "MICE_Pooled": round(mice_auc, 4),
            "MICE_SE": round(mice_auc_se, 4),
            "Difference": round(mice_auc - cc_auc, 4),
        })

        # Fairness comparison by group
        mice_fairness = pooled["fairness"]
        for _, mrow in mice_fairness.iterrows():
            group = mrow["Group"]
            cc_row = cc_fairness[cc_fairness["Group"] == group]
            if len(cc_row) == 0:
                continue
            cc_row = cc_row.iloc[0]

            for metric in ["TPR", "FPR", "PPV"]:
                if metric in mrow and metric in cc_row:
                    cc_val = pd.to_numeric(cc_row[metric], errors="coerce")
                    mice_val = pd.to_numeric(mrow[metric], errors="coerce")
                    if pd.notna(cc_val) and pd.notna(mice_val):
                        rows.append({
                            "Metric": metric,
                            "Group": group,
                            "Complete_Case": round(float(cc_val), 4),
                            "MICE_Pooled": round(float(mice_val), 4),
                            "MICE_SE": round(
                                float(mrow.get(f"{metric}_SE", np.nan)), 4
                            ),
                            "Difference": round(
                                float(mice_val) - float(cc_val), 4
                            ),
                        })

        return pd.DataFrame(rows)


class IPWAnalyzer:
    """
    Inverse probability weighting to correct for selection into
    the complete-case sample.
    """

    def __init__(
        self,
        config: dict,
        weight_cap_percentile: int = 99,
        stabilize: bool = True,
        random_state: int = 42,
    ):
        self.config = config
        self.weight_cap_percentile = weight_cap_percentile
        self.stabilize = stabilize
        self.random_state = random_state

    def compute_weights(
        self,
        df_full: pd.DataFrame,
        predictor_cols: List[str],
        outcome_col: str,
        ipw_predictors: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame, dict]:
        """
        Compute inverse probability weights.

        Args:
            df_full: Full dataset with missing values
            predictor_cols: All predictor columns (for defining completeness)
            outcome_col: Outcome column
            ipw_predictors: Columns to use for predicting completeness
                (should have low missingness themselves)

        Returns:
            Tuple of (weights for complete cases, diagnostics DataFrame, summary dict)
        """
        if ipw_predictors is None:
            ipw_predictors = [
                "X_RACETH_R", "X_CHSEX_R", "X1SESQ5",
                "X1RTHETK", "X1MTHETK", "X12LANGST",
            ]

        # Define completeness (same as prepare_modeling_data)
        available = [c for c in predictor_cols if c in df_full.columns]
        X_all = df_full[available]
        y_all = df_full[outcome_col] if outcome_col in df_full.columns else None
        if y_all is not None:
            is_complete = X_all.notna().all(axis=1) & y_all.notna()
        else:
            is_complete = X_all.notna().all(axis=1)

        # Build prediction features (from low-missingness vars)
        ipw_available = [c for c in ipw_predictors if c in df_full.columns]
        X_ipw = df_full[ipw_available].copy()
        for col in X_ipw.columns:
            X_ipw[col] = pd.to_numeric(X_ipw[col], errors="coerce")

        # Only fit on rows with complete IPW predictors
        ipw_valid = X_ipw.notna().all(axis=1)
        X_ipw_valid = X_ipw[ipw_valid]
        y_ipw_valid = is_complete[ipw_valid].astype(int)

        logger.info(
            f"IPW model: {ipw_valid.sum():,} rows with complete IPW predictors"
        )

        # Fit logistic regression
        lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        lr.fit(X_ipw_valid, y_ipw_valid)

        # Predict P(complete) for all valid rows
        p_complete = lr.predict_proba(X_ipw_valid)[:, 1]

        # Clip probabilities to avoid division by zero
        p_complete = np.clip(p_complete, 0.01, 0.99)

        # Weights for complete cases only
        complete_mask = is_complete[ipw_valid].values.astype(bool)
        weights_raw = 1.0 / p_complete[complete_mask]

        # Stabilize
        if self.stabilize:
            overall_rate = is_complete.mean()
            weights_raw = weights_raw * overall_rate

        # Trim extreme weights
        cap = np.percentile(weights_raw, self.weight_cap_percentile)
        weights_trimmed = np.clip(weights_raw, None, cap)

        # Diagnostics
        diagnostics = pd.DataFrame({
            "weight_min": [weights_trimmed.min()],
            "weight_max": [weights_trimmed.max()],
            "weight_mean": [weights_trimmed.mean()],
            "weight_median": [np.median(weights_trimmed)],
            "weight_sd": [weights_trimmed.std()],
            "n_trimmed": [int((weights_raw > cap).sum())],
            "cap_value": [cap],
        })

        summary = {
            "n_ipw_valid": int(ipw_valid.sum()),
            "n_complete": int(complete_mask.sum()),
            "model_accuracy": float(lr.score(X_ipw_valid, y_ipw_valid)),
            "weight_range": f"{weights_trimmed.min():.3f} - {weights_trimmed.max():.3f}",
        }

        logger.info(
            f"IPW weights: range={summary['weight_range']}, "
            f"mean={weights_trimmed.mean():.3f}, "
            f"trimmed={int((weights_raw > cap).sum())}"
        )

        return weights_trimmed, diagnostics, summary

    def train_with_weights(
        self,
        df_complete: pd.DataFrame,
        weights: np.ndarray,
        predictor_cols: List[str],
        outcome_col: str,
        group_col: str = "race_ethnicity",
        model_name: str = "elastic_net",
    ) -> dict:
        """
        Train model using IPW sample weights.

        Args:
            df_complete: Complete-case dataset
            weights: IPW weights aligned with df_complete
            predictor_cols: Feature columns
            outcome_col: Binary outcome
            group_col: Protected attribute
            model_name: Model to train

        Returns:
            Dict with performance and fairness metrics
        """
        ref_group = self.config.get("fairness", {}).get(
            "reference_groups", {}
        ).get("race", "White")

        X, y, groups = prepare_modeling_data(
            df_complete, predictor_cols, outcome_col, group_col
        )

        # Build a weight Series aligned with X's index
        # weights is a flat array aligned with df_complete's complete rows
        if len(weights) >= len(X):
            weight_series = pd.Series(
                weights[:len(X)], index=X.index
            )
        else:
            weight_series = pd.Series(
                np.ones(len(X)), index=X.index
            )

        trainer = ModelTrainer(
            random_state=self.config["model"]["random_state"],
            test_size=self.config["model"]["test_size"],
            cv_folds=self.config["model"]["cv_folds"],
        )
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        groups_test = groups.loc[X_test.index].reset_index(drop=True)

        # Align weights with train indices
        train_weights = weight_series.loc[X_train.index].values

        trainer.train_model(
            model_name, X_train, y_train, sample_weight=train_weights
        )
        perf_df = trainer.evaluate_all_models(X_test, y_test)

        model = trainer.models[model_name]
        y_pred, y_prob = trainer.get_predictions(model, X_test)

        evaluator = FairnessEvaluator(
            y_test.values,
            y_pred,
            y_prob,
            groups_test,
            reference_group=ref_group,
        )
        evaluator.compute_all_group_metrics()

        return {
            "auc_roc": perf_df.loc[model_name, "auc_roc"],
            "accuracy": perf_df.loc[model_name, "accuracy"],
            "group_metrics": evaluator.get_summary_table(),
        }

    def compare_with_unweighted(
        self,
        ipw_results: dict,
        cc_auc: float,
        cc_fairness: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compare IPW-weighted vs. unweighted (complete-case) results."""
        rows = []

        # Performance
        ipw_auc = ipw_results["auc_roc"]
        rows.append({
            "Metric": "AUC-ROC",
            "Group": "Overall",
            "Unweighted": round(cc_auc, 4),
            "IPW_Weighted": round(ipw_auc, 4),
            "Difference": round(ipw_auc - cc_auc, 4),
        })

        # Fairness by group
        ipw_fairness = ipw_results["group_metrics"]
        for _, irow in ipw_fairness.iterrows():
            group = irow["Group"]
            cc_row = cc_fairness[cc_fairness["Group"] == group]
            if len(cc_row) == 0:
                continue
            cc_row = cc_row.iloc[0]

            for metric in ["TPR", "FPR", "PPV"]:
                if metric in irow and metric in cc_row:
                    cc_val = pd.to_numeric(cc_row[metric], errors="coerce")
                    ipw_val = pd.to_numeric(irow[metric], errors="coerce")
                    if pd.notna(cc_val) and pd.notna(ipw_val):
                        rows.append({
                            "Metric": metric,
                            "Group": group,
                            "Unweighted": round(float(cc_val), 4),
                            "IPW_Weighted": round(float(ipw_val), 4),
                            "Difference": round(
                                float(ipw_val) - float(cc_val), 4
                            ),
                        })

        return pd.DataFrame(rows)


def run_missing_data_analysis(
    config: dict,
    df_full: pd.DataFrame,
    df_analytic: pd.DataFrame,
    complete_case_results: Optional[dict] = None,
) -> dict:
    """
    Run complete missing data sensitivity analysis.

    Args:
        config: Pipeline configuration
        df_full: Full dataset before complete-case filtering
        df_analytic: Analytic sample (after create_analytic_sample)
        complete_case_results: Results from primary analysis

    Returns:
        Dict with tables and summaries
    """
    logger.info("=" * 60)
    logger.info("MISSING DATA SENSITIVITY ANALYSIS")
    logger.info("=" * 60)

    vars_cfg = get_variable_lists(config)
    predictor_cols = vars_cfg["predictors"]
    outcome_var = config["variables"]["outcomes"]["reading"]
    outcome_col = f"{outcome_var}_at_risk"
    missing_cfg = config.get("missing_data", {})

    tables = {}

    # ----------------------------------------------------------------
    # 1. Attrition Analysis
    # ----------------------------------------------------------------
    logger.info("--- Attrition Analysis ---")
    attrition = AttritionAnalyzer(
        df_full, predictor_cols, outcome_col, group_col="race_ethnicity"
    )
    tables["attrition_comparison"] = attrition.compare_baseline_characteristics()
    tables["attrition_demographics"] = attrition.compare_demographics()
    attrition_summary = attrition.get_summary()
    logger.info(f"Attrition summary: {attrition_summary}")

    # Extract complete-case AUC and fairness for comparisons
    cc_auc = None
    cc_fairness = None
    if complete_case_results:
        cc_auc = complete_case_results.get("auc_roc")
        cc_fairness = complete_case_results.get("group_metrics")

    # ----------------------------------------------------------------
    # 2. MICE Analysis
    # ----------------------------------------------------------------
    mice_cfg = missing_cfg.get("mice", {})
    if mice_cfg.get("enabled", True):
        logger.info("--- MICE Analysis ---")
        n_imp = mice_cfg.get("n_imputations", 10)
        max_iter = mice_cfg.get("max_iter", 10)
        model_name = mice_cfg.get("model_for_comparison", "elastic_net")

        mice = MICEAnalyzer(
            config, n_imputations=n_imp, max_iter=max_iter, random_state=42
        )

        # Impute the full sample
        # Need outcome column in the df for imputation
        imputed = mice.create_imputed_datasets(
            df_full, predictor_cols, outcome_var
        )

        # Create at-risk indicator on each imputed dataset
        pct = config["variables"]["at_risk_percentile"]
        for i, df_imp in enumerate(imputed):
            imputed[i] = create_at_risk_indicator(
                df_imp, outcome_var, pct, name=outcome_col
            )

        # Train on each
        imp_results = mice.train_on_imputed(
            imputed, predictor_cols, outcome_col,
            model_name=model_name,
        )

        # Pool
        pooled = mice.pool_results(imp_results)
        tables["mice_pooled_fairness"] = pooled.get("fairness", pd.DataFrame())

        # Compare
        if cc_auc is not None and cc_fairness is not None:
            tables["mice_vs_complete_case"] = mice.compare_with_complete_case(
                pooled, cc_auc, cc_fairness
            )

        logger.info(
            f"MICE pooled AUC: {pooled['performance']['auc_roc']:.4f} "
            f"(SE={pooled['performance']['auc_roc_se']:.4f})"
        )

    # ----------------------------------------------------------------
    # 3. IPW Analysis
    # ----------------------------------------------------------------
    ipw_cfg = missing_cfg.get("ipw", {})
    if ipw_cfg.get("enabled", True):
        logger.info("--- IPW Analysis ---")
        model_name = ipw_cfg.get("model_for_comparison", "elastic_net")
        ipw_predictors = missing_cfg.get("ipw_predictors", None)

        ipw = IPWAnalyzer(
            config,
            weight_cap_percentile=ipw_cfg.get("weight_cap_percentile", 99),
            stabilize=ipw_cfg.get("stabilize", True),
        )

        weights, diagnostics, ipw_summary = ipw.compute_weights(
            df_full, predictor_cols, outcome_col,
            ipw_predictors=ipw_predictors,
        )
        tables["ipw_weight_diagnostics"] = diagnostics

        # Train with weights on the analytic sample
        ipw_results = ipw.train_with_weights(
            df_analytic, weights, predictor_cols, outcome_col,
            model_name=model_name,
        )

        # Compare
        if cc_auc is not None and cc_fairness is not None:
            tables["ipw_vs_unweighted"] = ipw.compare_with_unweighted(
                ipw_results, cc_auc, cc_fairness
            )

        logger.info(
            f"IPW AUC: {ipw_results['auc_roc']:.4f} "
            f"(unweighted: {cc_auc:.4f})" if cc_auc else ""
        )

    return {
        "tables": tables,
        "attrition_summary": attrition_summary,
    }
