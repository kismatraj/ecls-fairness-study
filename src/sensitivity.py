"""
Sensitivity Analysis Module
============================

Evaluate robustness of findings across different at-risk thresholds
and outcome definitions.
"""

import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging

from .models import ModelTrainer
from .fairness import (
    FairnessEvaluator,
    FairnessConfidenceIntervals,
)
from .data_loader import prepare_modeling_data

logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    """
    Run the modeling pipeline across multiple at-risk threshold
    percentiles to assess robustness of performance and fairness findings.
    """

    def __init__(
        self,
        config: dict,
        percentiles: Optional[List[int]] = None,
    ):
        self.config = config
        self.percentiles = percentiles or [10, 20, 25, 30]
        self.results: Dict[int, Dict] = {}

    def run_threshold_sensitivity(
        self,
        df: pd.DataFrame,
        outcome_var: str = "X9RTHETA",
        group_col: str = "race_ethnicity",
        model_names: Optional[List[str]] = None,
    ) -> Dict[int, Dict]:
        """
        Run full pipeline for each percentile threshold.

        Args:
            df: Full analytic sample (before at-risk indicator creation)
            outcome_var: Continuous outcome variable
            group_col: Protected attribute column
            model_names: Models to train (None = all enabled)

        Returns:
            Dictionary mapping percentile to results dict
        """
        from .data_loader import get_variable_lists

        vars_cfg = get_variable_lists(self.config)
        predictor_cols = vars_cfg["predictors"]

        if model_names is None:
            model_names = [
                name
                for name, cfg in self.config["model"]["algorithms"].items()
                if cfg.get("enabled", True)
            ]

        ref_group = self.config["fairness"]["reference_groups"]["race"]

        for pct in self.percentiles:
            logger.info(f"=== Sensitivity: {pct}th percentile threshold ===")

            # Create at-risk indicator at this threshold
            df_pct = df.copy()
            threshold = df_pct[outcome_var].quantile(pct / 100)
            outcome_col = f"{outcome_var}_at_risk"
            df_pct[outcome_col] = (df_pct[outcome_var] < threshold).astype(int)
            prevalence = df_pct[outcome_col].mean()
            logger.info(f"  Threshold={threshold:.2f}, prevalence={prevalence:.1%}")

            # Prepare modeling data
            X, y, groups = prepare_modeling_data(
                df_pct, predictor_cols, outcome_col, group_col
            )

            # Train
            trainer = ModelTrainer(
                random_state=self.config["model"]["random_state"],
                test_size=self.config["model"]["test_size"],
                cv_folds=self.config["model"]["cv_folds"],
            )
            X_train, X_test, y_train, y_test = trainer.split_data(X, y)
            groups_test = groups.loc[X_test.index].reset_index(drop=True)

            trainer.train_all_models(X_train, y_train, model_names)
            perf_df = trainer.evaluate_all_models(X_test, y_test)

            best_name = perf_df["auc_roc"].idxmax()
            best_model = trainer.models[best_name]
            y_pred, y_prob = trainer.get_predictions(best_model, X_test)

            # Fairness
            evaluator = FairnessEvaluator(
                y_test.values,
                y_pred,
                y_prob,
                groups_test,
                reference_group=ref_group,
            )
            evaluator.compute_all_group_metrics()
            evaluator.compute_fairness_metrics()

            n_boot = self.config.get("fairness", {}).get("bootstrap_iterations", 200)
            ci_analyzer = FairnessConfidenceIntervals(
                y_test.values, y_prob, groups_test, n_bootstrap=n_boot
            )
            metrics_ci = ci_analyzer.bootstrap_group_metrics()

            self.results[pct] = {
                "percentile": pct,
                "threshold": threshold,
                "prevalence": prevalence,
                "performance_df": perf_df,
                "best_model": best_name,
                "best_auc": perf_df.loc[best_name, "auc_roc"],
                "group_metrics": evaluator.get_summary_table(),
                "disparity_table": evaluator.get_disparity_table(),
                "fairness_criteria": evaluator.check_fairness_criteria(),
                "metrics_with_ci": metrics_ci,
            }

            logger.info(
                f"  Best: {best_name} AUC={perf_df.loc[best_name, 'auc_roc']:.4f}, "
                f"criteria: {evaluator.check_fairness_criteria()}"
            )

        return self.results

    # ------------------------------------------------------------------
    # Comparison tables
    # ------------------------------------------------------------------

    def compare_performance(self) -> pd.DataFrame:
        """Best model performance across thresholds."""
        rows = []
        for pct, res in sorted(self.results.items()):
            best = res["performance_df"].loc[res["best_model"]]
            rows.append(
                {
                    "percentile": pct,
                    "prevalence": f"{res['prevalence']:.1%}",
                    "best_model": res["best_model"],
                    "auc_roc": best["auc_roc"],
                    "accuracy": best["accuracy"],
                    "f1": best["f1"],
                    "recall": best["recall"],
                    "precision": best["precision"],
                }
            )
        return pd.DataFrame(rows)

    def compare_fairness(self) -> pd.DataFrame:
        """Group fairness metrics across thresholds."""
        frames = []
        for pct, res in sorted(self.results.items()):
            ci = res["metrics_with_ci"].copy()
            ci.insert(0, "percentile", pct)
            frames.append(ci)
        return pd.concat(frames, ignore_index=True)

    def compare_disparities(self) -> pd.DataFrame:
        """Disparity ratios across thresholds."""
        frames = []
        for pct, res in sorted(self.results.items()):
            dt = res["disparity_table"].copy()
            dt.insert(0, "percentile", pct)
            frames.append(dt)
        return pd.concat(frames, ignore_index=True)

    def compare_criteria(self) -> pd.DataFrame:
        """Fairness criteria pass/fail across thresholds."""
        rows = []
        for pct, res in sorted(self.results.items()):
            row = {"percentile": pct, "prevalence": f"{res['prevalence']:.1%}"}
            for criterion, passed in res["fairness_criteria"].items():
                row[criterion] = "PASS" if passed else "FAIL"
            rows.append(row)
        return pd.DataFrame(rows)

    def save_results(self, output_dir: str) -> List[str]:
        """Save all sensitivity analysis tables."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = []

        tables = {
            "sensitivity_performance.csv": self.compare_performance,
            "sensitivity_fairness.csv": self.compare_fairness,
            "sensitivity_disparities.csv": self.compare_disparities,
            "sensitivity_criteria.csv": self.compare_criteria,
        }

        for fname, method in tables.items():
            path = out / fname
            method().to_csv(path, index=False)
            saved.append(str(path))
            logger.info(f"Saved {path}")

        return saved


class OutcomeComparisonAnalyzer:
    """
    Compare model performance and fairness across different outcomes
    (e.g., reading vs math).
    """

    def __init__(self, config: dict):
        self.config = config
        self.results: Dict[str, Dict] = {}

    def run_outcome(
        self,
        df: pd.DataFrame,
        outcome_name: str,
        outcome_var: str,
        group_col: str = "race_ethnicity",
        model_names: Optional[List[str]] = None,
    ) -> Dict:
        """Run pipeline for a single outcome."""
        from .data_loader import get_variable_lists

        logger.info(f"=== Outcome: {outcome_name} ({outcome_var}) ===")

        vars_cfg = get_variable_lists(self.config)
        predictor_cols = vars_cfg["predictors"]
        pct = self.config["variables"]["at_risk_percentile"]
        ref_group = self.config["fairness"]["reference_groups"]["race"]

        if model_names is None:
            model_names = [
                name
                for name, cfg in self.config["model"]["algorithms"].items()
                if cfg.get("enabled", True)
            ]

        # Create at-risk indicator
        df_out = df.copy()
        threshold = df_out[outcome_var].quantile(pct / 100)
        outcome_col = f"{outcome_var}_at_risk"
        if outcome_col not in df_out.columns:
            df_out[outcome_col] = (df_out[outcome_var] < threshold).astype(int)

        X, y, groups = prepare_modeling_data(
            df_out, predictor_cols, outcome_col, group_col
        )

        trainer = ModelTrainer(
            random_state=self.config["model"]["random_state"],
            test_size=self.config["model"]["test_size"],
            cv_folds=self.config["model"]["cv_folds"],
        )
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        groups_test = groups.loc[X_test.index].reset_index(drop=True)

        trainer.train_all_models(X_train, y_train, model_names)
        perf_df = trainer.evaluate_all_models(X_test, y_test)

        best_name = perf_df["auc_roc"].idxmax()
        best_model = trainer.models[best_name]
        y_pred, y_prob = trainer.get_predictions(best_model, X_test)

        evaluator = FairnessEvaluator(
            y_test.values,
            y_pred,
            y_prob,
            groups_test,
            reference_group=ref_group,
        )
        evaluator.compute_all_group_metrics()
        evaluator.compute_fairness_metrics()

        n_boot = self.config.get("fairness", {}).get("bootstrap_iterations", 200)
        ci_analyzer = FairnessConfidenceIntervals(
            y_test.values, y_prob, groups_test, n_bootstrap=n_boot
        )

        result = {
            "outcome_name": outcome_name,
            "outcome_var": outcome_var,
            "performance_df": perf_df,
            "best_model": best_name,
            "best_auc": perf_df.loc[best_name, "auc_roc"],
            "group_metrics": evaluator.get_summary_table(),
            "disparity_table": evaluator.get_disparity_table(),
            "fairness_criteria": evaluator.check_fairness_criteria(),
            "metrics_with_ci": ci_analyzer.bootstrap_group_metrics(),
        }

        self.results[outcome_name] = result
        logger.info(
            f"  Best: {best_name} AUC={result['best_auc']:.4f}, "
            f"criteria: {result['fairness_criteria']}"
        )
        return result

    def run_all_outcomes(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Run pipeline for both reading and math."""
        outcomes = self.config["variables"]["outcomes"]
        for name, var in outcomes.items():
            self.run_outcome(df, name, var)
        return self.results

    def compare_performance(self) -> pd.DataFrame:
        """Compare best model performance across outcomes."""
        rows = []
        for name, res in self.results.items():
            best = res["performance_df"].loc[res["best_model"]]
            rows.append(
                {
                    "outcome": name,
                    "best_model": res["best_model"],
                    "auc_roc": best["auc_roc"],
                    "accuracy": best["accuracy"],
                    "f1": best["f1"],
                    "recall": best["recall"],
                }
            )
        return pd.DataFrame(rows)

    def compare_fairness(self) -> pd.DataFrame:
        """Compare fairness metrics across outcomes."""
        frames = []
        for name, res in self.results.items():
            ci = res["metrics_with_ci"].copy()
            ci.insert(0, "outcome", name)
            frames.append(ci)
        return pd.concat(frames, ignore_index=True)

    def compare_disparities(self) -> pd.DataFrame:
        """Compare disparity ratios across outcomes."""
        frames = []
        for name, res in self.results.items():
            dt = res["disparity_table"].copy()
            dt.insert(0, "outcome", name)
            frames.append(dt)
        return pd.concat(frames, ignore_index=True)

    def save_results(self, output_dir: str) -> List[str]:
        """Save comparison tables."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = []

        tables = {
            "outcome_performance_comparison.csv": self.compare_performance,
            "outcome_fairness_comparison.csv": self.compare_fairness,
            "outcome_disparity_comparison.csv": self.compare_disparities,
        }

        for fname, method in tables.items():
            path = out / fname
            method().to_csv(path, index=False)
            saved.append(str(path))
            logger.info(f"Saved {path}")

        return saved
