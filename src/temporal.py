"""
Temporal Generalization Module
==============================

Analyzes how model performance and fairness change when using
progressive feature sets from different developmental time windows,
all predicting the same 5th grade outcome.

Temporal scenarios (cumulative):
  k_only:        K Fall cognitive + demographics
  k_complete:    + K Spring cognitive
  k_through_1st: + 1st grade measures
  k_through_3rd: + 3rd grade measures (full model)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

from sklearn.model_selection import train_test_split

from .models import ModelTrainer
from .fairness import (
    FairnessEvaluator,
    FairnessConfidenceIntervals,
    CalibrationFairnessAnalyzer,
)

logger = logging.getLogger(__name__)


@dataclass
class TemporalScenario:
    """Definition of a temporal feature-set scenario."""
    name: str
    label: str
    cognitive_features: List[str]
    description: str = ""


@dataclass
class TemporalScenarioResult:
    """Results from running one temporal scenario."""
    scenario: TemporalScenario
    n_features: int
    performance_df: pd.DataFrame          # all models' test metrics
    best_model_name: str
    best_auc: float
    group_metrics: pd.DataFrame            # FairnessEvaluator summary
    disparity_table: pd.DataFrame          # FairnessEvaluator disparities
    fairness_criteria: Dict[str, bool]
    metrics_with_ci: pd.DataFrame          # bootstrap CI
    calibration_df: pd.DataFrame           # ECE/MCE by group
    trainer: Any = field(repr=False, default=None)
    y_pred: np.ndarray = field(repr=False, default=None)
    y_prob: np.ndarray = field(repr=False, default=None)


class TemporalGeneralizationAnalyzer:
    """
    Analyze how model performance and fairness change across
    progressive temporal feature sets.

    All scenarios use the same observations and train/test split
    so differences are solely attributable to the feature set.
    """

    def __init__(
        self,
        config: dict,
        random_state: int = 42,
        test_size: float = 0.30,
        cv_folds: int = 5,
    ):
        self.config = config
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds

        # Build scenarios from config
        self.scenarios = self._build_scenarios()
        self.demographics = self._get_demographics()

        # Populated by prepare_common_sample()
        self.df = None
        self.X_full = None
        self.y = None
        self.groups = None
        self.train_idx = None
        self.test_idx = None

        self.results: Dict[str, TemporalScenarioResult] = {}

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _build_scenarios(self) -> List[TemporalScenario]:
        """Build scenario list from config or use defaults."""
        temporal_cfg = self.config.get("temporal", {})
        scenario_defs = temporal_cfg.get("scenarios", None)

        if scenario_defs:
            return [
                TemporalScenario(
                    name=s["name"],
                    label=s["label"],
                    cognitive_features=s["features"],
                    description=s.get("description", ""),
                )
                for s in scenario_defs
            ]

        # Default scenario definitions
        return [
            TemporalScenario(
                name="k_only",
                label="K Fall Only",
                cognitive_features=["X1RTHETK", "X1MTHETK", "X1TCHAPP"],
                description="Earliest possible prediction using K Fall data",
            ),
            TemporalScenario(
                name="k_complete",
                label="K Fall + Spring",
                cognitive_features=[
                    "X1RTHETK", "X1MTHETK", "X1TCHAPP",
                    "X2RTHETK", "X2MTHETK", "X2TCHAPP",
                ],
                description="Full kindergarten year data",
            ),
            TemporalScenario(
                name="k_through_1st",
                label="K + 1st Grade",
                cognitive_features=[
                    "X1RTHETK", "X1MTHETK", "X1TCHAPP",
                    "X2RTHETK", "X2MTHETK", "X2TCHAPP",
                    "X4TCHAPP",
                ],
                description="Through 1st grade",
            ),
            TemporalScenario(
                name="k_through_3rd",
                label="K through 3rd",
                cognitive_features=[
                    "X1RTHETK", "X1MTHETK", "X1TCHAPP",
                    "X2RTHETK", "X2MTHETK", "X2TCHAPP",
                    "X4TCHAPP",
                    "X6DCCSSCR",
                ],
                description="Full model (matches main pipeline)",
            ),
        ]

    def _get_demographics(self) -> List[str]:
        """Get demographic feature columns."""
        temporal_cfg = self.config.get("temporal", {})
        return temporal_cfg.get(
            "demographics",
            ["X_CHSEX_R", "X_RACETH_R", "X1SESQ5", "X12LANGST"],
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_common_sample(
        self,
        df: pd.DataFrame,
        outcome_col: str = "X9RTHETA_at_risk",
        group_col: str = "race_ethnicity",
    ) -> int:
        """
        Filter to complete cases across ALL scenarios and create
        a single train/test split.

        Returns the number of complete-case observations.
        """
        # Collect every feature used across all scenarios + demographics
        all_features = set(self.demographics)
        for sc in self.scenarios:
            all_features.update(sc.cognitive_features)
        all_features = sorted(all_features)

        # Require outcome and group
        required = all_features + [outcome_col]
        if group_col in df.columns:
            required.append(group_col)

        available = [c for c in required if c in df.columns]
        missing = set(required) - set(available)
        if missing:
            logger.warning(f"Columns not in DataFrame, will drop: {missing}")

        subset = df[available].copy()
        valid_mask = subset.notna().all(axis=1)
        subset = subset[valid_mask].copy()

        logger.info(
            f"Temporal common sample: {len(subset):,} complete cases "
            f"(from {len(df):,} rows)"
        )

        self.df = subset
        self.y = subset[outcome_col].astype(int)
        self.groups = subset[group_col] if group_col in subset.columns else None

        # Build the full feature matrix (superset across all scenarios)
        feature_cols = sorted(
            set(all_features) & set(subset.columns) - {outcome_col, group_col}
        )
        self.X_full = subset[feature_cols]

        # Single stratified split (by group + outcome)
        stratify = self.groups.astype(str) + "_" + self.y.astype(str)
        idx = np.arange(len(self.y))
        self.train_idx, self.test_idx = train_test_split(
            idx,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        logger.info(
            f"  Train: {len(self.train_idx):,}  Test: {len(self.test_idx):,}"
        )
        return len(subset)

    # ------------------------------------------------------------------
    # Scenario execution
    # ------------------------------------------------------------------

    def _feature_cols_for_scenario(self, scenario: TemporalScenario) -> List[str]:
        """Return feature column list for a given scenario."""
        cols = list(scenario.cognitive_features) + list(self.demographics)
        # Keep only columns present in X_full
        return [c for c in cols if c in self.X_full.columns]

    def run_scenario(self, scenario: TemporalScenario) -> TemporalScenarioResult:
        """Train models, evaluate performance & fairness for one scenario."""
        logger.info(f"--- Scenario: {scenario.label} ({scenario.name}) ---")

        feature_cols = self._feature_cols_for_scenario(scenario)
        logger.info(f"  Features ({len(feature_cols)}): {feature_cols}")

        X = self.X_full[feature_cols]
        X_train = X.iloc[self.train_idx]
        X_test = X.iloc[self.test_idx]
        y_train = self.y.iloc[self.train_idx]
        y_test = self.y.iloc[self.test_idx]
        groups_test = self.groups.iloc[self.test_idx] if self.groups is not None else None

        # Fresh trainer (fresh scaler)
        trainer = ModelTrainer(
            random_state=self.random_state,
            test_size=self.test_size,
            cv_folds=self.cv_folds,
        )

        # Determine which model algorithms to train
        model_names = [
            name
            for name, cfg in self.config["model"]["algorithms"].items()
            if cfg.get("enabled", True)
        ]

        # Train
        trainer.train_all_models(X_train, y_train, model_names)

        # Evaluate
        perf_df = trainer.evaluate_all_models(X_test, y_test)
        best_name = perf_df["auc_roc"].idxmax()
        best_model = trainer.models[best_name]
        best_auc = perf_df.loc[best_name, "auc_roc"]

        # Predictions from best model
        y_pred, y_prob = trainer.get_predictions(best_model, X_test)

        # -- Fairness evaluation --
        ref_group = self.config["fairness"]["reference_groups"]["race"]

        evaluator = FairnessEvaluator(
            y_true=y_test.values,
            y_pred=y_pred,
            y_prob=y_prob,
            groups=groups_test,
            reference_group=ref_group,
        )
        evaluator.compute_all_group_metrics()
        evaluator.compute_fairness_metrics()

        group_metrics = evaluator.get_summary_table()
        disparity_table = evaluator.get_disparity_table()
        fairness_criteria = evaluator.check_fairness_criteria()

        # Bootstrap CI
        n_boot = self.config.get("fairness", {}).get("bootstrap_iterations", 200)
        ci_analyzer = FairnessConfidenceIntervals(
            y_test.values, y_prob, groups_test, n_bootstrap=n_boot
        )
        metrics_with_ci = ci_analyzer.bootstrap_group_metrics()

        # Calibration
        calib_analyzer = CalibrationFairnessAnalyzer(
            y_test.values, y_prob, groups_test
        )
        calibration_df = calib_analyzer.analyze_calibration_by_group()

        result = TemporalScenarioResult(
            scenario=scenario,
            n_features=len(feature_cols),
            performance_df=perf_df,
            best_model_name=best_name,
            best_auc=best_auc,
            group_metrics=group_metrics,
            disparity_table=disparity_table,
            fairness_criteria=fairness_criteria,
            metrics_with_ci=metrics_with_ci,
            calibration_df=calibration_df,
            trainer=trainer,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        self.results[scenario.name] = result
        logger.info(
            f"  Best model: {best_name} (AUC={best_auc:.4f}), "
            f"criteria: {fairness_criteria}"
        )
        return result

    def run_all_scenarios(self) -> Dict[str, TemporalScenarioResult]:
        """Run all temporal scenarios sequentially."""
        if self.X_full is None:
            raise RuntimeError("Call prepare_common_sample() first")

        for scenario in self.scenarios:
            self.run_scenario(scenario)

        return self.results

    # ------------------------------------------------------------------
    # Comparison tables
    # ------------------------------------------------------------------

    def compare_performance(self) -> pd.DataFrame:
        """All models x all scenarios performance table."""
        rows = []
        for sc_name, res in self.results.items():
            for model_name in res.performance_df.index:
                row = {"scenario": sc_name, "scenario_label": res.scenario.label}
                row["model"] = model_name
                row["n_features"] = res.n_features
                for col in res.performance_df.columns:
                    row[col] = res.performance_df.loc[model_name, col]
                rows.append(row)
        return pd.DataFrame(rows)

    def get_best_model_comparison(self) -> pd.DataFrame:
        """Best model per scenario summary."""
        rows = []
        for sc_name, res in self.results.items():
            best = res.performance_df.loc[res.best_model_name]
            rows.append({
                "scenario": sc_name,
                "scenario_label": res.scenario.label,
                "n_features": res.n_features,
                "best_model": res.best_model_name,
                "auc_roc": best["auc_roc"],
                "accuracy": best["accuracy"],
                "f1": best["f1"],
                "precision": best["precision"],
                "recall": best["recall"],
                "brier_score": best["brier_score"],
            })
        return pd.DataFrame(rows)

    def compare_fairness(self) -> pd.DataFrame:
        """Group fairness metrics with CI across all scenarios."""
        rows = []
        for sc_name, res in self.results.items():
            ci_df = res.metrics_with_ci.copy()
            ci_df.insert(0, "scenario", sc_name)
            ci_df.insert(1, "scenario_label", res.scenario.label)
            rows.append(ci_df)
        return pd.concat(rows, ignore_index=True)

    def compare_disparities(self) -> pd.DataFrame:
        """Disparity ratios across all scenarios."""
        rows = []
        for sc_name, res in self.results.items():
            dt = res.disparity_table.copy()
            dt.insert(0, "scenario", sc_name)
            dt.insert(1, "scenario_label", res.scenario.label)
            rows.append(dt)
        return pd.concat(rows, ignore_index=True)

    def compare_calibration(self) -> pd.DataFrame:
        """Calibration metrics across all scenarios."""
        rows = []
        for sc_name, res in self.results.items():
            cal = res.calibration_df.copy()
            cal.insert(0, "scenario", sc_name)
            cal.insert(1, "scenario_label", res.scenario.label)
            rows.append(cal)
        return pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_results(self, output_dir: str) -> List[str]:
        """Write all comparison tables and per-scenario tables to CSV."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = []

        # Cross-scenario comparison tables
        tables = {
            "temporal_performance_comparison.csv": self.compare_performance,
            "temporal_best_model_summary.csv": self.get_best_model_comparison,
            "temporal_fairness_comparison.csv": self.compare_fairness,
            "temporal_disparity_comparison.csv": self.compare_disparities,
            "temporal_calibration_comparison.csv": self.compare_calibration,
        }

        for filename, method in tables.items():
            path = out / filename
            method().to_csv(path, index=False)
            saved.append(str(path))
            logger.info(f"Saved {path}")

        # Per-scenario group metrics
        for sc_name, res in self.results.items():
            path = out / f"temporal_scenario_{sc_name}_group_metrics.csv"
            res.group_metrics.to_csv(path, index=False)
            saved.append(str(path))

        logger.info(f"Saved {len(saved)} temporal analysis tables")
        return saved
