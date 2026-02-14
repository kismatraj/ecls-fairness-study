"""
Explainability Module (2025 State-of-the-Art)
==============================================

Model interpretability and explainability methods including:
- SHAP values and interaction analysis
- Permutation importance with confidence intervals
- Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE)
- Fairness-aware SHAP analysis by protected groups
- Counterfactual explanations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
import warnings

from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.utils import resample

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("shap not installed. SHAP explanations unavailable.")

try:
    import lime
    import lime.lime_tabular

    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    warnings.warn("lime not installed. LIME explanations unavailable.")

try:
    import dice_ml

    HAS_DICE = True
except ImportError:
    HAS_DICE = False
    warnings.warn("dice-ml not installed. Counterfactual explanations unavailable.")

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based model explanations with fairness-aware analysis.

    Provides global and local explanations using SHAP (SHapley Additive
    exPlanations) values, with additional methods for fairness analysis
    across protected groups.
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        explainer_type: str = "auto",
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained sklearn-compatible model
            X_train: Training data for background distribution
            feature_names: Feature names (defaults to X_train columns)
            explainer_type: Type of explainer ('auto', 'tree', 'kernel', 'linear')
        """
        if not HAS_SHAP:
            raise ImportError("shap package required. Install with: pip install shap")

        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or list(X_train.columns)

        # Auto-select explainer type based on model
        self.explainer = self._create_explainer(explainer_type)
        self.shap_values = None
        self.expected_value = None

    def _create_explainer(self, explainer_type: str) -> shap.Explainer:
        """Create appropriate SHAP explainer based on model type."""
        model_name = type(self.model).__name__.lower()

        if explainer_type == "auto":
            # Auto-detect best explainer
            if any(
                tree in model_name
                for tree in ["forest", "xgb", "lgbm", "catboost", "gradient"]
            ):
                explainer_type = "tree"
            elif any(
                linear in model_name
                for linear in ["logistic", "linear", "sgd", "elastic"]
            ):
                explainer_type = "linear"
            else:
                explainer_type = "kernel"

        logger.info(f"Using {explainer_type} SHAP explainer for {model_name}")

        if explainer_type == "tree":
            return shap.TreeExplainer(self.model)
        elif explainer_type == "linear":
            return shap.LinearExplainer(self.model, self.X_train)
        else:
            # Kernel explainer (model-agnostic, slower)
            # Use k-means to summarize background data
            background = shap.kmeans(self.X_train, 50)
            return shap.KernelExplainer(self.model.predict_proba, background)

    def compute_shap_values(
        self, X: pd.DataFrame, check_additivity: bool = False
    ) -> np.ndarray:
        """
        Compute SHAP values for given data.

        Args:
            X: Data to explain
            check_additivity: Whether to check SHAP additivity property

        Returns:
            SHAP values array (n_samples, n_features)
        """
        logger.info(f"Computing SHAP values for {len(X)} samples...")

        try:
            shap_values = self.explainer.shap_values(
                X, check_additivity=check_additivity
            )
        except TypeError:
            # Newer SHAP versions removed check_additivity for some explainers
            shap_values = self.explainer.shap_values(X)

        # Handle binary classification (take positive class)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]

        self.shap_values = shap_values
        self.expected_value = self.explainer.expected_value
        if isinstance(self.expected_value, np.ndarray):
            self.expected_value = (
                self.expected_value[1]
                if len(self.expected_value) == 2
                else self.expected_value[0]
            )

        return shap_values

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance (mean |SHAP|)
        """
        if self.shap_values is None:
            raise ValueError("Call compute_shap_values() first")

        importance = np.abs(self.shap_values).mean(axis=0)

        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": importance,
                "std_shap": np.std(self.shap_values, axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)

        return df.head(top_n)

    def get_shap_interaction_values(
        self, X: pd.DataFrame, max_samples: int = 500
    ) -> np.ndarray:
        """
        Compute SHAP interaction values (pairwise feature interactions).

        Args:
            X: Data to explain
            max_samples: Maximum samples (interactions are O(n*f^2))

        Returns:
            Interaction values array (n_samples, n_features, n_features)
        """
        if not hasattr(self.explainer, "shap_interaction_values"):
            logger.warning("Explainer does not support interaction values")
            return None

        if len(X) > max_samples:
            logger.info(
                f"Sampling {max_samples} from {len(X)} for interaction computation"
            )
            X = X.sample(n=max_samples, random_state=42)

        logger.info("Computing SHAP interaction values (this may take time)...")
        return self.explainer.shap_interaction_values(X)

    def get_top_interactions(
        self, interaction_values: np.ndarray, top_n: int = 10
    ) -> pd.DataFrame:
        """
        Extract top feature interactions from interaction values.

        Args:
            interaction_values: SHAP interaction values
            top_n: Number of top interactions

        Returns:
            DataFrame with top feature pairs and interaction strength
        """
        # Average absolute interaction across samples
        mean_interactions = np.abs(interaction_values).mean(axis=0)

        # Zero out diagonal (self-interactions)
        np.fill_diagonal(mean_interactions, 0)

        # Get top interactions
        interactions = []
        for i in range(len(self.feature_names)):
            for j in range(i + 1, len(self.feature_names)):
                interactions.append(
                    {
                        "feature_1": self.feature_names[i],
                        "feature_2": self.feature_names[j],
                        "interaction_strength": mean_interactions[i, j],
                    }
                )

        df = pd.DataFrame(interactions)
        return df.sort_values("interaction_strength", ascending=False).head(top_n)

    def fairness_aware_shap(
        self, X: pd.DataFrame, groups: pd.Series, protected_attribute: str = "race"
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute SHAP-based feature importance stratified by protected groups.

        This reveals whether the model relies on different features for
        different demographic groups, which can indicate bias.

        Args:
            X: Data to explain
            groups: Group labels aligned with X
            protected_attribute: Name of protected attribute

        Returns:
            Dictionary with feature importance per group
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        groups = pd.Series(groups).reset_index(drop=True)
        unique_groups = groups.unique()

        results = {}

        for group in unique_groups:
            mask = groups == group
            group_shap = self.shap_values[mask]

            importance = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "mean_abs_shap": np.abs(group_shap).mean(axis=0),
                    "mean_shap": group_shap.mean(axis=0),  # Direction of effect
                    "std_shap": group_shap.std(axis=0),
                }
            ).sort_values("mean_abs_shap", ascending=False)

            results[group] = importance

        # Compute differential importance (which features differ most between groups)
        if len(unique_groups) >= 2:
            reference = list(unique_groups)[0]
            differential = pd.DataFrame({"feature": self.feature_names})

            for group in unique_groups:
                if group != reference:
                    ref_imp = results[reference].set_index("feature")["mean_abs_shap"]
                    grp_imp = results[group].set_index("feature")["mean_abs_shap"]
                    differential[f"diff_vs_{group}"] = np.abs(ref_imp - grp_imp)

            results["differential_importance"] = differential

        logger.info(f"Computed fairness-aware SHAP for {len(unique_groups)} groups")
        return results

    def plot_summary(
        self, X: pd.DataFrame, max_display: int = 10, output_path: Optional[str] = None
    ) -> None:
        """Generate SHAP summary plot (Nature/Science style)."""
        import matplotlib.pyplot as plt
        from src.visualization import set_publication_style, _map_feature_names, _save_figure, FIG_SINGLE

        if self.shap_values is None:
            self.compute_shap_values(X)

        set_publication_style()
        mapped_names = _map_feature_names(self.feature_names)

        plt.figure(figsize=(FIG_SINGLE, 4.0))
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=mapped_names,
            max_display=max_display,
            show=False,
            plot_size=None,
        )
        ax = plt.gca()
        ax.set_xlabel("SHAP value")
        ax.set_title("")

        if output_path:
            _save_figure(plt.gcf(), output_path)
        plt.close()

    def plot_dependence(
        self,
        feature: str,
        X: pd.DataFrame,
        interaction_feature: Optional[str] = "auto",
        output_path: Optional[str] = None,
    ) -> None:
        """Generate SHAP dependence plot for a feature."""
        import matplotlib.pyplot as plt

        if self.shap_values is None:
            self.compute_shap_values(X)

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature,
            self.shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False,
        )

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"SHAP dependence plot saved to {output_path}")
        plt.close()


class PermutationImportanceAnalyzer:
    """
    Permutation importance with bootstrap confidence intervals.

    Model-agnostic feature importance that measures the decrease in
    model performance when a feature's values are randomly shuffled.
    """

    def __init__(
        self, model: Any, X: pd.DataFrame, y: np.ndarray, scoring: str = "roc_auc"
    ):
        self.model = model
        self.X = X
        self.y = y
        self.scoring = scoring
        self.feature_names = list(X.columns)
        self.importance_results = None

    def compute_importance(self, n_repeats: int = 30, n_jobs: int = -1) -> pd.DataFrame:
        """
        Compute permutation importance with confidence intervals.

        Args:
            n_repeats: Number of permutation repeats
            n_jobs: Parallel jobs (-1 = all cores)

        Returns:
            DataFrame with importance mean, std, and CI
        """
        logger.info(f"Computing permutation importance with {n_repeats} repeats...")

        result = permutation_importance(
            self.model,
            self.X,
            self.y,
            n_repeats=n_repeats,
            n_jobs=n_jobs,
            scoring=self.scoring,
            random_state=42,
        )

        self.importance_results = result

        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
                "ci_lower": result.importances_mean - 1.96 * result.importances_std,
                "ci_upper": result.importances_mean + 1.96 * result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)

        return df

    def bootstrap_importance(
        self, n_bootstrap: int = 100, sample_fraction: float = 0.8
    ) -> pd.DataFrame:
        """
        Bootstrap confidence intervals for permutation importance.

        More robust than single-run CI, accounts for data variability.

        Args:
            n_bootstrap: Number of bootstrap iterations
            sample_fraction: Fraction of data per bootstrap sample

        Returns:
            DataFrame with bootstrap CI
        """
        logger.info(f"Computing bootstrap importance ({n_bootstrap} iterations)...")

        n_samples = int(len(self.X) * sample_fraction)
        all_importances = []

        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = resample(
                range(len(self.X)), n_samples=n_samples, random_state=42 + i
            )
            X_boot = self.X.iloc[indices]
            y_boot = self.y[indices]

            # Compute importance on bootstrap sample
            result = permutation_importance(
                self.model,
                X_boot,
                y_boot,
                n_repeats=10,
                n_jobs=-1,
                scoring=self.scoring,
                random_state=42 + i,
            )
            all_importances.append(result.importances_mean)

        importances = np.array(all_importances)

        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance_mean": importances.mean(axis=0),
                "importance_std": importances.std(axis=0),
                "ci_2.5": np.percentile(importances, 2.5, axis=0),
                "ci_97.5": np.percentile(importances, 97.5, axis=0),
            }
        ).sort_values("importance_mean", ascending=False)

        return df


class PartialDependenceAnalyzer:
    """
    Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE).

    PDPs show the marginal effect of features on model predictions.
    ICE plots show individual prediction trajectories.
    """

    def __init__(self, model: Any, X: pd.DataFrame):
        self.model = model
        self.X = X
        self.feature_names = list(X.columns)

    def plot_pdp(
        self,
        features: List[Union[str, Tuple[str, str]]],
        output_path: Optional[str] = None,
        kind: str = "both",  # 'average', 'individual', or 'both'
        subsample: int = 100,
    ) -> None:
        """
        Generate PDP/ICE plots for specified features.

        Args:
            features: List of feature names or pairs for 2D plots
            output_path: Path to save figure
            kind: 'average' for PDP, 'individual' for ICE, 'both'
            subsample: Number of ICE lines to show
        """
        import matplotlib.pyplot as plt

        # Convert feature names to indices
        feature_indices = []
        for f in features:
            if isinstance(f, tuple):
                feature_indices.append(
                    tuple(self.feature_names.index(name) for name in f)
                )
            else:
                feature_indices.append(self.feature_names.index(f))

        fig, ax = plt.subplots(figsize=(12, 4 * len(features)))

        PartialDependenceDisplay.from_estimator(
            self.model,
            self.X,
            feature_indices,
            feature_names=self.feature_names,
            kind=kind,
            subsample=subsample,
            random_state=42,
            ax=ax,
        )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"PDP plot saved to {output_path}")
        plt.close()

    def compute_pdp_values(
        self, feature: str, grid_resolution: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute PDP values for a single feature.

        Args:
            feature: Feature name
            grid_resolution: Number of grid points

        Returns:
            Tuple of (grid_values, pdp_values)
        """
        from sklearn.inspection import partial_dependence

        feature_idx = self.feature_names.index(feature)

        result = partial_dependence(
            self.model,
            self.X,
            [feature_idx],
            kind="average",
            grid_resolution=grid_resolution,
        )

        return result["grid_values"][0], result["average"][0]


class CounterfactualExplainer:
    """
    Counterfactual explanations using DiCE (Diverse Counterfactual Explanations).

    Generates minimal changes to input features that would flip the prediction,
    revealing what changes would be needed for different outcomes.
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        continuous_features: List[str],
        outcome_name: str = "outcome",
    ):
        if not HAS_DICE:
            raise ImportError("dice-ml required. Install with: pip install dice-ml")

        self.model = model
        self.X_train = X_train
        self.continuous_features = continuous_features
        self.outcome_name = outcome_name

        # Create DiCE data object
        self.dice_data = dice_ml.Data(
            dataframe=X_train,
            continuous_features=continuous_features,
            outcome_name=outcome_name,
        )

        # Create DiCE model object
        self.dice_model = dice_ml.Model(model=model, backend="sklearn")

        # Create explainer
        self.explainer = dice_ml.Dice(self.dice_data, self.dice_model, method="random")

    def generate_counterfactuals(
        self,
        query_instance: pd.DataFrame,
        total_CFs: int = 5,
        desired_class: int = 1,
        features_to_vary: Optional[List[str]] = None,
        permitted_range: Optional[Dict[str, List]] = None,
    ):
        """
        Generate counterfactual explanations for a query instance.

        Args:
            query_instance: Single instance to explain
            total_CFs: Number of counterfactuals to generate
            desired_class: Target class for counterfactuals
            features_to_vary: Features allowed to change (None = all)
            permitted_range: Allowed range for each feature

        Returns:
            DiCE counterfactual explanations object
        """
        cf = self.explainer.generate_counterfactuals(
            query_instance,
            total_CFs=total_CFs,
            desired_class=desired_class,
            features_to_vary=features_to_vary or "all",
            permitted_range=permitted_range,
        )

        return cf

    def get_counterfactual_summary(
        self, cf_explanations, query_instance: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Summarize counterfactual changes.

        Args:
            cf_explanations: DiCE counterfactual output
            query_instance: Original instance

        Returns:
            DataFrame showing feature changes
        """
        cf_df = cf_explanations.cf_examples_list[0].final_cfs_df

        changes = []
        for idx, cf_row in cf_df.iterrows():
            row_changes = {}
            for col in query_instance.columns:
                original = query_instance[col].values[0]
                cf_val = cf_row[col]
                if original != cf_val:
                    row_changes[col] = {"original": original, "counterfactual": cf_val}
            changes.append(row_changes)

        return changes


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations).

    Explains individual predictions by approximating the model locally
    with an interpretable model.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        mode: str = "classification",
        class_names: Optional[List[str]] = None,
    ):
        if not HAS_LIME:
            raise ImportError("lime required. Install with: pip install lime")

        self.X_train = X_train
        self.feature_names = list(X_train.columns)
        self.class_names = class_names or ["Not At-Risk", "At-Risk"]

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=mode,
            random_state=42,
        )

    def explain_instance(
        self,
        model: Any,
        instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000,
    ) -> lime.explanation.Explanation:
        """
        Generate LIME explanation for a single instance.

        Args:
            model: Trained model with predict_proba
            instance: Single instance to explain (1D array)
            num_features: Number of features in explanation
            num_samples: Number of samples for local approximation

        Returns:
            LIME explanation object
        """
        exp = self.explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
        )
        return exp

    def get_explanation_df(
        self, explanation: lime.explanation.Explanation
    ) -> pd.DataFrame:
        """Convert LIME explanation to DataFrame."""
        exp_list = explanation.as_list()
        return pd.DataFrame(exp_list, columns=["feature_condition", "weight"])


def compare_explanations(
    shap_importance: pd.DataFrame, perm_importance: pd.DataFrame, top_n: int = 15
) -> pd.DataFrame:
    """
    Compare feature importance across different explanation methods.

    Args:
        shap_importance: SHAP-based importance DataFrame
        perm_importance: Permutation importance DataFrame
        top_n: Number of top features to compare

    Returns:
        Combined comparison DataFrame
    """
    # Normalize importances to 0-1 scale
    shap_imp = shap_importance.head(top_n).copy()
    shap_imp["shap_normalized"] = (
        shap_imp["mean_abs_shap"] / shap_imp["mean_abs_shap"].max()
    )

    perm_imp = perm_importance.copy()
    perm_imp["perm_normalized"] = (
        perm_imp["importance_mean"] / perm_imp["importance_mean"].max()
    )
    perm_imp["perm_normalized"] = perm_imp["perm_normalized"].clip(
        lower=0
    )  # Handle negative

    # Merge
    comparison = (
        shap_imp[["feature", "shap_normalized"]]
        .merge(perm_imp[["feature", "perm_normalized"]], on="feature", how="outer")
        .fillna(0)
    )

    # Compute agreement score
    comparison["agreement"] = 1 - np.abs(
        comparison["shap_normalized"] - comparison["perm_normalized"]
    )

    # Add ranks
    comparison["shap_rank"] = comparison["shap_normalized"].rank(ascending=False)
    comparison["perm_rank"] = comparison["perm_normalized"].rank(ascending=False)
    comparison["rank_diff"] = np.abs(comparison["shap_rank"] - comparison["perm_rank"])

    return comparison.sort_values("shap_normalized", ascending=False)


def generate_explainability_report(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    groups: pd.Series,
    output_dir: str,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive explainability report with all methods.

    Args:
        model: Trained model
        X_train: Training data
        X_test: Test data
        y_test: Test labels
        groups: Group labels for fairness-aware analysis
        output_dir: Directory to save outputs
        feature_names: Feature names

    Returns:
        Dictionary with all explainability results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. SHAP Analysis
    if HAS_SHAP:
        logger.info("Running SHAP analysis...")
        shap_explainer = SHAPExplainer(model, X_train, feature_names)
        shap_explainer.compute_shap_values(X_test)

        results["shap_importance"] = shap_explainer.get_feature_importance(top_n=20)
        results["shap_importance"].to_csv(
            output_dir / "shap_importance.csv", index=False
        )

        # Fairness-aware SHAP
        results["fairness_shap"] = shap_explainer.fairness_aware_shap(X_test, groups)
        for group, df in results["fairness_shap"].items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(output_dir / f"shap_importance_{group}.csv", index=False)

        # SHAP plots
        shap_explainer.plot_summary(
            X_test, output_path=str(output_dir / "shap_summary.png")
        )

    # 2. Permutation Importance
    logger.info("Running permutation importance...")
    perm_analyzer = PermutationImportanceAnalyzer(model, X_test, y_test)
    results["perm_importance"] = perm_analyzer.compute_importance()
    results["perm_importance"].to_csv(
        output_dir / "permutation_importance.csv", index=False
    )

    # Bootstrap CI
    results["perm_bootstrap"] = perm_analyzer.bootstrap_importance(n_bootstrap=50)
    results["perm_bootstrap"].to_csv(
        output_dir / "permutation_importance_bootstrap.csv", index=False
    )

    # 3. Compare methods
    if HAS_SHAP:
        results["explanation_comparison"] = compare_explanations(
            results["shap_importance"], results["perm_importance"]
        )
        results["explanation_comparison"].to_csv(
            output_dir / "explanation_comparison.csv", index=False
        )

    logger.info(f"Explainability report saved to {output_dir}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Explainability module ready")
    print(f"  SHAP available: {HAS_SHAP}")
    print(f"  LIME available: {HAS_LIME}")
    print(f"  DiCE available: {HAS_DICE}")
