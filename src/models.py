"""
Models Module
=============

Machine learning model training, tuning, and evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
from datetime import datetime

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except (ImportError, OSError):
    HAS_XGBOOST = False

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except (ImportError, OSError):
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier as _CatBoostClassifier
    from sklearn.base import BaseEstimator, ClassifierMixin

    class CatBoostClassifierWrapper(BaseEstimator, ClassifierMixin):
        """Wrapper to make CatBoost compatible with sklearn >= 1.7 GridSearchCV."""

        _estimator_type = "classifier"

        def __init__(
            self,
            random_state=42,
            verbose=False,
            allow_writing_files=False,
            iterations=100,
            depth=6,
            learning_rate=0.1,
            **kwargs,
        ):
            self.random_state = random_state
            self.verbose = verbose
            self.allow_writing_files = allow_writing_files
            self.iterations = iterations
            self.depth = depth
            self.learning_rate = learning_rate
            self._extra_kwargs = kwargs

        def __sklearn_tags__(self):
            from sklearn.utils._tags import ClassifierTags

            tags = super().__sklearn_tags__()
            tags.estimator_type = "classifier"
            tags.classifier_tags = ClassifierTags()
            return tags

        def _get_cb(self):
            return _CatBoostClassifier(
                random_seed=self.random_state,
                verbose=self.verbose,
                allow_writing_files=self.allow_writing_files,
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                **self._extra_kwargs,
            )

        def fit(self, X, y, **kwargs):
            self._model = self._get_cb()
            self._model.fit(X, y, **kwargs)
            self.classes_ = np.array([0, 1])
            return self

        def predict(self, X):
            return self._model.predict(X).flatten().astype(int)

        def predict_proba(self, X):
            return self._model.predict_proba(X)

    HAS_CATBOOST = True
except (ImportError, OSError):
    HAS_CATBOOST = False

try:
    from tabpfn import TabPFNClassifier

    HAS_TABPFN = True
except (ImportError, OSError):
    HAS_TABPFN = False

try:
    from sklearn.ensemble import HistGradientBoostingClassifier

    HAS_HISTGB = True
except (ImportError, OSError):
    HAS_HISTGB = False

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Train and evaluate ML models for binary classification.

    Attributes:
        random_state: Random seed for reproducibility
        test_size: Fraction of data for test set
        cv_folds: Number of cross-validation folds
    """

    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.3,
        cv_folds: int = 5,
        sample_weights: Optional[np.ndarray] = None,
    ):
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.sample_weights = sample_weights

        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()

    def get_model_configs(self) -> Dict[str, Dict]:
        """Get default model configurations."""
        configs = {
            "logistic_regression": {
                "model": LogisticRegression(
                    random_state=self.random_state, max_iter=1000, l1_ratio=0
                ),
                "params": {"C": [0.01, 0.1, 1.0, 10.0]},
            },
            "elastic_net": {
                "model": SGDClassifier(
                    loss="log_loss",
                    penalty="elasticnet",
                    random_state=self.random_state,
                    max_iter=1000,
                ),
                "params": {"alpha": [0.0001, 0.001, 0.01], "l1_ratio": [0.2, 0.5, 0.8]},
            },
            "random_forest": {
                "model": RandomForestClassifier(
                    random_state=self.random_state, n_jobs=-1
                ),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, 15],
                    "min_samples_leaf": [5, 10],
                },
            },
        }

        if HAS_XGBOOST:
            configs["xgboost"] = {
                "model": xgb.XGBClassifier(
                    random_state=self.random_state, eval_metric="logloss"
                ),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1],
                },
            }

        # 2025 State-of-the-Art Models

        if HAS_LIGHTGBM:
            configs["lightgbm"] = {
                "model": lgb.LGBMClassifier(
                    random_state=self.random_state, verbose=-1, force_col_wise=True
                ),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, -1],
                    "learning_rate": [0.05, 0.1],
                    "num_leaves": [31],
                },
            }

        if HAS_CATBOOST:
            configs["catboost"] = {
                "model": CatBoostClassifierWrapper(
                    random_state=self.random_state,
                    verbose=False,
                    allow_writing_files=False,
                ),
                "params": {
                    "iterations": [100, 200],
                    "depth": [4, 6],
                    "learning_rate": [0.05, 0.1],
                },
            }

        if HAS_HISTGB:
            configs["hist_gradient_boosting"] = {
                "model": HistGradientBoostingClassifier(
                    random_state=self.random_state,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
                "params": {
                    "max_iter": [100, 200],
                    "max_depth": [5, 7],
                    "learning_rate": [0.05, 0.1],
                },
            }

        if HAS_TABPFN:
            # TabPFN - Transformer for Tabular Data (2024-2025 breakthrough)
            # Pre-trained on synthetic data, requires no hyperparameter tuning
            # Best for small-medium datasets (<10k samples, <100 features)
            configs["tabpfn"] = {
                "model": TabPFNClassifier(
                    device="cpu", n_estimators=32, random_state=self.random_state
                ),
                "params": {},  # TabPFN requires no tuning
            }

        return configs

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        stratify_cols: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Args:
            X: Feature matrix
            y: Target variable
            stratify_cols: Columns to stratify by (combined)

        Returns:
            X_train, X_test, y_train, y_test
        """
        if stratify_cols is not None:
            # Create combined stratification variable
            stratify = stratify_cols.astype(str).agg("_".join, axis=1)
        else:
            stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        logger.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

        return X_train, X_test, y_train, y_test

    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        scale_features: bool = True,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Dict]:
        """
        Train a single model with hyperparameter tuning.

        Args:
            model_name: Name of model to train
            X_train: Training features
            y_train: Training target
            param_grid: Hyperparameter grid (None = use defaults)
            scale_features: Whether to standardize features
            sample_weight: Optional sample weights for training

        Returns:
            Tuple of (trained model, CV results)
        """
        configs = self.get_model_configs()

        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}")

        config = configs[model_name]
        model = config["model"]
        params = param_grid or config["params"]

        # Scale features
        if scale_features:
            X_scaled = self.scaler.fit_transform(X_train)
        else:
            X_scaled = X_train.values

        # Grid search with CV
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        logger.info(f"Training {model_name} with {self.cv_folds}-fold CV...")

        # Use sample weights if provided
        sw = sample_weight if sample_weight is not None else self.sample_weights
        fit_params = {}
        if sw is not None:
            # Align weights with training indices
            if len(sw) == len(X_train):
                fit_params["sample_weight"] = sw
                logger.info(f"  Using sample weights (sum={sw.sum():.0f})")

        grid_search = GridSearchCV(
            model, params, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0
        )

        grid_search.fit(X_scaled, y_train, **fit_params)

        best_model = grid_search.best_estimator_

        cv_results = {
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
            "cv_scores": grid_search.cv_results_["mean_test_score"],
        }

        logger.info(f"  Best AUC: {cv_results['best_cv_score']:.4f}")
        logger.info(f"  Best params: {cv_results['best_params']}")

        self.models[model_name] = best_model

        return best_model, cv_results

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_names: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict]:
        """
        Train all specified models.

        Args:
            X_train: Training features
            y_train: Training target
            model_names: List of models to train (None = all)
            sample_weight: Optional sample weights for training

        Returns:
            Dictionary of results for each model
        """
        configs = self.get_model_configs()

        if model_names is None:
            model_names = list(configs.keys())

        all_results = {}

        for name in model_names:
            try:
                model, results = self.train_model(
                    name, X_train, y_train, sample_weight=sample_weight
                )
                all_results[name] = results
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                continue

        return all_results

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        scale_features: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            scale_features: Whether to scale features

        Returns:
            Dictionary of performance metrics
        """
        if scale_features:
            X_scaled = self.scaler.transform(X_test)
        else:
            X_scaled = X_test.values

        # Predictions
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        # Metrics
        metrics = {
            "auc_roc": roc_auc_score(y_test, y_prob),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "brier_score": brier_score_loss(y_test, y_prob),
        }

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

        return metrics

    def evaluate_all_models(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Evaluate all trained models.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            DataFrame with metrics for each model
        """
        results = []

        for name, model in self.models.items():
            metrics = self.evaluate_model(model, X_test, y_test)
            metrics["model"] = name
            results.append(metrics)

        df = pd.DataFrame(results)
        df = df.set_index("model")

        # Identify best model
        best_idx = df["auc_roc"].idxmax()
        self.best_model = self.models[best_idx]
        logger.info(f"Best model: {best_idx} (AUC={df.loc[best_idx, 'auc_roc']:.4f})")

        return df

    def get_predictions(
        self, model: Any, X: pd.DataFrame, scale_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions from a model.

        Args:
            model: Trained model
            X: Features
            scale_features: Whether to scale features

        Returns:
            Tuple of (predicted labels, predicted probabilities)
        """
        if scale_features:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        return y_pred, y_prob

    def save_model(
        self, model: Any, filepath: str, include_scaler: bool = True
    ) -> None:
        """Save model to disk."""
        save_dict = {"model": model}
        if include_scaler:
            save_dict["scaler"] = self.scaler

        joblib.dump(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> Any:
        """Load model from disk."""
        save_dict = joblib.load(filepath)
        self.scaler = save_dict.get("scaler", self.scaler)
        return save_dict["model"]


def get_feature_importance(
    model: Any, feature_names: List[str], top_n: int = 20
) -> pd.DataFrame:
    """
    Extract feature importance from model.

    Args:
        model: Trained model
        feature_names: Names of features
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, "feature_importances_"):
        # Tree-based models
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Linear models
        importance = np.abs(model.coef_).flatten()
    else:
        logger.warning("Model does not have feature importance attribute")
        return pd.DataFrame()

    df = pd.DataFrame({"feature": feature_names, "importance": importance})

    df = df.sort_values("importance", ascending=False)

    return df.head(top_n)


def create_model_card(
    model_name: str,
    metrics: Dict[str, float],
    training_info: Dict[str, Any],
    output_path: Optional[str] = None,
) -> Dict:
    """
    Create a model card documenting the model.

    Args:
        model_name: Name of model
        metrics: Performance metrics
        training_info: Training details
        output_path: Optional path to save JSON

    Returns:
        Model card dictionary
    """
    card = {
        "model_name": model_name,
        "created_at": datetime.now().isoformat(),
        "task": "Binary classification (at-risk prediction)",
        "dataset": "ECLS-K:2011 Public Use File",
        "training": training_info,
        "performance": metrics,
        "intended_use": "Research on algorithmic fairness in education",
        "limitations": [
            "Trained on single cohort (2010-2016)",
            "Public-use data has some variables suppressed",
            "Should not be used for individual-level decisions",
        ],
        "ethical_considerations": [
            "Fairness across demographic groups must be evaluated",
            "Model may perpetuate existing inequities",
            "Requires careful validation before any deployment",
        ],
    }

    if output_path:
        import json

        # Custom encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                return super().default(obj)

        with open(output_path, "w") as f:
            json.dump(card, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Model card saved to {output_path}")

    return card


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Models module ready")
