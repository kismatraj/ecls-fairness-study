"""Tests for models module."""

import pandas as pd
import numpy as np
import pytest

from src.models import ModelTrainer, get_feature_importance


def test_model_trainer_init():
    trainer = ModelTrainer(random_state=42, test_size=0.3, cv_folds=5)
    assert trainer.random_state == 42
    assert trainer.test_size == 0.3
    assert len(trainer.models) == 0


def test_split_data(sample_data):
    trainer = ModelTrainer(random_state=42, test_size=0.3)
    X = sample_data[["X1RTHETK", "X1MTHETK"]]
    y = sample_data["X9RTHETA_at_risk"]

    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    assert len(X_train) + len(X_test) == len(X)
    assert abs(len(X_test) / len(X) - 0.3) < 0.05


def test_train_logistic_regression(sample_data):
    trainer = ModelTrainer(random_state=42, test_size=0.3, cv_folds=3)
    X = sample_data[["X1RTHETK", "X1MTHETK", "X_CHSEX_R"]]
    y = sample_data["X9RTHETA_at_risk"]

    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    model, results = trainer.train_model("logistic_regression", X_train, y_train)

    assert model is not None
    assert "best_cv_score" in results
    assert 0 < results["best_cv_score"] < 1


def test_evaluate_model(sample_data):
    trainer = ModelTrainer(random_state=42, test_size=0.3, cv_folds=3)
    X = sample_data[["X1RTHETK", "X1MTHETK"]]
    y = sample_data["X9RTHETA_at_risk"]

    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    model, _ = trainer.train_model("logistic_regression", X_train, y_train)

    metrics = trainer.evaluate_model(model, X_test, y_test)
    assert "auc_roc" in metrics
    assert "accuracy" in metrics
    assert 0 <= metrics["auc_roc"] <= 1
    assert 0 <= metrics["accuracy"] <= 1


def test_get_predictions(sample_data):
    trainer = ModelTrainer(random_state=42, test_size=0.3, cv_folds=3)
    X = sample_data[["X1RTHETK", "X1MTHETK"]]
    y = sample_data["X9RTHETA_at_risk"]

    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    model, _ = trainer.train_model("logistic_regression", X_train, y_train)

    y_pred, y_prob = trainer.get_predictions(model, X_test)
    assert len(y_pred) == len(X_test)
    assert len(y_prob) == len(X_test)
    assert set(np.unique(y_pred)).issubset({0, 1})
    assert y_prob.min() >= 0 and y_prob.max() <= 1


def test_feature_importance(sample_data):
    trainer = ModelTrainer(random_state=42, test_size=0.3, cv_folds=3)
    features = ["X1RTHETK", "X1MTHETK"]
    X = sample_data[features]
    y = sample_data["X9RTHETA_at_risk"]

    X_train, _, y_train, _ = trainer.split_data(X, y)
    model, _ = trainer.train_model("logistic_regression", X_train, y_train)

    imp = get_feature_importance(model, features)
    assert len(imp) == 2
    assert "feature" in imp.columns
    assert "importance" in imp.columns


def test_sample_weights_accepted():
    """Verify ModelTrainer accepts sample_weights parameter."""
    weights = np.ones(100)
    trainer = ModelTrainer(sample_weights=weights)
    assert trainer.sample_weights is not None
    assert len(trainer.sample_weights) == 100
