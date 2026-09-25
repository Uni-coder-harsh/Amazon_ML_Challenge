"""
model.py
LightGBM training, persistence, and batch probability inference engine
for high-precision entity resolution.
"""

import os
import joblib
import numpy as np
import lightgbm as lgb


def build_model(
    n_estimators: int = 180,
    learning_rate: float = 0.08,
    num_leaves: int = 31,
    random_state: int = 42,
    n_jobs: int = 6
) -> lgb.LGBMClassifier:
    """
    Constructs a precision-oriented LightGBM Classifier.
    """
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1
    )


def save_model(clf: lgb.LGBMClassifier, filepath: str):
    """
    Saves trained LightGBM model artifact to disk.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    joblib.dump(clf, filepath)


def load_model(filepath: str) -> lgb.LGBMClassifier:
    """
    Loads saved LightGBM model artifact from disk.
    """
    return joblib.load(filepath)
