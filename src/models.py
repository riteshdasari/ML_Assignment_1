"""Shared data validation and the two selected polynomial pipelines."""

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    1: {"model": "Lasso", "degree": 5, "alpha": 0.00975},
    2: {"model": "Ridge", "degree": 12, "alpha": 2.5},
}


def read_csv(path):
    """Read a numeric CSV, rejecting missing or infinite values."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        columns = next(reader)
        values = np.asarray(list(reader), dtype=float)
    if values.ndim != 2 or values.shape[1] != len(columns) or not len(values):
        raise ValueError(f"Invalid CSV shape: {path}")
    if len(set(columns)) != len(columns) or not np.isfinite(values).all():
        raise ValueError(f"Duplicate columns or non-finite values: {path}")
    return columns, values


def load_training(data_dir, problem):
    columns, values = read_csv(Path(data_dir) / f"IMT2024080_train_var{problem}(in).csv")
    if columns.count("y") != 1:
        raise ValueError("Training data must contain exactly one target column named y")
    target = columns.index("y")
    features = [name for name in columns if name != "y"]
    indices = [columns.index(name) for name in features]
    if len(features) != {1: 6, 2: 3}[problem]:
        raise ValueError(f"Unexpected feature count for question {problem}")
    return features, values[:, indices], values[:, target]


def make_model(problem):
    config = SETTINGS[problem]
    estimator = (
        Lasso(alpha=config["alpha"], max_iter=50000)
        if problem == 1 else Ridge(alpha=config["alpha"])
    )
    return make_pipeline(
        PolynomialFeatures(degree=config["degree"], include_bias=False),
        StandardScaler(),
        estimator,
    )
