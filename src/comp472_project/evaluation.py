"""Metrics, a common train/validate harness, and result logging."""

from collections.abc import Sequence
from time import perf_counter

import numpy as np
import pandas as pd

from comp472_project.config import RESULTS_DIR
from comp472_project.data import TaskData


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """``cm[i, j]`` = number of examples with true class i predicted as j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """Recall of each class (diagonal over row sums)."""
    return np.diag(cm) / np.maximum(cm.sum(axis=1), 1)


def most_confused_pairs(cm: np.ndarray, top: int = 10) -> pd.DataFrame:
    """The largest off-diagonal entries of a confusion matrix."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:top]
    true, pred = np.unravel_index(flat, off.shape)
    return pd.DataFrame({"true": true, "pred": pred, "count": off[true, pred]})


def evaluate(
    model,
    data: TaskData,
    transforms: Sequence = (),
    train_metrics: bool = True,
) -> tuple[dict, np.ndarray]:
    """Fit ``model`` on the training split and score it on the validation split.

    ``transforms`` are fitted on the training split only, then applied to both
    splits in order. Set ``train_metrics=False`` when predicting on the
    training set is slow or meaningless (e.g. k-NN with k=1).

    Returns ``(metrics, val_pred)``.
    """
    X_train, X_val = data.X_train, data.X_val
    for t in transforms:
        X_train = t.fit_transform(X_train, data.y_train)
        X_val = t.transform(X_val)

    start = perf_counter()
    model.fit(X_train, data.y_train)
    fit_time = perf_counter() - start

    start = perf_counter()
    val_pred = np.asarray(model.predict(X_val))
    predict_time = perf_counter() - start

    metrics = {
        "task": data.task,
        "val_acc": accuracy(data.y_val, val_pred),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }
    if train_metrics:
        metrics["train_acc"] = accuracy(data.y_train, model.predict(X_train))
    return metrics, val_pred


def log_results(rows: dict | Sequence[dict], name: str) -> pd.DataFrame:
    """Append result rows (e.g. hyperparameters + metrics) to ``results/<name>.csv``.

    Rows may have different keys; missing values become empty cells.
    Returns the full table.
    """
    rows = [rows] if isinstance(rows, dict) else list(rows)
    path = RESULTS_DIR / f"{name}.csv"
    new = pd.DataFrame(rows)
    table = pd.concat([pd.read_csv(path), new], ignore_index=True) if path.exists() else new
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table
