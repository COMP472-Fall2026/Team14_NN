"""Loading the competition CSVs and the fixed train/validation split.

The CSVs are parsed once and cached as ``data/cache/<task>_<part>.npz``; the
cache is rebuilt automatically if the CSV is newer than it.

The train/validation split is stored in ``data/splits/<task>.csv`` (columns
``id,split``) and committed to git, so everyone evaluates on exactly the same
validation examples. Create it with ``scripts/prepare_data.py``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from comp472_project.config import (
    CACHE_DIR,
    DATA_DIR,
    SEED,
    SPLITS_DIR,
    TASKS,
    VAL_FRACTION,
    check_task,
)


def _read_csv(task: str, part: str) -> dict[str, np.ndarray]:
    path = DATA_DIR / f"{task}_{part}.csv"
    has_label = part == "train"
    feature_cols = [f"feature_{i}" for i in range(TASKS[task]["n_features"])]
    expected = ["id", *feature_cols, *(["label"] if has_label else [])]

    df = pd.read_csv(path)
    if list(df.columns) != expected:
        raise ValueError(f"{path.name}: unexpected columns")

    arrays = {
        "ids": df["id"].to_numpy(dtype=str),
        "X": df[feature_cols].to_numpy(dtype=np.float32),
    }
    if has_label:
        arrays["y"] = df["label"].to_numpy(dtype=np.int64)
    return arrays


def _load(task: str, part: str) -> dict[str, np.ndarray]:
    check_task(task)
    csv_path = DATA_DIR / f"{task}_{part}.csv"
    cache_path = CACHE_DIR / f"{task}_{part}.npz"

    if cache_path.exists() and cache_path.stat().st_mtime >= csv_path.stat().st_mtime:
        with np.load(cache_path) as f:
            return {k: f[k] for k in f.files}

    arrays = _read_csv(task, part)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **arrays)
    return arrays


def load_train(task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(ids, X, y)`` for the full labelled training file."""
    a = _load(task, "train")
    return a["ids"], a["X"], a["y"]


def load_test(task: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ids, X)`` for the unlabelled test file (public + private)."""
    a = _load(task, "test")
    return a["ids"], a["X"]


def stratified_split(
    y: np.ndarray, val_fraction: float = VAL_FRACTION, seed: int = SEED
) -> np.ndarray:
    """Return a boolean mask that is True for validation rows.

    Each class contributes ``round(n_class * val_fraction)`` rows to
    validation, so class proportions are the same in both subsets.
    """
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        is_val[idx[: round(len(idx) * val_fraction)]] = True
    return is_val


def split_path(task: str) -> Path:
    check_task(task)
    return SPLITS_DIR / f"{task}.csv"


def create_split(
    task: str,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
    overwrite: bool = False,
) -> None:
    """Write ``data/splits/<task>.csv``. Refuses to overwrite unless asked."""
    path = split_path(task)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    ids, _, y = load_train(task)
    is_val = stratified_split(y, val_fraction, seed)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "split": np.where(is_val, "val", "train")}).to_csv(
        path, index=False
    )


@dataclass(frozen=True)
class TaskData:
    """Training and validation arrays for one task."""

    task: str
    ids_train: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    ids_val: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray

    @property
    def n_classes(self) -> int:
        return TASKS[self.task]["n_classes"]


def load_split(task: str) -> TaskData:
    """Load the training data divided according to the saved split."""
    path = split_path(task)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `uv run python scripts/prepare_data.py` first"
        )
    ids, X, y = load_train(task)
    split = pd.read_csv(path).set_index("id")["split"].reindex(ids)
    if split.isna().any() or not split.isin(["train", "val"]).all():
        raise ValueError(f"{path} does not match {task}_train.csv")
    is_val = (split == "val").to_numpy()
    tr = ~is_val
    return TaskData(task, ids[tr], X[tr], y[tr], ids[is_val], X[is_val], y[is_val])
