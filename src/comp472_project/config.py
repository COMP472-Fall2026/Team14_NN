"""Project-wide paths, constants, and seeding."""

import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SUBMISSIONS_DIR = ROOT / "submissions"

SEED = 42
VAL_FRACTION = 0.2

TASKS = {
    "image": {"n_classes": 50, "n_features": 500, "test_id_prefix": "img_"},
    "sentiment": {"n_classes": 3, "n_features": 768, "test_id_prefix": "text_"},
}

SENTIMENT_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}


def check_task(task: str) -> None:
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASKS)}")


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
