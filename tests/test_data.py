import numpy as np
import pytest

from comp472_project.config import TASKS
from comp472_project.data import load_split, split_path, stratified_split


def test_stratified_split_keeps_class_proportions():
    y = np.repeat(np.arange(5), 100)
    is_val = stratified_split(y, val_fraction=0.2, seed=0)
    assert is_val.sum() == 100
    assert (np.bincount(y[is_val]) == 20).all()


def test_stratified_split_is_deterministic():
    y = np.repeat(np.arange(3), 50)
    assert (stratified_split(y, seed=1) == stratified_split(y, seed=1)).all()
    assert not (stratified_split(y, seed=1) == stratified_split(y, seed=2)).all()


@pytest.mark.parametrize("task", TASKS)
def test_saved_split_is_disjoint_and_complete(task):
    if not split_path(task).exists():
        pytest.skip("run scripts/prepare_data.py first")
    data = load_split(task)
    assert not set(data.ids_train) & set(data.ids_val)
    assert data.X_train.shape[1] == TASKS[task]["n_features"]
    assert set(np.unique(data.y_val)) == set(range(TASKS[task]["n_classes"]))
