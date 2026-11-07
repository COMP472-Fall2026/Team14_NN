"""Random Forest, built from the from-scratch DecisionTreeClassifier."""

import numpy as np

from comp472_project.models.decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    """
    Parameters
    ----------
    n_estimators : number of trees.
    max_depth, min_samples_split, min_samples_leaf, criterion : passed to each tree.
    max_features : features considered at each split in each tree.
    bootstrap : if True, each tree trains on a bootstrap sample of the rows.
    seed : seed for bootstrap sampling and per-tree seeds.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        max_features: int | float | str | None = "sqrt",
        bootstrap: bool = True,
        seed: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        # TODO: train n_estimators DecisionTreeClassifier instances, each on a
        # bootstrap sample and with its own seed.
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: majority vote across trees.
        raise NotImplementedError
