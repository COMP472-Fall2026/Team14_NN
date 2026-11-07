"""Decision Tree classifier, implemented from scratch with NumPy."""

import numpy as np


class DecisionTreeClassifier:
    """
    Parameters
    ----------
    max_depth : maximum depth of the tree (None = grow until pure or other
        stopping rules apply).
    min_samples_split : minimum samples a node needs to be considered for a split.
    min_samples_leaf : minimum samples each child must keep after a split.
    criterion : "gini" or "entropy".
    max_features : features considered at each split: None (all), "sqrt",
        "log2", an int, or a float fraction. Random Forest uses this.
    seed : seed for the feature subsampling.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        max_features: int | float | str | None = None,
        seed: int | None = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        # TODO: recursively choose the (feature, threshold) split that most
        # reduces impurity, until a stopping rule is met.
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: route each row from the root to a leaf; return the leaf's class.
        raise NotImplementedError
