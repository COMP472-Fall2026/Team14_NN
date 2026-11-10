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
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        X = np.asarray(X)
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                max_features=self.max_features,
                seed=int(rng.integers(0, 2**32)),
            )

            if self.bootstrap:
                # Pick training rows with replacement for this tree.
                rows = rng.integers(0, len(y), size=len(y))
                tree.fit(X[rows], y[rows])
            else:
                tree.fit(X, y)
            trees.append(tree)
        self.trees_ = trees
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: majority vote across trees.
        if not hasattr(self, "trees_"):
            raise ValueError("Call fit before predict")
        votes = np.zeros((len(X), len(self.classes_)), dtype=int)
        rows = np.arange(len(X))
        for tree in self.trees_:
            labels = tree.predict(X)
            # Find the vote column for each predicted label.
            columns = np.searchsorted(self.classes_, labels)
            votes[rows, columns] += 1
        return self.classes_[votes.argmax(axis=1)]
