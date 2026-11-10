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
        X = np.asarray(X, dtype=float)
        classes, labels = np.unique(y, return_inverse=True)
        n_rows, n_features = X.shape

        if n_rows == 0 or n_features == 0:
            raise ValueError("X must have at least one row and one feature")
        if self.criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")

        setting = self.max_features
        if setting is None:
            n_choices = n_features
        elif setting == "sqrt":
            n_choices = max(1, int(np.sqrt(n_features)))
        elif setting == "log2":
            n_choices = max(1, int(np.log2(n_features)))
        elif isinstance(setting, (int, np.integer)):
            n_choices = setting
        elif isinstance(setting, (float, np.floating)) and 0 < setting <= 1:
            n_choices = max(1, int(setting * n_features))
        else:
            raise ValueError("Invalid max_features setting")

        if not 1 <= n_choices <= n_features:
            raise ValueError("max_features must select between 1 and all features")

        rng = np.random.default_rng(self.seed)
        one_hot = np.eye(len(classes), dtype=np.int32)[labels]

        def impurity(counts):
            probs = counts / counts.sum(axis=-1, keepdims=True)
            if self.criterion == "gini":
                return 1 - (probs ** 2).sum(axis=-1)
            log_probs = np.zeros_like(probs)
            np.log2(probs, out=log_probs, where=probs > 0)
            return -(probs * log_probs).sum(axis=-1)

        def grow(rows, depth):
            size = len(rows)
            counts = one_hot[rows].sum(axis=0)
            majority = int(classes[counts.argmax()])
            if (
                np.count_nonzero(counts) == 1
                or (self.max_depth is not None and depth >= self.max_depth)
                or size < self.min_samples_split
                or size < 2 * self.min_samples_leaf
            ):
                return majority
            features = range(n_features)
            if n_choices < n_features:
                features = rng.choice(n_features, size=n_choices, replace=False)

            best_score = impurity(counts) - 1e-12
            best_split = None
            for feature in features:
                sorted_rows = rows[np.argsort(X[rows, feature])]
                values = X[sorted_rows, feature]
                # only cut between different values
                cuts = np.flatnonzero(values[:-1] < values[1:]) + 1
                cuts = cuts[
                    (cuts >= self.min_samples_leaf)
                    & (size - cuts >= self.min_samples_leaf)
                ]
                if cuts.size == 0:
                    continue

                # count the labels on each side of each cut
                running_counts = np.cumsum(one_hot[sorted_rows], axis=0, dtype=np.int32)
                left_counts = running_counts[cuts - 1]
                right_counts = counts - left_counts
                scores = (
                    cuts * impurity(left_counts)
                    + (size - cuts) * impurity(right_counts)
                ) / size
                best = scores.argmin()
                if scores[best] < best_score:
                    best_score = scores[best]
                    cut = cuts[best]
                    lower, upper = values[cut - 1], values[cut]
                    threshold = lower / 2 + upper / 2
                    if threshold >= upper:
                        threshold = lower
                    best_split = (feature, threshold, sorted_rows, cut)
            if best_split is None:
                return majority
            feature, threshold, sorted_rows, cut = best_split
            left = grow(sorted_rows[:cut], depth + 1)
            right = grow(sorted_rows[cut:], depth + 1)
            return feature, threshold, left, right
        self._tree = grow(np.arange(n_rows), depth=0)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: route each row from the root to a leaf; return the leaf's class.
        if not hasattr(self, "_tree"):
            raise ValueError("Call fit before predict")
        X = np.asarray(X, dtype=np.float64)
        predictions = np.empty(len(X), dtype=int)
        for i, row in enumerate(X):
            node = self._tree
            while isinstance(node, tuple):
                feature, threshold, left, right = node
                node = left if row[feature] <= threshold else right
            predictions[i] = node
        return predictions
