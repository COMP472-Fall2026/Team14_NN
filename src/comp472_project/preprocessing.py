"""Feature transforms with a fit/transform interface.

Anything that learns parameters (means, stds, PCA components, ...) must be
fitted on the training subset only and then applied to validation/test data.
``evaluation.evaluate`` does this for you if you pass ``transforms=[...]``.
"""

import numpy as np


class Standardizer:
    """Shift each feature to zero mean and scale it to unit variance."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Constant features (std == 0) are only centred, not divided by ~0.
        return (X - self.mean_) / np.maximum(self.std_, self.eps)

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class SqrtTransform:
    """Element-wise square root of non-negative features (learns nothing).

    For count histograms like the image features (bag-of-visual-words), this
    shrinks the gap between frequent and rare visual words, so a few very
    common words no longer dominate distances and dot products. Comparing
    square-rooted histograms is the classic "Hellinger" trick.
    """

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SqrtTransform":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if (X < 0).any():
            raise ValueError("SqrtTransform needs non-negative features")
        return np.sqrt(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class L2Normalizer:
    """Scale each row to unit Euclidean norm (learns nothing from the data)."""

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "L2Normalizer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True) # calculates the euclidean length (L2 norm) of each row vector in X
        return X / np.maximum(norms, self.eps) # normalize each row to unit length, avoiding division by zero by choosing `self.eps` instead if a row has length 0

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
