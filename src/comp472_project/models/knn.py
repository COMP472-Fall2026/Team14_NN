"""k-Nearest Neighbours, implemented from scratch with NumPy."""

import numpy as np

METRICS = ("euclidean", "manhattan", "cosine")

# Added to every distance before taking 1 / distance, so a neighbour at
# distance 0 (an exact duplicate of the query) gets a huge weight instead of
# causing a division by zero.
WEIGHT_EPS = 1e-8

# Manhattan distance needs a temporary (rows x n_train x n_features) array.
# We cap its size at this many numbers (64M float64 values = 512 MB).
MANHATTAN_MAX_ELEMENTS = 64_000_000


class KNNClassifier:
    """
    Parameters
    ----------
    k : number of neighbours.
    metric : "euclidean", "manhattan" or "cosine".
    weighted : if True, neighbours vote with weight 1 / (distance + eps).
    batch_size : test rows processed at a time, to bound memory use of the
        (batch_size x n_train) distance matrix.

    Ties: if two or more classes get the same (largest) vote, the prediction
    is the tied class that owns the closest neighbour.
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        weighted: bool = False,
        batch_size: int = 1024,
    ):
        self.k = k
        self.metric = metric
        self.weighted = weighted
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """k-NN does no learning: it just remembers the training set."""
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {self.metric!r}")
        if self.k < 1:
            raise ValueError(f"k must be at least 1, got {self.k}")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if len(X) == 0:
            raise ValueError("Training dataset cannot be empty")

        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D with one row per label in y")

        if self.k > len(X):
            raise ValueError(f"k={self.k} exceeds the number of training samples ({len(X)})")

        # Store the labels as positions 0..n_classes-1 in classes_, so they can
        # be used directly as column indices when counting votes.
        # predict() maps them back to the original labels at the end.
        self.classes_, self.y_train_ = np.unique(y, return_inverse=True)
        self.X_train_ = X

        # Pre-compute per-row quantities the distance formulas reuse for
        # every test batch (see _distances).
        if self.metric == "euclidean":
            self.train_sq_norms_ = np.sum(X**2, axis=1)  # ||x||^2 per row
        elif self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            self.X_train_unit_ = X / np.maximum(norms, 1e-12)  # each row length 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        preds = np.empty(len(X), dtype=np.int64)
        for start in range(0, len(X), self.batch_size):
            batch = X[start : start + self.batch_size]
            preds[start : start + len(batch)] = self._predict_batch(batch)
        return self.classes_[preds]

    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        dist = self._distances(X)  # shape (batch, n_train)
        rows = np.arange(len(X))[:, None]  # column of row numbers, for indexing

        # 1. Find the k nearest training points of every test row.
        #    argpartition only guarantees that the first k columns are the k
        #    smallest distances (in some order), which is faster than a full
        #    sort over all n_train points. We then sort just those k, closest
        #    first, because the tie-break below needs that order.
        nearest = np.argpartition(dist, self.k - 1, axis=1)[:, :self.k] # indices of the nearest k indices, unordered
        nearest_dist = dist[rows, nearest] # stores the actual nearest distances, unordered
        order = np.argsort(nearest_dist, axis=1) # ordered indices within k nearest
        nearest = nearest[rows, order] # stores indices for the nearest distances in the train set
        nearest_dist = nearest_dist[rows, order] # stores the acutal nearest distances, ordered
        nearest_labels = self.y_train_[nearest]  # shape (batch, k)

        # 2. Each neighbour votes for its class. Plain vote: weight 1.
        #    Weighted vote: weight 1 / distance, so closer neighbours count more.
        if self.weighted:
            weights = 1.0 / (nearest_dist + WEIGHT_EPS)
        else:
            weights = np.ones_like(nearest_dist)
        votes = np.zeros((len(X), len(self.classes_)))
        # votes[r, nearest_labels[r, j]] += weights[r, j] for every r, j.
        # (np.add.at, unlike votes[...] += ..., adds correctly when the same
        # class appears several times in one row.)
        np.add.at(votes, (np.broadcast_to(rows, nearest_labels.shape), nearest_labels), weights)

        # 3. Pick the class with the most votes. To break ties, go through the
        #    neighbours closest first and take the first one whose class is
        #    among the tied winners. With no tie, this is simply the winner.
        is_winner = votes == votes.max(axis=1, keepdims=True)  # (batch, n_classes)
        neighbour_is_winner = is_winner[rows, nearest_labels]  # (batch, k)
        first = neighbour_is_winner.argmax(axis=1)  # index of first True
        return nearest_labels[np.arange(len(X)), first]

    def _distances(self, X: np.ndarray) -> np.ndarray:
        """Distance from every row of X to every training row: (len(X), n_train)."""
        if self.metric == "euclidean":
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            # This turns the work into one matrix product (fast) instead of
            # building a (batch x n_train x n_features) array of differences.
            sq = (
                np.sum(X**2, axis=1)[:, None]
                + self.train_sq_norms_[None, :]
                - 2.0 * X @ self.X_train_.T
            )
            # Rounding can make a true 0 come out as a tiny negative number;
            # clip before the square root.
            return np.sqrt(np.maximum(sq, 0.0))

        if self.metric == "cosine":
            # cosine distance = 1 - cos(angle between a and b).
            # For length-1 vectors, cos(angle) is just the dot product.
            # 0 = same direction, 1 = perpendicular, 2 = opposite directions.
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X_unit = X / np.maximum(norms, 1e-12)
            return np.maximum(1.0 - X_unit @ self.X_train_unit_.T, 0.0)

        # Manhattan: sum of |a_i - b_i|. There is no matrix-product shortcut,
        # so take the differences directly, a few test rows at a time to keep
        # the temporary 3-D array within MANHATTAN_MAX_ELEMENTS.
        n_train, n_features = self.X_train_.shape
        step = max(1, MANHATTAN_MAX_ELEMENTS // (n_train * n_features))
        out = np.empty((len(X), n_train))
        for s in range(0, len(X), step):
            diff = X[s : s + step, None, :] - self.X_train_[None, :, :]
            # out=diff takes the absolute value in place. Plain np.abs(diff)
            # would make a second array the same size and double peak memory.
            np.abs(diff, out=diff)
            out[s : s + step] = diff.sum(axis=2)
        return out
