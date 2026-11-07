"""k-Nearest Neighbours, implemented from scratch with NumPy."""

import numpy as np


class KNNClassifier:
    """
    Parameters
    ----------
    k : number of neighbours.
    metric : "euclidean", "manhattan" or "cosine".
    weighted : if True, neighbours vote with weight 1 / (distance + eps).
    batch_size : test rows processed at a time, to bound memory use of the
        (batch_size x n_train) distance matrix.
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
        # TODO: store the training data.
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: for each batch of rows, compute distances to all training
        # points, take the k nearest, and vote (optionally distance-weighted).
        raise NotImplementedError
