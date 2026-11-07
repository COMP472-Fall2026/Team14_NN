"""Clustering for the unsupervised experiment (not part of the Kaggle score)."""

import numpy as np


class KMeans:
    """
    Parameters
    ----------
    n_clusters : number of clusters k.
    n_init : number of restarts from different initial centroids; the run
        with the lowest inertia is kept.
    max_iter : maximum assignment/update iterations per restart.
    tol : stop when centroids move less than this.
    seed : seed for centroid initialisation.

    After ``fit``: ``cluster_centers_``, ``labels_``, ``inertia_``.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        seed: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KMeans":
        # TODO: initialise centroids (e.g. k-means++), then alternate
        # assigning points to the nearest centroid and recomputing centroids.
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: index of the nearest centroid for each row.
        raise NotImplementedError
