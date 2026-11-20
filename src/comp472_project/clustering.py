"""Clustering for the unsupervised experiment (not part of the Kaggle score).

k-Means, implemented from scratch with NumPy.
"""

import numpy as np


def _sq_distances(X: np.ndarray, X_sq_norms: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance from every row of X to every centre: (n, k).

    Uses ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b so the work is one matrix
    product instead of an (n x k x n_features) array of differences.
    """
    sq = X_sq_norms[:, None] + np.sum(centers**2, axis=1)[None, :] - 2.0 * X @ centers.T
    # Rounding can make a true 0 come out as a tiny negative number.
    return np.maximum(sq, 0.0)


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
    Also ``n_iter_``: iterations used by the kept run.

    Initialisation is k-means++. Convergence: a run stops when no centroid
    moved more than ``tol`` (Euclidean distance) in the last update.
    Inertia is the sum of squared distances from each point to its centroid.
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
        """Cluster X. ``y`` is ignored (accepted so the API matches the classifiers)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a non-empty 2-D array")
        if not 1 <= self.n_clusters <= len(X):
            raise ValueError(
                f"n_clusters must be between 1 and the number of samples ({len(X)}), "
                f"got {self.n_clusters}"
            )
        if self.n_init < 1 or self.max_iter < 1:
            raise ValueError("n_init and max_iter must be at least 1")

        rng = np.random.default_rng(self.seed)
        X_sq_norms = np.sum(X**2, axis=1)  # reused by every distance computation

        best = None
        for _ in range(self.n_init):
            run = self._single_run(X, X_sq_norms, rng)
            if best is None or run[2] < best[2]:
                best = run
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Index of the nearest centroid for each row of X."""
        X = np.asarray(X, dtype=np.float64)
        dist = _sq_distances(X, np.sum(X**2, axis=1), self.cluster_centers_)
        return dist.argmin(axis=1)

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X).labels_

    # ------------------------------------------------------------------ #

    def _single_run(self, X, X_sq_norms, rng):
        """One k-means run (Lloyd's algorithm) from a k-means++ start.

        Returns (centers, labels, inertia, n_iter).
        """
        centers = self._init_kmeans_pp(X, X_sq_norms, rng)
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            # 1. Assignment step: each point joins its nearest centroid.
            dist = _sq_distances(X, X_sq_norms, centers)
            labels = dist.argmin(axis=1)

            # 2. Update step: each centroid moves to the mean of its points.
            new_centers = self._update_centers(X, labels, dist, centers)

            # 3. Stop once the centroids have (almost) stopped moving.
            shift = np.sqrt(np.sum((new_centers - centers) ** 2, axis=1)).max()
            centers = new_centers
            if shift < self.tol:
                break

        # Final assignment with the final centroids, so labels_ and inertia_
        # are consistent with cluster_centers_ (and labels_ == predict(X)).
        dist = _sq_distances(X, X_sq_norms, centers)
        labels = dist.argmin(axis=1)
        inertia = dist[np.arange(len(X)), labels].sum()
        return centers, labels, inertia, n_iter

    def _init_kmeans_pp(self, X, X_sq_norms, rng):
        """k-means++: spread the initial centroids out.

        The first centroid is a random point. Each next one is a random point
        chosen with probability proportional to its squared distance to the
        nearest centroid picked so far, so far-away points are favoured.
        """
        n = len(X)
        centers = np.empty((self.n_clusters, X.shape[1]))
        centers[0] = X[rng.integers(n)]
        # Squared distance from every point to its closest chosen centroid.
        closest = _sq_distances(X, X_sq_norms, centers[:1])[:, 0]
        for c in range(1, self.n_clusters):
            total = closest.sum()
            if total > 0:
                idx = rng.choice(n, p=closest / total)
            else:
                # Every point already coincides with a centroid (duplicates);
                # any point will do.
                idx = rng.integers(n)
            centers[c] = X[idx]
            closest = np.minimum(closest, _sq_distances(X, X_sq_norms, centers[c : c + 1])[:, 0])
        return centers

    def _update_centers(self, X, labels, dist, old_centers):
        """Mean of the points in each cluster.

        A cluster that lost all its points would have no mean. Its centroid is
        moved to the point that is currently worst served (farthest from its
        own centroid), which is the usual fix and lowers inertia.
        """
        k = self.n_clusters
        counts = np.bincount(labels, minlength=k)
        # sums[c] = sum of the points in cluster c. Built as one_hot.T @ X,
        # where one_hot[i, c] = 1 if point i is in cluster c: one matrix
        # product, much faster than looping (or np.add.at) over the points.
        one_hot = np.zeros((len(X), k))
        one_hot[np.arange(len(X)), labels] = 1.0
        sums = one_hot.T @ X

        new_centers = old_centers.copy()
        non_empty = counts > 0
        new_centers[non_empty] = sums[non_empty] / counts[non_empty, None]

        empty = np.flatnonzero(~non_empty)
        if len(empty):
            point_dist = dist[np.arange(len(X)), labels]
            # Farthest points first; each empty cluster takes a different one.
            far = np.argsort(point_dist)[::-1][: len(empty)]
            new_centers[empty] = X[far]
        return new_centers
