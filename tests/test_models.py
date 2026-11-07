"""Smoke tests: every model should solve an easy, well-separated toy problem.

Models that are not implemented yet are skipped, so these start passing as
the team fills in each algorithm.
"""

import numpy as np
import pytest

from comp472_project.clustering import KMeans
from comp472_project.evaluation import accuracy
from comp472_project.models import (
    DecisionTreeClassifier,
    KNNClassifier,
    NeuralNetClassifier,
    RandomForestClassifier,
)


def blobs(n_per_class=100, n_classes=3, n_features=10, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 5, size=(n_classes, n_features))
    X = np.concatenate([c + rng.normal(size=(n_per_class, n_features)) for c in centers])
    y = np.repeat(np.arange(n_classes), n_per_class)
    perm = rng.permutation(len(y))
    return X[perm].astype(np.float32), y[perm]


MODELS = {
    "knn": lambda: KNNClassifier(k=5),
    "decision_tree": lambda: DecisionTreeClassifier(max_depth=10, seed=0),
    "random_forest": lambda: RandomForestClassifier(n_estimators=10, seed=0),
    "neural_net": lambda: NeuralNetClassifier(epochs=20, seed=0),
}


@pytest.mark.parametrize("name", MODELS)
def test_classifier_on_blobs(name):
    X, y = blobs()
    try:
        model = MODELS[name]()
        model.fit(X[:240], y[:240])
        pred = np.asarray(model.predict(X[240:]))
    except NotImplementedError:
        pytest.skip(f"{name} not implemented yet")
    assert pred.shape == (60,)
    assert set(np.unique(pred)) <= {0, 1, 2}
    assert accuracy(y[240:], pred) > 0.9


def test_kmeans_recovers_blobs():
    X, y = blobs()
    try:
        km = KMeans(n_clusters=3, seed=0).fit(X)
    except NotImplementedError:
        pytest.skip("KMeans not implemented yet")
    # Each true class should map almost entirely to one cluster.
    for c in range(3):
        assert np.bincount(km.labels_[y == c]).max() / (y == c).sum() > 0.9
