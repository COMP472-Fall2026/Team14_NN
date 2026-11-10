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


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
@pytest.mark.parametrize("weighted", [False, True])
def test_knn_matches_sklearn(metric, weighted):
    """Our k-NN must give the same predictions as scikit-learn's.

    scikit-learn is used here only as a reference to check our implementation;
    it is never used to make the predictions we report or submit.

    The two implementations break vote ties differently (ours: class of the
    closest neighbour; scikit-learn: lowest class number), so the data is
    chosen so that ties cannot happen:
    - plain vote: 2 classes and an odd k, so the vote can never be split evenly;
    - weighted vote: random real-valued data, so two classes never get exactly
      the same total weight.
    The classes overlap heavily (noise std 4, centres about 1 apart per
    feature), so many neighbourhoods are mixed and the vote is often close:
    accuracy is only about 70% (plain) and 40% (weighted, 4 classes). On easy
    data every neighbour has the same class, and any k-NN would pass.
    """
    from sklearn.neighbors import KNeighborsClassifier

    n_classes, k = (2, 7) if not weighted else (4, 8)
    rng = np.random.default_rng(1)
    centers = rng.normal(0, 1, size=(n_classes, 20))
    n_per_class = 600 // n_classes  # 600 rows in total: 400 to fit, 200 to predict
    X = np.concatenate([c + rng.normal(0, 4, size=(n_per_class, 20)) for c in centers])
    y = np.repeat(np.arange(n_classes), n_per_class)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    X_train, y_train, X_test = X[:400], y[:400], X[400:]

    ours = KNNClassifier(k=k, metric=metric, weighted=weighted).fit(X_train, y_train).predict(X_test)
    reference = KNeighborsClassifier(
        n_neighbors=k,
        metric=metric,
        weights="distance" if weighted else "uniform",
        algorithm="brute",
    ).fit(X_train, y_train).predict(X_test)
    np.testing.assert_array_equal(ours, reference)


def test_kmeans_recovers_blobs():
    X, y = blobs()
    try:
        km = KMeans(n_clusters=3, seed=0).fit(X)
    except NotImplementedError:
        pytest.skip("KMeans not implemented yet")
    # Each true class should map almost entirely to one cluster.
    for c in range(3):
        assert np.bincount(km.labels_[y == c]).max() / (y == c).sum() > 0.9
