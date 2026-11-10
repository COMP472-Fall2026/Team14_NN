import numpy as np

from comp472_project.models import NeuralNetClassifier


def toy(n=300, n_features=8, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = rng.normal(size=(n, n_features)) + 3 * np.eye(3, n_features)[y]
    return X.astype(np.float32), y


def test_history_records_train_and_val_curves():
    X, y = toy()
    model = NeuralNetClassifier(epochs=5, seed=0).fit(X[:200], y[:200], X_val=X[200:], y_val=y[200:])
    assert [h["epoch"] for h in model.history_] == [1, 2, 3, 4, 5]
    for key in ("train_loss", "train_acc", "val_loss", "val_acc"):
        assert key in model.history_[-1]
    assert model.history_[-1]["train_loss"] < model.history_[0]["train_loss"]


def test_same_seed_gives_same_predictions():
    X, y = toy()
    a = NeuralNetClassifier(epochs=3, dropout=0.3, seed=1).fit(X, y).predict(X)
    b = NeuralNetClassifier(epochs=3, dropout=0.3, seed=1).fit(X, y).predict(X)
    assert np.array_equal(a, b)


def test_keeps_original_label_values():
    X, y = toy()
    labels = np.array([10, 20, 30])[y]
    model = NeuralNetClassifier(epochs=10, seed=0).fit(X, labels)
    assert set(np.unique(model.predict(X))) <= {10, 20, 30}
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1, atol=1e-5)
