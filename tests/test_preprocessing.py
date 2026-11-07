import numpy as np

from comp472_project.preprocessing import L2Normalizer, Standardizer


def test_standardizer_uses_training_statistics():
    rng = np.random.default_rng(0)
    X_train = rng.normal(5, 3, size=(1000, 4))
    X_val = rng.normal(5, 3, size=(10, 4))
    s = Standardizer().fit(X_train)
    Z = s.transform(X_train)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(Z.std(axis=0), 1, atol=1e-6)
    assert np.allclose(s.transform(X_val), (X_val - X_train.mean(0)) / X_train.std(0))


def test_standardizer_handles_constant_feature():
    X = np.column_stack([np.zeros(10), np.arange(10.0)])
    assert np.isfinite(Standardizer().fit_transform(X)).all()


def test_l2_normalizer_gives_unit_rows():
    X = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = L2Normalizer().fit_transform(X)
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], 0)
