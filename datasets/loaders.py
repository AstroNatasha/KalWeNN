from typing import Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder

Array = np.ndarray
Dataset = Tuple[Array, Array]


def _encode_labels(y) -> Array:
    if np.asarray(y).dtype.kind in ("U", "S", "O"):
        return LabelEncoder().fit_transform(y)
    return np.asarray(y, dtype=int)


# ── sklearn built-ins (no download) ───────────────────────────────────────────

def get_wine() -> Dataset:
    from sklearn.datasets import load_wine
    d = load_wine()
    return d.data, d.target


def get_iris() -> Dataset:
    from sklearn.datasets import load_iris
    d = load_iris()
    return d.data, d.target


def get_breast_cancer() -> Dataset:
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return d.data, d.target


def get_digits() -> Dataset:
    from sklearn.datasets import load_digits
    d = load_digits()
    return d.data, d.target


# ── OpenML datasets (downloaded and cached on first call) ─────────────────────

def _fetch(name: str, version: int = 1) -> Dataset:
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name, version=version, as_frame=False, parser="auto")
    return d.data.astype(float), _encode_labels(d.target)


def get_mnist(max_samples: int = 10_000) -> Dataset:
    from sklearn.datasets import fetch_openml
    d = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = d.data.astype(float), _encode_labels(d.target)
    if max_samples and max_samples < len(X):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def get_glass() -> Dataset:
    return _fetch("glass", version=4)


def get_vehicle() -> Dataset:
    return _fetch("vehicle", version=1)


def get_segment() -> Dataset:
    return _fetch("segment", version=1)


def get_vowel() -> Dataset:
    return _fetch("vowel", version=2)


def get_satellite() -> Dataset:
    return _fetch("satimage", version=1)
