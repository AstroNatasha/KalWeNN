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

def _fetch(name: str, version: int = 1, max_samples: int = None) -> Dataset:
    import pandas as pd
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name, version=version, as_frame=True, parser="auto")

    # mantém só atributos numéricos; descarta colunas de texto/identificadores
    # (ex.: nome do locutor no 'vowel'). Se não sobrar nada, codifica tudo.
    X_num = d.data.select_dtypes(include="number")
    if X_num.shape[1] == 0:
        X_num = d.data.apply(lambda c: pd.factorize(c)[0])
    X = X_num.to_numpy(dtype=float)

    y = np.asarray(d.target)
    if y.ndim > 1:  # raros casos de target multi-coluna
        y = y.argmax(axis=1)
    y = _encode_labels(y)

    if max_samples and max_samples < len(X):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    return X, y


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


# ── datasets extras (todos numéricos, via OpenML) ─────────────────────────────

def get_pendigits() -> Dataset:        # 10 classes, 16 atributos, ~11k
    return _fetch("pendigits", version=1)


def get_optdigits() -> Dataset:        # 10 classes, 64 atributos, ~5.6k
    return _fetch("optdigits", version=1)


def get_letter() -> Dataset:           # 26 classes, 16 atributos (limitado a 10k)
    return _fetch("letter", version=1, max_samples=10_000)


def get_texture() -> Dataset:          # 11 classes, 40 atributos, ~5.5k
    return _fetch("texture", version=1)


def get_waveform() -> Dataset:         # 3 classes, 40 atributos, 5k
    return _fetch("waveform-5000", version=1)


def get_spambase() -> Dataset:         # 2 classes, 57 atributos, ~4.6k
    return _fetch("spambase", version=1)


def get_ionosphere() -> Dataset:       # 2 classes, 34 atributos, 351
    return _fetch("ionosphere", version=1)


def get_sonar() -> Dataset:            # 2 classes, 60 atributos, 208
    return _fetch("sonar", version=1)


def get_yeast() -> Dataset:            # 10 classes, 8 atributos, ~1.5k
    return _fetch("yeast", version=1)


def get_mfeat_fourier() -> Dataset:    # 10 classes, 76 atributos, 2k
    return _fetch("mfeat-fourier", version=1)


def get_mfeat_karhunen() -> Dataset:   # 10 classes, 64 atributos, 2k
    return _fetch("mfeat-karhunen", version=1)
