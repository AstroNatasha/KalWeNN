"""
Loaders dos datasets de REGRESSÃO (alvo contínuo).

Mesmo contrato dos loaders de classificação (`datasets/loaders.py`): cada função
devolve `(X, y)` como arrays float. A diferença é que aqui `y` é contínuo — não
passa por LabelEncoder e não pode ser estratificado no split.

Regras aplicadas a todos:
  - só atributos numéricos entram em X (colunas categóricas viram códigos
    inteiros via factorize, como no loader de classificação);
  - linhas com NaN em X ou y são descartadas;
  - datasets grandes são subamostrados com semente fixa (rng=0) para manter o
    custo do estudo viável — o corte é anotado em REG_INFO e no README.

Uso:
    from datasets.loaders_regression import get_regression, REGRESSION_DATASETS
    X, y = get_regression("airfoil")
"""

import os
from typing import Tuple

import numpy as np

Array = np.ndarray
Dataset = Tuple[Array, Array]

# Teto por dataset (subamostragem com semente fixa). Ajustável por ambiente:
#   REG_MAX_SAMPLES=0 python3 ...   → sem teto (california usa as 20.640 linhas)
MAX_SAMPLES = int(os.environ.get("REG_MAX_SAMPLES", "5000")) or None


def _subsample(X: Array, y: Array, max_samples: int = MAX_SAMPLES) -> Dataset:
    if max_samples and max_samples < len(X):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def _clean(X: Array, y: Array) -> Dataset:
    """Remove linhas com NaN/inf em X ou y."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[ok], y[ok]


def _fetch_reg(name: str = None, data_id: int = None, version: int = 1,
               drop: list = None, max_samples: int = MAX_SAMPLES) -> Dataset:
    """Baixa um dataset do OpenML e prepara para regressão.

    Preferimos `data_id` da suíte OpenML-CTR23 (benchmark curado de regressão):
    as versões antigas por nome trazem o alvo já discretizado em categorias, o
    que descaracterizaria o problema. `drop` remove colunas que vazariam o alvo.
    """
    import pandas as pd
    from sklearn.datasets import fetch_openml

    d = (fetch_openml(data_id=data_id, as_frame=True, parser="auto") if data_id
         else fetch_openml(name, version=version, as_frame=True, parser="auto"))
    df, y = d.data.copy(), d.target

    if drop:
        df = df.drop(columns=[c for c in drop if c in df.columns])

    # categóricas → códigos inteiros (mesma convenção do loader de classificação)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.factorize(df[col])[0]

    X = df.to_numpy(dtype=float)
    y = np.asarray(y).astype(float)
    X, y = _clean(X, y)
    return _subsample(X, y, max_samples)


# ── os 8 datasets do estudo ───────────────────────────────────────────────────

def get_diabetes() -> Dataset:
    """sklearn built-in — progressão da diabetes em 1 ano (442 × 10)."""
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    return _clean(d.data, d.target)


def get_california() -> Dataset:
    """Valor mediano das casas na Califórnia (20.640 × 8 → subamostrado)."""
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing()
    X, y = _clean(d.data, d.target)
    return _subsample(X, y)


def get_airfoil() -> Dataset:
    """Ruído de aerofólio NASA — pressão sonora em dB (1.503 × 5)."""
    return _fetch_reg(data_id=44957)


def get_energy() -> Dataset:
    """Eficiência energética — carga de aquecimento (768 × 8).

    A carga de refrigeração (segunda saída do mesmo problema) já não está
    presente nesta versão; se estivesse, vazaria o alvo.
    """
    return _fetch_reg(data_id=44960)


def get_concrete() -> Dataset:
    """Resistência à compressão do concreto em MPa (1.030 × 8)."""
    return _fetch_reg(data_id=44959)


def get_abalone() -> Dataset:
    """Idade do abalone em nº de anéis (4.177 × 8; 'Sex' vira código)."""
    return _fetch_reg(data_id=44956)


def get_auto_mpg() -> Dataset:
    """Consumo de automóveis em milhas por galão (398 × 7)."""
    return _fetch_reg("autoMpg")


def get_yacht() -> Dataset:
    """Resistência residual de cascos de veleiro (308 × 6)."""
    return _fetch_reg("yacht_hydrodynamics")


_REG_LOADERS = {
    "diabetes":   get_diabetes,
    "california": get_california,
    "airfoil":    get_airfoil,
    "energy":     get_energy,
    "concrete":   get_concrete,
    "abalone":    get_abalone,
    "auto_mpg":   get_auto_mpg,
    "yacht":      get_yacht,
}

REGRESSION_DATASETS = list(_REG_LOADERS)


def get_regression(name: str) -> Dataset:
    if name not in _REG_LOADERS:
        raise ValueError(f"dataset de regressão desconhecido: {name}. "
                         f"Opções: {REGRESSION_DATASETS}")
    return _REG_LOADERS[name]()


if __name__ == "__main__":
    print(f"{'dataset':14s} {'n':>6s} {'d':>4s} {'y.min':>10s} {'y.max':>10s} "
          f"{'y.média':>10s} {'y.desvio':>10s}")
    for name in REGRESSION_DATASETS:
        X, y = get_regression(name)
        print(f"{name:14s} {len(X):6d} {X.shape[1]:4d} {y.min():10.3f} {y.max():10.3f} "
              f"{y.mean():10.3f} {y.std():10.3f}")
