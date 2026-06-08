"""
Utilidades compartilhadas do estudo comparativo v2.

Desenho experimental:
  - address_size e resolution FIXOS e IGUAIS para os 3 modelos (isola a regra de update).
  - arquitetura escolhida pelo "joelho" do custo-benefício da WiSARD clássica
    (menor/mais rápida que chega a 1pp do pico de acurácia).
  - 80% treino / 20% teste.  MNIST limitado a 10.000 exemplos.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split

from wisard.models_v2 import ClassicWiSARD, KalmanV2
from datasets.loaders import get_wine, get_mnist

# ── configuração global ──────────────────────────────────────────────────────────

TEST_SIZE   = 0.20
GRID_SEEDS  = list(range(5))    # busca de hiperparâmetros
FINAL_SEEDS = list(range(30))   # comparação final + testes de significância
CURVE_SEEDS = list(range(10))   # learning curve / épocas

PALETTE = {
    "Classica":         "#4C72B0",   # WiSARD clássica (bleach=0)
    "Classica+Bleach":  "#55A868",   # WiSARD clássica com bleaching
    "KalmanWiSARD":     "#DD8452",   # KWiSARD
}
MODEL_LABELS = {
    "Classica":        "WiSARD\nClássica",
    "Classica+Bleach": "Clássica\n+ Bleach",
    "KalmanWiSARD":    "Kalman\nWiSARD",
}

ROOT = Path(__file__).resolve().parents[1]
FIG  = ROOT / "results" / "figures" / "v2"
CFG  = ROOT / "results" / "v2"
FIG.mkdir(parents=True, exist_ok=True)
CFG.mkdir(parents=True, exist_ok=True)


def get_data(name: str):
    if name == "wine":
        return get_wine()
    if name == "mnist":
        return get_mnist(max_samples=10_000)
    raise ValueError(name)


def split(X, y, seed):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)


# ── avaliação genérica ───────────────────────────────────────────────────────────

def eval_over_seeds(make_model, X, y, seeds):
    """make_model(seed) -> modelo. Retorna dict com listas por seed."""
    accs, train_t, mem = [], [], []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = split(X, y, seed)
        model = make_model(seed)
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_t.append(time.perf_counter() - t0)
        accs.append(float((model.predict(X_te) == y_te).mean()))
        mem.append(model.memory_bytes())
    return {"acc": np.array(accs), "train_t": np.array(train_t), "mem": np.array(mem)}


# ── seleção de arquitetura (joelho do custo-benefício da clássica) ────────────────

def select_architecture(X, y, address_sizes, resolutions, seeds, tol=0.01):
    """Varre address_size × resolution na ClassicWiSARD (bleach=0).
    Escolhe a config mais barata (menor tempo de treino) dentro de `tol`
    do pico de acurácia. Retorna (best_addr, best_res, tabela)."""
    table = []
    for a in address_sizes:
        for r in resolutions:
            res = eval_over_seeds(
                lambda s, a=a, r=r: ClassicWiSARD(address_size=a, resolution=r, seed=s, bleach=0),
                X, y, seeds,
            )
            table.append({
                "address_size": a, "resolution": r,
                "mean_acc": res["acc"].mean(),
                "mean_train_t": res["train_t"].mean(),
            })
    peak = max(row["mean_acc"] for row in table)
    candidates = [row for row in table if row["mean_acc"] >= peak - tol]
    best = min(candidates, key=lambda row: row["mean_train_t"])
    return best["address_size"], best["resolution"], table


# ── confiança / curva de rejeição ────────────────────────────────────────────────

def accuracy_rejection_curve(confidence, correct, fractions):
    """Para cada fração rejeitada (descartando os menos confiantes),
    retorna a acurácia nas amostras remanescentes."""
    order = np.argsort(confidence)        # menos confiantes primeiro
    correct_sorted = np.asarray(correct)[order]
    n = len(correct_sorted)
    accs = []
    for f in fractions:
        keep = correct_sorted[int(f * n):]   # descarta a fração menos confiante
        accs.append(keep.mean() if len(keep) else np.nan)
    return np.array(accs)
