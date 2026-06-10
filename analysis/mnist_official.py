"""
TESTE (descartável) — MNIST com o split OFICIAL (60k treino / 10k teste).

O MNIST vem pré-dividido: as primeiras 60.000 amostras são treino e as
últimas 10.000 são teste (ordem canônica do openml 'mnist_784'). Aqui não
fazemos split aleatório — usamos exatamente essa divisão, para a acurácia
ficar comparável à literatura.

Arquitetura: tupla=28, termômetro=3 (sparse). Hiperparâmetros do Kalman lidos
da config da busca. Não toca em nada do pipeline. Pode apagar depois.

Uso:  python3 analysis/mnist_official.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from datasets.loaders import get_mnist
from wisard.models_sparse import SparseClassicWiSARD, SparseKalman

ADDR, RES = 28, 3

cfg = json.load(open("results/v2/mnist_full_sparse_config.json"))
k = cfg["kalman"]
print(f"Kalman da config: q={k['q']}, r={k['r']}\n")

# ── split OFICIAL (sem embaralhar) ────────────────────────────────────────────
X, y = get_mnist(max_samples=None)  # 70.000, ordem canônica
X_tr, y_tr = X[:60000], y[:60000]
X_te, y_te = X[60000:], y[60000:]
print(f"treino oficial = {len(X_tr)} | teste oficial = {len(X_te)}\n")

# ── Clássica (b=0) ────────────────────────────────────────────────────────────
c = SparseClassicWiSARD(ADDR, RES, seed=0).fit(X_tr, y_tr)
print(f"Clássica b=0: {(c.predict(X_te) == y_te).mean():.4f}\n")

# ── Kalman (3 seeds, para ver estabilidade) ───────────────────────────────────
for seed in range(3):
    m = SparseKalman(ADDR, RES, seed=seed,
                     q=k["q"], r=k["r"], p0=1.0, n_epochs=1).fit(X_tr, y_tr)
    print(f"Kalman seed {seed}: {(m.predict(X_te) == y_te).mean():.4f}")
