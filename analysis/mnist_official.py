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

# ── Clássica (b=0) e Kalman, ambos com N_SEEDS seeds ──────────────────────────
N_SEEDS = 30

acc_c, acc_k = [], []
for seed in range(N_SEEDS):
    c = SparseClassicWiSARD(ADDR, RES, seed=seed).fit(X_tr, y_tr)
    acc_c.append(float((c.predict(X_te) == y_te).mean()))
    m = SparseKalman(ADDR, RES, seed=seed,
                     q=k["q"], r=k["r"], p0=1.0, n_epochs=1).fit(X_tr, y_tr)
    acc_k.append(float((m.predict(X_te) == y_te).mean()))
    print(f"  seed {seed:2d}: clássica={acc_c[-1]:.4f} | kalman={acc_k[-1]:.4f}")

acc_c, acc_k = np.array(acc_c), np.array(acc_k)
print(f"\n{'='*60}")
for name, a in [("Clássica b=0", acc_c), ("Kalman", acc_k)]:
    cv = a.std() / a.mean() * 100
    print(f"{name:14s}: média={a.mean():.4f} ± {a.std():.4f} | "
          f"CV={cv:.2f}% | pior={a.min():.4f} | melhor={a.max():.4f}")
