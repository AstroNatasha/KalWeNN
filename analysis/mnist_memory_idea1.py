"""
TESTE (descartável) — Ideia 1 de redução de memória no MNIST: descartar o p.

Na inferência o p NÃO é usado (a predição soma só o x). Logo, após o treino,
o p pode ser descartado: a memória do Kalman cai de 2 valores por célula (x, p)
para 1 (só x) — ficando IGUAL à clássica, sem nenhuma perda de acurácia.

Split oficial do MNIST (60k/10k), tupla=28, termômetro=3 (sparse).
Não toca no pipeline. Pode apagar depois.

Uso:  python3 analysis/mnist_memory_idea1.py
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

X, y = get_mnist(max_samples=None)
X_tr, y_tr = X[:60000], y[:60000]
X_te, y_te = X[60000:], y[60000:]

c = SparseClassicWiSARD(ADDR, RES, seed=0).fit(X_tr, y_tr)
m = SparseKalman(ADDR, RES, seed=0,
                 q=k["q"], r=k["r"], p0=1.0, n_epochs=1).fit(X_tr, y_tr)

# nº de células efetivamente armazenadas (endereços vistos)
n_c = sum(len(d) for rams in c._counts for d in rams)
n_k = sum(len(d) for rams in m._mem for d in rams)

acc_c = float((c.predict(X_te) == y_te).mean())
acc_k = float((m.predict(X_te) == y_te).mean())   # usa só x → já é o "compact"

B = 8  # bytes por parâmetro (float64)
mb = lambda p: p * B / 1e6

print(f"\n== MNIST oficial | tupla={ADDR}, termômetro={RES} ==")
print(f"células armazenadas: clássica={n_c:,} | kalman={n_k:,}\n")
print(f"{'modelo':24s} | {'parâmetros':>12s} | {'memória':>11s} | acurácia")
print(f"{'Clássica b=0':24s} | {n_c:12,} | {mb(n_c):8.2f} MB | {acc_c:.4f}")
print(f"{'Kalman full (x + p)':24s} | {2*n_k:12,} | {mb(2*n_k):8.2f} MB | {acc_k:.4f}")
print(f"{'Kalman compact (só x)':24s} | {n_k:12,} | {mb(n_k):8.2f} MB | {acc_k:.4f}  (p descartado — idêntico)")
print(f"\n→ Descartar o p corta 50% da memória do Kalman, igualando o footprint")
print(f"  da clássica ({mb(n_k):.2f} ≈ {mb(n_c):.2f} MB), sem perda de acurácia.")
