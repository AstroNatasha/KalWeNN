"""
Comparativo de regimes de arquitetura — tuplas pequenas vs grandes.

Para cada dataset, em DOIS regimes de (tupla, termômetro):
  - pequeno: tupla=8,  termômetro=16   → memória DENSA
  - grande : tupla=20, termômetro=20   → memória ESPARSA
             (exceto MNIST no regime grande: tupla=28, termômetro=3)
roda:
  - WiSARD clássica (b=0)
  - Kalman-WiSARD com grid search de (q, r) sobre a tupla/termômetro FIXOS do regime

Split aleatório 80/20 com seeds.

Saída: UM relatório (CSV + tabela no terminal), SEM figuras, com colunas:
  dataset | regime | modelo | tupla | termometro | acuracia | desvio | CV_% |
  treino_s | teste_s | memoria_MB

Uso:  python3 analysis/v2_tuple_comparison.py                 (lista padrão)
      python3 analysis/v2_tuple_comparison.py wine iris mnist (datasets específicos)
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from v2_common import get_data, split, CFG
from datasets.loaders import get_mnist
from wisard.models_v2 import ClassicWiSARD, KalmanV2               # denso (tupla pequena)
from wisard.models_sparse import SparseClassicWiSARD, SparseKalman  # esparso (tupla grande)

MNIST_SET = ("mnist", "mnist_full")  # usam o split OFICIAL 60k/10k, não o 80/20

# ── configuração ──────────────────────────────────────────────────────────────

DATASETS_PADRAO = ["wine", "iris", "breast_cancer", "digits",
                   "vehicle", "segment", "vowel", "mnist"]

SMALL = (8, 16)           # (tupla, termômetro) — DENSO
LARGE = (20, 20)          # (tupla, termômetro) — ESPARSO
LARGE_MNIST = (28, 3)     # override só para mnist / mnist_full no regime grande

Q_GRID = [1e-4, 1e-3, 1e-2]
R_GRID = [1e-2, 1e-1, 5e-1, 1.0, 2.0]

SEEDS = list(range(int(os.environ.get("WISARD_SEEDS", "10"))))           # avaliação final
GRID_SEEDS = list(range(int(os.environ.get("WISARD_GRID_SEEDS", "3"))))  # busca q,r


def regime_arch(ds, regime):
    if regime == "pequeno":
        return SMALL
    return LARGE_MNIST if ds in ("mnist", "mnist_full") else LARGE


def models_for(regime):
    """Classes a usar: denso no regime pequeno, esparso no grande."""
    if regime == "pequeno":
        return ClassicWiSARD, KalmanV2
    return SparseClassicWiSARD, SparseKalman


def get_split(ds, X, y, seed):
    """MNIST: split OFICIAL fixo (60k/10k). Demais: aleatório 80/20 por seed."""
    if ds in MNIST_SET:
        return X[:60000], X[60000:], y[:60000], y[60000:]
    return split(X, y, seed)


def eval_model(make, ds, X, y, seeds):
    """make(seed)->modelo. Retorna acurácia média/desvio/cv + tempos + memória."""
    accs, tr, te, mem = [], [], [], []
    for s in seeds:
        X_tr, X_te, y_tr, y_te = get_split(ds, X, y, s)
        m = make(s)
        t0 = time.perf_counter(); m.fit(X_tr, y_tr); tr.append(time.perf_counter() - t0)
        t1 = time.perf_counter(); pred = m.predict(X_te); te.append(time.perf_counter() - t1)
        accs.append(float((pred == y_te).mean())); mem.append(m.memory_bytes())
    accs = np.array(accs)
    return {
        "acc": accs.mean(), "std": accs.std(),
        "cv": accs.std() / accs.mean() * 100 if accs.mean() else 0.0,
        "train_t": float(np.mean(tr)), "test_t": float(np.mean(te)),
        "mem": float(np.mean(mem)),
    }


def best_qr(KalmanCls, ds, X, y, addr, res):
    """Grid de (q,r) na arquitetura fixa; retorna o melhor par por acurácia média."""
    best = (-1.0, None)
    for q in Q_GRID:
        for r in R_GRID:
            out = eval_model(
                lambda s, q=q, r=r: KalmanCls(addr, res, seed=s,
                                              q=q, r=r, p0=1.0, n_epochs=1),
                ds, X, y, GRID_SEEDS)
            if out["acc"] > best[0]:
                best = (out["acc"], (q, r))
    return best[1]


def main(datasets):
    rows = []
    for ds in datasets:
        print(f"\n===== {ds} =====")
        # MNIST: carrega os 70k completos (split oficial é aplicado em get_split)
        X, y = get_mnist(max_samples=None) if ds in MNIST_SET else get_data(ds)
        if ds in MNIST_SET:
            print("  (MNIST: split OFICIAL 60k/10k, sem 80/20)")
        for regime in ("pequeno", "grande"):
            addr, res = regime_arch(ds, regime)
            Classic, Kalman = models_for(regime)
            backend = "denso" if regime == "pequeno" else "esparso"
            print(f"  [{regime} / {backend}] tupla={addr}, termometro={res}")

            print("    clássica...")
            rc = eval_model(lambda s: Classic(addr, res, seed=s, bleach=0), ds, X, y, SEEDS)
            rows.append({"dataset": ds, "regime": regime, "modelo": "Clássica",
                         "tupla": addr, "termometro": res, **rc})

            print("    kalman (grid q,r)...")
            q, r = best_qr(Kalman, ds, X, y, addr, res)
            rk = eval_model(lambda s: Kalman(addr, res, seed=s,
                                             q=q, r=r, p0=1.0, n_epochs=1), ds, X, y, SEEDS)
            print(f"      melhor (q,r)=({q:g},{r:g})")
            rows.append({"dataset": ds, "regime": regime, "modelo": "Kalman",
                         "tupla": addr, "termometro": res, **rk})

    table = pd.DataFrame(rows)
    rep = table.rename(columns={
        "acc": "acuracia", "std": "desvio", "cv": "CV_%",
        "train_t": "treino_s", "test_t": "teste_s"})
    rep["acuracia"] = rep["acuracia"].round(4)
    rep["desvio"] = rep["desvio"].round(4)
    rep["CV_%"] = rep["CV_%"].round(3)
    rep["treino_s"] = rep["treino_s"].round(4)
    rep["teste_s"] = rep["teste_s"].round(4)
    rep["memoria_MB"] = (rep["mem"] / 1e6).round(4)
    rep = rep[["dataset", "regime", "modelo", "tupla", "termometro",
               "acuracia", "desvio", "CV_%", "treino_s", "teste_s", "memoria_MB"]]

    out = CFG / "tuple_comparison_report.csv"
    rep.to_csv(out, index=False)
    print(f"\n{'='*70}\nRelatório salvo → {out}\n{'='*70}\n")
    print(rep.to_string(index=False))


if __name__ == "__main__":
    ds_list = sys.argv[1:] if len(sys.argv) > 1 else DATASETS_PADRAO
    main(ds_list)
