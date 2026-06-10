"""
TESTE (descartável) — reduções de memória do Kalman, sem alterar o pipeline.

Ideia 1 (lossless): na inferência o p NÃO é usado → após treinar, descartar o p.
                     Memória cai de 2x (x+p) para 1x (só x) — igual à clássica.
Ideia 2 (lossy):     quantizar o x (float64 → float16 → int8 → binário) e medir
                     quanta memória cai e qual o custo em acurácia.

Não importa nada do pipeline além de ler a config. Pode apagar este arquivo
sem efeito colateral.

Uso:  python3 analysis/v2_memory_experiment.py            (wine)
      python3 analysis/v2_memory_experiment.py mnist
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from v2_common import get_data, split, CFG
from wisard.models_v2 import KalmanV2


def acc_with_x(model, x_mem, X_te, y_te):
    """Avalia a acurácia usando uma versão alternativa do x (sem retreinar)."""
    old = model._score_memory
    model._score_memory = x_mem
    acc = float((model.predict(X_te) == y_te).mean())
    model._score_memory = old
    return acc


def run(ds):
    cfg = json.load(open(CFG / f"{ds}_config.json"))
    a, res, k = cfg["address_size"], cfg["resolution"], cfg["kalman"]
    X, y = get_data(ds)
    X_tr, X_te, y_tr, y_te = split(X, y, 0)

    m = KalmanV2(address_size=a, resolution=res, seed=0,
                 q=k["q"], r=k["r"], p0=k["p0"], n_epochs=k["n_epochs"]).fit(X_tr, y_tr)
    x = m._memory_x
    n = x.size  # nº de células

    x16 = x.astype(np.float16).astype(np.float64)
    xi8 = np.round(x * 255).astype(np.uint8).astype(np.float64) / 255.0
    xbin = (x > 0.5).astype(np.float64)

    variants = [
        ("full  (x + p) float64",   2 * n * 8, x,    "baseline atual (2x)"),
        ("compact (só x) float64",  1 * n * 8, x,    "descarta p — LOSSLESS"),
        ("x float16",               1 * n * 2, x16,  "metade do compact"),
        ("x int8 (0..255)",         1 * n * 1, xi8,  "1/8 do baseline"),
        ("x binário (x>0.5)",       n / 8.0,   xbin, "vira ~clássica binária"),
    ]

    base = 2 * n * 8
    print(f"\n== {ds}  (tupla={a}, termômetro={res}, células={n:,}) ==")
    print(f"  {'variante':24s} | {'memória':>11s} | {'% do baseline':>13s} | {'acurácia':>8s} | obs")
    for name, mem, xmem, obs in variants:
        acc = acc_with_x(m, xmem, X_te, y_te)
        print(f"  {name:24s} | {mem/1e6:8.3f} MB | {mem/base*100:11.1f}% | {acc:8.4f} | {obs}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "wine")
