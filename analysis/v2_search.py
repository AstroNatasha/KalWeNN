"""
Etapa 1 — Seleção de arquitetura + busca de hiperparâmetros.

Para cada dataset:
  1. Escolhe address_size/resolution pelo joelho do custo-benefício da clássica.
  2. Grid search dos hiperparâmetros SÓ do Kalman (q, r), na arquitetura fixa.
  3. Varre o bleaching b da clássica (reaproveitando 1 treino por seed).
  4. Salva a config escolhida em results/v2/{ds}_config.json.

Uso:  python3 analysis/v2_search.py wine
      python3 analysis/v2_search.py mnist
"""

import sys
import json
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from v2_common import (
    get_data, split, eval_over_seeds, select_architecture,
    GRID_SEEDS, PALETTE, FIG, CFG,
)
from wisard.models_v2 import ClassicWiSARD, KalmanV2

ADDR_SIZES  = [2, 3, 4, 5, 6, 7, 8]
RESOLUTIONS = [4, 8, 12, 16, 24, 32]

Q_VALS = [1e-4, 1e-3, 1e-2, 1e-1]
R_VALS = [1e-3, 1e-2, 5e-2, 1e-1]

BLEACH_VALS = [0, 1, 2, 3, 5, 8, 12, 20, 30, 50]


def savefig(ds, name):
    d = FIG / ds
    d.mkdir(parents=True, exist_ok=True)
    plt.savefig(d / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    salvo → {d / f'{name}.png'}")


# ── grid search Kalman (q × r) na arquitetura fixa ───────────────────────────────

def kalman_grid(X, y, addr, res, seeds):
    print("  [Kalman] grid q × r...")
    grid = np.zeros((len(Q_VALS), len(R_VALS)))
    best = (-1, None)
    for i, q in enumerate(Q_VALS):
        for j, r in enumerate(R_VALS):
            out = eval_over_seeds(
                lambda s, q=q, r=r: KalmanV2(address_size=addr, resolution=res,
                                             q=q, r=r, p0=1.0, n_epochs=1, seed=s),
                X, y, seeds)
            acc = out["acc"].mean()
            grid[i, j] = acc
            if acc > best[0]:
                best = (acc, {"q": q, "r": r})
    return grid, best


def plot_kalman_heatmap(grid, ds):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(R_VALS))); ax.set_yticks(range(len(Q_VALS)))
    ax.set_xticklabels([f"{v:.0e}" for v in R_VALS])
    ax.set_yticklabels([f"{v:.0e}" for v in Q_VALS])
    ax.set_xlabel("r (ruído de observação)"); ax.set_ylabel("q (ruído de processo)")
    ax.set_title(f"#9 Robustez do Kalman a q × r — {ds}\n(heatmap plano = robusto)")
    for i in range(len(Q_VALS)):
        for j in range(len(R_VALS)):
            ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); savefig(ds, "9_kalman_qr_heatmap")


# ── varredura de bleaching (1 treino por seed, reaproveitado) ─────────────────────

def bleach_sweep(X, y, addr, res, seeds):
    print("  [Clássica] varredura de bleaching...")
    accs = np.zeros((len(seeds), len(BLEACH_VALS)))
    for si, seed in enumerate(seeds):
        X_tr, X_te, y_tr, y_te = split(X, y, seed)
        model = ClassicWiSARD(address_size=addr, resolution=res, seed=seed).fit(X_tr, y_tr)
        for bi, b in enumerate(BLEACH_VALS):
            model.set_bleach(b)
            accs[si, bi] = (model.predict(X_te) == y_te).mean()
    mean_acc = accs.mean(axis=0)
    best_bi = int(mean_acc.argmax())
    return mean_acc, accs.std(axis=0), BLEACH_VALS[best_bi]


def run(ds):
    print(f"\n{'='*60}\n  BUSCA — {ds.upper()}\n{'='*60}")
    X, y = get_data(ds)

    print("  Selecionando arquitetura (joelho da clássica)...")
    addr, res, table = select_architecture(X, y, ADDR_SIZES, RESOLUTIONS, GRID_SEEDS)
    print(f"    → address_size={addr}, resolution={res}")
    pd.DataFrame(table).to_csv(CFG / f"{ds}_arch_table.csv", index=False)

    grid, (k_acc, k_params) = kalman_grid(X, y, addr, res, GRID_SEEDS)
    plot_kalman_heatmap(grid, ds)
    print(f"    melhor Kalman: {k_params} (acc grid={k_acc:.4f})")

    bleach_mean, bleach_std, best_b = bleach_sweep(X, y, addr, res, GRID_SEEDS)
    print(f"    melhor bleaching: b={best_b}")

    config = {
        "dataset": ds,
        "address_size": addr,
        "resolution": res,
        "kalman": {**k_params, "p0": 1.0, "n_epochs": 1},
        "best_bleach": best_b,
        "bleach_vals": BLEACH_VALS,
        "bleach_mean_acc": bleach_mean.tolist(),
        "bleach_std_acc": bleach_std.tolist(),
        "kalman_grid": grid.tolist(),
        "q_vals": Q_VALS, "r_vals": R_VALS,
    }
    with open(CFG / f"{ds}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  config salva → {CFG / f'{ds}_config.json'}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "wine"
    run(ds)
