"""
Grid search de hiperparâmetros — SimpleWiSARD vs KalmanWiSARD
Para cada dataset: encontra o melhor de cada modelo e os compara.

Espaço de busca:
  Simple : address_size × resolution
  Kalman : address_size × resolution × q × r

Grid search com GRID_SEEDS seeds; comparação final com FINAL_SEEDS seeds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from wisard.models import SimpleWiSARD, KalmanWiSARD
from datasets.loaders import get_wine, get_mnist

# ── configuração ───────────────────────────────────────────────────────────────

GRID_SEEDS  = list(range(5))   # seeds usadas no grid search
FINAL_SEEDS = list(range(20))  # seeds usadas na comparação final

SIMPLE_GRID = {
    "address_size": [2, 3, 4, 5, 6, 7, 8],
    "resolution":   [4, 8, 12, 16, 24, 32],
}

KALMAN_GRID = {
    "address_size": [3, 4, 5, 6],
    "resolution":   [8, 16, 32],
    "q":            [1e-4, 1e-3, 1e-2, 1e-1],
    "r":            [1e-3, 1e-2, 5e-2, 1e-1],
    "n_epochs":     [1],
    "p0":           [1.0],
}

PALETTE = {"SimpleWiSARD": "#4C72B0", "KalmanWiSARD": "#DD8452"}
OUT = Path("results/figures/best_vs_best")
OUT.mkdir(parents=True, exist_ok=True)


def savefig(name):
    path = OUT / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    salvo → {path}")


# ── avaliação ──────────────────────────────────────────────────────────────────

def evaluate(model_cls, X, y, seeds, **params):
    accs = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=seed, stratify=y)
        m = model_cls(seed=seed, **params)
        m.fit(X_train, y_train)
        accs.append((m.predict(X_test) == y_test).mean())
    return float(np.mean(accs)), float(np.std(accs))


def grid_search(model_cls, grid, X, y, seeds):
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    best_acc, best_params, results = -1, None, []
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        mean_acc, std_acc = evaluate(model_cls, X, y, seeds, **params)
        results.append({**params, "mean_acc": mean_acc, "std_acc": std_acc})
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_params = params
        if (i + 1) % 20 == 0:
            print(f"      {i+1}/{len(combos)} combinações testadas...")
    return best_params, best_acc, pd.DataFrame(results)


# ── gráficos ───────────────────────────────────────────────────────────────────

def plot_heatmap_simple(df_results, dataset_name):
    pivot = df_results.pivot_table(
        index="address_size", columns="resolution", values="mean_acc")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("resolution"); ax.set_ylabel("address_size")
    ax.set_title(f"Grid Search — SimpleWiSARD / {dataset_name}")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i,j]:.3f}",
                    ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    savefig(f"{dataset_name}_simple_heatmap")


def plot_heatmap_kalman(df_results, dataset_name):
    # marginaliza sobre address_size e resolution, mostra q × r
    pivot = df_results.groupby(["q", "r"])["mean_acc"].mean().unstack("r")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"{v:.0e}" for v in pivot.columns], fontsize=8)
    ax.set_yticklabels([f"{v:.0e}" for v in pivot.index], fontsize=8)
    ax.set_xlabel("r (ruído de observação)"); ax.set_ylabel("q (ruído de processo)")
    ax.set_title(f"Grid Search — KalmanWiSARD q×r / {dataset_name}\n(média sobre address_size e resolution)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i,j]:.3f}",
                    ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    savefig(f"{dataset_name}_kalman_heatmap_qr")


def plot_final_comparison(results_final, dataset_name):
    models = ["SimpleWiSARD", "KalmanWiSARD"]
    means  = [results_final[m]["mean"] for m in models]
    stds   = [results_final[m]["std"]  for m in models]
    n = len(FINAL_SEEDS)
    cis = [1.96 * s / np.sqrt(n) for s in stds]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Melhor Simple vs Melhor Kalman — {dataset_name} ({len(FINAL_SEEDS)} seeds)",
                 fontsize=12, fontweight="bold")

    # barras com IC
    ax = axes[0]
    bars = ax.bar(models, means, yerr=cis, capsize=10, width=0.5,
                  color=[PALETTE[m] for m in models], alpha=0.85, error_kw={"linewidth": 2})
    ax.set_ylabel("Acurácia média"); ax.set_title("Acurácia ± IC 95%")
    ax.set_xticklabels(["Melhor\nSimple", "Melhor\nKalman"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + max(cis) * 0.1 + 0.003,
                f"{mean:.2%}\n(±{std:.3f})", ha="center", va="bottom", fontsize=9)

    # boxplot com todos os runs
    ax = axes[1]
    bp = ax.boxplot([results_final[m]["all_accs"] for m in models],
                    patch_artist=True, widths=0.5)
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(PALETTE[m]); patch.set_alpha(0.8)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Melhor\nSimple", "Melhor\nKalman"])
    ax.set_ylabel("Acurácia"); ax.set_title("Distribuição (20 seeds)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    plt.tight_layout()
    savefig(f"{dataset_name}_best_vs_best")


def print_best_params(name, params, mean_acc, final_mean, final_std):
    print(f"\n  {name}:")
    print(f"    Melhores params : {params}")
    print(f"    Acurácia (grid) : {mean_acc:.4f}")
    print(f"    Acurácia (final): {final_mean:.4f} ± {final_std:.4f}")


# ── pipeline por dataset ───────────────────────────────────────────────────────

def run_dataset(name, X, y):
    print(f"\n{'='*60}")
    print(f"  {name.upper()}")
    print(f"{'='*60}")

    print(f"\n  [Simple] Grid search ({len(list(itertools.product(*SIMPLE_GRID.values())))} combos × {len(GRID_SEEDS)} seeds)...")
    best_s_params, best_s_acc, df_s = grid_search(SimpleWiSARD, SIMPLE_GRID, X, y, GRID_SEEDS)

    n_kalman = len(list(itertools.product(*KALMAN_GRID.values())))
    print(f"\n  [Kalman] Grid search ({n_kalman} combos × {len(GRID_SEEDS)} seeds)...")
    best_k_params, best_k_acc, df_k = grid_search(KalmanWiSARD, KALMAN_GRID, X, y, GRID_SEEDS)

    print(f"\n  Melhor Simple (grid): acc={best_s_acc:.4f} | params={best_s_params}")
    print(f"  Melhor Kalman (grid): acc={best_k_acc:.4f} | params={best_k_params}")

    # gráficos de grid
    plot_heatmap_simple(df_s, name)
    plot_heatmap_kalman(df_k, name)

    # avaliação final com 20 seeds
    print(f"\n  Avaliação final com {len(FINAL_SEEDS)} seeds...")
    accs_s, accs_k = [], []
    for seed in FINAL_SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=seed, stratify=y)

        ms = SimpleWiSARD(seed=seed, **best_s_params)
        ms.fit(X_train, y_train)
        accs_s.append((ms.predict(X_test) == y_test).mean())

        mk = KalmanWiSARD(seed=seed, **best_k_params)
        mk.fit(X_train, y_train)
        accs_k.append((mk.predict(X_test) == y_test).mean())

    results_final = {
        "SimpleWiSARD": {"mean": np.mean(accs_s), "std": np.std(accs_s), "all_accs": accs_s},
        "KalmanWiSARD": {"mean": np.mean(accs_k), "std": np.std(accs_k), "all_accs": accs_k},
    }

    print_best_params("SimpleWiSARD", best_s_params, best_s_acc,
                      results_final["SimpleWiSARD"]["mean"], results_final["SimpleWiSARD"]["std"])
    print_best_params("KalmanWiSARD", best_k_params, best_k_acc,
                      results_final["KalmanWiSARD"]["mean"], results_final["KalmanWiSARD"]["std"])

    plot_final_comparison(results_final, name)

    # salva resultados em CSV
    summary = pd.DataFrame([
        {"dataset": name, "model": "SimpleWiSARD", "best_params": str(best_s_params),
         "grid_acc": best_s_acc, "final_mean": results_final["SimpleWiSARD"]["mean"],
         "final_std": results_final["SimpleWiSARD"]["std"]},
        {"dataset": name, "model": "KalmanWiSARD", "best_params": str(best_k_params),
         "grid_acc": best_k_acc, "final_mean": results_final["KalmanWiSARD"]["mean"],
         "final_std": results_final["KalmanWiSARD"]["std"]},
    ])
    csv_path = Path("results") / f"{name}_best_params.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\n  Resultados salvos em {csv_path}")

    return results_final


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Grid Search: Melhor Simple vs Melhor Kalman ===")

    print("\nCarregando Wine...")
    X_wine, y_wine = get_wine()
    wine_results = run_dataset("wine", X_wine, y_wine)

    print("\nCarregando MNIST (pode demorar)...")
    X_mnist, y_mnist = get_mnist()
    mnist_results = run_dataset("mnist", X_mnist, y_mnist)

    # comparativo final entre datasets
    print("\n\n=== RESUMO FINAL ===")
    for ds, res in [("Wine", wine_results), ("MNIST", mnist_results)]:
        s = res["SimpleWiSARD"]
        k = res["KalmanWiSARD"]
        print(f"\n  {ds}:")
        print(f"    Melhor Simple : {s['mean']:.4f} ± {s['std']:.4f}")
        print(f"    Melhor Kalman : {k['mean']:.4f} ± {k['std']:.4f}")
        print(f"    Ganho Kalman  : {(k['mean']-s['mean'])*100:+.2f}pp")

    print(f"\nTodos os gráficos em {OUT}/")


if __name__ == "__main__":
    main()
