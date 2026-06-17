"""
Análise completa do MNIST — SimpleWiSARD vs KalmanWiSARD
(estrutura idêntica ao wine_analysis.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.model_selection import train_test_split

from wisard.models import SimpleWiSARD, KalmanWiSARD
from wisard.encoder import build_encoders, apply_encoders
from datasets.loaders import get_mnist

PALETTE = {"SimpleWiSARD": "#4C72B0", "KalmanWiSARD": "#DD8452"}
OUT = Path("results/figures/mnist")
OUT.mkdir(parents=True, exist_ok=True)

SIMPLE_PARAMS = dict(address_size=4, resolution=16)
KALMAN_PARAMS = dict(address_size=4, resolution=16, q=1e-3, r=5e-2, p0=1.0, n_epochs=1)
SEEDS = list(range(20))
CLASS_NAMES = [str(i) for i in range(10)]


def savefig(name):
    path = OUT / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  salvo → {path}")


def load_raw():
    return pd.read_csv("results/mnist_raw.csv")


# ── 1. Comparação de desempenho ───────────────────────────────────────────────

def sec1_performance(df):
    print("\n[1] Comparação de desempenho")
    models = ["SimpleWiSARD", "KalmanWiSARD"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("1. Comparação de Desempenho — MNIST", fontsize=13, fontweight="bold")

    ax = axes[0]
    bp = ax.boxplot([df[df.model == m]["accuracy"].values for m in models],
                    patch_artist=True, widths=0.5)
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(PALETTE[m]); patch.set_alpha(0.8)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Simple\nWiSARD", "Kalman\nWiSARD"])
    ax.set_ylabel("Acurácia"); ax.set_title("a) Boxplot de Acurácia")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    ax = axes[1]
    means = [df[df.model == m]["accuracy"].mean() for m in models]
    stds  = [df[df.model == m]["accuracy"].std()  for m in models]
    cis   = [1.96 * s / np.sqrt(len(SEEDS)) for s in stds]
    bars  = ax.bar(models, means, yerr=cis, capsize=8, width=0.5,
                   color=[PALETTE[m] for m in models], alpha=0.85, error_kw={"linewidth": 2})
    ax.set_ylabel("Acurácia média"); ax.set_title("b) Média ± IC 95%")
    ax.set_xticklabels(["Simple\nWiSARD", "Kalman\nWiSARD"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.005,
                f"{mean:.2%}", ha="center", va="bottom", fontsize=9)

    ax = axes[2]
    for m in models:
        vals = df[df.model == m].sort_values("seed")["accuracy"].values
        ax.plot(SEEDS, vals, marker="o", label=m, color=PALETTE[m], linewidth=2)
    ax.set_xlabel("Seed"); ax.set_ylabel("Acurácia"); ax.set_title("c) Acurácia por Seed")
    ax.legend(fontsize=8); ax.set_xticks(SEEDS)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    plt.tight_layout(); savefig("1_performance")


# ── 2. Trade-off desempenho × custo ─────────────────────────────────────────

def sec2_tradeoff(df):
    print("\n[2] Trade-off desempenho × custo")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("2. Trade-off Desempenho × Custo — MNIST", fontsize=13, fontweight="bold")

    ax = axes[0]
    for m in ["SimpleWiSARD", "KalmanWiSARD"]:
        sub = df[df.model == m]
        ax.scatter(sub["train_wall_s"], sub["accuracy"],
                   color=PALETTE[m], label=m, s=80, alpha=0.85, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("Tempo de treino (s)"); ax.set_ylabel("Acurácia")
    ax.set_title("a) Acurácia × Tempo de Treino"); ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    ax = axes[1]
    for m in ["SimpleWiSARD", "KalmanWiSARD"]:
        sub = df[df.model == m]
        ax.scatter(sub["train_peak_mb"], sub["accuracy"],
                   color=PALETTE[m], label=m, s=80, alpha=0.85, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("Memória de treino (MB)"); ax.set_ylabel("Acurácia")
    ax.set_title("b) Acurácia × Memória de Treino"); ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    ax = axes[2]
    for m in ["SimpleWiSARD", "KalmanWiSARD"]:
        sub = df[df.model == m]
        eff = sub["accuracy"] / sub["train_wall_s"]
        ax.bar(sub["seed"] + (0.2 if m == "KalmanWiSARD" else -0.2),
               eff, width=0.35, color=PALETTE[m], label=m, alpha=0.85)
    ax.set_xlabel("Seed"); ax.set_ylabel("Acurácia / s")
    ax.set_title("c) Eficiência por Seed"); ax.set_xticks(SEEDS); ax.legend(fontsize=8)

    plt.tight_layout(); savefig("2_tradeoff")


# ── 3. Estabilidade e variância ───────────────────────────────────────────────

def sec3_stability(df):
    print("\n[3] Estabilidade e variância")
    models = ["SimpleWiSARD", "KalmanWiSARD"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("3. Estabilidade e Variância — MNIST", fontsize=13, fontweight="bold")

    ax = axes[0]
    parts = ax.violinplot([df[df.model == m]["accuracy"].values for m in models],
                          positions=[1, 2], showmedians=True)
    for body, m in zip(parts["bodies"], models):
        body.set_facecolor(PALETTE[m]); body.set_alpha(0.7)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Simple\nWiSARD", "Kalman\nWiSARD"])
    ax.set_ylabel("Acurácia"); ax.set_title("a) Violin Plot")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    ax = axes[1]
    for m in models:
        vals = df[df.model == m]["accuracy"].values
        cv = vals.std() / vals.mean() * 100
        ax.bar(m, cv, color=PALETTE[m], alpha=0.85, width=0.5)
        ax.text(m, cv + 0.02, f"{cv:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("CV (%)"); ax.set_title("b) Coeficiente de Variação")
    ax.set_xticklabels(["Simple\nWiSARD", "Kalman\nWiSARD"])

    ax = axes[2]; ax.axis("off")
    s_acc = df[df.model == "SimpleWiSARD"]["accuracy"].values
    k_acc = df[df.model == "KalmanWiSARD"]["accuracy"].values
    t_stat, t_p = stats.ttest_ind(k_acc, s_acc, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(k_acc, s_acc, alternative="two-sided")
    lines = [
        "c) Testes Estatísticos", "",
        f"  SimpleWiSARD:",
        f"    Média : {s_acc.mean():.4f}",
        f"    Std   : {s_acc.std():.4f}",
        f"    CV    : {s_acc.std()/s_acc.mean()*100:.2f}%", "",
        f"  KalmanWiSARD:",
        f"    Média : {k_acc.mean():.4f}",
        f"    Std   : {k_acc.std():.4f}",
        f"    CV    : {k_acc.std()/k_acc.mean()*100:.2f}%", "",
        f"  t de Welch:",
        f"    t = {t_stat:.3f}",
        f"    p = {t_p:.4f}  {'*' if t_p < 0.05 else '(n.s.)'}", "",
        f"  Mann-Whitney U:",
        f"    U = {u_stat:.0f}",
        f"    p = {u_p:.4f}  {'*' if u_p < 0.05 else '(n.s.)'}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(); savefig("3_stability")


# ── 4. Hiperparâmetros do Kalman ──────────────────────────────────────────────

def _eval(cls, X, y, seeds, **params):
    accs = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=seed, stratify=y)
        m = cls(seed=seed, **params)
        m.fit(X_train, y_train)
        accs.append((m.predict(X_test) == y_test).mean())
    return float(np.mean(accs))


def sec4_hyperparams(X, y):
    print("\n[4] Hiperparâmetros (usando 5 seeds para agilizar...)")
    seeds5 = list(range(5))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("4. Análise de Hiperparâmetros — MNIST", fontsize=13, fontweight="bold")

    print("  4a. grid q × r...")
    ax = axes[0, 0]
    q_vals = [1e-4, 1e-3, 1e-2, 1e-1, 5e-1]
    r_vals = [1e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    grid = np.zeros((len(q_vals), len(r_vals)))
    for i, q in enumerate(q_vals):
        for j, r in enumerate(r_vals):
            grid[i, j] = _eval(KalmanWiSARD, X, y, seeds5,
                                address_size=4, resolution=16, q=q, r=r, p0=1.0, n_epochs=1)
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(r_vals))); ax.set_yticks(range(len(q_vals)))
    ax.set_xticklabels([f"{v:.0e}" for v in r_vals], fontsize=8)
    ax.set_yticklabels([f"{v:.0e}" for v in q_vals], fontsize=8)
    ax.set_xlabel("r (ruído de observação)"); ax.set_ylabel("q (ruído de processo)")
    ax.set_title("a) Heatmap Acurácia: q × r")
    for i in range(len(q_vals)):
        for j in range(len(r_vals)):
            ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    print("  4b. n_epochs...")
    ax = axes[0, 1]
    epochs_range = [1, 2, 3, 5]
    accs_k = [_eval(KalmanWiSARD, X, y, seeds5, address_size=4, resolution=16,
                    q=1e-3, r=5e-2, p0=1.0, n_epochs=e) for e in epochs_range]
    baseline = _eval(SimpleWiSARD, X, y, seeds5, address_size=4, resolution=16)
    ax.plot(epochs_range, accs_k, marker="o", color=PALETTE["KalmanWiSARD"], linewidth=2, label="KalmanWiSARD")
    ax.axhline(baseline, color=PALETTE["SimpleWiSARD"], linestyle="--", linewidth=1.5,
               label=f"SimpleWiSARD ({baseline:.3f})")
    ax.set_xlabel("n_epochs"); ax.set_ylabel("Acurácia média")
    ax.set_title("b) Acurácia × n_epochs"); ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    print("  4c. address_size...")
    ax = axes[1, 0]
    addr_range = [2, 3, 4, 5, 6, 7, 8]
    ax.plot(addr_range,
            [_eval(SimpleWiSARD, X, y, seeds5, address_size=a, resolution=16) for a in addr_range],
            marker="s", color=PALETTE["SimpleWiSARD"], linewidth=2, label="SimpleWiSARD")
    ax.plot(addr_range,
            [_eval(KalmanWiSARD, X, y, seeds5, address_size=a, resolution=16,
                   q=1e-3, r=5e-2, p0=1.0, n_epochs=1) for a in addr_range],
            marker="o", color=PALETTE["KalmanWiSARD"], linewidth=2, label="KalmanWiSARD")
    ax.set_xlabel("address_size"); ax.set_ylabel("Acurácia média")
    ax.set_title("c) Acurácia × address_size"); ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    print("  4d. resolution...")
    ax = axes[1, 1]
    res_range = [4, 8, 12, 16, 24, 32]
    ax.plot(res_range,
            [_eval(SimpleWiSARD, X, y, seeds5, address_size=4, resolution=r) for r in res_range],
            marker="s", color=PALETTE["SimpleWiSARD"], linewidth=2, label="SimpleWiSARD")
    ax.plot(res_range,
            [_eval(KalmanWiSARD, X, y, seeds5, address_size=4, resolution=r,
                   q=1e-3, r=5e-2, p0=1.0, n_epochs=1) for r in res_range],
            marker="o", color=PALETTE["KalmanWiSARD"], linewidth=2, label="KalmanWiSARD")
    ax.set_xlabel("resolution (bits por atributo)"); ax.set_ylabel("Acurácia média")
    ax.set_title("d) Acurácia × resolution"); ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    plt.tight_layout(); savefig("4_hyperparams")


# ── 5. Análise interna dos modelos ────────────────────────────────────────────

def sec5_internals(X, y):
    print("\n[5] Análise interna dos modelos")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0, stratify=y)

    simple = SimpleWiSARD(**SIMPLE_PARAMS, seed=0)
    simple.fit(X_train, y_train)

    kalman = KalmanWiSARD(**KALMAN_PARAMS, seed=0)
    kalman.fit(X_train, y_train)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("5. Análise Interna dos Modelos — MNIST", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # heatmap memórias — classe 0 apenas (representativa)
    for col, (name, mem) in enumerate([
        ("SimpleWiSARD", simple._memory),
        ("KalmanWiSARD", kalman._memory_x),
    ]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(mem[0], cmap="Blues", aspect="auto")
        ax.set_title(f"{name} — dígito 0", fontsize=9)
        ax.set_xlabel("Endereço RAM", fontsize=8); ax.set_ylabel("RAM", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # distribuição p (incerteza Kalman) — classes 0, 4, 9
    ax = fig.add_subplot(gs[0, 2])
    for cls_idx in [0, 4, 9]:
        ax.hist(kalman._memory_p[cls_idx].flatten(), bins=40, alpha=0.5,
                label=f"dígito {cls_idx}", density=True)
    ax.set_xlabel("Incerteza p"); ax.set_ylabel("Densidade")
    ax.set_title("5c. Distribuição de p (Kalman)"); ax.legend(fontsize=7)

    # scores por classe — Simple
    ax = fig.add_subplot(gs[1, 0])
    Xb = apply_encoders(X_test.astype(float), simple._encoders)
    scores_s = simple._compute_scores(simple._score_memory, Xb)
    for cls_idx in range(10):
        ax.hist(scores_s[:, cls_idx], bins=20, alpha=0.4, label=str(cls_idx), density=True)
    ax.set_xlabel("Score"); ax.set_ylabel("Densidade")
    ax.set_title("5d. Scores por classe (Simple)"); ax.legend(fontsize=6, ncol=2)

    # scores por classe — Kalman
    ax = fig.add_subplot(gs[1, 1])
    Xb_k = apply_encoders(X_test.astype(float), kalman._encoders)
    scores_k = kalman._compute_scores(kalman._score_memory, Xb_k)
    for cls_idx in range(10):
        ax.hist(scores_k[:, cls_idx], bins=20, alpha=0.4, label=str(cls_idx), density=True)
    ax.set_xlabel("Score"); ax.set_ylabel("Densidade")
    ax.set_title("5e. Scores por classe (Kalman)"); ax.legend(fontsize=6, ncol=2)

    # acurácia por dígito (ambos os modelos)
    ax = fig.add_subplot(gs[1, 2])
    y_pred_s = simple.predict(X_test)
    y_pred_k = kalman.predict(X_test)
    digits = list(range(10))
    acc_s = [(y_pred_s[y_test == d] == d).mean() for d in digits]
    acc_k = [(y_pred_k[y_test == d] == d).mean() for d in digits]
    x = np.arange(10)
    ax.bar(x - 0.2, acc_s, 0.35, color=PALETTE["SimpleWiSARD"], label="Simple", alpha=0.85)
    ax.bar(x + 0.2, acc_k, 0.35, color=PALETTE["KalmanWiSARD"], label="Kalman", alpha=0.85)
    ax.set_xlabel("Dígito"); ax.set_ylabel("Acurácia")
    ax.set_title("5f. Acurácia por Dígito"); ax.set_xticks(x); ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    plt.tight_layout(); savefig("5_internals")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Análise MNIST — SimpleWiSARD vs KalmanWiSARD ===")
    df = load_raw()
    print("Carregando MNIST...")
    X, y = get_mnist()

    sec1_performance(df)
    sec2_tradeoff(df)
    sec3_stability(df)
    sec4_hyperparams(X, y)
    sec5_internals(X, y)

    print(f"\nPronto! Gráficos em {OUT}/")


if __name__ == "__main__":
    main()
