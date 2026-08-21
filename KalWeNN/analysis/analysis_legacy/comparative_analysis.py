"""
Análise comparativa Wine vs MNIST — SimpleWiSARD vs KalmanWiSARD

Gráficos:
  A. Tabela comparativa (acurácia, std, CV, tempo)
  B. Barras agrupadas por dataset
  C. Ganho do Kalman sobre o Simple por dataset
  D. Radar chart: acurácia × estabilidade × eficiência
  E. Scatter: acurácia × tempo de treino (todos os runs)
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
from math import pi

PALETTE = {"SimpleWiSARD": "#4C72B0", "KalmanWiSARD": "#DD8452"}
DATASET_LABELS = {"wine": "Wine", "mnist": "MNIST"}
OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)


def savefig(name):
    path = OUT / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  salvo → {path}")


def load_data():
    frames = []
    for ds in ["wine", "mnist"]:
        path = Path(f"results/{ds}_raw.csv")
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def build_summary(df):
    rows = []
    for ds in df["dataset"].unique():
        for model in ["SimpleWiSARD", "KalmanWiSARD"]:
            sub = df[(df.dataset == ds) & (df.model == model)]["accuracy"]
            t = df[(df.dataset == ds) & (df.model == model)]["train_wall_s"]
            rows.append({
                "dataset": ds,
                "model": model,
                "mean_acc": sub.mean(),
                "std_acc": sub.std(),
                "cv": sub.std() / sub.mean() * 100,
                "mean_train_s": t.mean(),
            })
    return pd.DataFrame(rows)


# ── A. Tabela comparativa ─────────────────────────────────────────────────────

def plot_A_table(summary):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    fig.suptitle("A. Tabela Comparativa — Wine vs MNIST", fontsize=13, fontweight="bold")

    cols = ["Dataset", "Modelo", "Acurácia média", "Std", "CV (%)", "Tempo treino (s)"]
    rows = []
    for _, r in summary.iterrows():
        rows.append([
            DATASET_LABELS.get(r.dataset, r.dataset),
            r.model,
            f"{r.mean_acc:.4f}",
            f"{r.std_acc:.4f}",
            f"{r.cv:.2f}%",
            f"{r.mean_train_s:.4f}",
        ])

    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2d3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif rows[row - 1][1] == "KalmanWiSARD":
            cell.set_facecolor("#fdf0e6")
        else:
            cell.set_facecolor("#eef2f8")

    plt.tight_layout()
    savefig("comparative_A_table")


# ── B. Barras agrupadas por dataset ──────────────────────────────────────────

def plot_B_bars(df, summary):
    datasets = list(df["dataset"].unique())
    models = ["SimpleWiSARD", "KalmanWiSARD"]

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    fig.suptitle("B. Acurácia por Dataset — Média ± IC 95%", fontsize=13, fontweight="bold")

    for ax, ds in zip(axes, datasets):
        sub = summary[summary.dataset == ds]
        means = [sub[sub.model == m]["mean_acc"].values[0] for m in models]
        stds  = [sub[sub.model == m]["std_acc"].values[0] for m in models]
        n = len(df[(df.dataset == ds) & (df.model == models[0])])
        cis = [1.96 * s / np.sqrt(n) for s in stds]
        bars = ax.bar(models, means, yerr=cis, capsize=8, width=0.5,
                      color=[PALETTE[m] for m in models], alpha=0.85, error_kw={"linewidth": 2})
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12)
        ax.set_ylabel("Acurácia média")
        ax.set_xticklabels(["Simple\nWiSARD", "Kalman\nWiSARD"])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.005,
                    f"{mean:.2%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    savefig("comparative_B_bars")


# ── C. Ganho do Kalman sobre o Simple ────────────────────────────────────────

def plot_C_gain(summary):
    datasets = summary["dataset"].unique()
    gains_acc, gains_cv = [], []
    labels = []

    for ds in datasets:
        s = summary[(summary.dataset == ds) & (summary.model == "SimpleWiSARD")].iloc[0]
        k = summary[(summary.dataset == ds) & (summary.model == "KalmanWiSARD")].iloc[0]
        gains_acc.append((k.mean_acc - s.mean_acc) * 100)
        gains_cv.append(s.cv - k.cv)  # redução de CV (positivo = Kalman mais estável)
        labels.append(DATASET_LABELS.get(ds, ds))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("C. Ganho do KalmanWiSARD sobre SimpleWiSARD", fontsize=13, fontweight="bold")

    ax = axes[0]
    colors = ["#2ecc71" if g > 0 else "#e74c3c" for g in gains_acc]
    bars = ax.bar(labels, gains_acc, color=colors, alpha=0.85, width=0.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Ganho em acurácia (pp)")
    ax.set_title("Ganho em Acurácia (Kalman − Simple)")
    for bar, val in zip(bars, gains_acc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.2 if val >= 0 else -0.5),
                f"{val:+.2f}pp", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax = axes[1]
    colors2 = ["#2ecc71" if g > 0 else "#e74c3c" for g in gains_cv]
    bars2 = ax.bar(labels, gains_cv, color=colors2, alpha=0.85, width=0.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Redução de CV (pp)")
    ax.set_title("Redução de Instabilidade (Simple CV − Kalman CV)")
    for bar, val in zip(bars2, gains_cv):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.05 if val >= 0 else -0.2),
                f"{val:+.2f}pp", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    savefig("comparative_C_gain")


# ── D. Radar chart ────────────────────────────────────────────────────────────

def plot_D_radar(summary, df):
    datasets = list(summary["dataset"].unique())
    categories = ["Acurácia", "Estabilidade\n(1-CV)", "Eficiência\n(acc/tempo)"]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5),
                             subplot_kw=dict(polar=True))
    if len(datasets) == 1:
        axes = [axes]
    fig.suptitle("D. Radar Chart por Dataset", fontsize=13, fontweight="bold")

    for ax, ds in zip(axes, datasets):
        sub = summary[summary.dataset == ds]
        for model in ["SimpleWiSARD", "KalmanWiSARD"]:
            r = sub[sub.model == model].iloc[0]
            acc_norm = r.mean_acc
            stab_norm = 1 - r.cv / 100
            eff_norm = r.mean_acc / r.mean_train_s

            # normaliza eficiência pelo máximo do dataset
            max_eff = (summary[summary.dataset == ds]["mean_acc"] /
                       summary[summary.dataset == ds]["mean_train_s"]).max()
            eff_norm = eff_norm / max_eff

            values = [acc_norm, stab_norm, eff_norm]
            values += values[:1]
            ax.plot(angles, values, color=PALETTE[model], linewidth=2, label=model)
            ax.fill(angles, values, color=PALETTE[model], alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=11, pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout()
    savefig("comparative_D_radar")


# ── E. Scatter acurácia × tempo ───────────────────────────────────────────────

def plot_E_scatter(df):
    datasets = list(df["dataset"].unique())
    markers = {"wine": "o", "mnist": "s"}

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("E. Acurácia × Tempo de Treino — todos os runs", fontsize=13, fontweight="bold")

    for ds in datasets:
        for model in ["SimpleWiSARD", "KalmanWiSARD"]:
            sub = df[(df.dataset == ds) & (df.model == model)]
            ax.scatter(sub["train_wall_s"], sub["accuracy"],
                       color=PALETTE[model], marker=markers[ds], s=70, alpha=0.75,
                       edgecolors="k", linewidths=0.4,
                       label=f"{model} / {DATASET_LABELS.get(ds, ds)}")

    ax.set_xlabel("Tempo de treino (s)")
    ax.set_ylabel("Acurácia")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    savefig("comparative_E_scatter")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Análise Comparativa Wine vs MNIST ===")
    df = load_data()
    summary = build_summary(df)

    print("\nResumo:")
    print(summary.to_string(index=False))

    plot_A_table(summary)
    plot_B_bars(df, summary)
    plot_C_gain(summary)
    plot_D_radar(summary, df)
    plot_E_scatter(df)

    print(f"\nPronto! Gráficos comparativos em {OUT}/")


if __name__ == "__main__":
    main()
