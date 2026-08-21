"""
Comparação final entre TODOS os datasets — Clássica × Clássica+Bleach × KWiSARD.

Lê todos os results/v2/{ds}_final_summary.csv e gera:
  - tabela-mestre (results/v2/all_datasets_summary.csv)
  - A) barras por dataset (acurácia e F1-macro)
  - B) rank médio por métrica
  - C) diagrama de diferença crítica (Friedman + Nemenyi)  [acurácia e F1]
  - D) dominância: Kalman × Clássica+Bleach (ponto por dataset)
  - E) "sem tuning": Clássica(b=0) × Clássica(melhor b) × Kalman
  - F) estabilidade: boxplot de CV entre datasets
  - G) win/tie/loss par-a-par

Uso:  python3 analysis/v2_compare_all.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from v2_common import PALETTE, MODEL_LABELS, FIG, CFG

ORDER = ["Classica", "Classica+Bleach", "KalmanWiSARD"]
SHORT = {"Classica": "Clássica", "Classica+Bleach": "Cláss+Bleach", "KalmanWiSARD": "Kalman"}

# valores críticos de Nemenyi (q_alpha) para alpha=0.05
Q_ALPHA = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}

OUT = FIG / "_compare_all"
OUT.mkdir(parents=True, exist_ok=True)


def savefig(name):
    plt.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  salvo → {OUT / f'{name}.png'}")


# ── carrega todas as tabelas ──────────────────────────────────────────────────

def load_all():
    frames = []
    for p in sorted(CFG.glob("*_final_summary.csv")):
        ds = p.name.replace("_final_summary.csv", "")
        df = pd.read_csv(p)
        df.insert(0, "dataset", ds)
        frames.append(df)
    if not frames:
        print("Nenhum *_final_summary.csv encontrado. Rode v2_analysis.py antes.")
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def pivot(df, metric):
    """Retorna DataFrame datasets × modelos para a métrica."""
    return df.pivot(index="dataset", columns="model", values=metric)[ORDER]


# ── A) barras por dataset ─────────────────────────────────────────────────────

def fig_per_dataset(df, metric, name, title):
    piv = pivot(df, metric)
    datasets = list(piv.index)
    x = np.arange(len(datasets)); width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, 1.0*len(datasets)), 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for j, m in enumerate(ORDER):
        ax.bar(x + (j-1)*width, piv[m].values, width,
               color=PALETTE[m], alpha=0.88, label=SHORT[m])
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel(metric); ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout(); savefig(name)


# ── ranks e Friedman ──────────────────────────────────────────────────────────

def avg_ranks(df, metric):
    piv = pivot(df, metric)
    # rank por dataset: maior métrica = rank 1 (melhor)
    ranks = piv.rank(axis=1, ascending=False, method="average")
    return ranks.mean(axis=0), piv, ranks


def fig_avg_rank(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("B) Rank médio entre datasets (1 = melhor)", fontsize=13, fontweight="bold")
    for ax, metric in zip(axes, ["acc", "f1_macro"]):
        ar, _, _ = avg_ranks(df, metric)
        bars = ax.bar([SHORT[m] for m in ORDER], [ar[m] for m in ORDER],
                      color=[PALETTE[m] for m in ORDER], alpha=0.88)
        for bar, m in zip(bars, ORDER):
            ax.text(bar.get_x()+bar.get_width()/2, ar[m], f"{ar[m]:.2f}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Rank médio"); ax.set_title(metric)
        ax.invert_yaxis()  # rank 1 no topo
    plt.tight_layout(); savefig("B_avg_rank")


# ── C) diagrama de diferença crítica ──────────────────────────────────────────

def cd_diagram(df, metric, ax):
    ar, piv, _ = avg_ranks(df, metric)
    N, k = piv.shape
    # Friedman
    try:
        chi, p = stats.friedmanchisquare(*[piv[m].values for m in ORDER])
    except Exception:
        p = np.nan
    cd = Q_ALPHA.get(k, 3.0) * np.sqrt(k * (k + 1) / (6.0 * N))

    ranks_sorted = sorted(ORDER, key=lambda m: ar[m])  # melhor (menor) primeiro
    ax.set_xlim(0.8, k + 0.2); ax.set_ylim(0, 1)
    ax.hlines(0.6, 1, k, color="black", lw=1)
    for r in range(1, k + 1):
        ax.vlines(r, 0.58, 0.62, color="black", lw=1)
        ax.text(r, 0.52, str(r), ha="center", fontsize=8)
    for i, m in enumerate(ranks_sorted):
        ax.plot([ar[m], ar[m]], [0.6, 0.75 + i*0.06], color=PALETTE[m], lw=2)
        ax.text(ar[m], 0.78 + i*0.06, f"{SHORT[m]} ({ar[m]:.2f})",
                color=PALETTE[m], fontsize=9, ha="center", fontweight="bold")
    # barra de CD
    ax.hlines(0.4, 1, 1 + cd, color="red", lw=3)
    ax.text(1 + cd/2, 0.33, f"CD = {cd:.2f}", color="red", ha="center", fontsize=8)
    ax.set_title(f"{metric}  (Friedman p={p:.4f})", fontsize=10)
    ax.text(k, 0.52, "melhor →", ha="right", fontsize=7, style="italic")
    ax.axis("off")


def fig_cd(df):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle("C) Diferença Crítica — Friedman + Nemenyi (rank menor = melhor)",
                 fontsize=13, fontweight="bold")
    cd_diagram(df, "acc", axes[0])
    cd_diagram(df, "f1_macro", axes[1])
    plt.tight_layout(); savefig("C_critical_difference")


# ── D) dominância Kalman × Clássica+Bleach ────────────────────────────────────

def fig_dominance(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("D) Dominância: Kalman × Clássica+Bleach (acima da diagonal = Kalman vence)",
                 fontsize=13, fontweight="bold")
    for ax, metric in zip(axes, ["acc", "f1_macro"]):
        piv = pivot(df, metric)
        x = piv["Classica+Bleach"].values
        y = piv["KalmanWiSARD"].values
        lo = min(x.min(), y.min()) - 0.02
        hi = max(x.max(), y.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)
        ax.scatter(x, y, color=PALETTE["KalmanWiSARD"], s=60, edgecolors="k", linewidths=0.5)
        for xi, yi, ds in zip(x, y, piv.index):
            ax.annotate(ds, (xi, yi), fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        wins = int((y > x).sum())
        ax.set_xlabel("Clássica+Bleach"); ax.set_ylabel("KalmanWiSARD")
        ax.set_title(f"{metric}  (Kalman vence em {wins}/{len(x)})")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout(); savefig("D_dominance")


# ── E) "sem tuning": b=0 × melhor b × Kalman ──────────────────────────────────

def fig_no_tuning(df):
    piv = pivot(df, "acc")
    datasets = list(piv.index)
    x = np.arange(len(datasets)); width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, 1.0*len(datasets)), 6))
    fig.suptitle("E) Kalman alcança o melhor bleaching SEM ajustar nada",
                 fontsize=13, fontweight="bold")
    ax.bar(x - width, piv["Classica"].values, width, color="#BBBBBB",
           label="Clássica (b=0, sem tuning)")
    ax.bar(x, piv["Classica+Bleach"].values, width, color=PALETTE["Classica+Bleach"],
           label="Clássica (melhor b, com tuning)")
    ax.bar(x + width, piv["KalmanWiSARD"].values, width, color=PALETTE["KalmanWiSARD"],
           label="Kalman (sem b para ajustar)")
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel("Acurácia"); ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout(); savefig("E_no_tuning")


# ── F) estabilidade: CV entre datasets ────────────────────────────────────────

def fig_stability(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("F) Estabilidade entre datasets — CV da acurácia (menor = melhor)",
                 fontsize=13, fontweight="bold")
    data = [df[df.model == m]["cv_acc_%"].values for m in ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showmeans=True)
    for patch, m in zip(bp["boxes"], ORDER):
        patch.set_facecolor(PALETTE[m]); patch.set_alpha(0.85)
    ax.set_xticks(range(1, 4)); ax.set_xticklabels([SHORT[m] for m in ORDER])
    ax.set_ylabel("CV da acurácia (%)")
    plt.tight_layout(); savefig("F_stability_cv")


# ── G) win/tie/loss ───────────────────────────────────────────────────────────

def fig_win_tie_loss(df, metric="acc", eps=1e-3):
    piv = pivot(df, metric)
    k = len(ORDER)
    mat = np.zeros((k, k))  # mat[i,j] = datasets em que i vence j
    for i, a in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            if i == j:
                continue
            mat[i, j] = int(((piv[a] - piv[b]) > eps).sum())
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"G) Win/Tie/Loss — vitórias em {metric} (linha vence coluna)",
                 fontsize=12, fontweight="bold")
    im = ax.imshow(mat, cmap="Greens")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels([SHORT[m] for m in ORDER]); ax.set_yticklabels([SHORT[m] for m in ORDER])
    for i in range(k):
        for j in range(k):
            if i != j:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); savefig("G_win_tie_loss")


# ── H) custo relativo (Kalman ÷ Clássica) entre datasets ──────────────────────

def fig_cost(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("H) Custo do Kalman relativo à Clássica (razão por dataset)",
                 fontsize=13, fontweight="bold")
    panels = [("train_ms", "Tempo de treino"),
              ("test_ms", "Tempo de inferência"),
              ("mem_MB", "Memória do modelo")]
    for ax, (col, label) in zip(axes, panels):
        piv = pivot(df, col)
        ratio = (piv["KalmanWiSARD"] / piv["Classica"]).values
        bp = ax.boxplot(ratio, patch_artist=True, widths=0.5, showmeans=True)
        bp["boxes"][0].set_facecolor(PALETTE["KalmanWiSARD"]); bp["boxes"][0].set_alpha(0.85)
        ax.axhline(1.0, color="gray", linestyle="--", label="igual à clássica")
        ax.set_title(f"{label}\n(mediana {np.median(ratio):.2f}×)")
        ax.set_ylabel("Kalman ÷ Clássica (×)")
        ax.set_xticks([]); ax.legend(fontsize=8)
    plt.tight_layout(); savefig("H_cost_ratio")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_all()
    n_ds = df["dataset"].nunique()
    print(f"Datasets encontrados: {n_ds} → {sorted(df['dataset'].unique())}\n")

    df.to_csv(CFG / "all_datasets_summary.csv", index=False)
    print(f"Tabela-mestre → {CFG / 'all_datasets_summary.csv'}\n")

    # ranks resumidos no terminal
    for metric in ["acc", "f1_macro"]:
        ar, _, _ = avg_ranks(df, metric)
        print(f"Rank médio ({metric}): " +
              ", ".join(f"{SHORT[m]}={ar[m]:.2f}" for m in ORDER))
    print()

    fig_per_dataset(df, "acc", "A_accuracy_by_dataset", "A) Acurácia por dataset")
    fig_per_dataset(df, "f1_macro", "A_f1_by_dataset", "A) F1-macro por dataset")
    fig_avg_rank(df)
    fig_cd(df)
    fig_dominance(df)
    fig_no_tuning(df)
    fig_stability(df)
    fig_win_tie_loss(df, "acc")
    fig_cost(df)

    print(f"\nPronto! Figuras em {OUT}/")


if __name__ == "__main__":
    main()
