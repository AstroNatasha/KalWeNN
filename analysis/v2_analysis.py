"""
Etapa 2 — Comparação final + todas as análises.

Lê results/v2/{ds}_config.json (gerado por v2_search.py) e produz:
  #6 Acurácia ........ boxplot, barras+IC, por-seed, testes de significância
  #2 Estabilidade .... violin, coeficiente de variação
  #1 Robustez ........ acurácia × bleaching (clássica) com Kalman por cima
  #3 Incerteza ....... curva acurácia-rejeição (p nativo do Kalman)
  #4 Eficiência ...... learning curve (acurácia × tamanho do treino)
  #5 Épocas .......... acurácia × n_epochs (ganho exclusivo do Kalman)
  #7 Custo ........... tempo de treino + memória
  #8 Por classe ...... matrizes de confusão + acurácia por classe

Uso:  python3 analysis/v2_analysis.py wine
      python3 analysis/v2_analysis.py mnist
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix

from v2_common import (
    get_data, split, eval_over_seeds,
    make_classic, make_kalman,
    FINAL_SEEDS, CURVE_SEEDS, PALETTE, MODEL_LABELS, FIG, CFG, TAG,
)

ORDER = ["Classica", "Classica+Bleach", "KalmanWiSARD"]
EPOCHS_RANGE = [1, 2, 3, 5, 8]
EPOCH_SEEDS  = list(range(5))
TRAIN_FRACS  = [0.1, 0.25, 0.5, 0.75, 1.0]


def savefig(ds, name):
    d = FIG / f"{ds}{TAG}"
    d.mkdir(parents=True, exist_ok=True)
    plt.savefig(d / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    salvo → {d / f'{name}.png'}")


def make_models(cfg):
    a, r = cfg["address_size"], cfg["resolution"]
    k = cfg["kalman"]
    b = cfg["best_bleach"]
    return {
        "Classica":        lambda s: make_classic(address_size=a, resolution=r, seed=s, bleach=0),
        "Classica+Bleach": lambda s: make_classic(address_size=a, resolution=r, seed=s, bleach=b),
        "KalmanWiSARD":    lambda s: make_kalman(address_size=a, resolution=r, seed=s,
                                                 q=k["q"], r=k["r"], p0=k["p0"], n_epochs=k["n_epochs"]),
    }


# ── comparação final (30 seeds) ──────────────────────────────────────────────────

def final_comparison(cfg, X, y):
    print("  Comparação final (30 seeds)...")
    factories = make_models(cfg)
    return {name: eval_over_seeds(factories[name], X, y, FINAL_SEEDS) for name in ORDER}


# ── #6 Acurácia ──────────────────────────────────────────────────────────────────

def fig_accuracy(results, ds):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"#6 Acurácia — {ds.upper()} ({len(FINAL_SEEDS)} seeds)", fontsize=13, fontweight="bold")

    ax = axes[0]
    bp = ax.boxplot([results[m]["acc"] for m in ORDER], patch_artist=True, widths=0.6)
    for patch, m in zip(bp["boxes"], ORDER):
        patch.set_facecolor(PALETTE[m]); patch.set_alpha(0.85)
    ax.set_xticks(range(1, 4)); ax.set_xticklabels([MODEL_LABELS[m] for m in ORDER])
    ax.set_ylabel("Acurácia"); ax.set_title("a) Boxplot")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    ax = axes[1]
    means = [results[m]["acc"].mean() for m in ORDER]
    cis = [1.96 * results[m]["acc"].std() / np.sqrt(len(FINAL_SEEDS)) for m in ORDER]
    bars = ax.bar([MODEL_LABELS[m] for m in ORDER], means, yerr=cis, capsize=8,
                  color=[PALETTE[m] for m in ORDER], alpha=0.85, error_kw={"linewidth": 2})
    ax.set_ylabel("Acurácia média"); ax.set_title("b) Média ± IC 95%")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    for bar, m_ in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m_ + 0.002, f"{m_:.2%}",
                ha="center", va="bottom", fontsize=8)

    ax = axes[2]; ax.axis("off")
    c  = results["Classica"]["acc"]
    cb = results["Classica+Bleach"]["acc"]
    k  = results["KalmanWiSARD"]["acc"]
    def cmp(a, b, la, lb):
        t_p = stats.ttest_ind(a, b, equal_var=False).pvalue
        u_p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        return [f"  {la} vs {lb}:",
                f"    Δ média = {(a.mean()-b.mean())*100:+.2f}pp",
                f"    Welch  p = {t_p:.4f} {'*' if t_p<0.05 else 'n.s.'}",
                f"    MW-U   p = {u_p:.4f} {'*' if u_p<0.05 else 'n.s.'}", ""]
    lines = ["c) Testes de Significância", ""]
    lines += cmp(k, cb, "Kalman", "Cláss+Bleach")
    lines += cmp(k, c,  "Kalman", "Clássica")
    lines += cmp(cb, c, "Cláss+Bleach", "Clássica")
    ax.text(0.0, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=9,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(); savefig(ds, "6_accuracy")


# ── #2 Estabilidade ──────────────────────────────────────────────────────────────

def fig_stability(results, ds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"#2 Estabilidade entre seeds — {ds.upper()}", fontsize=13, fontweight="bold")

    ax = axes[0]
    parts = ax.violinplot([results[m]["acc"] for m in ORDER], positions=range(3), showmedians=True)
    for body, m in zip(parts["bodies"], ORDER):
        body.set_facecolor(PALETTE[m]); body.set_alpha(0.7)
    ax.set_xticks(range(3)); ax.set_xticklabels([MODEL_LABELS[m] for m in ORDER])
    ax.set_ylabel("Acurácia"); ax.set_title("a) Violin")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    ax = axes[1]
    cvs = [results[m]["acc"].std()/results[m]["acc"].mean()*100 for m in ORDER]
    bars = ax.bar([MODEL_LABELS[m] for m in ORDER], cvs,
                  color=[PALETTE[m] for m in ORDER], alpha=0.85)
    ax.set_ylabel("CV (%)"); ax.set_title("b) Coeficiente de Variação\n(menor = mais estável)")
    for bar, v in zip(bars, cvs):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout(); savefig(ds, "2_stability")


# ── #1 Robustez ao bleaching ─────────────────────────────────────────────────────

def fig_robustness(cfg, results, ds):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(f"#1 Robustez a hiperparâmetros — {ds.upper()}", fontsize=13, fontweight="bold")

    b_vals = cfg["bleach_vals"]
    b_mean = np.array(cfg["bleach_mean_acc"])
    b_std  = np.array(cfg["bleach_std_acc"])
    ax.errorbar(b_vals, b_mean, yerr=b_std, marker="o", color=PALETTE["Classica+Bleach"],
                capsize=4, linewidth=2, label="WiSARD clássica (varia com b)")
    ax.axvline(cfg["best_bleach"], color="gray", linestyle=":", alpha=0.7,
               label=f"melhor b = {cfg['best_bleach']}")

    k_mean = results["KalmanWiSARD"]["acc"].mean()
    ax.axhline(k_mean, color=PALETTE["KalmanWiSARD"], linestyle="--", linewidth=2,
               label=f"KalmanWiSARD (sem b para ajustar) = {k_mean:.3f}")

    ax.set_xlabel("limiar de bleaching  b"); ax.set_ylabel("Acurácia")
    ax.set_title("Clássica depende de achar o melhor b; Kalman entrega o pico automaticamente")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    plt.tight_layout(); savefig(ds, "1_robustness_bleaching")


# ── #3 Incerteza / rejeição ──────────────────────────────────────────────────────

def fig_uncertainty(cfg, X, y, ds):
    print("  #3 curva acurácia-rejeição...")
    X_tr, X_te, y_tr, y_te = split(X, y, 0)
    fracs = np.linspace(0, 0.9, 19)
    from v2_common import accuracy_rejection_curve

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(f"#3 Confiança / Rejeição — {ds.upper()}", fontsize=13, fontweight="bold")

    factories = make_models(cfg)
    # clássicas: confiança por margem
    for name in ["Classica", "Classica+Bleach"]:
        m = factories[name](0).fit(X_tr, y_tr)
        scores = m.decision_scores(X_te)
        conf = m.margin_confidence(scores)
        correct = (m.classes_[scores.argmax(1)] == y_te)
        accs = accuracy_rejection_curve(conf, correct, fracs)
        ax.plot(fracs*100, accs, marker=".", color=PALETTE[name],
                label=f"{name.replace('_',' ')} (margem)")

    # kalman: margem E incerteza p nativa
    mk = factories["KalmanWiSARD"](0).fit(X_tr, y_tr)
    scores_k = mk.decision_scores(X_te)
    correct_k = (mk.classes_[scores_k.argmax(1)] == y_te)
    accs_margin = accuracy_rejection_curve(mk.margin_confidence(scores_k), correct_k, fracs)
    ax.plot(fracs*100, accs_margin, marker=".", color=PALETTE["KalmanWiSARD"], alpha=0.5,
            label="Kalman (margem)")
    _, unc = mk.predicted_uncertainty(X_te)
    accs_p = accuracy_rejection_curve(-unc, correct_k, fracs)  # menor incerteza = mais confiante
    ax.plot(fracs*100, accs_p, marker="o", color=PALETTE["KalmanWiSARD"], linewidth=2.5,
            label="Kalman (incerteza p nativa)")

    ax.set_xlabel("% de amostras rejeitadas (menos confiantes)")
    ax.set_ylabel("Acurácia nas remanescentes")
    ax.set_title("Só o Kalman tem incerteza p nativa — a clássica não")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    plt.tight_layout(); savefig(ds, "3_uncertainty_rejection")


# ── #4 Learning curve ────────────────────────────────────────────────────────────

def fig_learning_curve(cfg, X, y, ds):
    print("  #4 learning curve...")
    factories = make_models(cfg)
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(f"#4 Eficiência de dados — {ds.upper()}", fontsize=13, fontweight="bold")

    for name in ORDER:
        means, stds = [], []
        for frac in TRAIN_FRACS:
            accs = []
            for seed in CURVE_SEEDS:
                X_tr, X_te, y_tr, y_te = split(X, y, seed)
                n = max(int(frac * len(X_tr)), len(np.unique(y)))
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(X_tr), n, replace=False)
                m = factories[name](seed).fit(X_tr[idx], y_tr[idx])
                accs.append((m.predict(X_te) == y_te).mean())
            means.append(np.mean(accs)); stds.append(np.std(accs))
        means, stds = np.array(means), np.array(stds)
        ax.plot([f*100 for f in TRAIN_FRACS], means, marker="o", color=PALETTE[name],
                linewidth=2, label=name.replace("_", " "))
        ax.fill_between([f*100 for f in TRAIN_FRACS], means-stds, means+stds,
                        color=PALETTE[name], alpha=0.12)

    ax.set_xlabel("% do conjunto de treino usado"); ax.set_ylabel("Acurácia")
    ax.set_title("Acurácia × volume de dados de treino")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    plt.tight_layout(); savefig(ds, "4_learning_curve")


# ── #5 Épocas ────────────────────────────────────────────────────────────────────

def fig_epochs(cfg, X, y, ds):
    print("  #5 acurácia × n_epochs...")
    a, r = cfg["address_size"], cfg["resolution"]
    k = cfg["kalman"]
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(f"#5 Refinamento por épocas — {ds.upper()}", fontsize=13, fontweight="bold")

    means, stds = [], []
    for e in EPOCHS_RANGE:
        accs = []
        for seed in EPOCH_SEEDS:
            X_tr, X_te, y_tr, y_te = split(X, y, seed)
            m = make_kalman(address_size=a, resolution=r, seed=seed,
                            q=k["q"], r=k["r"], p0=k["p0"], n_epochs=e).fit(X_tr, y_tr)
            accs.append((m.predict(X_te) == y_te).mean())
        means.append(np.mean(accs)); stds.append(np.std(accs))
    means, stds = np.array(means), np.array(stds)
    ax.errorbar(EPOCHS_RANGE, means, yerr=stds, marker="o", color=PALETTE["KalmanWiSARD"],
                linewidth=2, capsize=4, label="KalmanWiSARD (refina com épocas)")

    # clássicas são one-pass → linhas horizontais de referência
    for name in ["Classica", "Classica+Bleach"]:
        accs = []
        for seed in EPOCH_SEEDS:
            X_tr, X_te, y_tr, y_te = split(X, y, seed)
            m = make_models(cfg)[name](seed).fit(X_tr, y_tr)
            accs.append((m.predict(X_te) == y_te).mean())
        ax.axhline(np.mean(accs), color=PALETTE[name], linestyle="--",
                   label=f"{name.replace('_',' ')} (one-pass)")

    ax.set_xlabel("n_epochs"); ax.set_ylabel("Acurácia")
    ax.set_title("Kalman pode refinar com múltiplas passagens; a clássica não")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    plt.tight_layout(); savefig(ds, "5_epochs")


# ── #7 Custo ─────────────────────────────────────────────────────────────────────

def fig_cost(results, ds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"#7 Custo Computacional — {ds.upper()}", fontsize=13, fontweight="bold")

    ax = axes[0]
    means = [results[m]["train_t"].mean()*1000 for m in ORDER]
    bars = ax.bar([MODEL_LABELS[m] for m in ORDER], means,
                  color=[PALETTE[m] for m in ORDER], alpha=0.85)
    ax.set_ylabel("Tempo de treino (ms)"); ax.set_title("a) Tempo de Treino")
    for bar, v in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    mems = [results[m]["mem"].mean()/1e6 for m in ORDER]
    bars = ax.bar([MODEL_LABELS[m] for m in ORDER], mems,
                  color=[PALETTE[m] for m in ORDER], alpha=0.85)
    ax.set_ylabel("Memória do modelo (MB)"); ax.set_title("b) Footprint de Memória\n(Kalman guarda x e p)")
    for bar, v in zip(bars, mems):
        ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout(); savefig(ds, "7_cost")


# ── #8 Por classe (confusão + acurácia por classe) ───────────────────────────────

def fig_per_class(cfg, X, y, ds):
    print("  #8 matrizes de confusão...")
    X_tr, X_te, y_tr, y_te = split(X, y, 0)
    factories = make_models(cfg)
    classes = np.unique(y)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"#8 Análise por Classe — {ds.upper()}", fontsize=13, fontweight="bold")

    per_class = {}
    for i, name in enumerate(ORDER):
        m = factories[name](0).fit(X_tr, y_tr)
        pred = m.predict(X_te)
        cm = confusion_matrix(y_te, pred, labels=classes)
        ax = axes[i]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name.replace('_',' ')}", fontsize=10)
        ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
        if len(classes) <= 12:
            ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, fontsize=6); ax.set_yticklabels(classes, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        per_class[name] = cm.diagonal() / cm.sum(axis=1)

    ax = axes[3]
    width = 0.25
    xpos = np.arange(len(classes))
    for j, name in enumerate(ORDER):
        ax.bar(xpos + (j-1)*width, per_class[name], width,
               color=PALETTE[name], alpha=0.85, label=name.replace("_", " "))
    ax.set_xlabel("Classe"); ax.set_ylabel("Acurácia"); ax.set_title("Acurácia por classe")
    ax.set_xticks(xpos); ax.set_xticklabels(classes, fontsize=6)
    ax.legend(fontsize=7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    plt.tight_layout(); savefig(ds, "8_per_class")


# ── main ─────────────────────────────────────────────────────────────────────────

def run(ds):
    cfg_path = CFG / f"{ds}{TAG}_config.json"
    if not cfg_path.exists():
        print(f"ERRO: rode primeiro  python3 analysis/v2_search.py {ds}")
        sys.exit(1)
    cfg = json.load(open(cfg_path))
    print(f"\n{'='*60}\n  ANÁLISE — {ds.upper()}")
    print(f"  arquitetura: address_size={cfg['address_size']}, resolution={cfg['resolution']}")
    print(f"  kalman: {cfg['kalman']} | melhor bleach: {cfg['best_bleach']}\n{'='*60}")

    X, y = get_data(ds)
    results = final_comparison(cfg, X, y)

    # tabela-resumo (todas as métricas)
    summary = pd.DataFrame([{
        "model": m,
        "acc": results[m]["acc"].mean(),
        "f1_macro": results[m]["f1_macro"].mean(),
        "precision_macro": results[m]["precision_macro"].mean(),
        "recall_macro": results[m]["recall_macro"].mean(),
        "balanced_acc": results[m]["balanced_acc"].mean(),
        "kappa": results[m]["kappa"].mean(),
        "cv_acc_%": results[m]["acc"].std()/results[m]["acc"].mean()*100,
        "train_ms": results[m]["train_t"].mean()*1000,
        "test_ms": results[m]["test_t"].mean()*1000,
        "mem_MB": results[m]["mem"].mean()/1e6,
    } for m in ORDER])
    summary.to_csv(CFG / f"{ds}{TAG}_final_summary.csv", index=False)
    print("\n" + summary.to_string(index=False) + "\n")

    fig_accuracy(results, ds)
    fig_stability(results, ds)
    fig_robustness(cfg, results, ds)
    fig_uncertainty(cfg, X, y, ds)
    fig_learning_curve(cfg, X, y, ds)
    fig_epochs(cfg, X, y, ds)
    fig_cost(results, ds)
    fig_per_class(cfg, X, y, ds)
    print(f"\nPronto! Figuras em {FIG / ds}/")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "wine"
    run(ds)
