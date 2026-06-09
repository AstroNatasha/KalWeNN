"""
Experimento rápido — fusão de evidência ponderada pela incerteza.

Padrão : score_c = Σ_ram  x[c, ram, addr]                  (peso igual por RAM)
Pesado : score_c = Σ_ram  (1 - p[c, ram, addr]) · x[...]   (peso = confiança da célula)

A ideia: usar o p (incerteza) não só no treino, mas TAMBÉM na decisão —
células confiantes votam mais, incertas votam menos. Primeira forma de
fusão de evidência ponderada na KWiSARD.

Uso:  python3 analysis/v2_fusion_experiment.py wine
      python3 analysis/v2_fusion_experiment.py mnist
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

from v2_common import get_data, split, CFG, FIG, PALETTE
from wisard.models_v2 import KalmanV2
from wisard.encoder import apply_encoders

SEEDS = list(range(10))


def weighted_scores(model, X):
    """Score com cada célula ponderada por (1 - p) = confiança."""
    Xb = apply_encoders(np.asarray(X, dtype=float), model._encoders)
    weighted_mem = (1.0 - model._memory_p) * model._memory_x
    return model._compute_scores(weighted_mem, Xb)


def run(ds):
    cfg = json.load(open(CFG / f"{ds}_config.json"))
    a, r, k = cfg["address_size"], cfg["resolution"], cfg["kalman"]
    X, y = get_data(ds)

    std_accs, w_accs = [], []
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = split(X, y, seed)
        m = KalmanV2(address_size=a, resolution=r, seed=seed,
                     q=k["q"], r=k["r"], p0=k["p0"], n_epochs=k["n_epochs"]).fit(X_tr, y_tr)

        # padrão (peso igual)
        std_accs.append((m.predict(X_te) == y_te).mean())

        # ponderado por confiança
        sc = weighted_scores(m, X_te)
        w_accs.append((m.classes_[sc.argmax(1)] == y_te).mean())

    std_accs, w_accs = np.array(std_accs), np.array(w_accs)
    print(f"\n=== Fusão por incerteza — {ds.upper()} ({len(SEEDS)} seeds) ===")
    print(f"  Padrão  (peso igual)     : {std_accs.mean():.4f} ± {std_accs.std():.4f}")
    print(f"  Pesado  (conf. = 1 - p)  : {w_accs.mean():.4f} ± {w_accs.std():.4f}")
    print(f"  Δ média                  : {(w_accs.mean()-std_accs.mean())*100:+.3f}pp")
    wins = (w_accs > std_accs).sum()
    print(f"  Pesado venceu em         : {wins}/{len(SEEDS)} seeds")
    return {"dataset": ds, "std": std_accs, "weighted": w_accs}


def save_and_plot(all_results):
    # CSV (uma linha por seed)
    rows = []
    for res in all_results:
        for i, seed in enumerate(SEEDS):
            rows.append({"dataset": res["dataset"], "seed": seed,
                         "acc_padrao": res["std"][i], "acc_pesado": res["weighted"][i]})
    csv_path = CFG / "fusion_experiment.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nCSV salvo → {csv_path}")

    # gráfico: barras padrão vs pesado por dataset
    datasets = [r["dataset"] for r in all_results]
    x = np.arange(len(datasets)); width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Fusão de evidência por incerteza — padrão vs ponderado",
                 fontsize=13, fontweight="bold")

    std_means = [r["std"].mean() for r in all_results]
    w_means   = [r["weighted"].mean() for r in all_results]
    std_ci = [1.96*r["std"].std()/np.sqrt(len(SEEDS)) for r in all_results]
    w_ci   = [1.96*r["weighted"].std()/np.sqrt(len(SEEDS)) for r in all_results]

    b1 = ax.bar(x-width/2, std_means, width, yerr=std_ci, capsize=6,
                color="#999999", alpha=0.85, label="Padrão (peso igual)")
    b2 = ax.bar(x+width/2, w_means, width, yerr=w_ci, capsize=6,
                color=PALETTE["KalmanWiSARD"], alpha=0.9, label="Ponderado (conf. = 1−p)")
    for bars, means in [(b1, std_means), (b2, w_means)]:
        for bar, m in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, m, f"{m:.2%}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_ylabel("Acurácia média"); ax.legend(fontsize=9)
    ax.set_title("Ponderar votos pela confiança da célula (p) — só possível na KWiSARD")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout()
    out = FIG / "fusion_experiment.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo → {out}")

    # versão com ZOOM por dataset (faixa relevante de cada um) ───────────────────
    fig, axes = plt.subplots(1, len(all_results), figsize=(6*len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]
    fig.suptitle("Fusão por incerteza — zoom na faixa relevante",
                 fontsize=13, fontweight="bold")
    for ax, res in zip(axes, all_results):
        s_m, w_m = res["std"].mean(), res["weighted"].mean()
        s_ci = 1.96*res["std"].std()/np.sqrt(len(SEEDS))
        w_ci = 1.96*res["weighted"].std()/np.sqrt(len(SEEDS))
        bars = ax.bar(["Padrão", "Ponderado"], [s_m, w_m], yerr=[s_ci, w_ci],
                      capsize=8, color=["#999999", PALETTE["KalmanWiSARD"]], alpha=0.9,
                      width=0.5)
        for bar, m in zip(bars, [s_m, w_m]):
            ax.text(bar.get_x()+bar.get_width()/2, m, f"{m:.3%}",
                    ha="center", va="bottom", fontsize=9)
        # zoom: centra na faixa dos dois valores, com folga
        lo, hi = min(s_m, w_m), max(s_m, w_m)
        pad = max(hi - lo, 0.002) * 3
        ax.set_ylim(lo - pad, hi + pad)
        wins = (res["weighted"] > res["std"]).sum()
        ax.set_title(f"{res['dataset'].upper()}  "
                     f"(Δ={(w_m-s_m)*100:+.3f}pp, {wins}/{len(SEEDS)} seeds)")
        ax.set_ylabel("Acurácia média")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2%}"))
    plt.tight_layout()
    out_zoom = FIG / "fusion_experiment_zoom.png"
    plt.savefig(out_zoom, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Gráfico (zoom) salvo → {out_zoom}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        results = [run("wine"), run("mnist")]
        save_and_plot(results)
