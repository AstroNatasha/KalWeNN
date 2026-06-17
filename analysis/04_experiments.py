"""
04_experiments — testes e experimentos extras (fora do fluxo principal).

Reúne os experimentos que fizemos além da comparação central. Cada um é
independente e selecionado por subcomando.

  fusion [ds|all]   Fusão de evidência ponderada pela incerteza na INFERÊNCIA.
                    Em vez de cada RAM votar igual, pesa o voto pela confiança
                    (1 - p) da célula. Compara padrão vs ponderado por seed,
                    salva CSV + figuras. Requer {ds}_config.json (rode a busca
                    densa antes). Ex.: python3 analysis/04_experiments.py fusion wine

  mnist-official    MNIST com o split OFICIAL (60k/10k), tupla=28, termômetro=3
                    (sparse). Roda 30 seeds e imprime média/CV de clássica e
                    Kalman, para comparar com a literatura (~93%).

  memory            Redução de memória no MNIST: como o p não é usado na
                    inferência, pode ser descartado após o treino — a memória
                    do Kalman cai pela metade, igualando a clássica, sem perda
                    de acurácia. Imprime a comparação de footprint.

Uso:  python3 analysis/04_experiments.py fusion all
      python3 analysis/04_experiments.py mnist-official
      python3 analysis/04_experiments.py memory
"""

import sys
import json
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np

# ── carrega o 01_common (nome começa com dígito → via importlib) ──────────────
_spec = importlib.util.spec_from_file_location(
    "common01", Path(__file__).resolve().parent / "01_common.py")
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)

CFG, FIG, PALETTE = common.CFG, common.FIG, common.PALETTE


# ══════════════════════════════════════════════════════════════════════════════
# fusão de evidência ponderada pela incerteza (na inferência)
# ══════════════════════════════════════════════════════════════════════════════
FUSION_SEEDS = list(range(10))


def _weighted_scores(model, X):
    """Score com cada célula ponderada por (1 - p) = confiança."""
    from wisard.encoder import apply_encoders
    Xb = apply_encoders(np.asarray(X, dtype=float), model._encoders)
    weighted_mem = (1.0 - model._memory_p) * model._memory_x
    return model._compute_scores(weighted_mem, Xb)


def _fusion_run(ds):
    from wisard.models import KalmanV2
    get_data, split = common.get_data, common.split
    cfg = json.load(open(CFG / f"{ds}_config.json"))
    a, r, k = cfg["address_size"], cfg["resolution"], cfg["kalman"]
    X, y = get_data(ds)

    std_accs, w_accs = [], []
    for seed in FUSION_SEEDS:
        X_tr, X_te, y_tr, y_te = split(X, y, seed)
        m = KalmanV2(address_size=a, resolution=r, seed=seed,
                     q=k["q"], r=k["r"], p0=k["p0"], n_epochs=k["n_epochs"]).fit(X_tr, y_tr)
        std_accs.append((m.predict(X_te) == y_te).mean())
        sc = _weighted_scores(m, X_te)
        w_accs.append((m.classes_[sc.argmax(1)] == y_te).mean())

    std_accs, w_accs = np.array(std_accs), np.array(w_accs)
    print(f"\n=== Fusão por incerteza — {ds.upper()} ({len(FUSION_SEEDS)} seeds) ===")
    print(f"  Padrão  (peso igual)     : {std_accs.mean():.4f} ± {std_accs.std():.4f}")
    print(f"  Pesado  (conf. = 1 - p)  : {w_accs.mean():.4f} ± {w_accs.std():.4f}")
    print(f"  Δ média                  : {(w_accs.mean()-std_accs.mean())*100:+.3f}pp")
    print(f"  Pesado venceu em         : {(w_accs > std_accs).sum()}/{len(FUSION_SEEDS)} seeds")
    return {"dataset": ds, "std": std_accs, "weighted": w_accs}


def _fusion_save_and_plot(all_results):
    import pandas as pd
    import matplotlib.pyplot as plt
    rows = []
    for res in all_results:
        for i, seed in enumerate(FUSION_SEEDS):
            rows.append({"dataset": res["dataset"], "seed": seed,
                         "acc_padrao": res["std"][i], "acc_pesado": res["weighted"][i]})
    csv_path = CFG / "fusion_experiment.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nCSV salvo → {csv_path}")

    datasets = [r["dataset"] for r in all_results]
    x = np.arange(len(datasets)); width = 0.35
    fig, ax = plt.subplots(figsize=(max(9, 1.1*len(datasets)), 6))
    fig.suptitle("Fusão de evidência por incerteza — padrão vs ponderado",
                 fontsize=13, fontweight="bold")
    std_means = [r["std"].mean() for r in all_results]
    w_means   = [r["weighted"].mean() for r in all_results]
    std_ci = [1.96*r["std"].std()/np.sqrt(len(FUSION_SEEDS)) for r in all_results]
    w_ci   = [1.96*r["weighted"].std()/np.sqrt(len(FUSION_SEEDS)) for r in all_results]
    b1 = ax.bar(x-width/2, std_means, width, yerr=std_ci, capsize=6,
                color="#999999", alpha=0.85, label="Padrão (peso igual)")
    b2 = ax.bar(x+width/2, w_means, width, yerr=w_ci, capsize=6,
                color=PALETTE["KalmanWiSARD"], alpha=0.9, label="Ponderado (conf. = 1−p)")
    for bars, means in [(b1, std_means), (b2, w_means)]:
        for bar, m in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, m, f"{m:.2%}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([d.upper() for d in datasets], rotation=45, ha="right")
    ax.set_ylabel("Acurácia média"); ax.legend(fontsize=9)
    ax.set_title("Ponderar votos pela confiança da célula (p) — só possível na KWiSARD")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout()
    out = FIG / "fusion_experiment.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo → {out}")

    # versão com ZOOM por dataset
    n = len(all_results)
    ncols = min(4, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.2*nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Fusão por incerteza — zoom na faixa relevante",
                 fontsize=13, fontweight="bold")
    for ax, res in zip(axes, all_results):
        s_m, w_m = res["std"].mean(), res["weighted"].mean()
        s_ci = 1.96*res["std"].std()/np.sqrt(len(FUSION_SEEDS))
        w_ci = 1.96*res["weighted"].std()/np.sqrt(len(FUSION_SEEDS))
        bars = ax.bar(["Padrão", "Ponderado"], [s_m, w_m], yerr=[s_ci, w_ci],
                      capsize=8, color=["#999999", PALETTE["KalmanWiSARD"]], alpha=0.9, width=0.5)
        for bar, m in zip(bars, [s_m, w_m]):
            ax.text(bar.get_x()+bar.get_width()/2, m, f"{m:.3%}",
                    ha="center", va="bottom", fontsize=9)
        lo, hi = min(s_m, w_m), max(s_m, w_m)
        pad = max(hi - lo, 0.002) * 3
        ax.set_ylim(lo - pad, hi + pad)
        wins = (res["weighted"] > res["std"]).sum()
        ax.set_title(f"{res['dataset'].upper()}  (Δ={(w_m-s_m)*100:+.3f}pp, {wins}/{len(FUSION_SEEDS)} seeds)")
        ax.set_ylabel("Acurácia média")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2%}"))
    plt.tight_layout()
    out_zoom = FIG / "fusion_experiment_zoom.png"
    plt.savefig(out_zoom, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Gráfico (zoom) salvo → {out_zoom}")


def run_fusion(arg="all"):
    if arg == "all":
        names = sorted(p.name.replace("_config.json", "") for p in CFG.glob("*_config.json")
                       if "_sparse_" not in p.name)
    else:
        names = [arg]
    results = []
    for ds in names:
        if not (CFG / f"{ds}_config.json").exists():
            print(f"  pulando {ds}: sem config (rode a busca densa antes)")
            continue
        results.append(_fusion_run(ds))
    if results:
        _fusion_save_and_plot(results)


# ══════════════════════════════════════════════════════════════════════════════
# MNIST split oficial (60k/10k) — comparável à literatura
# ══════════════════════════════════════════════════════════════════════════════
def run_mnist_official():
    from datasets.loaders import get_mnist
    from wisard.models import SparseClassicWiSARD, SparseKalman
    ADDR, RES, N_SEEDS = 28, 3, 30
    cfg = json.load(open(CFG / "mnist_full_sparse_config.json"))
    k = cfg["kalman"]
    print(f"Kalman da config: q={k['q']}, r={k['r']}\n")

    X, y = get_mnist(max_samples=None)  # 70.000, ordem canônica
    X_tr, y_tr = X[:60000], y[:60000]
    X_te, y_te = X[60000:], y[60000:]
    print(f"treino oficial = {len(X_tr)} | teste oficial = {len(X_te)}\n")

    acc_c, acc_k = [], []
    for seed in range(N_SEEDS):
        c = SparseClassicWiSARD(ADDR, RES, seed=seed).fit(X_tr, y_tr)
        acc_c.append(float((c.predict(X_te) == y_te).mean()))
        m = SparseKalman(ADDR, RES, seed=seed, q=k["q"], r=k["r"], p0=1.0, n_epochs=1).fit(X_tr, y_tr)
        acc_k.append(float((m.predict(X_te) == y_te).mean()))
        print(f"  seed {seed:2d}: clássica={acc_c[-1]:.4f} | kalman={acc_k[-1]:.4f}")

    acc_c, acc_k = np.array(acc_c), np.array(acc_k)
    print(f"\n{'='*60}")
    for name, a in [("Clássica b=0", acc_c), ("Kalman", acc_k)]:
        cv = a.std() / a.mean() * 100
        print(f"{name:14s}: média={a.mean():.4f} ± {a.std():.4f} | "
              f"CV={cv:.2f}% | pior={a.min():.4f} | melhor={a.max():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# redução de memória no MNIST — descartar o p
# ══════════════════════════════════════════════════════════════════════════════
def run_memory():
    from datasets.loaders import get_mnist
    from wisard.models import SparseClassicWiSARD, SparseKalman
    ADDR, RES = 28, 3
    cfg = json.load(open(CFG / "mnist_full_sparse_config.json"))
    k = cfg["kalman"]

    X, y = get_mnist(max_samples=None)
    X_tr, y_tr = X[:60000], y[:60000]
    X_te, y_te = X[60000:], y[60000:]

    c = SparseClassicWiSARD(ADDR, RES, seed=0).fit(X_tr, y_tr)
    m = SparseKalman(ADDR, RES, seed=0, q=k["q"], r=k["r"], p0=1.0, n_epochs=1).fit(X_tr, y_tr)

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


# ── dispatcher ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fusion":
        run_fusion(sys.argv[2] if len(sys.argv) > 2 else "all")
    elif cmd == "mnist-official":
        run_mnist_official()
    elif cmd == "memory":
        run_memory()
    else:
        print(__doc__)
        print("subcomandos: fusion [ds|all] | mnist-official | memory")
