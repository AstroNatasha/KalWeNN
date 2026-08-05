"""
06_regression_analysis — figuras e tabelas do estudo de regressão (roda depois do 05).

Lê results/regression/regression_report.csv (+ grid e baselines) e escreve em
results/figures/regression/. Cada figura sai como SVG transparente (entregável)
e PNG com fundo (conferência), no mesmo estilo neon do resto do trabalho.

  1  rmse_normalizado   RMSE/σ do alvo por dataset — melhor ReW vs melhor KalWeNN
  2  estabilidade_cv    CV% do RMSE entre as 30 seeds (dumbbell), ReW vs KalWeNN
  3  escopo_p           efeito de ONDE o p é mantido (célula vs compartilhado)
  4  inferencia         efeito de COMO as RAMs são agregadas (5 modos)
  5  estimadores_rew    as 6 formas de agregação da ReW
  6  heatmap_qr         RMSE de validação no grid (q, r), por dataset
  7  efeito_q           o ruído de processo q ajuda? (curva RMSE × q)
  tabela                tabela síntese (CSV + Markdown) com todas as métricas

Uso:  python3 analysis/06_regression_analysis.py            (tudo)
      python3 analysis/06_regression_analysis.py 2 4 tabela
"""

import sys
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 01_common: estilo neon e utilidades de figura ─────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "common01", Path(__file__).resolve().parent / "01_common.py")
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
NEON, set_neon_style, nospines, save_fig = (
    common.NEON, common.set_neon_style, common.nospines, common.save_fig)

from datasets.loaders_regression import get_regression, REGRESSION_DATASETS

ROOT    = Path(__file__).resolve().parents[1]
REG_DIR = ROOT / "results" / "regression"
REG_FIG = ROOT / "results" / "figures" / "regression"
REG_FIG.mkdir(parents=True, exist_ok=True)

DS_ORDER = REGRESSION_DATASETS
set_neon_style()


def load_report():
    df = pd.read_csv(REG_DIR / "regression_report.csv")
    df["dataset"] = pd.Categorical(df["dataset"], DS_ORDER, ordered=True)
    return df.sort_values("dataset")


def sigma_of(ds, _cache={}):
    """Desvio-padrão do alvo — usado para normalizar o RMSE entre datasets."""
    if ds not in _cache:
        _cache[ds] = float(get_regression(ds)[1].std())
    return _cache[ds]


def best_of(df, familia):
    """Melhor configuração (menor RMSE médio) de cada família, por dataset."""
    sub = df[df["familia"] == familia]
    return sub.loc[sub.groupby("dataset", observed=True)["rmse_mean"].idxmin()]


# ══════════════════════════════════════════════════════════════════════════════
def fig1_rmse_normalizado():
    df = load_report()
    rew, kal = best_of(df, "ReW"), best_of(df, "KalWeNN")
    ds = [d for d in DS_ORDER if d in set(rew["dataset"])]
    a = [rew.set_index("dataset").loc[d, "rmse_mean"] / sigma_of(d) for d in ds]
    b = [kal.set_index("dataset").loc[d, "rmse_mean"] / sigma_of(d) for d in ds]

    x = np.arange(len(ds)); w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - w/2, a, w, label="ReW (melhor variante)", color=NEON["CYAN"], alpha=.9)
    ax.bar(x + w/2, b, w, label="KalWeNN (melhor variante)", color=NEON["MAGENTA"], alpha=.9)
    ax.axhline(1.0, color=NEON["YELLOW"], ls="--", lw=1.2)
    ax.text(len(ds) - .4, 1.02, "prever a média", color=NEON["YELLOW"], fontsize=10, ha="right")
    for i, (va, vb) in enumerate(zip(a, b)):
        ax.text(i - w/2, va + .01, f"{va:.2f}", ha="center", fontsize=9, color=NEON["TEXTC"])
        ax.text(i + w/2, vb + .01, f"{vb:.2f}", ha="center", fontsize=9, color=NEON["TEXTC"])
    ax.set_xticks(x); ax.set_xticklabels(ds, rotation=20, ha="right")
    ax.set_ylabel("RMSE / σ do alvo  (menor é melhor)")
    ax.set_title("Erro normalizado por dataset")
    ax.legend(frameon=False); nospines(ax)
    save_fig(fig, "1_rmse_normalizado", out=REG_FIG)


def fig2_estabilidade_cv():
    df = load_report()
    rew, kal = best_of(df, "ReW"), best_of(df, "KalWeNN")
    ds = [d for d in DS_ORDER if d in set(rew["dataset"])]
    a = [rew.set_index("dataset").loc[d, "rmse_cv"] for d in ds]
    b = [kal.set_index("dataset").loc[d, "rmse_cv"] for d in ds]

    y = np.arange(len(ds))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (va, vb) in enumerate(zip(a, b)):
        ax.plot([va, vb], [i, i], color=NEON["LILAC"], lw=1.5, alpha=.6, zorder=1)
    # ReW como círculo vazado: onde os dois coincidem, os dois continuam visíveis
    ax.scatter(a, y, s=230, facecolors="none", edgecolors=NEON["CYAN"],
               linewidths=2.2, label="ReW", zorder=3)
    ax.scatter(b, y, s=110, color=NEON["MAGENTA"], label="KalWeNN", zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(ds)
    ax.set_xlabel("CV% do RMSE entre as 30 seeds  (menor = mais estável)")
    ax.set_title("Estabilidade entre execuções\n"
                 "(nos 7 datasets em que os círculos coincidem, a melhor KalWeNN "
                 "é o caso limite da ReW)", fontsize=14)
    ax.legend(frameon=False); nospines(ax)
    save_fig(fig, "2_estabilidade_cv", out=REG_FIG)


def fig3_escopo_p():
    """Para cada dataset, o melhor RMSE alcançado com p por célula vs p compartilhado."""
    df = load_report()
    kal = df[df["familia"] == "KalWeNN"]
    ds = [d for d in DS_ORDER if d in set(kal["dataset"])]
    scopes = ["cell", "ram", "global"]
    cols = [NEON["MAGENTA"], NEON["CYAN"], NEON["GREEN"]]

    x = np.arange(len(ds)); w = 0.26
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for j, (sc, c) in enumerate(zip(scopes, cols)):
        sub = kal[kal["p_scope"] == sc]
        vals = [sub[sub["dataset"] == d]["rmse_mean"].min() / sigma_of(d) for d in ds]
        ax.bar(x + (j - 1) * w, vals, w, label=f"p por {sc}", color=c, alpha=.9)
    ax.set_xticks(x); ax.set_xticklabels(ds, rotation=20, ha="right")
    ax.set_ylabel("RMSE / σ do alvo")
    ax.set_title("Onde manter a incerteza p  (ram e global são idênticos por construção)")
    ax.legend(frameon=False); nospines(ax)
    save_fig(fig, "3_escopo_p", out=REG_FIG)


def fig4_inferencia():
    """Efeito do modo de agregação, com p por célula (o único escopo em que
    conf/fusion se distinguem da média)."""
    df = load_report()
    kal = df[(df["familia"] == "KalWeNN") & (df["p_scope"] == "cell")]
    ds = [d for d in DS_ORDER if d in set(kal["dataset"])]
    modes = ["mean", "count", "conf", "fusion", "kalman_seq"]
    cols = [NEON["CYAN"], NEON["YELLOW"], NEON["GREEN"], NEON["MAGENTA"], NEON["LILAC"]]

    x = np.arange(len(ds)); w = 0.16
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for j, (m, c) in enumerate(zip(modes, cols)):
        sub = kal[kal["inference"] == m].set_index("dataset")
        vals = [sub.loc[d, "rmse_mean"] / sigma_of(d) if d in sub.index else np.nan for d in ds]
        ax.bar(x + (j - 2) * w, vals, w, label=m, color=c, alpha=.9)
    ax.set_xticks(x); ax.set_xticklabels(ds, rotation=20, ha="right")
    ax.set_ylabel("RMSE / σ do alvo")
    ax.set_title("Como agregar as RAMs na inferência (p por célula)")
    ax.legend(frameon=False, ncol=5); nospines(ax)
    save_fig(fig, "4_inferencia", out=REG_FIG)


def fig5_estimadores_rew():
    df = load_report()
    rew = df[df["familia"] == "ReW"]
    ds = [d for d in DS_ORDER if d in set(rew["dataset"])]
    from wisard.models_regression import REW_ESTIMATORS as ests
    cmap = plt.get_cmap("cool")
    cols = [cmap(i / max(len(ests) - 1, 1)) for i in range(len(ests))]

    x = np.arange(len(ds)); w = 0.8 / len(ests)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for j, (e, c) in enumerate(zip(ests, cols)):
        sub = rew[rew["inference"] == e].set_index("dataset")
        vals = [sub.loc[d, "rmse_mean"] / sigma_of(d) if d in sub.index else np.nan for d in ds]
        ax.bar(x + (j - (len(ests) - 1) / 2) * w, vals, w, label=e, color=c, alpha=.9)
    ax.set_xticks(x); ax.set_xticklabels(ds, rotation=20, ha="right")
    ax.set_ylabel("RMSE / σ do alvo")
    ax.set_title("Agregadores da ReW (fórmulas do wisardpkg v2.0.0a7)")
    ax.legend(frameon=False, ncol=8, fontsize=10); nospines(ax)
    save_fig(fig, "5_estimadores_rew", out=REG_FIG)


def fig6_heatmap_qr():
    """Mapa de calor do RMSE de validação no grid (q, r), p por célula,
    inferência 'count' — um painel por dataset."""
    files = sorted(REG_DIR.glob("*_grid.csv"))
    if not files:
        print("  sem *_grid.csv — rode 05_regression.py grid")
        return
    n = len(files)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, files):
        d = pd.read_csv(f)
        d = d[(d["p_scope"] == "cell") & (d["inference"] == "count")]
        piv = d.pivot_table(index="q", columns="r", values="val_rmse")
        im = ax.imshow(piv.values, cmap="magma", aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns], fontsize=8, rotation=45)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{i:g}" for i in piv.index], fontsize=8)
        ax.set_title(f.name.replace("_grid.csv", ""), fontsize=12)
        ax.set_xlabel("r", fontsize=9); ax.set_ylabel("q", fontsize=9)
        best = np.unravel_index(np.nanargmin(piv.values), piv.shape)
        ax.scatter([best[1]], [best[0]], marker="*", s=150,
                   color=NEON["CYAN"], edgecolor="white", linewidth=.5)
        fig.colorbar(im, ax=ax, fraction=.046)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("RMSE de validação no grid (q, r) — p por célula, inferência 'count'",
                 fontsize=15, color=NEON["TEXTC"])
    save_fig(fig, "6_heatmap_qr", out=REG_FIG)


def fig7_efeito_q():
    """A pergunta central do estudo, isolada: o ruído de processo q ajuda?

    Para cada dataset e cada valor de q, o menor RMSE de validação sobre todos
    os r (p por célula). q=0 é o caso limite ReW — a curva mostra o que se ganha
    ou se perde ao deixar a célula esquecer o passado.
    """
    files = sorted(REG_DIR.glob("*_grid.csv"))
    if not files:
        print("  sem *_grid.csv — rode 05_regression.py grid")
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("cool")
    for i, f in enumerate(files):
        ds = f.name.replace("_grid.csv", "")
        d = pd.read_csv(f)
        d = d[d["p_scope"] == "cell"]
        curve = d.groupby("q")["val_rmse"].min()
        base = curve.loc[0.0] if 0.0 in curve.index else curve.iloc[0]
        qs = curve.index.to_numpy()
        # q=0 no eixo log: desenhado como um ponto à esquerda do menor q>0
        qs_plot = np.where(qs == 0, min(q for q in qs if q > 0) / 10, qs)
        ax.plot(qs_plot, curve.to_numpy() / base * 100, "o-",
                color=cmap(i / max(len(files) - 1, 1)), label=ds, lw=2, ms=6)
    ax.axhline(100, color=NEON["YELLOW"], ls="--", lw=1.2)
    ax.text(ax.get_xlim()[0], 100.4, "referência: q=0 (limite ReW)",
            color=NEON["YELLOW"], fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("ruído de processo q  (o ponto mais à esquerda é q = 0)")
    ax.set_ylabel("RMSE de validação, % do valor em q=0")
    ax.set_title("O ruído de processo do Kalman ajuda na regressão?")
    ax.legend(frameon=False, ncol=4, fontsize=10); nospines(ax)
    save_fig(fig, "7_efeito_q", out=REG_FIG)


def _to_markdown(df):
    """Tabela em Markdown sem depender do pacote `tabulate`."""
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


def tabela():
    """Tabela síntese: melhor ReW vs melhor KalWeNN por dataset, todas as métricas."""
    df = load_report()
    rew, kal = best_of(df, "ReW"), best_of(df, "KalWeNN")
    base = (pd.read_csv(REG_DIR / "baselines.csv")
            if (REG_DIR / "baselines.csv").exists() else None)

    rows = []
    for d in [x for x in DS_ORDER if x in set(rew["dataset"])]:
        a = rew.set_index("dataset").loc[d]
        b = kal.set_index("dataset").loc[d]
        rec = {
            "dataset": d, "σ do alvo": round(sigma_of(d), 4),
            "tupla": int(a["address_size"]), "resolução": int(a["resolution"]),
            "ReW (variante)": a["inference"],
            "ReW RMSE": round(a["rmse_mean"], 4), "ReW MAE": round(a["mae_mean"], 4),
            "ReW MSE": round(a["mse_mean"], 4), "ReW R²": round(a["r2_mean"], 4),
            "ReW acc±10%": round(a["acc_10pct_mean"], 4),
            "ReW acc±0,25σ": round(a["acc_sigma_mean"], 4),
            "ReW CV%": round(a["rmse_cv"], 2),
            "KalWeNN (variante)": f"{b['p_scope']}/{b['inference']}",
            "KalWeNN q": b["q"], "KalWeNN r": b["r"],
            "KalWeNN RMSE": round(b["rmse_mean"], 4), "KalWeNN MAE": round(b["mae_mean"], 4),
            "KalWeNN MSE": round(b["mse_mean"], 4), "KalWeNN R²": round(b["r2_mean"], 4),
            "KalWeNN acc±10%": round(b["acc_10pct_mean"], 4),
            "KalWeNN acc±0,25σ": round(b["acc_sigma_mean"], 4),
            "KalWeNN CV%": round(b["rmse_cv"], 2),
            "Δ RMSE %": round((a["rmse_mean"] - b["rmse_mean"]) / a["rmse_mean"] * 100, 2),
            "Δ CV %": round(a["rmse_cv"] - b["rmse_cv"], 2),
        }
        if base is not None:
            sub = base[base["dataset"] == d]
            for m in ("linear", "kNN (k=5)", "árvore", "random forest"):
                v = sub[sub["modelo"] == m]["rmse_mean"]
                rec[f"RMSE {m}"] = round(float(v.iloc[0]), 4) if len(v) else np.nan
        rows.append(rec)

    t = pd.DataFrame(rows)
    t.to_csv(REG_DIR / "synthesis_regression.csv", index=False)
    (REG_DIR / "synthesis_regression.md").write_text(_to_markdown(t), encoding="utf-8")
    print(t.to_string(index=False))
    print(f"\n  salvo → {REG_DIR/'synthesis_regression.csv'}")
    print(f"  salvo → {REG_DIR/'synthesis_regression.md'}")


FIGS = {
    "1": fig1_rmse_normalizado, "2": fig2_estabilidade_cv, "3": fig3_escopo_p,
    "4": fig4_inferencia, "5": fig5_estimadores_rew, "6": fig6_heatmap_qr,
    "7": fig7_efeito_q, "tabela": tabela,
}

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    todo = list(FIGS) if args == ["all"] else [a for a in args if a in FIGS]
    for key in todo:
        print(f"\n[{key}]")
        FIGS[key]()
