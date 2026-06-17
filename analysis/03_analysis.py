"""
03_analysis — todas as figuras e tabelas do trabalho (rodar depois do 02).

Lê results/v2/tuple_comparison_report.csv (gerado pelo 02_search) e os
*_sparse_config.json da busca, e escreve em results/figures/presentation/.
Cada saída sai como SVG transparente (entregável) + PNG (conferência).

Figuras:
  1  mecanismo            — o filtro de Kalman em cada célula (x sobe, p cai)
  2  estabilidade_cv      — CV clássica vs KalWeNN (dumbbell)
  3  scatter_acc_cv       — acurácia × estabilidade (4 modelos)
  4  slope_acuracia       — clássica → KalWeNN
  5  ganho_dificuldade    — ganho do KalWeNN cresce nos problemas difíceis
  6  ganho_acuracia       — barras horizontais de ganho (regime denso)
  7  ganho_kalman_k       — ganho de Kalman k ao longo do treino (config MNIST)
  8  acuracia_barras      — acurácia por dataset, 4 modelos (± desvio)
  9  heatmaps_qr          — busca de q×r (4 datasets mais diferentes)
  10 cv_dominancia        — dominância de estabilidade (CV log-log)
  11 cv_barras            — CV por dataset, 4 modelos
  12 acuracia_por_seed    — acurácia × seed (recomputa; demonstra estabilidade)
  table                   — tabela_6datasets + tabela_completa

Uso:  python3 analysis/03_analysis.py            (tudo)
      python3 analysis/03_analysis.py 8 11 table  (apenas algumas)
      python3 analysis/03_analysis.py 12 vowel    (fig 12 com outro dataset)
"""

import sys
import json
import glob
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── carrega o 01_common (nome começa com dígito → via importlib) ──────────────
_spec = importlib.util.spec_from_file_location(
    "common01", Path(__file__).resolve().parent / "01_common.py")
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)

NEON, PRESENTATION, REPORT_CSV, CFG = common.NEON, common.PRESENTATION, common.REPORT_CSV, common.CFG
MODEL_KEYS, MODEL_DISPLAY = common.MODEL_KEYS, common.MODEL_DISPLAY
nospines, save_fig = common.nospines, common.save_fig

# paleta como nomes curtos
BG, CARD, BAND = NEON["BG"], NEON["CARD"], NEON["BAND"]
MAGENTA, CYAN, YELLOW, GREEN = NEON["MAGENTA"], NEON["CYAN"], NEON["YELLOW"], NEON["GREEN"]
TEXTC, LILAC, VIOLET = NEON["TEXTC"], NEON["LILAC"], NEON["VIOLET"]
CLASSIC_C, KALMAN_C, TITLE_C = LILAC, GREEN, CYAN

# regime grande (mesmos nomes do CSV) — usado nas figuras "clássica vs KalWeNN"
CLASSIC, KALMAN = "DictWisard", "Dict Kalman WNN"


def _df():
    return pd.read_csv(REPORT_CSV)


# ── 1–5: figuras da apresentação ──────────────────────────────────────────────

def fig_mechanism():
    q, p0 = 1e-4, 1.0
    n = np.arange(0, 16)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 7.5))
    for r, col in [(0.01, CYAN), (0.1, GREEN), (1.0, YELLOW), (2.0, MAGENTA)]:
        x, p, xs, ps = 0.0, p0, [], []
        for _ in n:
            xs.append(x); ps.append(p)
            k = (p+q)/(p+q+r); x = x + k*(1-x); p = (1-k)*(p+q)
        a1.plot(n, xs, marker="o", ms=4, lw=2.5, color=col, label=f"r = {r:g}")
        a2.plot(n, ps, marker="o", ms=4, lw=2.5, color=col, label=f"r = {r:g}")
    a1.set_title("Estimativa $x$ cresce", color=TEXTC)
    a1.set_xlabel("nº de ativações"); a1.set_ylabel("$x$")
    a2.set_title("Incerteza $p$ cai", color=TEXTC)
    a2.set_xlabel("nº de ativações"); a2.set_ylabel("$p$")
    for ax in (a1, a2):
        nospines(ax)
        leg = ax.legend(frameon=False, fontsize=12)
        for t in leg.get_texts(): t.set_color(TEXTC)
    fig.suptitle("O filtro de Kalman em cada célula", fontsize=19, fontweight="bold", color=TITLE_C)
    save_fig(fig, "1_mecanismo")


def fig_stability():
    cv = _df().pivot(index="dataset", columns="modelo", values="CV_%")
    d = cv[[CLASSIC, KALMAN]].dropna().sort_values(CLASSIC)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for yi, (ds, row) in zip(y, d.iterrows()):
        ax.plot([row[KALMAN], row[CLASSIC]], [yi, yi], color=VIOLET, lw=2.5, zorder=1)
    ax.scatter(d[CLASSIC], y, s=95, color=CLASSIC_C, zorder=2, label="WiSARD clássica")
    ax.scatter(d[KALMAN], y, s=95, color=KALMAN_C, zorder=3, label="KalWeNN")
    ax.set_yticks(y); ax.set_yticklabels(d.index, color=TEXTC)
    ax.set_xlabel("Coeficiente de variação (%)  —  menor é melhor")
    ax.set_title("Estabilidade entre execuções", fontsize=19, fontweight="bold", color=TITLE_C)
    nospines(ax)
    leg = ax.legend(frameon=False, loc="lower right")
    for t in leg.get_texts(): t.set_color(TEXTC)
    save_fig(fig, "2_estabilidade_cv")


def fig_scatter():
    df = _df()
    fig, ax = plt.subplots(figsize=(10, 7.5))
    styles = {
        "Wisard": (CLASSIC_C, "o", "Clássica (pequena)"),
        "DictWisard": (CLASSIC_C, "s", "Clássica (grande)"),
        "Kalman WNN": (KALMAN_C, "o", "KalWeNN (pequena)"),
        "Dict Kalman WNN": (KALMAN_C, "s", "KalWeNN (grande)"),
    }
    for modelo, (col, mk, lab) in styles.items():
        sub = df[df.modelo == modelo]
        ax.scatter(sub["CV_%"], sub["acuracia"], s=85, color=col, marker=mk,
                   edgecolors=BG, linewidths=0.9, alpha=0.95, label=lab)
    ax.set_xlabel("CV (%)  —  menor = mais estável")
    ax.set_ylabel("Acurácia  —  maior = melhor")
    ax.set_title("Acurácia × estabilidade", fontsize=19, fontweight="bold", color=TITLE_C)
    ax.annotate("melhor", xy=(0.2, 0.97), xytext=(2.2, 0.86),
                fontsize=14, color=MAGENTA, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=MAGENTA, lw=2.2))
    nospines(ax)
    leg = ax.legend(frameon=False, fontsize=11, loc="lower right")
    for t in leg.get_texts(): t.set_color(TEXTC)
    save_fig(fig, "3_scatter_acc_cv")


def fig_slope():
    acc = _df().pivot(index="dataset", columns="modelo", values="acuracia")
    d = acc[[CLASSIC, KALMAN]].dropna().sort_values(KALMAN)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for ds, row in d.iterrows():
        col = KALMAN_C if row[KALMAN] - row[CLASSIC] > 0.002 else LILAC
        ax.plot([0, 1], [row[CLASSIC], row[KALMAN]], color=col, lw=2.2, marker="o", ms=7)
        ax.text(1.04, row[KALMAN], ds, va="center", fontsize=11, color=TEXTC)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["WiSARD\nclássica", "KalWeNN"], color=TEXTC)
    ax.set_xlim(-0.15, 1.45); ax.set_ylabel("Acurácia")
    ax.set_title("Acurácia: clássica → KalWeNN", fontsize=19, fontweight="bold", color=TITLE_C)
    nospines(ax)
    save_fig(fig, "4_slope_acuracia")


def fig_gain_difficulty():
    acc = _df().pivot(index="dataset", columns="modelo", values="acuracia")
    d = acc[[CLASSIC, KALMAN]].dropna()
    gain = (d[KALMAN] - d[CLASSIC]) * 100
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.scatter(d[CLASSIC], gain, s=95, color=KALMAN_C, edgecolors=BG, linewidths=0.9, zorder=3)
    for ds in d.index:
        ax.annotate(ds, (d[CLASSIC][ds], gain[ds]), xytext=(6, 4),
                    textcoords="offset points", fontsize=10, color=LILAC)
    z = np.polyfit(d[CLASSIC], gain, 1); xs = np.linspace(d[CLASSIC].min(), d[CLASSIC].max(), 50)
    ax.plot(xs, np.polyval(z, xs), "--", color=YELLOW, lw=2.2, zorder=2)
    ax.axhline(0, color=VIOLET, lw=1.2)
    ax.set_xlabel("Acurácia da clássica  (mais difícil →← mais fácil)")
    ax.set_ylabel("Ganho do KalWeNN (pontos %)")
    ax.set_title("O ganho cresce nos problemas difíceis", fontsize=19, fontweight="bold", color=TITLE_C)
    nospines(ax)
    save_fig(fig, "5_ganho_dificuldade")


# ── 6: ganho em barras (regime denso) ─────────────────────────────────────────

def fig_gain_bars():
    piv = _df().pivot(index="dataset", columns="modelo", values="acuracia")
    gain = ((piv["Kalman WNN"] - piv["Wisard"]) * 100).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.barh(gain.index, gain.values, color=GREEN, height=0.66, zorder=3)
    for ds, v in gain.items():
        ax.text(v + 0.3, ds, f"+{v:.1f}", va="center", ha="left",
                fontsize=12, color=TEXTC, fontweight="bold")
    ax.axvline(0, color=VIOLET, lw=1.2)
    ax.set_xlabel("Ganho de acurácia (pontos %)")
    ax.set_title("Quanto o KalWeNN ganha sobre a WiSARD clássica",
                 fontsize=19, fontweight="bold", color=CYAN)
    ax.set_xlim(0, gain.max() * 1.15)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(colors=LILAC)
    for t in ax.get_yticklabels(): t.set_color(TEXTC)
    save_fig(fig, "6_ganho_acuracia")


# ── 7: ganho de Kalman k ao longo do treino (config do MNIST) ─────────────────

def fig_gain_k():
    q, r, p0 = 1e-4, 2.0, 1.0
    n = np.arange(1, 21)
    p, ks = p0, []
    for _ in n:
        p_pred = p + q
        k = p_pred / (p_pred + r)
        ks.append(k)
        p = (1 - k) * p_pred
    ks = np.array(ks)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.plot(n, ks, marker="o", ms=7, lw=3, color=GREEN, zorder=3)
    ax.axhline(0, color=VIOLET, lw=1.0)
    ax.annotate("ganho alto:\nconfia na 1ª observação", xy=(n[0], ks[0]), xytext=(3.6, 0.295),
                fontsize=12, color=CYAN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.8))
    ax.annotate("já confiante:\ncada ativação mexe menos", xy=(n[-1], ks[-1]), xytext=(12.5, ks[-1]+0.06),
                fontsize=12, color=YELLOW, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1.8))
    ax.set_xlabel("nº de ativações da célula")
    ax.set_ylabel("ganho de Kalman  $k$")
    ax.set_title("Ganho de Kalman ($k$) ao longo do treino — configuração do MNIST",
                 fontsize=18, fontweight="bold", color=CYAN)
    ax.set_xticks(n[::2])
    ax.text(0.99, 0.95, "$q = 10^{-4}$,  $r = 2{,}0$", transform=ax.transAxes,
            ha="right", va="top", fontsize=13, color=LILAC)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(colors=LILAC)
    save_fig(fig, "7_ganho_kalman_k")
    print(f"  k inicial={ks[0]:.3f}, k final={ks[-1]:.3f}")


# ── 8: acurácia em barras, 4 modelos (± desvio) ───────────────────────────────

def fig_acc_bars():
    df = _df()
    COLORS = [LILAC, CYAN, VIOLET, GREEN]
    ds_order = list(dict.fromkeys(df["dataset"]))
    acc = df.pivot(index="dataset", columns="modelo", values="acuracia")[MODEL_KEYS].loc[ds_order]
    std = df.pivot(index="dataset", columns="modelo", values="desvio")[MODEL_KEYS].loc[ds_order]
    x = np.arange(len(acc)); w = 0.2
    fig, ax = plt.subplots(figsize=(12, 6.75))
    for i, (m, c) in enumerate(zip(MODEL_KEYS, COLORS)):
        ax.bar(x + (i-1.5)*w, acc[m], w*0.92, yerr=std[m], label=MODEL_DISPLAY[m], color=c, zorder=3,
               error_kw=dict(ecolor=TEXTC, elinewidth=1, capsize=2, alpha=0.8))
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xticks(x); ax.set_xticklabels(acc.index, rotation=40, ha="right", color=TEXTC)
    ax.set_ylabel("Acurácia")
    ax.set_title("Acurácia por dataset — 4 modelos (± desvio)", fontsize=18, fontweight="bold",
                 color=CYAN, pad=12)
    ax.grid(axis="y", color=VIOLET, alpha=0.3, lw=0.7, zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(colors=LILAC)
    leg = ax.legend(frameon=False, loc="upper center", ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.18))
    for t in leg.get_texts(): t.set_color(TEXTC)
    save_fig(fig, "8_acuracia_barras")


# ── 9: heatmaps da busca q×r (4 datasets mais diferentes) ─────────────────────

def fig_heatmaps():
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.spatial.distance import pdist, squareform
    CMAP = LinearSegmentedColormap.from_list("neon", ["#241046", VIOLET, CYAN, GREEN])

    grids, qv, rv = {}, None, None
    for f in sorted(glob.glob(str(CFG / "*_sparse_config.json"))):
        ds = Path(f).name.replace("_sparse_config.json", "").replace("_full", "")
        c = json.load(open(f)); g = np.array(c["kalman_grid"])
        if g.shape != (4, 10): continue
        grids[ds] = g; qv, rv = c["q_vals"], c["r_vals"]

    keys = list(grids)
    norm = lambda g: ((g-g.min())/(g.max()-g.min()+1e-9)).ravel()
    X = np.array([norm(grids[k]) for k in keys])
    D = squareform(pdist(X))
    i, j = np.unravel_index(np.argmax(D), D.shape); sel = [i, j]
    while len(sel) < 4:
        rest = [k for k in range(len(keys)) if k not in sel]
        sel.append(max(rest, key=lambda k: min(D[k][s] for s in sel)))
    chosen = [keys[i] for i in sel]
    print("  escolhidos:", chosen)

    qlab = [f"{v:g}" for v in qv]; rlab = [f"{v:g}" for v in rv]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.0))
    for ax, ds in zip(axes.ravel(), chosen):
        g = grids[ds]
        ax.imshow(g, cmap=CMAP, aspect="auto", vmin=g.min(), vmax=g.max())
        bi, bj = np.unravel_index(g.argmax(), g.shape)
        ax.add_patch(plt.Rectangle((bj-0.5, bi-0.5), 1, 1, fill=False, edgecolor=MAGENTA, lw=2.5))
        for a in range(g.shape[0]):
            for b in range(g.shape[1]):
                nv = (g[a, b]-g.min())/(g.max()-g.min()+1e-9)
                ax.text(b, a, f"{g[a,b]*100:.1f}", ha="center", va="center", fontsize=6,
                        color=BG if nv > 0.6 else TEXTC)
        ax.set_xticks(range(len(rlab))); ax.set_xticklabels(rlab, fontsize=7, rotation=45)
        ax.set_yticks(range(len(qlab))); ax.set_yticklabels(qlab, fontsize=8)
        ax.set_xlabel("r (ruído de observação)", fontsize=9)
        ax.set_ylabel("q", fontsize=9)
        ax.set_title(ds, fontsize=14, fontweight="bold", color=CYAN)
        for sp in ax.spines.values(): sp.set_color(VIOLET)
    fig.suptitle("Busca dos hiperparâmetros do Kalman (acurácia por q × r)",
                 fontsize=17, fontweight="bold", color=CYAN, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PRESENTATION/"9_heatmaps_qr.svg", format="svg", transparent=True, bbox_inches="tight")
    fig.savefig(PRESENTATION/"9_heatmaps_qr.png", facecolor=BG, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  salvo → 9_heatmaps_qr.svg / .png")


# ── 10: dominância de estabilidade (CV log-log) ───────────────────────────────

def fig_cv_scatter():
    cv = _df().pivot(index="dataset", columns="modelo", values="CV_%")
    xC, yK = cv["Wisard"], cv["Kalman WNN"]
    lo, hi = 0.1, 6.0
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.fill([lo, hi, hi], [lo, hi, lo], color=GREEN, alpha=0.08, zorder=0)
    ax.plot([lo, hi], [lo, hi], "--", color=LILAC, lw=1.5, zorder=1)
    ax.scatter(xC, yK, s=120, color=GREEN, edgecolors=BG, linewidths=1.2, zorder=3)
    for ds in cv.index:
        ax.annotate(ds, (xC[ds], yK[ds]), xytext=(7, 3), textcoords="offset points",
                    fontsize=10, color=TEXTC)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("CV da WiSARD (%)"); ax.set_ylabel("CV do KalWeNN (%)")
    ax.set_title("Estabilidade: cada ponto abaixo da linha = KalWeNN mais estável",
                 fontsize=16, fontweight="bold", color=CYAN)
    ax.annotate("KalWeNN mais estável", xy=(2.0, 0.23), fontsize=13, color=GREEN, fontweight="bold")
    ax.annotate("(menor CV = melhor)", xy=(0.13, 4.2), fontsize=11, color=LILAC, style="italic")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(colors=LILAC, which="both")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5]); ax.set_xticklabels(["0,1", "0,2", "0,5", "1", "2", "5"])
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5]); ax.set_yticklabels(["0,1", "0,2", "0,5", "1", "2", "5"])
    save_fig(fig, "10_cv_dominancia")


# ── 11: CV em barras, 4 modelos ───────────────────────────────────────────────

def fig_cv_bars():
    df = _df()
    COLORS = [LILAC, CYAN, VIOLET, GREEN]
    ds_order = list(dict.fromkeys(df["dataset"]))
    cv = df.pivot(index="dataset", columns="modelo", values="CV_%")[MODEL_KEYS].loc[ds_order]
    x = np.arange(len(cv)); w = 0.2
    fig, ax = plt.subplots(figsize=(12, 6.75))
    for i, (m, c) in enumerate(zip(MODEL_KEYS, COLORS)):
        ax.bar(x + (i-1.5)*w, cv[m], w*0.92, label=MODEL_DISPLAY[m], color=c, zorder=3)
    ax.set_ylim(0, cv.values.max()*1.12)
    ax.set_xticks(x); ax.set_xticklabels(cv.index, rotation=40, ha="right", color=TEXTC)
    ax.set_ylabel("Coeficiente de variação (%)")
    ax.set_title("Estabilidade por dataset — CV dos 4 modelos (menor = melhor)",
                 fontsize=18, fontweight="bold", color=CYAN, pad=12)
    ax.grid(axis="y", color=VIOLET, alpha=0.3, lw=0.7, zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(colors=LILAC)
    leg = ax.legend(frameon=False, loc="upper center", ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.18))
    for t in leg.get_texts(): t.set_color(TEXTC)
    save_fig(fig, "11_cv_barras")


# ── 12: acurácia × seed (recomputa; demonstra estabilidade) ───────────────────

def fig_acc_seed(ds="letter"):
    from wisard.models import ClassicWiSARD, KalmanV2
    get_data, split = common.get_data, common.split
    ADDR, RES = 8, 16
    X, y = get_data(ds)

    Q, R = [1e-4, 1e-3, 1e-2], [1e-2, 1e-1, 5e-1, 1.0, 2.0]
    best = (-1, None)
    for q in Q:
        for r in R:
            accs = []
            for sd in range(3):
                Xtr, Xte, ytr, yte = split(X, y, sd)
                m = KalmanV2(ADDR, RES, seed=sd, q=q, r=r, p0=1.0, n_epochs=1).fit(Xtr, ytr)
                accs.append((m.predict(Xte) == yte).mean())
            if np.mean(accs) > best[0]: best = (np.mean(accs), (q, r))
    q, r = best[1]
    print(f"  melhor (q,r) = {q:g},{r:g}")

    aw, ak = [], []
    for sd in range(30):
        Xtr, Xte, ytr, yte = split(X, y, sd)
        aw.append((ClassicWiSARD(ADDR, RES, seed=sd, bleach=0).fit(Xtr, ytr).predict(Xte) == yte).mean())
        ak.append((KalmanV2(ADDR, RES, seed=sd, q=q, r=r, p0=1.0, n_epochs=1).fit(Xtr, ytr).predict(Xte) == yte).mean())
    aw, ak = np.array(aw)*100, np.array(ak)*100
    cvw, cvk = aw.std()/aw.mean()*100, ak.std()/ak.mean()*100

    xx = np.arange(len(aw))
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.plot(xx, aw, "-o", ms=5, lw=2, color=LILAC, label=f"WiSARD  (CV {cvw:.2f}%)")
    ax.plot(xx, ak, "-o", ms=5, lw=2, color=GREEN, label=f"KalWeNN  (CV {cvk:.2f}%)")
    ax.axhline(aw.mean(), color=LILAC, lw=1, ls="--", alpha=0.6)
    ax.axhline(ak.mean(), color=GREEN, lw=1, ls="--", alpha=0.6)
    ax.set_ylim(min(aw.min(), ak.min()) - 2, max(aw.max(), ak.max()) + 2)
    ax.set_xlim(-0.5, len(aw)-0.5)
    ax.set_xlabel("seed"); ax.set_ylabel("Acurácia (%)")
    ax.set_title(f"Acurácia por seed — {ds} (regime denso)", fontsize=18, fontweight="bold", color=CYAN)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(colors=LILAC)
    leg = ax.legend(frameon=False, loc="center right", fontsize=13)
    for t in leg.get_texts(): t.set_color(TEXTC)
    save_fig(fig, "12_acuracia_por_seed")


# ── tabela (6 datasets em destaque + completa) ────────────────────────────────

def _table(df_all, selected, name):
    df = df_all[df_all["dataset"].isin(selected)].copy()
    df["__o"] = df["dataset"].map({d: i for i, d in enumerate(selected)})
    df = df.sort_values("__o", kind="mergesort").drop(columns="__o")

    headers = ["Dataset", "Modelo", "Tupla", "Term.", "Acur.", "Desvio", "CV %", "Treino s", "Teste s", "Mem. MB"]
    widths  = [0.155, 0.165, 0.066, 0.066, 0.084, 0.084, 0.075, 0.092, 0.085, 0.085]

    rows, ds_order = [], []
    for _, r in df.iterrows():
        rows.append([
            r["dataset"], MODEL_DISPLAY.get(r["modelo"], r["modelo"]),
            f'{int(r["tupla"])}', f'{int(r["termometro"])}',
            f'{r["acuracia"]:.3f}', f'{r["desvio"]:.3f}', f'{r["CV_%"]:.2f}',
            f'{r["treino_s"]:.3f}', f'{r["teste_s"]:.3f}', f'{r["memoria_MB"]:.3f}',
        ])
        if r["dataset"] not in ds_order:
            ds_order.append(r["dataset"])

    # negrito: melhor acurácia (col 4) e menor CV (col 6) por dataset
    HL, base = set(), 0
    for ds, g in df.groupby("dataset", sort=False):
        HL.add((base + int(g["acuracia"].values.argmax()), 4))
        HL.add((base + int(g["CV_%"].values.argmin()), 6))
        base += len(g)

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98]); ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers, colWidths=widths, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(13)
    tbl.scale(1, 1.7)
    for (rr, cc), cell in tbl.get_celld().items():
        cell.set_edgecolor(VIOLET); cell.set_linewidth(0.4)
        if rr == 0:
            cell.set_facecolor(CARD)
            cell.get_text().set_color(CYAN); cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(14)
            continue
        modelo, ds = rows[rr-1][1], rows[rr-1][0]
        cell.set_facecolor(BAND if ds_order.index(ds) % 2 == 0 else "none")
        t = cell.get_text()
        if cc == 0:
            t.set_color(LILAC); t.set_fontweight("bold")
        elif cc == 1:
            t.set_color(GREEN if "KalWeNN" in modelo else LILAC); t.set_fontweight("bold")
        else:
            t.set_color(TEXTC)
        if (rr-1, cc) in HL:
            t.set_fontweight("bold")
    for fname, kw in [(f"{name}.svg", dict(transparent=True)), (f"{name}.png", dict(facecolor=BG))]:
        fig.savefig(PRESENTATION/fname, bbox_inches="tight", **kw)
        print("  salvo →", PRESENTATION/fname)
    plt.close(fig)


def fig_tables():
    df = _df()
    seis = ["mnist", "letter", "satellite", "vowel", "pendigits", "texture"]
    seis = [d for d in seis if d in set(df["dataset"])]
    _table(df, seis, "tabela_6datasets")
    _table(df, list(dict.fromkeys(df["dataset"])), "tabela_completa")


# ── dispatcher ────────────────────────────────────────────────────────────────
FIGS = {
    "1": fig_mechanism, "2": fig_stability, "3": fig_scatter, "4": fig_slope,
    "5": fig_gain_difficulty, "6": fig_gain_bars, "7": fig_gain_k, "8": fig_acc_bars,
    "9": fig_heatmaps, "10": fig_cv_scatter, "11": fig_cv_bars, "12": fig_acc_seed,
    "table": fig_tables,
}

if __name__ == "__main__":
    common.set_neon_style()
    args = sys.argv[1:] or ["all"]
    # fig 12 aceita um dataset opcional: "12 vowel"
    seed_ds = None
    if "12" in args:
        idx = args.index("12")
        if idx + 1 < len(args) and args[idx+1] not in FIGS:
            seed_ds = args.pop(idx+1)

    todo = list(FIGS) if args == ["all"] else [a for a in args if a in FIGS]
    print(f"Gerando: {', '.join(todo)}")
    for key in todo:
        print(f"[{key}]")
        if key == "12" and seed_ds:
            fig_acc_seed(seed_ds)
        else:
            FIGS[key]()
    print("Pronto.")
