"""
01_common — base compartilhada de todo o estudo (rodar primeiro).

Reúne tudo que os outros arquivos numerados reaproveitam:
  - carregamento dos datasets e split treino/teste;
  - backend de memória (denso / esparso) e fábricas dos modelos;
  - avaliação genérica sobre várias seeds (acurácia, F1, tempos, memória);
  - nomes e paleta neon usados nas figuras (02/03);
  - caminhos padrão de resultados e figuras.

Não é executável por conta própria — é importado pelos demais via importlib
(o nome começa com dígito, então não dá para "import 01_common").

Desenho experimental:
  - address_size e resolution FIXOS e IGUAIS entre os modelos (isola o update).
  - 80% treino / 20% teste; MNIST usa o split OFICIAL 60k/10k.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split

from wisard.models import ClassicWiSARD, KalmanV2, SparseClassicWiSARD, SparseKalman
from datasets.loaders import (
    get_wine, get_iris, get_breast_cancer, get_digits, get_glass,
    get_vehicle, get_segment, get_vowel, get_satellite, get_mnist,
    get_pendigits, get_optdigits, get_letter, get_texture, get_waveform,
    get_spambase, get_ionosphere, get_sonar, get_yeast,
    get_mfeat_fourier, get_mfeat_karhunen,
)

# ── caminhos ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
FIG          = ROOT / "results" / "figures" / "experiments"
CFG          = ROOT / "results" / "results"
PRESENTATION = ROOT / "results" / "figures" / "presentation"
REPORT_CSV   = CFG / "tuple_comparison_report.csv"
for _d in (FIG, CFG, PRESENTATION):
    _d.mkdir(parents=True, exist_ok=True)

# ── configuração global ───────────────────────────────────────────────────────
TEST_SIZE   = 0.20
# nº de seeds ajustável por ambiente (útil para runs pesados em modo sparse):
#   WISARD_GRID_SEEDS=2 WISARD_FINAL_SEEDS=5 python3 ...
GRID_SEEDS  = list(range(int(os.environ.get("WISARD_GRID_SEEDS", "5"))))    # busca de hiperparâmetros
FINAL_SEEDS = list(range(int(os.environ.get("WISARD_FINAL_SEEDS", "30"))))  # comparação final
CURVE_SEEDS = list(range(int(os.environ.get("WISARD_CURVE_SEEDS", "10"))))  # learning curve / épocas

# ── nomes dos 4 modelos ───────────────────────────────────────────────────────
# Chaves internas (como aparecem no CSV) → nomes de exibição (figuras/tabelas).
MODEL_KEYS = ["Wisard", "Kalman WNN", "DictWisard", "Dict Kalman WNN"]
MODEL_DISPLAY = {
    "Wisard":          "WiSARD",
    "Kalman WNN":      "KalWeNN",
    "DictWisard":      "WiSARD (dict)",
    "Dict Kalman WNN": "KalWeNN (dict)",
}

# ── paleta neon/dark (figuras da apresentação, SVG transparente) ──────────────
NEON = dict(
    BG="#1E0B36", CARD="#3A1D6E", BAND="#2A1450",
    MAGENTA="#FF2E92", CYAN="#00E5FF", YELLOW="#FFD23F", GREEN="#3DFF8E",
    TEXTC="#F4EFFF", LILAC="#B9A7E0", VIOLET="#6C2BD9",
)

# paleta antiga (usada por figuras de experimento em 04)
PALETTE = {
    "Classica":         "#4C72B0",   # WiSARD clássica (bleach=0)
    "Classica+Bleach":  "#55A868",   # WiSARD clássica com bleaching
    "KalmanWiSARD":     "#DD8452",   # KWiSARD
}
MODEL_LABELS = {
    "Classica":        "WiSARD\nClássica",
    "Classica+Bleach": "Clássica\n+ Bleach",
    "KalmanWiSARD":    "Kalman\nWiSARD",
}


def set_neon_style():
    """Aplica o estilo neon base do matplotlib (fundo transparente, eixos lilás)."""
    import matplotlib.pyplot as plt
    n = NEON
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 18, "axes.labelsize": 14,
        "axes.edgecolor": n["VIOLET"], "axes.linewidth": 1.2,
        "axes.facecolor": "none", "figure.facecolor": "none",
        "xtick.color": n["LILAC"], "ytick.color": n["LILAC"],
        "text.color": n["TEXTC"], "axes.labelcolor": n["TEXTC"],
        "savefig.dpi": 150, "font.family": "DejaVu Sans",
    })


def nospines(ax):
    """Remove as bordas de cima/direita e pinta os ticks de lilás."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=NEON["LILAC"])


def save_fig(fig, name, out=PRESENTATION, tight=True):
    """Salva SVG transparente (entregável) + PNG com fundo (conferência)."""
    if tight:
        fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.svg", format="svg", transparent=True, bbox_inches="tight")
    fig.savefig(out / f"{name}.png", facecolor=NEON["BG"], bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("  salvo →", out / f"{name}.svg")


# ── backend de memória: "dense" (padrão) ou "sparse" (tuplas grandes) ─────────
# Selecionado por variável de ambiente, sem editar código:
#   WISARD_BACKEND=sparse python3 analysis/02_search.py mnist
BACKEND = os.environ.get("WISARD_BACKEND", "dense").lower()
if BACKEND not in ("dense", "sparse"):
    raise ValueError(f"WISARD_BACKEND inválido: {BACKEND!r} (use 'dense' ou 'sparse')")
TAG = "" if BACKEND == "dense" else f"_{BACKEND}"

_CLASSIC = {"dense": ClassicWiSARD, "sparse": SparseClassicWiSARD}
_KALMAN  = {"dense": KalmanV2,       "sparse": SparseKalman}


def make_classic(**kw):
    return _CLASSIC[BACKEND](**kw)


def make_kalman(**kw):
    return _KALMAN[BACKEND](**kw)


# ── datasets ──────────────────────────────────────────────────────────────────
_LOADERS = {
    "wine": get_wine,
    "iris": get_iris,
    "breast_cancer": get_breast_cancer,
    "digits": get_digits,
    "glass": get_glass,
    "vehicle": get_vehicle,
    "segment": get_segment,
    "vowel": get_vowel,
    "satellite": get_satellite,
    # extras
    "pendigits": get_pendigits,
    "optdigits": get_optdigits,
    "letter": get_letter,
    "texture": get_texture,
    "waveform": get_waveform,
    "spambase": get_spambase,
    "ionosphere": get_ionosphere,
    "sonar": get_sonar,
    "yeast": get_yeast,
    "mfeat_fourier": get_mfeat_fourier,
    "mfeat_karhunen": get_mfeat_karhunen,
}
DATASETS = list(_LOADERS) + ["mnist"]


def get_data(name: str):
    if name == "mnist":
        return get_mnist(max_samples=10_000)
    if name == "mnist_full":
        return get_mnist(max_samples=None)  # 70.000 amostras
    if name in _LOADERS:
        return _LOADERS[name]()
    raise ValueError(f"dataset desconhecido: {name}. Opções: {DATASETS}")


def split(X, y, seed):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)


# ── avaliação genérica ────────────────────────────────────────────────────────
METRIC_KEYS = ["acc", "f1_macro", "precision_macro", "recall_macro",
               "balanced_acc", "kappa", "train_t", "test_t", "mem"]
METRIC_LABELS = {
    "acc": "Acurácia", "f1_macro": "F1 (macro)",
    "precision_macro": "Precisão (macro)", "recall_macro": "Recall (macro)",
    "balanced_acc": "Acur. balanceada", "kappa": "Cohen's κ",
    "train_t": "Tempo treino (s)", "test_t": "Tempo inferência (s)",
    "mem": "Memória (bytes)",
}


def eval_over_seeds(make_model, X, y, seeds):
    """make_model(seed) -> modelo. Retorna dict de arrays (uma entrada por seed)
    com acurácia, F1/precisão/recall macro, acurácia balanceada, kappa,
    tempos de treino/inferência e memória."""
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        balanced_accuracy_score, cohen_kappa_score,
    )
    m = {k: [] for k in METRIC_KEYS}
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = split(X, y, seed)
        model = make_model(seed)

        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        m["train_t"].append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        pred = model.predict(X_te)
        m["test_t"].append(time.perf_counter() - t1)

        m["acc"].append(float((pred == y_te).mean()))
        m["f1_macro"].append(f1_score(y_te, pred, average="macro", zero_division=0))
        m["precision_macro"].append(precision_score(y_te, pred, average="macro", zero_division=0))
        m["recall_macro"].append(recall_score(y_te, pred, average="macro", zero_division=0))
        m["balanced_acc"].append(balanced_accuracy_score(y_te, pred))
        m["kappa"].append(cohen_kappa_score(y_te, pred))
        m["mem"].append(model.memory_bytes())
    return {k: np.array(v) for k, v in m.items()}


# ── seleção de arquitetura (joelho do custo-benefício da clássica) ────────────
def select_architecture(X, y, address_sizes, resolutions, seeds, tol=0.01):
    """Varre address_size × resolution na ClassicWiSARD (bleach=0).
    Escolhe a config mais barata (menor tempo de treino) dentro de `tol`
    do pico de acurácia. Retorna (best_addr, best_res, tabela)."""
    table = []
    for a in address_sizes:
        for r in resolutions:
            res = eval_over_seeds(
                lambda s, a=a, r=r: make_classic(address_size=a, resolution=r, seed=s, bleach=0),
                X, y, seeds,
            )
            table.append({
                "address_size": a, "resolution": r,
                "mean_acc": res["acc"].mean(),
                "mean_train_t": res["train_t"].mean(),
            })
    peak = max(row["mean_acc"] for row in table)
    candidates = [row for row in table if row["mean_acc"] >= peak - tol]
    best = min(candidates, key=lambda row: row["mean_train_t"])
    return best["address_size"], best["resolution"], table


# ── confiança / curva de rejeição ─────────────────────────────────────────────
def accuracy_rejection_curve(confidence, correct, fractions):
    """Para cada fração rejeitada (descartando os menos confiantes),
    retorna a acurácia nas amostras remanescentes."""
    order = np.argsort(confidence)        # menos confiantes primeiro
    correct_sorted = np.asarray(correct)[order]
    n = len(correct_sorted)
    accs = []
    for f in fractions:
        keep = correct_sorted[int(f * n):]   # descarta a fração menos confiante
        accs.append(keep.mean() if len(keep) else np.nan)
    return np.array(accs)
