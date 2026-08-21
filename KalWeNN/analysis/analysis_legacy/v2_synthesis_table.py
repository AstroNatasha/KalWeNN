"""
Tabela-síntese de todos os resultados (uma linha por dataset × modelo).

Colunas:
  dataset | tipo (binário/multiclasse) | n_train | n_test | n_features | n_classes
  | tupla (address_size) | termometro (resolution) | modelo
  | acuracia | acc_std | CV_% (estabilidade) | mem_MB | treino_ms | teste_ms

Estabilidade = CV (coeficiente de variação) da acurácia entre seeds:
  menor CV = mais estável (vantagem do Kalman).

Lê os dense por padrão. Salva CSV + imprime markdown.
Uso:  python3 analysis/v2_synthesis_table.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from v2_common import get_data, TEST_SIZE, CFG, FIG

ORDER = ["Classica", "Classica+Bleach", "KalmanWiSARD"]
SHORT = {"Classica": "Clássica", "Classica+Bleach": "Clássica+Bleach", "KalmanWiSARD": "KalmanWiSARD"}


def dataset_meta(ds):
    """type, n_train, n_test, n_features, n_classes — calculado do dataset real."""
    X, y = get_data(ds)
    n = len(X)
    n_classes = len(np.unique(y))
    n_test = int(round(n * TEST_SIZE))
    return {
        "tipo": "binário" if n_classes == 2 else "multiclasse",
        "n_train": n - n_test,
        "n_test": n_test,
        "n_features": X.shape[1],
        "n_classes": n_classes,
    }


def save_figure(table, path):
    n_rows, n_cols = table.shape
    fig, ax = plt.subplots(figsize=(min(1.0 * n_cols + 2, 26), 0.42 * n_rows + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=table.values, colLabels=table.columns,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.auto_set_column_width(range(n_cols))
    tbl.scale(1, 1.3)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2d3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            modelo = table.iloc[r - 1]["modelo"]
            cell.set_facecolor(
                "#fdece1" if modelo == "KalmanWiSARD"
                else "#edf6ed" if modelo == "Clássica+Bleach"
                else "#eef2f8")
    fig.suptitle("Tabela-síntese — Clássica × Clássica+Bleach × Kalman "
                 "(linhas alaranjadas = Kalman)", fontsize=11, fontweight="bold")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    # summaries densos (ignora *_sparse_*)
    files = sorted(p for p in CFG.glob("*_final_summary.csv") if "_sparse_" not in p.name)
    if not files:
        print("Nenhum *_final_summary.csv encontrado. Rode v2_analysis.py antes.")
        sys.exit(1)

    rows = []
    for f in files:
        ds = f.name.replace("_final_summary.csv", "")
        summ = pd.read_csv(f)
        cfg = json.load(open(CFG / f"{ds}_config.json"))
        meta = dataset_meta(ds)

        for m in ORDER:
            r = summ[summ.model == m]
            if r.empty:
                continue
            r = r.iloc[0]
            acc = r["acc"]
            cv = r["cv_acc_%"]
            std = cv / 100.0 * acc          # std reconstruído a partir do CV
            rows.append({
                "dataset": ds,
                "tipo": meta["tipo"],
                "n_train": meta["n_train"],
                "n_test": meta["n_test"],
                "n_features": meta["n_features"],
                "n_classes": meta["n_classes"],
                "tupla": cfg["address_size"],
                "termometro": cfg["resolution"],
                "modelo": SHORT[m],
                "acuracia": round(acc, 4),
                "acc_std": round(std, 4),
                "CV_%": round(cv, 3),
                "mem_MB": round(r["mem_MB"], 4),
                "treino_ms": round(r["train_ms"], 2),
                "teste_ms": round(r["test_ms"], 2),
            })

    table = pd.DataFrame(rows)
    out_csv = CFG / "synthesis_table.csv"
    table.to_csv(out_csv, index=False)
    print(f"CSV salvo → {out_csv}\n")

    # markdown (sem depender de 'tabulate')
    cols = list(table.columns)
    md = ["| " + " | ".join(cols) + " |",
          "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in table.iterrows():
        md.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    out_md = CFG / "synthesis_table.md"
    out_md.write_text("\n".join(md))
    print(f"Markdown salvo → {out_md}\n")

    # figura PNG
    out_png = FIG / "synthesis_table.png"
    save_figure(table, out_png)
    print(f"Figura salva → {out_png}\n")

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
