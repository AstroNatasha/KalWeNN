"""
08_wisardpkg_models — ReW e ClusReW da biblioteca de referência como baseline.

O `07_wisardpkg_check.py` prova que a nossa ReW é numericamente igual à da
referência. Este script responde outra pergunta: como os nossos números se
comparam aos dos modelos que a biblioteca oferece e que **não** reimplementamos —
em particular a `ClusRegressionWisard`, o ensemble com clusterização, que na
literatura costuma superar a ReW pura.

Roda, nos mesmos splits 70/30 e com a mesma binarização por termômetro:

  ReW (wisardpkg)      RegressionWisard com o melhor agregador do nosso estudo
  ClusReW (wisardpkg)  ClusRegressionWisard, com os limiares de crescimento
                       (minScore, threshold, discriminatorLimit) varridos numa
                       grade pequena e escolhidos na validação interna

Requer o ambiente do wisardpkg (README_REGRESSAO.md §3.2):
    /tmp/wpkg-env/bin/python analysis/08_wisardpkg_models.py
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets.loaders_regression import get_regression, REGRESSION_DATASETS
from wisard.models_regression import RegressionWiSARD
from wisard.encoder import apply_encoders

try:
    import wisardpkg as wp
except ImportError:
    sys.exit("wisardpkg não encontrado — ver README_REGRESSAO.md §3.2.")

ROOT    = Path(__file__).resolve().parents[1]
REG_DIR = ROOT / "results" / "regression"
OUT     = REG_DIR / "wisardpkg_models.csv"

TEST_SIZE, VAL_SIZE = 0.30, 0.30
SEEDS = list(range(10))
SEL_SEEDS = [0, 1]

# grade do ClusReW: quando criar um novo discriminador e quantos permitir
CLUS_GRID = [
    {"minScore": 0.1, "threshold": 20, "discriminatorLimit": 5},
    {"minScore": 0.3, "threshold": 20, "discriminatorLimit": 5},
    {"minScore": 0.5, "threshold": 50, "discriminatorLimit": 10},
    {"minScore": 0.7, "threshold": 100, "discriminatorLimit": 10},
]

MEAN_OF = {
    "simple": lambda: wp.SimpleMean(), "unweighted": lambda: wp.PowerMean(1),
    "power2": lambda: wp.PowerMean(2), "harmonic": lambda: wp.HarmonicMean(),
    "harmonic_power2": lambda: wp.HarmonicPowerMean(2),
    "geometric": lambda: wp.GeometricMean(), "exponential": lambda: wp.ExponentialMean(),
}
# `Median` fica de fora: o defeito de indexação da referência devolve NaN
# (README_REGRESSAO.md §3.2). Quando o melhor agregador do dataset é a mediana,
# a ReW de referência roda com o melhor agregador comparável.


def rmse(a, b):
    return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).mean()))


def prepare(ds, seed, a, res):
    """Split + binarização + normalização do alvo, idênticos aos nossos."""
    X, y = get_regression(ds)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
    ref = RegressionWiSARD(a, res, seed=seed).fit(X_tr, y_tr)   # dá encoders/mapping
    binz = lambda M: [[int(b) for b in row]
                      for row in apply_encoders(np.asarray(M, float), ref._encoders)]
    return ref, binz(X_tr), binz(X_te), ref._normalize(y_tr), X_te, y_te


def run(ds, cfg, best_est, ref_est):
    a, res = cfg["address_size"], cfg["resolution"]

    # ── ReW da referência e a nossa, nas MESMAS seeds e no mesmo mapeamento ───
    rew_scores, ours_scores = [], []
    for seed in SEEDS:
        ref, btr, bte, ytrn, X_te, y_te = prepare(ds, seed, a, res)
        mapping = [int(v) for v in ref._mapping.ravel()]
        m = wp.RegressionWisard(a, mean=MEAN_OF[ref_est](), mapping=mapping,
                                completeAddressing=True, orderedMapping=True)
        m.train(wp.DataSet(btr, [float(v) for v in ytrn]))
        rew_scores.append(rmse(y_te, ref._denormalize(np.asarray(m.predict(wp.DataSet(bte))))))
        ours_scores.append(rmse(y_te, ref.set_estimator(best_est).predict(X_te)))

    # ── ClusReW: escolhe os limiares na validação interna ─────────────────────
    best_cfg, best_val = None, np.inf
    for params in CLUS_GRID:
        vals = []
        for seed in SEL_SEEDS:
            X, y = get_regression(ds)
            X_tr, _, y_tr, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
            X_in, X_va, y_in, y_va = train_test_split(X_tr, y_tr, test_size=VAL_SIZE,
                                                      random_state=1000 + seed)
            ref = RegressionWiSARD(a, res, seed=seed).fit(X_in, y_in)
            binz = lambda M: [[int(b) for b in row]
                              for row in apply_encoders(np.asarray(M, float), ref._encoders)]
            m = wp.ClusRegressionWisard(a, params["minScore"], params["threshold"],
                                        params["discriminatorLimit"],
                                        mean=wp.SimpleMean(), completeAddressing=True)
            m.train(wp.DataSet(binz(X_in), [float(v) for v in ref._normalize(y_in)]))
            pred = ref._denormalize(np.asarray(m.predict(wp.DataSet(binz(X_va)))))
            vals.append(rmse(y_va, pred))
        if np.mean(vals) < best_val:
            best_val, best_cfg = float(np.mean(vals)), params

    clus_scores = []
    for seed in SEEDS:
        ref, btr, bte, ytrn, X_te, y_te = prepare(ds, seed, a, res)
        m = wp.ClusRegressionWisard(a, best_cfg["minScore"], best_cfg["threshold"],
                                    best_cfg["discriminatorLimit"],
                                    mean=wp.SimpleMean(), completeAddressing=True)
        m.train(wp.DataSet(btr, [float(v) for v in ytrn]))
        clus_scores.append(rmse(y_te, ref._denormalize(np.asarray(m.predict(wp.DataSet(bte))))))

    return ours_scores, rew_scores, clus_scores, best_cfg


if __name__ == "__main__":
    picked = [a for a in sys.argv[1:] if a in REGRESSION_DATASETS] or REGRESSION_DATASETS
    report = pd.read_csv(REG_DIR / "regression_report.csv")

    rows = []
    print("Modelos do wisardpkg 2.0.0a7 nos mesmos splits 70/30 "
          f"({len(SEEDS)} seeds)\n")
    for ds in picked:
        cfg = json.load(open(REG_DIR / f"{ds}_config.json"))
        sub = report[(report["dataset"] == ds) & (report["familia"] == "ReW")]
        best_est = sub.nsmallest(1, "rmse_mean").iloc[0]["inference"]
        # melhor agregador que a referência também sabe calcular
        ref_est = sub[sub["inference"].isin(MEAN_OF)].nsmallest(1, "rmse_mean").iloc[0]["inference"]

        t0 = time.perf_counter()
        ours_s, rew, clus, ccfg = run(ds, cfg, best_est, ref_est)
        ours = float(np.mean(ours_s))
        rows.append({
            "dataset": ds, "agregador_nosso": best_est, "agregador_referencia": ref_est,
            "nossa_ReW": ours,
            "wisardpkg_ReW": float(np.mean(rew)),
            "wisardpkg_ClusReW": float(np.mean(clus)),
            "ClusReW_cv%": float(np.std(clus) / np.mean(clus) * 100),
            "ClusReW_params": json.dumps(ccfg),
            "segundos": round(time.perf_counter() - t0, 1),
        })
        print(f"{ds:11s} nossa[{best_est[:9]:>9s}]={ours:9.4f}  "
              f"wisardpkg ReW[{ref_est[:9]:>9s}]={np.mean(rew):9.4f}  "
              f"ClusReW={np.mean(clus):9.4f}  ({rows[-1]['segundos']}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\n→ {OUT}")
    print(f"\nClusReW melhor que a ReW em "
          f"{int((df.wisardpkg_ClusReW < df.wisardpkg_ReW).sum())}/{len(df)} datasets")
