"""
05_regression — estudo de regressão: ReW vs KalWeNN (rodar depois do 01).

Pipeline em quatro etapas, cada uma um subcomando. As três primeiras produzem
os arquivos que a seguinte consome.

  arch       Seleção de arquitetura (tupla × resolução do termômetro) por
             dataset, usando a ReW de média simples. A escolha é feita numa
             VALIDAÇÃO INTERNA (30% do treino), nunca no teste.
             → results/regression/{ds}_arch.csv, {ds}_config.json

  grid       Busca de (q, r) do Kalman por dataset, para cada um dos 3 escopos
             de p. Também na validação interna. Cada modelo treinado é avaliado
             nos 5 modos de inferência de uma vez (a inferência não altera o
             treino), então o melhor (q, r) é escolhido por par escopo×inferência.
             → results/regression/{ds}_grid.csv, {ds}_config.json (atualizado)

  final      Comparação final no conjunto de TESTE, 30 seeds, split 70/30:
             ReW nos 6 estimadores + KalWeNN nos 3 escopos × 5 inferências.
             → results/regression/regression_report.csv

  baselines  Modelos de referência do scikit-learn (linear, kNN, árvore,
             random forest) nos mesmos splits, só para dar escala aos números.
             → results/regression/baselines.csv

  epochs     Repassar o treino ajuda o KalWeNN? (n_epochs = 1, 2, 3, 5)
             → results/regression/epochs.csv

  strat      Estratificar o split por faixas de quantil do alvo reduz a
             variação entre seeds? → results/regression/stratification.csv

  signif     Wilcoxon pareado por seed: melhor ReW vs melhor KalWeNN.
             → results/regression/significance.csv

Uso:  python3 analysis/05_regression.py all
      python3 analysis/05_regression.py arch grid final
      python3 analysis/05_regression.py final yacht concrete

Variáveis de ambiente:
  REG_FINAL_SEEDS (30)  seeds da comparação final
  REG_SEL_SEEDS   (3)   seeds das etapas de seleção (arch e grid)
  REG_MAX_SAMPLES (5000) teto de amostras por dataset; 0 remove o teto
"""

import os
import sys
import json
import time
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets.loaders_regression import get_regression, REGRESSION_DATASETS
from wisard.models_regression import (
    RegressionWiSARD, KalmanRegressionWiSARD,
    REW_ESTIMATORS, P_SCOPES, KAL_INFERENCES,
)

# ── caminhos ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
REG_DIR = ROOT / "results" / "regression"
REG_FIG = ROOT / "results" / "figures" / "regression"
for _d in (REG_DIR, REG_FIG):
    _d.mkdir(parents=True, exist_ok=True)

REPORT    = REG_DIR / "regression_report.csv"
BASELINES = REG_DIR / "baselines.csv"

# ── protocolo ─────────────────────────────────────────────────────────────────
TEST_SIZE = 0.30          # 70/30, como pedido
VAL_SIZE  = 0.30          # validação interna: 30% do treino
FINAL_SEEDS = list(range(int(os.environ.get("REG_FINAL_SEEDS", "30"))))
SEL_SEEDS   = list(range(int(os.environ.get("REG_SEL_SEEDS", "3"))))

# grades de busca
ADDRESS_SIZES = [16, 32, 48, 64]   # 64 é o teto: acima disso o endereço não
                                   # cabe em 64 bits (ver models_regression)
RESOLUTIONS   = [16, 32, 64, 128, 256]
Q_GRID  = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
R_GRID  = [1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 2.0]
P0_GRID = [0.1, 1.0, 10.0]
EPOCHS_GRID = [1, 2, 3, 5]
STRAT_BINS = 10          # faixas de y usadas no split estratificado


def _y_bins(y, n_bins=STRAT_BINS):
    """Faixas por quantil do alvo, para estratificar um problema contínuo.
    Faixas com menos de 2 amostras são fundidas com a vizinha."""
    q = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
    b = np.digitize(y, np.unique(q))
    vals, counts = np.unique(b, return_counts=True)
    for v in vals[counts < 2]:
        b[b == v] = vals[0]
    return b


def split_test(X, y, seed, stratify_bins=False):
    """Split externo 70/30.

    Por padrão sem estratificação — não existe estratificação para alvo
    contínuo. Com `stratify_bins`, estratifica por faixas de quantil de y, o que
    reduz a variação entre seeds em datasets pequenos (ver etapa `strat`)."""
    strat = _y_bins(y) if stratify_bins else None
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed,
                            stratify=strat)


def split_val(X_tr, y_tr, seed):
    """Split interno para seleção de hiperparâmetros (nunca toca o teste)."""
    return train_test_split(X_tr, y_tr, test_size=VAL_SIZE, random_state=1000 + seed)


# ── métricas ──────────────────────────────────────────────────────────────────
METRIC_KEYS = ["mae", "mse", "rmse", "r2", "acc_10pct", "acc_sigma"]
METRIC_LABELS = {
    "mae": "MAE", "mse": "MSE", "rmse": "RMSE", "r2": "R²",
    "acc_10pct": "Acurácia ±10%", "acc_sigma": "Acurácia ±0,25σ",
}


def metrics(y_true, y_pred):
    """MAE, MSE, RMSE, R² e duas acurácias por tolerância.

    Regressão não tem acerto/erro, então "acurácia" precisa de uma tolerância:
      acc_10pct  fração das predições com erro ≤ 10% do valor verdadeiro
                 (relativa — indefinida perto de y=0, por isso a segunda);
      acc_sigma  fração com erro ≤ 0,25 desvio-padrão do alvo no teste
                 (absoluta, funciona para alvos que passam por zero).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mse = float((err ** 2).mean())
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum()) or 1e-12
    sigma = float(y_true.std()) or 1e-12
    denom = np.maximum(np.abs(y_true), 1e-9)
    return {
        "mae": float(np.abs(err).mean()),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": 1.0 - ss_res / ss_tot,
        "acc_10pct": float((np.abs(err) / denom <= 0.10).mean()),
        "acc_sigma": float((np.abs(err) <= 0.25 * sigma).mean()),
    }


def agg(rows):
    """Agrega uma lista de dicts de métricas → média, desvio e CV% por métrica."""
    out = {}
    for k in METRIC_KEYS:
        v = np.array([r[k] for r in rows], dtype=float)
        out[f"{k}_mean"] = v.mean()
        out[f"{k}_std"] = v.std()
        out[f"{k}_cv"] = (v.std() / abs(v.mean()) * 100) if abs(v.mean()) > 1e-12 else np.nan
    for k in ("train_t", "test_t", "mem", "cells"):
        if k in rows[0]:
            out[f"{k}_mean"] = float(np.mean([r[k] for r in rows]))
    return out


# ── config por dataset ────────────────────────────────────────────────────────
def cfg_path(ds):
    return REG_DIR / f"{ds}_config.json"


def load_cfg(ds):
    p = cfg_path(ds)
    return json.load(open(p)) if p.exists() else {}


def save_cfg(ds, **kw):
    cfg = load_cfg(ds)
    cfg.update(kw)
    json.dump(cfg, open(cfg_path(ds), "w"), indent=2)
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# etapa 1 — arquitetura (tupla × resolução), na validação interna
# ══════════════════════════════════════════════════════════════════════════════

def stage_arch(datasets):
    for ds in datasets:
        X, y = get_regression(ds)
        print(f"\n[arch] {ds}  (n={len(X)}, d={X.shape[1]})")
        rows = []
        for a in ADDRESS_SIZES:
            for res in RESOLUTIONS:
                rmses, r2s, ts = [], [], []
                for seed in SEL_SEEDS:
                    X_tr, _, y_tr, _ = split_test(X, y, seed)
                    X_in, X_va, y_in, y_va = split_val(X_tr, y_tr, seed)
                    t0 = time.perf_counter()
                    m = RegressionWiSARD(a, res, seed=seed, estimator="simple").fit(X_in, y_in)
                    ts.append(time.perf_counter() - t0)
                    mm = metrics(y_va, m.predict(X_va))
                    rmses.append(mm["rmse"])
                    r2s.append(mm["r2"])
                rows.append({"address_size": a, "resolution": res, "n_rams": m.n_rams_,
                             "val_rmse": float(np.mean(rmses)), "val_r2": float(np.mean(r2s)),
                             "fit_s": float(np.mean(ts)), "cells": m.n_cells()})
                print(f"   tupla={a:3d} res={res:4d} rams={m.n_rams_:3d} "
                      f"RMSE_val={np.mean(rmses):9.4f} R²={np.mean(r2s):6.3f}")
        df = pd.DataFrame(rows).sort_values("val_rmse")
        df.to_csv(REG_DIR / f"{ds}_arch.csv", index=False)
        best = df.iloc[0]
        save_cfg(ds, address_size=int(best.address_size), resolution=int(best.resolution),
                 arch_val_rmse=float(best.val_rmse), arch_val_r2=float(best.val_r2))
        print(f"   → escolhido: tupla={int(best.address_size)} resolução={int(best.resolution)} "
              f"(RMSE_val={best.val_rmse:.4f}, R²_val={best.val_r2:.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# etapa 2 — grid de (q, r) por escopo de p, na validação interna
# ══════════════════════════════════════════════════════════════════════════════

def stage_grid(datasets):
    for ds in datasets:
        cfg = load_cfg(ds)
        if "address_size" not in cfg:
            print(f"[grid] {ds}: rode 'arch' antes. Pulando.")
            continue
        a, res = cfg["address_size"], cfg["resolution"]
        X, y = get_regression(ds)
        print(f"\n[grid] {ds}  tupla={a} res={res}  "
              f"({len(Q_GRID)}×{len(R_GRID)} combos × {len(P_SCOPES)} escopos)")

        rows = []
        for scope in P_SCOPES:
            for q in Q_GRID:
              for p0 in P0_GRID:
                for r in R_GRID:
                    per_inf = {inf: [] for inf in KAL_INFERENCES}
                    for seed in SEL_SEEDS:
                        X_tr, _, y_tr, _ = split_test(X, y, seed)
                        X_in, X_va, y_in, y_va = split_val(X_tr, y_tr, seed)
                        model = KalmanRegressionWiSARD(
                            a, res, seed=seed, q=q, r=r, p0=p0,
                            p_scope=scope).fit(X_in, y_in)
                        for inf in KAL_INFERENCES:
                            model.set_inference(inf)
                            per_inf[inf].append(metrics(y_va, model.predict(X_va))["rmse"])
                    for inf in KAL_INFERENCES:
                        rows.append({"p_scope": scope, "inference": inf,
                                     "q": q, "r": r, "p0": p0,
                                     "val_rmse": float(np.mean(per_inf[inf]))})
            best_scope = min((row for row in rows if row["p_scope"] == scope),
                             key=lambda row: row["val_rmse"])
            print(f"   escopo={scope:6s} melhor: inferência={best_scope['inference']:10s} "
                  f"q={best_scope['q']:g} r={best_scope['r']:g} p0={best_scope['p0']:g} "
                  f"RMSE_val={best_scope['val_rmse']:.4f}")

        df = pd.DataFrame(rows)
        df.to_csv(REG_DIR / f"{ds}_grid.csv", index=False)
        # melhor (q, r, p0) para cada par escopo × inferência
        best = {}
        for (scope, inf), sub in df.groupby(["p_scope", "inference"]):
            row = sub.loc[sub["val_rmse"].idxmin()]
            best[f"{scope}|{inf}"] = {"q": float(row.q), "r": float(row.r),
                                      "p0": float(row.p0),
                                      "val_rmse": float(row.val_rmse)}
        save_cfg(ds, kalman_best=best)


# ══════════════════════════════════════════════════════════════════════════════
# etapa 3 — comparação final no teste, 30 seeds
# ══════════════════════════════════════════════════════════════════════════════

def stage_final(datasets):
    all_rows = []
    for ds in datasets:
        cfg = load_cfg(ds)
        if "address_size" not in cfg or "kalman_best" not in cfg:
            print(f"[final] {ds}: rode 'arch' e 'grid' antes. Pulando.")
            continue
        a, res, kbest = cfg["address_size"], cfg["resolution"], cfg["kalman_best"]
        X, y = get_regression(ds)
        print(f"\n[final] {ds}  tupla={a} res={res}  {len(FINAL_SEEDS)} seeds")

        # ── ReW: um treino por seed serve para os 6 estimadores ───────────────
        per_model = {f"ReW/{e}": [] for e in REW_ESTIMATORS}
        for seed in FINAL_SEEDS:
            X_tr, X_te, y_tr, y_te = split_test(X, y, seed)
            t0 = time.perf_counter()
            model = RegressionWiSARD(a, res, seed=seed, estimator="simple").fit(X_tr, y_tr)
            train_t = time.perf_counter() - t0
            for est in REW_ESTIMATORS:
                model.set_estimator(est)
                t1 = time.perf_counter()
                pred = model.predict(X_te)
                row = metrics(y_te, pred)
                row.update(train_t=train_t, test_t=time.perf_counter() - t1,
                           mem=model.memory_bytes(), cells=model.n_cells())
                per_model[f"ReW/{est}"].append(row)

        # ── KalWeNN: um treino por (seed, escopo, q, r) usado nas inferências ──
        # (q, r) muda por par escopo×inferência, então agrupamos os pares que
        # compartilham o mesmo (escopo, q, r) para não retreinar à toa.
        groups = {}
        for key, hp in kbest.items():
            scope, inf = key.split("|")
            groups.setdefault((scope, hp["q"], hp["r"], hp.get("p0", 1.0)), []).append(inf)

        for (scope, q, r, p0), infs in groups.items():
            names = {inf: f"KalWeNN/{scope}/{inf}" for inf in infs}
            for n in names.values():
                per_model[n] = []
            for seed in FINAL_SEEDS:
                X_tr, X_te, y_tr, y_te = split_test(X, y, seed)
                t0 = time.perf_counter()
                model = KalmanRegressionWiSARD(a, res, seed=seed, q=q, r=r, p0=p0,
                                               p_scope=scope).fit(X_tr, y_tr)
                train_t = time.perf_counter() - t0
                for inf in infs:
                    model.set_inference(inf)
                    t1 = time.perf_counter()
                    pred = model.predict(X_te)
                    row = metrics(y_te, pred)
                    row.update(train_t=train_t, test_t=time.perf_counter() - t1,
                               mem=model.memory_bytes(), cells=model.n_cells(),
                               mem_no_p=model.memory_bytes_no_p())
                    per_model[names[inf]].append(row)

        for name, rows in per_model.items():
            if not rows:
                continue
            fam = name.split("/")[0]
            rec = {"dataset": ds, "modelo": name, "familia": fam,
                   "address_size": a, "resolution": res, "n_seeds": len(rows)}
            if fam == "KalWeNN":
                _, scope, inf = name.split("/")
                hp = kbest[f"{scope}|{inf}"]
                rec.update(p_scope=scope, inference=inf, q=hp["q"], r=hp["r"],
                           p0=hp.get("p0", 1.0),
                           mem_no_p_mean=float(np.mean([r_["mem_no_p"] for r_ in rows])))
            else:
                rec.update(p_scope="-", inference=name.split("/")[1], q=np.nan, r=np.nan)
            rec.update(agg(rows))
            all_rows.append(rec)
            print(f"   {name:28s} RMSE={rec['rmse_mean']:9.4f} ± {rec['rmse_std']:7.4f} "
                  f"(CV {rec['rmse_cv']:5.2f}%)  MAE={rec['mae_mean']:9.4f}  R²={rec['r2_mean']:6.3f}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        if REPORT.exists():   # mescla com o que já existe, por dataset+modelo
            old = pd.read_csv(REPORT)
            old = old[~old.set_index(["dataset", "modelo"]).index.isin(
                df.set_index(["dataset", "modelo"]).index)]
            df = pd.concat([old, df], ignore_index=True)
        df.sort_values(["dataset", "rmse_mean"]).to_csv(REPORT, index=False)
        print(f"\n→ {REPORT}  ({len(df)} linhas)")


# ══════════════════════════════════════════════════════════════════════════════
# etapa extra — número de épocas do KalWeNN
# ══════════════════════════════════════════════════════════════════════════════

def stage_epochs(datasets):
    """Repassar o treino ajuda? Só faz sentido no KalWeNN: a ReW soma, então
    N épocas apenas multiplicam contadores e somas por N e a predição não muda.
    Com q > 0 a memória esquece, e a ordem/repetição passa a importar."""
    seeds = FINAL_SEEDS[:10]
    rows = []
    for ds in datasets:
        cfg = load_cfg(ds)
        if "kalman_best" not in cfg:
            print(f"[epochs] {ds}: rode 'grid' antes. Pulando.")
            continue
        a, res = cfg["address_size"], cfg["resolution"]
        hp = cfg["kalman_best"]["cell|mean"]
        X, y = get_regression(ds)
        print(f"\n[epochs] {ds}  q={hp['q']:g} r={hp['r']:g} p0={hp.get('p0',1.0):g}")
        for n_ep in EPOCHS_GRID:
            per = []
            for seed in seeds:
                X_tr, X_te, y_tr, y_te = split_test(X, y, seed)
                m = KalmanRegressionWiSARD(a, res, seed=seed, q=hp["q"], r=hp["r"],
                                           p0=hp.get("p0", 1.0), n_epochs=n_ep,
                                           p_scope="cell", inference="mean").fit(X_tr, y_tr)
                per.append(metrics(y_te, m.predict(X_te)))
            rec = {"dataset": ds, "n_epochs": n_ep}
            rec.update(agg(per))
            rows.append(rec)
            print(f"   épocas={n_ep}: RMSE={rec['rmse_mean']:9.4f} (CV {rec['rmse_cv']:5.2f}%)")
    if rows:
        pd.DataFrame(rows).to_csv(REG_DIR / "epochs.csv", index=False)
        print(f"\n→ {REG_DIR/'epochs.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# etapa extra — estratificação do split por faixas do alvo
# ══════════════════════════════════════════════════════════════════════════════

def stage_strat(datasets):
    """O split 70/30 puro não controla a distribuição do alvo entre treino e
    teste, o que infla a variação entre seeds nos datasets pequenos. Compara o
    CV do RMSE com e sem estratificação por faixas de quantil de y."""
    rows = []
    for ds in datasets:
        cfg = load_cfg(ds)
        if "address_size" not in cfg:
            print(f"[strat] {ds}: rode 'arch' antes. Pulando.")
            continue
        a, res = cfg["address_size"], cfg["resolution"]
        X, y = get_regression(ds)
        print(f"\n[strat] {ds}")
        for strat in (False, True):
            per = []
            for seed in FINAL_SEEDS:
                X_tr, X_te, y_tr, y_te = split_test(X, y, seed, stratify_bins=strat)
                m = RegressionWiSARD(a, res, seed=seed, estimator="unweighted").fit(X_tr, y_tr)
                per.append(metrics(y_te, m.predict(X_te)))
            rec = {"dataset": ds, "estratificado": strat}
            rec.update(agg(per))
            rows.append(rec)
            print(f"   {'com' if strat else 'sem'} estratificação: "
                  f"RMSE={rec['rmse_mean']:9.4f} CV={rec['rmse_cv']:5.2f}%")
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(REG_DIR / "stratification.csv", index=False)
        piv = df.pivot_table(index="dataset", columns="estratificado", values="rmse_cv")
        piv.columns = ["CV sem", "CV com"]
        piv["redução %"] = ((piv["CV sem"] - piv["CV com"]) / piv["CV sem"] * 100).round(1)
        print("\n" + piv.round(2).to_string())
        print(f"\n→ {REG_DIR/'stratification.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# etapa 4 — baselines do scikit-learn (contexto, mesmos splits)
# ══════════════════════════════════════════════════════════════════════════════

def stage_baselines(datasets):
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.dummy import DummyRegressor

    makers = {
        "média (dummy)":  lambda s: DummyRegressor(strategy="mean"),
        "linear":         lambda s: LinearRegression(),
        "kNN (k=5)":      lambda s: KNeighborsRegressor(5),
        "árvore":         lambda s: DecisionTreeRegressor(random_state=s),
        "random forest":  lambda s: RandomForestRegressor(100, random_state=s, n_jobs=-1),
    }
    seeds = FINAL_SEEDS[:10]   # baselines são só referência de escala
    rows = []
    for ds in datasets:
        X, y = get_regression(ds)
        print(f"\n[baselines] {ds}")
        for name, make in makers.items():
            per = []
            for seed in seeds:
                X_tr, X_te, y_tr, y_te = split_test(X, y, seed)
                per.append(metrics(y_te, make(seed).fit(X_tr, y_tr).predict(X_te)))
            rec = {"dataset": ds, "modelo": name, "n_seeds": len(seeds)}
            rec.update(agg(per))
            rows.append(rec)
            print(f"   {name:16s} RMSE={rec['rmse_mean']:9.4f}  R²={rec['r2_mean']:6.3f}")
    pd.DataFrame(rows).to_csv(BASELINES, index=False)
    print(f"\n→ {BASELINES}")


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# etapa extra — significância estatística (Wilcoxon pareado por seed)
# ══════════════════════════════════════════════════════════════════════════════

def stage_signif(datasets):
    """As 30 seeds são pareadas (mesmo split para os dois modelos), então o teste
    de postos sinalizados de Wilcoxon é o adequado: não pressupõe normalidade e
    respeita o pareamento. Compara, por dataset, a melhor ReW contra o melhor
    KalWeNN — ambos escolhidos pelo relatório final."""
    from scipy.stats import wilcoxon

    if not REPORT.exists():
        print("[signif] rode 'final' antes.")
        return
    rep = pd.read_csv(REPORT)
    rows = []
    for ds in datasets:
        sub = rep[rep["dataset"] == ds]
        if sub.empty:
            continue
        cfg = load_cfg(ds)
        a, res = cfg["address_size"], cfg["resolution"]
        X, y = get_regression(ds)

        rew_row = sub[sub["familia"] == "ReW"].nsmallest(1, "rmse_mean").iloc[0]
        kal_row = sub[sub["familia"] == "KalWeNN"].nsmallest(1, "rmse_mean").iloc[0]
        hp = cfg["kalman_best"][f"{kal_row['p_scope']}|{kal_row['inference']}"]

        e_rew, e_kal = [], []
        for seed in FINAL_SEEDS:
            X_tr, X_te, y_tr, y_te = split_test(X, y, seed)
            m1 = RegressionWiSARD(a, res, seed=seed,
                                  estimator=rew_row["inference"]).fit(X_tr, y_tr)
            m2 = KalmanRegressionWiSARD(a, res, seed=seed, q=hp["q"], r=hp["r"],
                                        p0=hp.get("p0", 1.0),
                                        p_scope=kal_row["p_scope"],
                                        inference=kal_row["inference"]).fit(X_tr, y_tr)
            e_rew.append(metrics(y_te, m1.predict(X_te))["rmse"])
            e_kal.append(metrics(y_te, m2.predict(X_te))["rmse"])

        e_rew, e_kal = np.array(e_rew), np.array(e_kal)
        diff = e_kal - e_rew
        if np.allclose(diff, 0):
            stat, p = np.nan, 1.0     # idênticos: o teste não se aplica
        else:
            stat, p = wilcoxon(e_rew, e_kal)
        rows.append({
            "dataset": ds, "ReW": rew_row["modelo"], "KalWeNN": kal_row["modelo"],
            "rmse_rew": e_rew.mean(), "rmse_kal": e_kal.mean(),
            "delta_%": (e_rew.mean() - e_kal.mean()) / e_rew.mean() * 100,
            "vitorias_rew": int((e_rew < e_kal).sum()), "n": len(FINAL_SEEDS),
            "wilcoxon_p": p, "significativo_5%": bool(p < 0.05),
        })
        print(f"   {ds:11s} ReW={e_rew.mean():9.4f} KalWeNN={e_kal.mean():9.4f} "
              f"| ReW vence {rows[-1]['vitorias_rew']}/{len(FINAL_SEEDS)} seeds "
              f"| p={p:.2e} {'*' if p < 0.05 else '(n.s.)'}")

    if rows:
        pd.DataFrame(rows).to_csv(REG_DIR / "significance.csv", index=False)
        print(f"\n→ {REG_DIR/'significance.csv'}")


STAGES = {"arch": stage_arch, "grid": stage_grid, "final": stage_final,
          "baselines": stage_baselines, "epochs": stage_epochs,
          "strat": stage_strat, "signif": stage_signif}

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    todo = [a for a in args if a in STAGES] or list(STAGES)
    if "all" in args:
        todo = list(STAGES)
    picked = [a for a in args if a in REGRESSION_DATASETS] or REGRESSION_DATASETS
    print(f"etapas: {todo}\ndatasets: {picked}")
    for stage in todo:
        t0 = time.perf_counter()
        STAGES[stage](picked)
        print(f"\n== etapa '{stage}' concluída em {time.perf_counter() - t0:.1f}s ==")
