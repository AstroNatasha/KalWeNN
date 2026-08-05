"""
07_wisardpkg_check — valida a nossa ReW contra a implementação de referência.

Compara `wisard.models_regression.RegressionWiSARD` (nossa, em Python) com a
`RegressionWisard` do **wisardpkg v2.0.0a7**, a biblioteca do grupo de WiSARD da
UFRJ, forçando as duas a usarem:

  - a MESMA entrada binarizada (o termômetro do nosso `wisard/encoder.py`);
  - o MESMO mapeamento bit→RAM (o wisardpkg aceita o kwarg `mapping`, um vetor
    achatado em que a RAM i fica com mapping[i*tupla : (i+1)*tupla] — exatamente
    o layout do nosso `_mapping`);
  - o MESMO alvo normalizado em [0,1].

Com isso, as predições têm de bater dígito a dígito para cada agregador.

Requisito: o wisardpkg não está no PyPI com a parte de regressão (a 1.6.3, de
2019, só tem classificação). É preciso compilar a branch `develop` do
repositório oficial. Passo a passo no README_REGRESSAO.md §3.2.

Uso:  python3 analysis/07_wisardpkg_check.py
      python3 analysis/07_wisardpkg_check.py yacht auto_mpg
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split

from datasets.loaders_regression import get_regression, REGRESSION_DATASETS
from wisard.models_regression import RegressionWiSARD, REW_ESTIMATORS
from wisard.encoder import apply_encoders

try:
    import wisardpkg as wp
except ImportError:
    sys.exit("wisardpkg não encontrado — ver README_REGRESSAO.md §3.2 para compilar.")

# nosso nome → objeto Mean equivalente na biblioteca de referência.
# `median` fica de fora: a referência tem defeito de indexação (ver
# models_regression._aggregate), então não há o que comparar.
MEAN_OF = {
    "simple":          lambda: wp.SimpleMean(),
    "unweighted":      lambda: wp.PowerMean(1),
    "power2":          lambda: wp.PowerMean(2),
    "harmonic":        lambda: wp.HarmonicMean(),
    "harmonic_power2": lambda: wp.HarmonicPowerMean(2),
    "geometric":       lambda: wp.GeometricMean(),
    "exponential":     lambda: wp.ExponentialMean(),
}

FALLBACK = (16, 16)   # usado se o dataset ainda não tiver config da etapa `arch`
TOL = 1e-9

ROOT = Path(__file__).resolve().parents[1]
REG_DIR = ROOT / "results" / "regression"


def arch_of(ds):
    """A tupla/resolução realmente usada no estudo, para validar a configuração
    de verdade e não uma de brinquedo."""
    p = REG_DIR / f"{ds}_config.json"
    if p.exists():
        cfg = json.load(open(p))
        if "address_size" in cfg:
            return cfg["address_size"], cfg["resolution"]
    return FALLBACK


def binarize(model, X):
    """Entrada binária no formato que o wisardpkg espera: lista de listas de int."""
    Xb = apply_encoders(np.asarray(X, dtype=float), model._encoders)
    return [[int(b) for b in row] for row in Xb]


def covered(model, X):
    """Máscara das amostras com ao menos uma célula endereçada conhecida.

    Para as demais, os dois modelos divergem por opção declarada: a referência
    devolve 0, nós devolvemos a média do alvo no treino (ver README §3.2). Essa
    diferença é de projeto, não de fórmula, então é isolada aqui."""
    addr = model._addresses_of(X)
    return np.array([any(int(addr[i, r]) in model._mem[r]
                         for r in range(model.n_rams_))
                     for i in range(len(addr))])


def check(ds, seed):
    a, res = arch_of(ds)
    X, y = get_regression(ds)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)

    ours = RegressionWiSARD(a, res, seed=seed, estimator="simple").fit(X_tr, y_tr)

    # mesma binarização e mesmo alvo normalizado que o nosso modelo usou
    bin_tr, bin_te = binarize(ours, X_tr), binarize(ours, X_te)
    y_tr_norm = ours._normalize(y_tr)
    mapping = [int(v) for v in ours._mapping.ravel()]
    mask = covered(ours, X_te)
    span = float(y_te.max() - y_te.min()) or 1.0

    out = {}
    for est, make_mean in MEAN_OF.items():
        ref = wp.RegressionWisard(a, mean=make_mean(), mapping=mapping,
                                  completeAddressing=True, orderedMapping=True)
        ref.train(wp.DataSet(bin_tr, [float(v) for v in y_tr_norm]))
        pred_ref = ours._denormalize(np.asarray(ref.predict(wp.DataSet(bin_te))))

        pred_ours = ours.set_estimator(est).predict(X_te)
        d = np.abs(pred_ours - pred_ref)[mask]
        out[est] = float(d.max()) / span if d.size else 0.0
    out["_sem_cobertura"] = int((~mask).sum())
    out["_n_teste"] = int(len(mask))
    out["_arch"] = (a, res)
    return out


if __name__ == "__main__":
    picked = [a for a in sys.argv[1:] if a in REGRESSION_DATASETS] or REGRESSION_DATASETS
    print("Nossa ReW  vs  wisardpkg 2.0.0a7 RegressionWisard")
    print("(mesma binarização, mesmo mapeamento, na tupla/resolução escolhidas para\n"
          " cada dataset; maior diferença absoluta / amplitude do alvo)\n")

    ests = list(MEAN_OF)
    print(f"{'dataset':11s} {'tupla/res':>10s} {'seed':>4s} "
          + " ".join(f"{e[:9]:>10s}" for e in ests) + f" {'s/cobert.':>10s}")
    worst = {e: 0.0 for e in ests}
    uncovered = 0
    for ds in picked:
        for seed in range(2):
            r = check(ds, seed)
            for e in ests:
                worst[e] = max(worst[e], r[e])
            uncovered += r["_sem_cobertura"]
            arch = f"{r['_arch'][0]}/{r['_arch'][1]}"
            print(f"{ds:11s} {arch:>10s} {seed:4d} "
                  + " ".join(f"{r[e]:10.2e}" for e in ests)
                  + f" {r['_sem_cobertura']:4d}/{r['_n_teste']:<5d}")

    print("\npior caso por agregador:")
    for e in ests:
        print(f"  {e:16s} {worst[e]:9.2e}  [{'IGUAL' if worst[e] < TOL else 'DIFERE'}]")
    print(f"\n{'TODOS IGUAIS' if max(worst.values()) < TOL else 'HÁ DIVERGÊNCIAS'} "
          f"(tolerância {TOL:g})")
    print(f"amostras excluídas por não ter nenhuma célula conhecida: {uncovered} "
          f"(ali a referência devolve 0 e nós devolvemos a média do treino)")
