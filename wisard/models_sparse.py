"""
Versões com MEMÓRIA ESPARSA — viabilizam tuplas grandes (address_size alto)
sem alocar 2^address_size células densas.

NÃO altera models.py nem models_v2.py. São matematicamente equivalentes às
versões densas (ClassicWiSARD / KalmanV2): guardam apenas os endereços vistos;
endereço ausente = nunca visto (contagem 0 / x=0, p=p0). Mesma seed → mesmos
números das densas (use o __main__ deste arquivo para validar).

  SparseClassicWiSARD : clássica (contagem + bleaching), em dicionários.
  SparseKalman        : KalmanWiSARD, com x e p em dicionários.

Validar:  python3 -m wisard.models_sparse
"""

import numpy as np

from .models import BaseWiSARD, KalmanWiSARD
from .encoder import apply_encoders


def _margin_confidence(scores: np.ndarray) -> np.ndarray:
    srt = np.sort(scores, axis=1)
    return (srt[:, -1] - srt[:, -2]) / (srt[:, -1] + 1e-9)


# ── WiSARD clássica esparsa (contagem + bleaching) ──────────────────────────────

class SparseClassicWiSARD(BaseWiSARD):
    def __init__(self, address_size=4, resolution=16, seed=42, bleach=0):
        super().__init__(address_size, resolution, seed)
        self.bleach = bleach

    def fit(self, X, y):
        X, y, Xb = self._setup(X, y)
        addresses = self._all_addresses(Xb)  # (n_amostras, n_rams)
        # _counts[c][r] = {endereco: contagem}
        self._counts = []
        for c in self.classes_:
            pos = addresses[y == c]
            rams = []
            for r in range(self.n_rams_):
                a, cnt = np.unique(pos[:, r], return_counts=True)
                rams.append(dict(zip(a.tolist(), cnt.tolist())))
            self._counts.append(rams)
        return self

    def set_bleach(self, bleach):
        self.bleach = bleach
        return self

    def decision_scores(self, X):
        Xb = apply_encoders(np.asarray(X, dtype=float), self._encoders)
        addresses = self._all_addresses(Xb)
        n, nc = len(addresses), len(self.classes_)
        scores = np.zeros((n, nc))
        for ci in range(nc):
            rams = self._counts[ci]
            for r in range(self.n_rams_):
                d, col = rams[r], addresses[:, r]
                scores[:, ci] += np.fromiter(
                    (1.0 if d.get(int(a), 0) > self.bleach else 0.0 for a in col),
                    dtype=float, count=n,
                )
        return scores

    def predict(self, X):
        return self.classes_[self.decision_scores(X).argmax(axis=1)]

    @staticmethod
    def margin_confidence(scores):
        return _margin_confidence(scores)

    def memory_bytes(self):
        entries = sum(len(d) for rams in self._counts for d in rams)
        return int(entries * 16)  # ~ chave (8) + contagem (8)


# ── KalmanWiSARD esparsa ────────────────────────────────────────────────────────

class SparseKalman(KalmanWiSARD):
    def _step(self, x, p):
        p_pred = p + self.q
        k = p_pred / (p_pred + self.r)
        return x + k * (1.0 - x), (1.0 - k) * p_pred

    def fit(self, X, y):
        X, y, Xb = self._setup(X, y)
        addresses = self._all_addresses(Xb)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        nc = len(self.classes_)
        # _mem[c][r] = {endereco: (x, p)}
        self._mem = [[dict() for _ in range(self.n_rams_)] for _ in range(nc)]

        for _ in range(self.n_epochs):
            for i in self._rng.permutation(len(Xb)):
                ci = class_to_idx[y[i]]
                row, memc = addresses[i], self._mem[class_to_idx[y[i]]]
                for r in range(self.n_rams_):
                    a = int(row[r])
                    x, p = memc[r].get(a, (0.0, self.p0))
                    memc[r][a] = self._step(x, p)
        return self

    def decision_scores(self, X):
        Xb = apply_encoders(np.asarray(X, dtype=float), self._encoders)
        addresses = self._all_addresses(Xb)
        n, nc = len(addresses), len(self.classes_)
        scores = np.zeros((n, nc))
        for ci in range(nc):
            memc = self._mem[ci]
            for r in range(self.n_rams_):
                d, col = memc[r], addresses[:, r]
                scores[:, ci] += np.fromiter(
                    (d[int(a)][0] if int(a) in d else 0.0 for a in col),
                    dtype=float, count=n,
                )
        return scores

    def predict(self, X):
        return self.classes_[self.decision_scores(X).argmax(axis=1)]

    def predicted_uncertainty(self, X):
        Xb = apply_encoders(np.asarray(X, dtype=float), self._encoders)
        addresses = self._all_addresses(Xb)
        scores = self.decision_scores(X)
        pred_idx = scores.argmax(axis=1)
        unc = np.empty(len(addresses))
        for i in range(len(addresses)):
            memc, row = self._mem[pred_idx[i]], addresses[i]
            ps = [memc[r].get(int(row[r]), (0.0, self.p0))[1] for r in range(self.n_rams_)]
            unc[i] = float(np.mean(ps))
        return self.classes_[pred_idx], unc

    @staticmethod
    def margin_confidence(scores):
        return _margin_confidence(scores)

    def memory_bytes(self):
        entries = sum(len(d) for rams in self._mem for d in rams)
        return int(entries * 24)  # ~ chave (8) + x (8) + p (8)


# ── validação: esparsa deve dar IGUAL à densa ───────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_wine
    from sklearn.model_selection import train_test_split
    from .models_v2 import ClassicWiSARD, KalmanV2

    print("Validando esparsa == densa (mesma seed, address_size=8, resolution=16)\n")
    for name, load in [("iris", load_iris), ("wine", load_wine)]:
        d = load(); X, y = d.data, d.target
        print(f"== {name} ==")
        for seed in range(5):
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y)

            cd = ClassicWiSARD(8, 16, seed=seed, bleach=2).fit(Xtr, ytr)
            cs = SparseClassicWiSARD(8, 16, seed=seed, bleach=2).fit(Xtr, ytr)
            acd = (cd.predict(Xte) == yte).mean()
            acs = (cs.predict(Xte) == yte).mean()

            kd = KalmanV2(8, 16, seed=seed, q=1e-4, r=1e-1).fit(Xtr, ytr)
            ks = SparseKalman(8, 16, seed=seed, q=1e-4, r=1e-1).fit(Xtr, ytr)
            akd = (kd.predict(Xte) == yte).mean()
            aks = (ks.predict(Xte) == yte).mean()

            ok_c = "OK " if abs(acd - acs) < 1e-9 else "DIFF"
            ok_k = "OK " if abs(akd - aks) < 1e-9 else "DIFF"
            print(f"  seed {seed}: classica densa={acd:.4f} esparsa={acs:.4f} [{ok_c}] | "
                  f"kalman densa={akd:.4f} esparsa={aks:.4f} [{ok_k}]")
        print()
