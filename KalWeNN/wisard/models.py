"""
Base e utilidades
  BaseWiSARD          : codificação, mapeamento bit→RAM e cálculo de score.
  _ScoreMixin         : decision_scores + margin_confidence (densos).

Densos (memória em array, tuplas pequenas)
  SimpleWiSARD        : contagem crua (versão antiga, pré-v2).
  ClassicWiSARD       : WiSARD clássica correta (reconhecimento binário + bleaching).
                        bleach=0 → clássica pura; bleach=b → filtra ruído (contagem > b).
  KalmanWiSARD        : filtro de Kalman por célula (base do KalWeNN).
  KalmanV2            : KalmanWiSARD + score, confiança e incerteza nativa.

Esparsos (memória em dicionários, viabilizam tuplas grandes)
  SparseClassicWiSARD : clássica (contagem + bleaching), só endereços vistos.
  SparseKalman        : KalmanWiSARD esparsa (x e p em dicionários).

As versões esparsas são matematicamente equivalentes às densas (mesma seed →
mesmos números). Validar:  python3 -m wisard.models
"""

import numpy as np
from .encoder import build_encoders, apply_encoders


# ══════════════════════════════════════════════════════════════════════════════
# base
# ══════════════════════════════════════════════════════════════════════════════

class BaseWiSARD:
    def __init__(self, address_size: int = 4, resolution: int = 16, seed: int = 42):
        self.address_size = address_size
        self.resolution = resolution
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.classes_ = None
        self._encoders = None
        self._mapping = None
        self._powers = None
        self.n_rams_ = None
        self._score_memory = None

    def _setup(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self._encoders = build_encoders(X, self.resolution)
        Xb = apply_encoders(X, self._encoders)
        n_bits = Xb.shape[1]
        self.n_rams_ = int(np.ceil(n_bits / self.address_size))
        total = self.n_rams_ * self.address_size
        indices = np.arange(n_bits)
        if total > n_bits:
            indices = np.concatenate(
                [indices, self._rng.choice(indices, total - n_bits, replace=True)]
            )
        self._rng.shuffle(indices)
        self._mapping = indices.reshape(self.n_rams_, self.address_size)
        self._powers = 1 << np.arange(self.address_size - 1, -1, -1)
        return X, y, Xb

    def _all_addresses(self, Xb: np.ndarray) -> np.ndarray:
        bits = Xb[:, self._mapping]  # (n_samples, n_rams, address_size)
        return (bits * self._powers).sum(axis=2).astype(int)  # (n_samples, n_rams)

    def _compute_scores(self, memory: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        addresses = self._all_addresses(Xb)  # (n_samples, n_rams)
        n_addr = 2 ** self.address_size
        # flat index: ram r, address a → r * n_addr + a
        flat = addresses + np.arange(self.n_rams_) * n_addr  # (n_samples, n_rams)
        mem_flat = memory.reshape(len(self.classes_), -1)  # (n_classes, n_rams * n_addr)
        return mem_flat[:, flat].sum(axis=2).T  # (n_samples, n_classes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xb = apply_encoders(np.asarray(X, dtype=float), self._encoders)
        scores = self._compute_scores(self._score_memory, Xb)
        return self.classes_[scores.argmax(axis=1)]


class _ScoreMixin:
    """Utilidades de score/confiança para os modelos densos."""

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        Xb = apply_encoders(np.asarray(X, dtype=float), self._encoders)
        return self._compute_scores(self._score_memory, Xb)

    @staticmethod
    def margin_confidence(scores: np.ndarray) -> np.ndarray:
        """Confiança = (top1 - top2) / (top1 + eps), por amostra."""
        srt = np.sort(scores, axis=1)
        top1, top2 = srt[:, -1], srt[:, -2]
        return (top1 - top2) / (top1 + 1e-9)


def _margin_confidence(scores: np.ndarray) -> np.ndarray:
    srt = np.sort(scores, axis=1)
    return (srt[:, -1] - srt[:, -2]) / (srt[:, -1] + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# densos
# ══════════════════════════════════════════════════════════════════════════════

class SimpleWiSARD(BaseWiSARD):
    """Contagem crua (versão antiga, pré-v2)."""

    def __init__(
        self,
        address_size: int = 4,
        resolution: int = 16,
        seed: int = 42,
        use_negative_update: bool = False,
    ):
        super().__init__(address_size, resolution, seed)
        self.use_negative_update = use_negative_update

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleWiSARD":
        X, y, Xb = self._setup(X, y)
        n_classes = len(self.classes_)
        n_addr = 2 ** self.address_size
        self._memory = np.zeros((n_classes, self.n_rams_, n_addr))
        addresses = self._all_addresses(Xb)  # (n_samples, n_rams)

        for c_idx, c in enumerate(self.classes_):
            pos = addresses[y == c]
            for r in range(self.n_rams_):
                np.add.at(self._memory[c_idx, r], pos[:, r], 1.0)
            if self.use_negative_update:
                neg = addresses[y != c]
                for r in range(self.n_rams_):
                    np.add.at(self._memory[c_idx, r], neg[:, r], -1.0)

        self._score_memory = self._memory
        return self


class ClassicWiSARD(_ScoreMixin, BaseWiSARD):
    """WiSARD clássica correta — reconhecimento binário + bleaching."""

    def __init__(
        self,
        address_size: int = 4,
        resolution: int = 16,
        seed: int = 42,
        bleach: int = 0,
    ):
        super().__init__(address_size, resolution, seed)
        self.bleach = bleach

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicWiSARD":
        X, y, Xb = self._setup(X, y)
        n_classes = len(self.classes_)
        n_addr = 2 ** self.address_size
        self._memory = np.zeros((n_classes, self.n_rams_, n_addr))
        addresses = self._all_addresses(Xb)

        for c_idx, c in enumerate(self.classes_):
            pos = addresses[y == c]
            for r in range(self.n_rams_):
                np.add.at(self._memory[c_idx, r], pos[:, r], 1.0)

        self.set_bleach(self.bleach)
        return self

    def set_bleach(self, bleach: int) -> "ClassicWiSARD":
        """Re-binariza a memória com novo limiar, sem retreinar.
        Permite varrer o bleaching reaproveitando um único treino."""
        self.bleach = bleach
        self._score_memory = (self._memory > bleach).astype(float)
        return self

    def memory_bytes(self) -> int:
        # guarda apenas as contagens (1 array)
        return int(self._memory.nbytes)


class KalmanWiSARD(BaseWiSARD):
    """Filtro de Kalman por célula (base do KalWeNN)."""

    def __init__(
        self,
        address_size: int = 4,
        resolution: int = 16,
        q: float = 1e-3,
        r: float = 5e-2,
        p0: float = 1.0,
        n_epochs: int = 1,
        seed: int = 42,
        use_negative_update: bool = False,
    ):
        super().__init__(address_size, resolution, seed)
        self.q = q
        self.r = r
        self.p0 = p0
        self.n_epochs = n_epochs
        self.use_negative_update = use_negative_update

    def _kalman_step(self, x: np.ndarray, p: np.ndarray, z: float):
        p_pred = p + self.q
        k = p_pred / (p_pred + self.r)
        return x + k * (z - x), (1.0 - k) * p_pred

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KalmanWiSARD":
        X, y, Xb = self._setup(X, y)
        n_classes = len(self.classes_)
        n_addr = 2 ** self.address_size

        self._memory_x = np.zeros((n_classes, self.n_rams_, n_addr))
        self._memory_p = np.full((n_classes, self.n_rams_, n_addr), self.p0)

        addresses = self._all_addresses(Xb)  # (n_samples, n_rams)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        ri = np.arange(self.n_rams_)

        for _ in range(self.n_epochs):
            for i in self._rng.permutation(len(Xb)):
                addrs = addresses[i]
                c_idx = class_to_idx[y[i]]

                self._memory_x[c_idx, ri, addrs], self._memory_p[c_idx, ri, addrs] = (
                    self._kalman_step(
                        self._memory_x[c_idx, ri, addrs],
                        self._memory_p[c_idx, ri, addrs],
                        1.0,
                    )
                )

                if self.use_negative_update:
                    for other in range(n_classes):
                        if other == c_idx:
                            continue
                        self._memory_x[other, ri, addrs], self._memory_p[other, ri, addrs] = (
                            self._kalman_step(
                                self._memory_x[other, ri, addrs],
                                self._memory_p[other, ri, addrs],
                                0.0,
                            )
                        )

        self._score_memory = self._memory_x
        return self


class KalmanV2(_ScoreMixin, KalmanWiSARD):
    """KalmanWiSARD + score, confiança e incerteza nativa por amostra."""

    def predicted_uncertainty(self, X: np.ndarray):
        """Retorna (predicoes, incerteza_por_amostra).

        Incerteza = média de p nas células endereçadas da classe prevista.
        p baixo → modelo confiante. Capacidade que a clássica não possui."""
        Xb = apply_encoders(np.asarray(X, dtype=float), self._encoders)
        scores = self._compute_scores(self._score_memory, Xb)
        pred_idx = scores.argmax(axis=1)
        addresses = self._all_addresses(Xb)  # (n_amostras, n_rams)
        ri = np.arange(self.n_rams_)
        unc = np.array([
            self._memory_p[pred_idx[i], ri, addresses[i]].mean()
            for i in range(len(addresses))
        ])
        return self.classes_[pred_idx], unc

    def memory_bytes(self) -> int:
        # guarda estado x e incerteza p (2 arrays) → ~2x a clássica
        return int(self._memory_x.nbytes + self._memory_p.nbytes)


# ══════════════════════════════════════════════════════════════════════════════
# esparsos (dicionários — viabilizam tuplas grandes)
# ══════════════════════════════════════════════════════════════════════════════

class SparseClassicWiSARD(BaseWiSARD):
    """Clássica (contagem + bleaching), guardando só os endereços vistos."""

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


class SparseKalman(KalmanWiSARD):
    """KalmanWiSARD esparsa (x e p em dicionários)."""

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


# ══════════════════════════════════════════════════════════════════════════════
# validação: esparsa deve dar IGUAL à densa
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_wine
    from sklearn.model_selection import train_test_split

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
