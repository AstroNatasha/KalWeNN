import numpy as np
from .encoder import build_encoders, apply_encoders


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


class SimpleWiSARD(BaseWiSARD):
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


class KalmanWiSARD(BaseWiSARD):
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

    def _kalman_step(
        self, x: np.ndarray, p: np.ndarray, z: float
    ):
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
