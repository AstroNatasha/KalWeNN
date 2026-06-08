"""
Variantes para o estudo comparativo v2 — NÃO altera wisard/models.py.

  ClassicWiSARD : WiSARD clássica correta (reconhecimento binário + bleaching).
                  bleach=0 → clássica pura;  bleach=b → filtra ruído (contagem > b).
  KalmanV2      : KalmanWiSARD + métodos de score, confiança e incerteza.

Ambas expõem:
  - decision_scores(X)      → matriz (n_amostras, n_classes)
  - margin_confidence(s)    → confiança por margem (top1-top2)
  - memory_bytes()          → footprint de memória do modelo treinado
KalmanV2 adiciona:
  - predicted_uncertainty(X) → incerteza p nativa por amostra (capacidade exclusiva)
"""

import numpy as np

from .models import BaseWiSARD, KalmanWiSARD
from .encoder import apply_encoders


# ── mixin de utilidades de score/confiança ──────────────────────────────────────

class _ScoreMixin:
    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        Xb = apply_encoders(np.asarray(X, dtype=float), self._encoders)
        return self._compute_scores(self._score_memory, Xb)

    @staticmethod
    def margin_confidence(scores: np.ndarray) -> np.ndarray:
        """Confiança = (top1 - top2) / (top1 + eps), por amostra."""
        srt = np.sort(scores, axis=1)
        top1, top2 = srt[:, -1], srt[:, -2]
        return (top1 - top2) / (top1 + 1e-9)


# ── WiSARD clássica (binária + bleaching) ───────────────────────────────────────

class ClassicWiSARD(_ScoreMixin, BaseWiSARD):
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


# ── KalmanWiSARD estendida ──────────────────────────────────────────────────────

class KalmanV2(_ScoreMixin, KalmanWiSARD):
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
