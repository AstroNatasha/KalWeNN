import numpy as np


class ThermometerEncoder:
    def __init__(self, minimum: float, maximum: float, resolution: int):
        self.minimum = minimum
        self.maximum = maximum
        self.resolution = resolution
        self._range = (maximum - minimum) + 1e-9

    def encode(self, column: np.ndarray) -> np.ndarray:
        col = np.clip(column, self.minimum, self.maximum)
        k = np.clip(
            ((col - self.minimum) / self._range * self.resolution).astype(int),
            0,
            self.resolution,
        )
        return (np.arange(self.resolution) < k[:, np.newaxis]).astype(np.uint8)


def build_encoders(X: np.ndarray, resolution: int) -> list:
    mins, maxs = X.min(axis=0), X.max(axis=0)
    return [ThermometerEncoder(mins[j], maxs[j], resolution) for j in range(X.shape[1])]


def apply_encoders(X: np.ndarray, encoders: list) -> np.ndarray:
    return np.concatenate([enc.encode(X[:, j]) for j, enc in enumerate(encoders)], axis=1)
