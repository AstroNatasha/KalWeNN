import glob
import os
import threading
import time
from pathlib import Path
from typing import Optional, Sequence, Type

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ── RAPL energy (Linux Intel) ──────────────────────────────────────────────────

def _find_rapl_file() -> Optional[str]:
    patterns = [
        "/sys/class/powercap/intel-rapl*/energy_uj",
        "/sys/class/powercap/intel-rapl:*/*/energy_uj",
        "/sys/class/powercap/intel-rapl:*/energy_uj",
    ]
    candidates = sorted({f for p in patterns for f in glob.glob(p)})
    return candidates[0] if candidates else None


def _read_energy_uj(path: Optional[str]) -> Optional[int]:
    if path is None:
        return None
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None


def _read_max_range_uj(path: Optional[str]) -> Optional[int]:
    if path is None:
        return None
    try:
        return int((Path(path).parent / "max_energy_range_uj").read_text().strip())
    except Exception:
        return None


def _delta_energy_j(
    start: Optional[int],
    end: Optional[int],
    max_range: Optional[int],
) -> Optional[float]:
    if start is None or end is None:
        return None
    if end >= start:
        delta = end - start
    elif max_range is not None:
        delta = max_range - start + end
    else:
        return None
    return delta / 1e6


# ── Peak RSS monitor ───────────────────────────────────────────────────────────

class _PeakRSSMonitor:
    def __init__(self, interval: float = 0.001):
        self._interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.peak_bytes: int = 0
        self._proc = psutil.Process(os.getpid())

    def __enter__(self) -> "_PeakRSSMonitor":
        self.peak_bytes = self._proc.memory_info().rss
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while self._running:
            try:
                rss = self._proc.memory_info().rss
                if rss > self.peak_bytes:
                    self.peak_bytes = rss
            except Exception:
                pass
            time.sleep(self._interval)

    def __exit__(self, *_) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / 1e6


# ── Profiler context manager ───────────────────────────────────────────────────

class Profiler:
    def __init__(self, rapl_file: Optional[str], rapl_max: Optional[int]):
        self._rapl_file = rapl_file
        self._rapl_max = rapl_max
        self._mem = _PeakRSSMonitor()
        self.wall_s: float = 0.0
        self.cpu_s: float = 0.0
        self.peak_mb: float = 0.0
        self.energy_j: Optional[float] = None

    def __enter__(self) -> "Profiler":
        self._wall_t0 = time.perf_counter()
        self._cpu_t0 = time.process_time()
        self._energy_start = _read_energy_uj(self._rapl_file)
        self._mem.__enter__()
        return self

    def __exit__(self, *_) -> None:
        self.wall_s = time.perf_counter() - self._wall_t0
        self.cpu_s = time.process_time() - self._cpu_t0
        self._mem.__exit__()
        self.peak_mb = self._mem.peak_mb
        energy_end = _read_energy_uj(self._rapl_file)
        self.energy_j = _delta_energy_j(self._energy_start, energy_end, self._rapl_max)


# ── Experiment runner ──────────────────────────────────────────────────────────

def run_experiment(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    simple_params: dict,
    kalman_params: dict,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    test_size: float = 0.30,
    output_dir: str = "results",
) -> pd.DataFrame:
    from .models import SimpleWiSARD, KalmanWiSARD

    rapl_file = _find_rapl_file()
    rapl_max = _read_max_range_uj(rapl_file)
    rapl_available = rapl_file is not None

    rows = []
    models = [
        ("SimpleWiSARD", SimpleWiSARD, simple_params),
        ("KalmanWiSARD", KalmanWiSARD, kalman_params),
    ]

    for model_name, model_cls, params in models:
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )
            model = model_cls(seed=seed, **params)

            with Profiler(rapl_file, rapl_max) as train_p:
                model.fit(X_train, y_train)

            with Profiler(rapl_file, rapl_max) as test_p:
                y_pred = model.predict(X_test)

            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": seed,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "train_wall_s": train_p.wall_s,
                    "test_wall_s": test_p.wall_s,
                    "train_cpu_s": train_p.cpu_s,
                    "test_cpu_s": test_p.cpu_s,
                    "train_peak_mb": train_p.peak_mb,
                    "test_peak_mb": test_p.peak_mb,
                    "train_energy_j": train_p.energy_j,
                    "test_energy_j": test_p.energy_j,
                    "rapl_available": rapl_available,
                }
            )

    df = pd.DataFrame(rows)

    Path(output_dir).mkdir(exist_ok=True)
    df.to_csv(f"{output_dir}/{dataset_name}_raw.csv", index=False)

    agg = {
        "mean_accuracy": ("accuracy", "mean"),
        "std_accuracy": ("accuracy", "std"),
        "mean_train_wall_s": ("train_wall_s", "mean"),
        "mean_test_wall_s": ("test_wall_s", "mean"),
        "mean_train_cpu_s": ("train_cpu_s", "mean"),
        "mean_test_cpu_s": ("test_cpu_s", "mean"),
        "mean_train_peak_mb": ("train_peak_mb", "mean"),
        "mean_test_peak_mb": ("test_peak_mb", "mean"),
        "mean_train_energy_j": ("train_energy_j", "mean"),
        "mean_test_energy_j": ("test_energy_j", "mean"),
    }
    summary = df.groupby("model").agg(**agg).reset_index()
    summary.insert(0, "dataset", dataset_name)
    summary.to_csv(f"{output_dir}/{dataset_name}_summary.csv", index=False)

    print(f"\n{'=' * 60}")
    print(f"  {dataset_name.upper()}")
    print(f"{'=' * 60}")
    print(summary.to_string(index=False))

    return df
