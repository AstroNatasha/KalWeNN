import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.loaders import get_wine
from wisard.metrics import run_experiment

SIMPLE_PARAMS = dict(address_size=4, resolution=16)
KALMAN_PARAMS = dict(address_size=4, resolution=16, q=1e-3, r=5e-2, p0=1.0, n_epochs=1)
SEEDS = list(range(20))


def run():
    X, y = get_wine()
    return run_experiment(X, y, "wine", SIMPLE_PARAMS, KALMAN_PARAMS, SEEDS)


if __name__ == "__main__":
    run()
