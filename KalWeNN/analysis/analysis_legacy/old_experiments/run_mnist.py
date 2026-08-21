import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.loaders import get_mnist
from wisard.metrics import run_experiment

# Larger address_size and lower resolution for 784-dimensional input.
# max_samples=10_000 keeps training tractable; increase to use more data.
SIMPLE_PARAMS = dict(address_size=8, resolution=8)
KALMAN_PARAMS = dict(address_size=8, resolution=8, q=1e-3, r=5e-2, p0=1.0, n_epochs=1)
SEEDS = list(range(20))
MAX_SAMPLES = 10_000


def run():
    X, y = get_mnist(max_samples=MAX_SAMPLES)
    return run_experiment(X, y, "mnist", SIMPLE_PARAMS, KALMAN_PARAMS, SEEDS)


if __name__ == "__main__":
    run()
