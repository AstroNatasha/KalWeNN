import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from experiments import (
    run_wine,
    run_iris,
    run_breast_cancer,
    run_digits,
    run_mnist,
    run_glass,
    run_vehicle,
    run_segment,
    run_vowel,
    run_satellite,
)

EXPERIMENTS = [
    run_wine,
    run_iris,
    run_breast_cancer,
    run_digits,
    run_glass,
    run_vehicle,
    run_segment,
    run_vowel,
    run_satellite,
    run_mnist,  # last — requires download and is the slowest
]


def main():
    all_frames = [module.run() for module in EXPERIMENTS]
    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv("results/all_raw.csv", index=False)

    summary = (
        combined.groupby(["dataset", "model"])
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_train_wall_s=("train_wall_s", "mean"),
            mean_test_wall_s=("test_wall_s", "mean"),
            mean_train_energy_j=("train_energy_j", "mean"),
            mean_test_energy_j=("test_energy_j", "mean"),
        )
        .reset_index()
    )
    summary.to_csv("results/all_summary.csv", index=False)
    print("\n\n===== COMBINED SUMMARY =====")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
