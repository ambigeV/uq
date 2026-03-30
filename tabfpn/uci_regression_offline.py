#!/usr/bin/env python3
"""
Offline TabPFN regression benchmark on small UCI datasets.

Uses the local tabpfn package (no API client). For each test point we
retrieve the mean prediction plus uncertainty estimates derived from
the predicted quantile distribution.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm as scipy_norm
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ShuffleSplit
from tabpfn import TabPFNRegressor


MAX_TRAIN = 500

# z-score for the 80% CI (10th–90th percentile) under Gaussian assumption
_Z_80 = scipy_norm.ppf(0.90)   # 1.2816


def tabpfn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    return_std: bool = False,
    n_estimators: int = 8,
    random_state: int = 42,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Fit TabPFN on (X_train, y_train) and predict on X_test.

    Parameters
    ----------
    X_train, y_train : training data
    X_test           : test features
    return_std       : if True, also return GP-equivalent predictive std
                       computed as (q90 - q10) / (2 * 1.2816)
    n_estimators     : TabPFN ensemble size
    random_state     : RNG seed

    Returns
    -------
    y_pred           : shape (n_test,)
    std              : shape (n_test,)  — only when return_std=True
    """
    model = TabPFNRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        ignore_pretraining_limits=True,
    )
    model.fit(X_train, y_train)

    if not return_std:
        return model.predict(X_test)

    output = model.predict(X_test, output_type="main")
    y_pred = output["mean"]
    quantiles = np.array(output["quantiles"])   # (n_quantiles, n_test)
    q10, q90 = quantiles[0], quantiles[-1]
    std = (q90 - q10) / (2.0 * _Z_80)          # GP-equivalent σ
    return y_pred, std


# ---------------------------------------------------------------------------
# Benchmark helpers (unchanged logic, now delegated to tabpfn_predict)
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    name: str
    X: np.ndarray
    y: np.ndarray


def get_regression_datasets() -> list[DatasetSpec]:
    diabetes = load_diabetes()
    cal = fetch_california_housing()

    rng = np.random.default_rng(0)
    idx = rng.choice(len(cal.target), size=2000, replace=False)

    return [
        DatasetSpec("diabetes", diabetes.data, diabetes.target),
        DatasetSpec("cal_housing_2k", cal.data[idx], cal.target[idx]),
    ]


def evaluate_one_dataset(
    data: DatasetSpec,
    train_sizes: list[int],
    n_splits: int,
    random_state: int,
    n_estimators: int,
) -> list[dict]:
    X, y = data.X, data.y
    results = []

    for train_size in train_sizes:
        train_size = min(train_size, MAX_TRAIN)
        if train_size >= len(y):
            continue

        splitter = ShuffleSplit(
            n_splits=n_splits,
            train_size=train_size,
            test_size=len(y) - train_size,
            random_state=random_state,
        )

        rmse_list, mae_list, mean_std_list = [], [], []

        for train_idx, test_idx in splitter.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            y_pred, std = tabpfn_predict(
                X_train, y_train, X_test,
                return_std=True,
                n_estimators=n_estimators,
                random_state=random_state,
            )

            rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae_list.append(mean_absolute_error(y_test, y_pred))
            mean_std_list.append(float(np.mean(std)))

        results.append({
            "dataset": data.name,
            "train_size": train_size,
            "n_splits": n_splits,
            "rmse": float(np.mean(rmse_list)),
            "mae": float(np.mean(mae_list)),
            "mean_std": float(np.mean(mean_std_list)),
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline TabPFN regression benchmark on UCI datasets."
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=[64, 128, 256, 500],
        help=f"Train sizes (capped at {MAX_TRAIN}).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Number of random splits per train size.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=8,
    )
    args = parser.parse_args()

    all_results = []
    for dataset in get_regression_datasets():
        print(f"Evaluating {dataset.name} ...")
        all_results.extend(
            evaluate_one_dataset(
                data=dataset,
                train_sizes=args.train_sizes,
                n_splits=args.n_splits,
                random_state=args.random_state,
                n_estimators=args.n_estimators,
            )
        )

    print("\nOffline TabPFN Regression Benchmark\n")
    print(
        f"{'dataset':<20}{'train_size':>12}{'splits':>8}"
        f"{'RMSE':>12}{'MAE':>12}{'mean_std(GP)':>16}"
    )
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['dataset']:<20}{r['train_size']:>12}{r['n_splits']:>8}"
            f"{r['rmse']:>12.4f}{r['mae']:>12.4f}{r['mean_std']:>16.4f}"
        )


if __name__ == "__main__":
    main()
