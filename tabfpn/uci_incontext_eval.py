#!/usr/bin/env python3
"""
Quick TabPFN client (API) benchmark on small UCI datasets.

This script evaluates how performance changes with different
"on-demand" train set sizes, which is useful for probing
in-context learning behavior.
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from tabpfn_client import TabPFNClassifier, init, set_access_token


@dataclass
class DatasetSpec:
    name: str
    X: np.ndarray
    y: np.ndarray


def get_uci_like_datasets() -> list[DatasetSpec]:
    iris = load_iris()
    wine = load_wine()
    breast_cancer = load_breast_cancer()

    return [
        DatasetSpec("iris", iris.data, iris.target),
        DatasetSpec("wine", wine.data, wine.target),
        DatasetSpec("breast_cancer", breast_cancer.data, breast_cancer.target),
    ]


def evaluate_one_dataset(
    model: TabPFNClassifier,
    data: DatasetSpec,
    train_sizes: list[int],
    n_splits: int,
    random_state: int,
) -> list[tuple[str, int, int, float, float]]:
    X, y = data.X, data.y
    results: list[tuple[str, int, int, float, float]] = []

    n_classes = len(np.unique(y))
    min_required = n_classes

    for train_size in train_sizes:
        if train_size < min_required or train_size >= len(y):
            continue

        splitter = StratifiedShuffleSplit(
            n_splits=n_splits,
            train_size=train_size,
            test_size=len(y) - train_size,
            random_state=random_state,
        )

        acc_scores: list[float] = []
        f1_scores: list[float] = []

        for train_idx, test_idx in splitter.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average="macro"))

        results.append(
            (
                data.name,
                train_size,
                n_splits,
                float(np.mean(acc_scores)),
                float(np.mean(f1_scores)),
            )
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TabPFN client API on small UCI datasets "
            "with varying train sizes."
        )
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=[16, 32, 64, 96, 128],
        help="Train sizes for on-demand train/test splits.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of random stratified splits per train size.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="v2.5_default",
        help=(
            "TabPFN server model identifier. "
            "Use v2.5_default to explicitly target TabPFN v2.5."
        ),
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=8,
        help="Number of estimators to request from the API.",
    )
    parser.add_argument(
        "--access-token",
        type=str,
        default=None,
        help=(
            "TabPFN API access token. "
            "If omitted, TABPFN_ACCESS_TOKEN env var is used when available."
        ),
    )
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Skip tabpfn_client.init() (useful if already authenticated).",
    )
    args = parser.parse_args()

    token = args.access_token or os.getenv("TABPFN_ACCESS_TOKEN")
    if token:
        set_access_token(token)
        print("Using API token from --access-token/TABPFN_ACCESS_TOKEN.")

    if not args.skip_init:
        init()

    print(f"Using TabPFN client model_path: {args.model_path}")
    model = TabPFNClassifier(
        model_path=args.model_path,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    all_results: list[tuple[str, int, int, float, float]] = []

    for dataset in get_uci_like_datasets():
        all_results.extend(
            evaluate_one_dataset(
                model=model,
                data=dataset,
                train_sizes=args.train_sizes,
                n_splits=args.n_splits,
                random_state=args.random_state,
            )
        )

    if not all_results:
        raise RuntimeError("No valid evaluations were run. Check train sizes and dataset sizes.")

    print("\nTabPFN Client (API) In-Context Learning Quick Benchmark\n")
    print(f"{'dataset':<16}{'train_size':>12}{'splits':>10}{'acc_mean':>12}{'f1_macro':>12}")
    print("-" * 62)
    for ds, ts, sp, acc, f1m in all_results:
        print(f"{ds:<16}{ts:>12}{sp:>10}{acc:>12.4f}{f1m:>12.4f}")


if __name__ == "__main__":
    main()
