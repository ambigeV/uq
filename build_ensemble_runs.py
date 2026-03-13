import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence


EXPECTED_KEYS = ("Pos_Prob", "Neg_Prob")
DEFAULT_FOLDERS = ("tox21_test", "cytosafe_val")
DEFAULT_RUNS = 5


def _load_model_json(path: Path) -> Dict[str, List[float]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")

    for key in EXPECTED_KEYS:
        if key not in data:
            raise ValueError(f"{path} missing required key: {key}")
        if not isinstance(data[key], list):
            raise ValueError(f"{path} key {key} must be a list.")

    return data


def _validate_and_collect(folder: Path) -> List[Dict[str, List[float]]]:
    json_paths = sorted(folder.glob("*.json"))
    if len(json_paths) != 5:
        raise ValueError(f"{folder} must contain exactly 5 json files, found {len(json_paths)}.")

    model_outputs = [_load_model_json(path) for path in json_paths]

    expected_len = len(model_outputs[0]["Pos_Prob"])
    for idx, model in enumerate(model_outputs):
        if len(model["Pos_Prob"]) != expected_len or len(model["Neg_Prob"]) != expected_len:
            raise ValueError(
                f"{folder} json index {idx} has inconsistent list lengths "
                f"(Pos_Prob={len(model['Pos_Prob'])}, Neg_Prob={len(model['Neg_Prob'])}, expected={expected_len})."
            )
    return model_outputs


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values))


def _variance(values: Sequence[float]) -> float:
    mu = _mean(values)
    return sum((v - mu) ** 2 for v in values) / float(len(values))


def _build_rows(model_outputs: List[Dict[str, List[float]]]) -> List[List[float]]:
    n_samples = len(model_outputs[0]["Pos_Prob"])
    rows: List[List[float]] = []

    for i in range(n_samples):
        pos_probs = [float(m["Pos_Prob"][i]) for m in model_outputs]
        neg_probs = [float(m["Neg_Prob"][i]) for m in model_outputs]

        prob_class1_positive = _mean(pos_probs)
        prob_class2_negative = _mean(neg_probs)
        epistemic_uncertainty = _variance(pos_probs)
        aleatoric_uncertainty = _mean([p * (1.0 - p) for p in pos_probs])

        rows.append(
            [
                prob_class1_positive,
                prob_class2_negative,
                epistemic_uncertainty,
                aleatoric_uncertainty,
            ]
        )
    return rows


def _write_run_csv(path: Path, rows: List[List[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "prob_class1_positive",
                "prob_class2_negative",
                "epistemic_uncertainty",
                "aleatoric_uncertainty",
            ]
        )
        writer.writerows(rows)


def process_folder(folder: Path, runs: int) -> None:
    model_outputs = _validate_and_collect(folder)
    rows = _build_rows(model_outputs)
    for run_idx in range(1, runs + 1):
        out_path = folder / f"ensemble_run_{run_idx}.csv"
        _write_run_csv(out_path, rows)
    print(f"[ok] {folder}: wrote {runs} files with {len(rows)} rows each.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ensemble CSV runs from 5 base-model JSON outputs."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root containing target folders.",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=list(DEFAULT_FOLDERS),
        help="Folder names under repo root to process.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Number of duplicate run csv files to emit per folder.",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    for folder_name in args.folders:
        folder_path = args.repo_root / folder_name
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        process_folder(folder_path, args.runs)


if __name__ == "__main__":
    main()
