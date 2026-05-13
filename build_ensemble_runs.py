import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


EXPECTED_KEYS = ("Pos_Prob", "Neg_Prob")
DEFAULT_FOLDERS = ("tox21_test", "cytosafe_val")
DEFAULT_RUNS = 5

# Column pairs (Pos_Prob_col, Neg_Prob_col) in gbm_prediction.csv
GBM_MODEL_COLS = [
    ("LGBM_BSM_Pos_Prob",   "LGBM_BSM_Neg_Prob"),
    ("LGBM_Naive_Pos_Prob", "LGBM_Naive_Neg_Prob"),
    ("LGBM_Unb_Pos_Prob",   "LGBM_Unb_Neg_Prob"),
    ("kNN_Pos_Prob",         "kNN_Neg_Prob"),
    ("XGB_Pos_Prob",         "XGB_Neg_Prob"),
]


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


def _load_model_outputs_from_csv(csv_path: Path) -> List[Dict[str, List[float]]]:
    df = pd.read_csv(csv_path)
    model_outputs = []
    for pos_col, neg_col in GBM_MODEL_COLS:
        model_outputs.append({
            "Pos_Prob": df[pos_col].tolist(),
            "Neg_Prob": df[neg_col].tolist(),
        })
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


def process_gbm_csv(csv_path: Path, out_folder: Path, runs: int) -> None:
    out_folder.mkdir(parents=True, exist_ok=True)
    model_outputs = _load_model_outputs_from_csv(csv_path)
    rows = _build_rows(model_outputs)
    for run_idx in range(1, runs + 1):
        out_path = out_folder / f"ensemble_run_{run_idx}.csv"
        _write_run_csv(out_path, rows)
    print(f"[ok] {csv_path.name} -> {out_folder}: wrote {runs} files with {len(rows)} rows each.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ensemble CSV runs from 5 base-model JSON outputs or a gbm_prediction CSV."
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
        help="Folder names under repo root to process (JSON mode).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Number of duplicate run csv files to emit per folder.",
    )
    parser.add_argument(
        "--gbm-csv",
        type=Path,
        default=None,
        help="Path to gbm_prediction.csv; if given, runs CSV mode instead of JSON mode.",
    )
    parser.add_argument(
        "--gbm-out",
        type=Path,
        default=None,
        help="Output folder for CSV mode (default: <repo-root>/inference_outputs/BII).",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    if args.gbm_csv is not None:
        out_folder = args.gbm_out or (args.repo_root / "inference_outputs" / "BII")
        process_gbm_csv(args.gbm_csv, out_folder, args.runs)
    else:
        for folder_name in args.folders:
            folder_path = args.repo_root / folder_name
            if not folder_path.exists():
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            process_folder(folder_path, args.runs)


if __name__ == "__main__":
    main()
