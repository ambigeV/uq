import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import torch


PROB_COL = "prob_class1_positive"
EPI_COL = "epistemic_uncertainty"
ALE_COL = "aleatoric_uncertainty"


def _safe_read_pickle(path: Path) -> pd.DataFrame:
    try:
        obj = pd.read_pickle(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read pickle: {path}") from exc
    if not isinstance(obj, pd.DataFrame):
        raise ValueError(f"Expected DataFrame in {path}, got {type(obj)}")
    return obj


def _roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int).reshape(-1)
    y_score = y_score.reshape(-1)
    n = len(y_true)
    if n == 0:
        return float("nan")
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    pos_rank_sum = ranks[y_true == 1].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _classification_metrics(
    y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    y = y_true.reshape(-1).astype(int)
    p = np.clip(prob.reshape(-1).astype(float), 1e-8, 1.0 - 1e-8)
    pred = (p >= float(threshold)).astype(int)
    acc = float((pred == y).mean())
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    auc = _roc_auc_binary(y_true=y.astype(int), y_score=p)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    tn = float(np.sum((pred == 0) & (y == 0)))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-12))
    mcc_denom = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = float(((tp * tn) - (fp * fn)) / mcc_denom) if mcc_denom > 0 else float("nan")
    return {"acc": acc, "brier": brier, "logloss": logloss, "auc": auc, "f1": f1, "mcc": mcc}


def _rankdata_average_ties(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1)
    order = np.argsort(a, kind="mergesort")
    ranks = np.zeros_like(a, dtype=float)
    sorted_vals = a[order]
    n = len(a)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def _spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) != len(y) or len(x) == 0:
        return float("nan")
    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    if denom <= 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    p = np.asarray(probs, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=float).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    overconfident_bins = 0
    underconfident_bins = 0
    non_empty_bins = 0
    per_bin: List[Dict[str, Any]] = []
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        count = int(mask.sum())
        if count == 0:
            continue
        p_bin = p[mask]
        y_bin = y[mask]
        conf = float(np.mean(p_bin))
        acc = float(np.mean((p_bin >= float(threshold)).astype(float) == y_bin))
        gap = acc - conf
        ece += (count / max(n, 1)) * abs(gap)
        non_empty_bins += 1
        if gap < 0:
            overconfident_bins += 1
            calib_state = "overconfident"
        elif gap > 0:
            underconfident_bins += 1
            calib_state = "underconfident"
        else:
            calib_state = "calibrated"
        per_bin.append(
            {
                "bin_index": int(i),
                "left": float(left),
                "right": float(right),
                "count": int(count),
                "empirical_accuracy": float(acc),
                "predicted_confidence": float(conf),
                "gap_empirical_minus_confidence": float(gap),
                "calibration_state": calib_state,
            }
        )
    return {
        "ECE": float(ece),
        "ECE_Overconfident_Bin_Count": float(overconfident_bins),
        "ECE_Underconfident_Bin_Count": float(underconfident_bins),
        "ECE_NonEmpty_Bin_Count": float(non_empty_bins),
        "ECE_Per_Bin_Details_JSON": json.dumps(per_bin, separators=(",", ":")),
    }


def _compute_confusion_matrix_binary(
    y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(probs, dtype=float).reshape(-1)
    pred = (p >= float(threshold)).astype(int)
    return {
        "confusion_tn": float(np.sum((pred == 0) & (y == 0))),
        "confusion_fp": float(np.sum((pred == 1) & (y == 0))),
        "confusion_fn": float(np.sum((pred == 0) & (y == 1))),
        "confusion_tp": float(np.sum((pred == 1) & (y == 1))),
    }


def _evaluate_like_main_active(
    y_true: np.ndarray,
    prob: np.ndarray,
    uncertainty: np.ndarray,
    n_bins: int = 10,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    cls = _classification_metrics(y_true, prob, threshold=threshold)
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.clip(np.asarray(prob, dtype=float).reshape(-1), 1e-8, 1.0 - 1e-8)
    u = np.asarray(uncertainty, dtype=float).reshape(-1)
    sq_err = (y - p) ** 2
    ece_detail = _compute_ece(p, y, n_bins=n_bins, threshold=threshold)
    out = {
        "AUC": float(cls["auc"]),
        "F1": float(cls["f1"]),
        "MCC": float(cls["mcc"]),
        "Accuracy": float(cls["acc"]),
        "NLL": float(cls["logloss"]),
        "Brier": float(cls["brier"]),
        "ECE": float(ece_detail["ECE"]),
        "ECE_Overconfident_Bin_Count": float(ece_detail["ECE_Overconfident_Bin_Count"]),
        "ECE_Underconfident_Bin_Count": float(ece_detail["ECE_Underconfident_Bin_Count"]),
        "ECE_NonEmpty_Bin_Count": float(ece_detail["ECE_NonEmpty_Bin_Count"]),
        "ECE_Per_Bin_Details_JSON": str(ece_detail["ECE_Per_Bin_Details_JSON"]),
        "Avg_Entropy": float(np.mean(u)),
        "Spearman_Err_Unc": float(_spearman_rank_corr(sq_err, u)),
        "Decision_Threshold": float(threshold),
    }
    out.update(_compute_confusion_matrix_binary(y, p, threshold=threshold))
    return out


def _select_threshold_by_val_pr(
    y_val: np.ndarray, prob_val: np.ndarray, target_recall: float = 0.95
) -> Tuple[float, float, float]:
    y = np.asarray(y_val, dtype=int).reshape(-1)
    p = np.asarray(prob_val, dtype=float).reshape(-1)
    if int((y == 1).sum()) == 0:
        return 0.5, float("nan"), float("nan")

    thresholds = np.unique(p)
    best = None
    for thr in thresholds:
        pred = p >= thr
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        if recall >= float(target_recall):
            candidate = (precision, recall, float(thr))
            if best is None or candidate[0] > best[0] or (
                candidate[0] == best[0] and candidate[1] > best[1]
            ) or (
                candidate[0] == best[0] and candidate[1] == best[1] and candidate[2] > best[2]
            ):
                best = candidate
    if best is not None:
        return best[2], best[0], best[1]

    fallback = None
    for thr in thresholds:
        pred = p >= thr
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        candidate = (recall, precision, float(thr))
        if fallback is None or candidate[0] > fallback[0] or (
            candidate[0] == fallback[0] and candidate[1] > fallback[1]
        ):
            fallback = candidate
    return fallback[2], fallback[1], fallback[0]


def _load_one_prediction(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    df = pd.read_csv(path)
    for col in [PROB_COL, EPI_COL, ALE_COL]:
        if col not in df.columns:
            raise ValueError(f"{path} missing column: {col}")
    prob = df[PROB_COL].to_numpy(dtype=np.float64)
    epi = np.clip(df[EPI_COL].to_numpy(dtype=np.float64), 1e-12, None)
    ale = np.clip(df[ALE_COL].to_numpy(dtype=np.float64), 1e-12, None)
    unc = np.clip(epi + ale, 1e-12, None)
    return prob, epi, ale, unc


def _load_split_predictions(
    pred_root: Path,
    split_name: str,
    methods: Sequence[str],
    run_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    probs, epi_uncs, ale_uncs, uncs, tags = [], [], [], [], []
    split_dir = pred_root / split_name
    for method in methods:
        path = split_dir / f"{method}_run_{run_id}.csv"
        p, epi, ale, u = _load_one_prediction(path)
        probs.append(p)
        epi_uncs.append(epi)
        ale_uncs.append(ale)
        uncs.append(u)
        tags.append(method)
    p_stack = np.stack(probs, axis=0)  # (K, N)
    u_epi_stack = np.stack(epi_uncs, axis=0)  # (K, N)
    u_ale_stack = np.stack(ale_uncs, axis=0)  # (K, N)
    u_stack = np.stack(uncs, axis=0)   # (K, N)
    return p_stack, u_epi_stack, u_ale_stack, u_stack, tags


def _fit_weights_objective(
    p_val: np.ndarray,
    y_val: np.ndarray,
    objective: str,
    l2: float = 1e-3,
    max_iters: int = 200,
    tol: float = 1e-7,
) -> np.ndarray:
    p_t = torch.as_tensor(np.clip(p_val, 1e-8, 1.0 - 1e-8), dtype=torch.double)  # (K, N)
    y_t = torch.as_tensor(y_val.reshape(-1), dtype=torch.double)                   # (N,)
    k, n = p_t.shape
    logits = torch.nn.Parameter(torch.zeros(k, dtype=torch.double))
    opt = torch.optim.LBFGS([logits], lr=1.0, max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        w = torch.softmax(logits, dim=0)
        p_ens = (w[:, None] * p_t).sum(dim=0)
        p_ens = torch.clamp(p_ens, min=1e-8, max=1.0 - 1e-8)
        if objective == "brier":
            loss_data = ((p_ens - y_t) ** 2).mean()
        elif objective == "logloss":
            loss_data = -(y_t * torch.log(p_ens) + (1.0 - y_t) * torch.log(1.0 - p_ens)).mean()
        else:
            raise ValueError(f"Unknown objective: {objective}")
        reg = 0.5 * l2 * (logits ** 2).sum() / max(n, 1)
        loss = loss_data + reg
        loss.backward()
        return loss

    prev = float("inf")
    for _ in range(max_iters):
        loss = opt.step(closure)
        cur = loss.item()
        if abs(prev - cur) < tol:
            break
        prev = cur

    with torch.no_grad():
        w = torch.softmax(logits, dim=0).cpu().numpy()
    return w.astype(np.float64)


def _fit_weights_softmax_brier_per_model(p_val: np.ndarray, y_val: np.ndarray, temp: float = 12.0) -> np.ndarray:
    y = y_val.reshape(-1)
    brier = ((p_val - y[None, :]) ** 2).mean(axis=1)
    score = -float(temp) * brier
    score = score - np.max(score)
    w = np.exp(score)
    return w / w.sum()


def _fit_weights_uncertainty_inverse_isotonic(
    p_val: np.ndarray,
    u_val: np.ndarray,
    u_test: np.ndarray,
    y_val: np.ndarray,
    target_recall: float = 0.95,
    tau: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calibrate per-method uncertainty via isotonic regression against validation error.

    x: uncertainty score
    y: 1 if prediction is wrong at that method's threshold, else 0
    """
    y = np.asarray(y_val, dtype=int).reshape(-1)
    calibrated = np.zeros_like(u_val, dtype=np.float64)
    iso_coeff_val = np.zeros(p_val.shape[0], dtype=np.float64)
    iso_coeff_test = np.zeros(p_val.shape[0], dtype=np.float64)

    for k in range(p_val.shape[0]):
        p_k = np.asarray(p_val[k], dtype=float).reshape(-1)
        u_k = np.asarray(u_val[k], dtype=float).reshape(-1)
        thr_k, _, _ = _select_threshold_by_val_pr(
            y_val=y, prob_val=p_k, target_recall=target_recall
        )
        pred_k = (p_k >= float(thr_k)).astype(int)
        err_k = (pred_k != y).astype(float)

        if np.allclose(u_k, u_k[0]):
            calibrated[k] = np.full_like(u_k, float(np.mean(err_k)), dtype=np.float64)
            ratio_val = calibrated[k] / np.clip(u_k, 1e-12, None)
            iso_coeff_val[k] = float(np.mean(ratio_val))
            u_test_k = np.asarray(u_test[k], dtype=float).reshape(-1)
            cal_test_k = np.full_like(u_test_k, float(np.mean(err_k)), dtype=np.float64)
            ratio_test = cal_test_k / np.clip(u_test_k, 1e-12, None)
            iso_coeff_test[k] = float(np.mean(ratio_test))
            continue

        iso = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
            y_min=0.0,
            y_max=1.0,
        )
        calibrated[k] = iso.fit_transform(u_k, err_k).astype(np.float64)
        ratio_val = calibrated[k] / np.clip(u_k, 1e-12, None)
        iso_coeff_val[k] = float(np.mean(ratio_val))
        u_test_k = np.asarray(u_test[k], dtype=float).reshape(-1)
        cal_test_k = iso.predict(u_test_k).astype(np.float64)
        ratio_test = cal_test_k / np.clip(u_test_k, 1e-12, None)
        iso_coeff_test[k] = float(np.mean(ratio_test))

    inv = (1.0 / np.clip(calibrated, 1e-6, None)).mean(axis=1)
    score = np.maximum(inv, 1e-12) ** float(tau)
    return score / score.sum(), iso_coeff_val, iso_coeff_test


def _apply_weights(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (w[:, None] * p).sum(axis=0)


def _load_labels(label_dir: Path, split_name: str, label_column: str) -> np.ndarray:
    pkl_path = label_dir / f"{split_name}.pkl"
    df = _safe_read_pickle(pkl_path)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' missing in {pkl_path}")
    return df[label_column].to_numpy(dtype=np.float64).reshape(-1)


def _ensure_same_length(y: np.ndarray, p: np.ndarray, name: str) -> None:
    if len(y) != p.shape[1]:
        raise ValueError(
            f"Length mismatch on {name}: labels={len(y)} vs predictions={p.shape[1]}. "
            "Current BIi CSVs do not include row ids, so strict alignment is required."
        )


def _format_metric_line(metrics: Dict[str, Any]) -> str:
    ordered = [
        "AUC", "F1", "MCC", "Accuracy", "NLL", "Brier", "ECE",
        "Avg_Entropy", "Spearman_Err_Unc",
        "Decision_Threshold",
        "confusion_tn", "confusion_fp", "confusion_fn", "confusion_tp",
    ]
    parts = []
    for k in ordered:
        v = metrics.get(k, float("nan"))
        if "confusion_" in k:
            parts.append(f"{k}={int(round(v))}")
        else:
            parts.append(f"{k}={v:.6f}")
    return " | ".join(parts)


def _append_metric_row(
    rows: List[Dict[str, object]],
    *,
    row_type: str,
    run_id: str,
    split: str,
    model_family: str,
    model_name: str,
    metrics: Dict[str, Any],
    weights: str = "",
) -> None:
    row = {
        "row_type": row_type,
        "run_id": run_id,
        "split": split,
        "model_family": model_family,
        "model_name": model_name,
        "weights": weights,
    }
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.number)):
            row[k] = float(v)
        else:
            row[k] = v
    rows.append(row)


def _display_model_name(model_key: str) -> str:
    if "::" in model_key:
        family, name = model_key.split("::", 1)
    else:
        family, name = "base", model_key
    lname = name.lower()
    if family == "base":
        if lname == "ensemble":
            return "base_gbm"
        if "mc" in lname:
            return "base_mc"
        if "evd" in lname or "new" in lname:
            return "base_evd"
        return f"base_{name}"
    if family == "stack":
        mapped = {
            "uniform": "ensemble_uniform",
            "uncertainty_inverse_isotonic": "ensemble_uncertainty_iso",
            "softmax_brier_model_score": "ensemble_softmax",
            "brier_opt": "ensemble_brier",
            "logloss_opt": "ensemble_logloss",
        }
        return mapped.get(name, f"ensemble_{name}")
    if family == "baseline":
        mapped = {
            "always_pos": "baseline_all_pos",
            "always_neg": "baseline_all_neg",
        }
        return mapped.get(name, f"baseline_{name}")
    return model_key


def _cm_counts_from_metrics(metrics: Dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(metrics.get("confusion_tn", 0.0)),
                float(metrics.get("confusion_fp", 0.0)),
            ],
            [
                float(metrics.get("confusion_fn", 0.0)),
                float(metrics.get("confusion_tp", 0.0)),
            ],
        ],
        dtype=float,
    )


def _accumulate_uncertainty_values(
    store: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    *,
    scenario: str,
    uncertainty_type: str,
    model_family: str,
    model_name: str,
    split: str,
    values: np.ndarray,
) -> None:
    model_key = f"{model_family}::{model_name}"
    scenario_store = store.setdefault(scenario, {})
    unc_store = scenario_store.setdefault(uncertainty_type, {})
    payload = unc_store.setdefault(
        model_key,
        {
            "model_family": model_family,
            "model_name": model_name,
            "val": [],
            "test": [],
        },
    )
    payload[split].append(np.asarray(values, dtype=float).reshape(-1))


def _accumulate_auc_cutoff_inputs(
    store: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    *,
    scenario: str,
    uncertainty_type: str,
    model_family: str,
    model_name: str,
    split: str,
    uncertainty: np.ndarray,
    prob: np.ndarray,
    labels: np.ndarray,
) -> None:
    model_key = f"{model_family}::{model_name}"
    scenario_store = store.setdefault(scenario, {})
    unc_store = scenario_store.setdefault(uncertainty_type, {})
    payload = unc_store.setdefault(
        model_key,
        {
            "model_family": model_family,
            "model_name": model_name,
            "val_unc": [],
            "val_prob": [],
            "val_y": [],
            "test_unc": [],
            "test_prob": [],
            "test_y": [],
        },
    )
    payload[f"{split}_unc"].append(np.asarray(uncertainty, dtype=float).reshape(-1))
    payload[f"{split}_prob"].append(np.asarray(prob, dtype=float).reshape(-1))
    payload[f"{split}_y"].append(np.asarray(labels, dtype=float).reshape(-1))


def _export_uncertainty_distributions(
    store: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    out_dir: Path,
    hist_bins: int,
) -> None:
    _ = hist_bins  # kept for CLI backward compatibility
    out_dir.mkdir(parents=True, exist_ok=True)
    for scenario in ["all", "conformal"]:
        scenario_store = store.get(scenario, {})
        for uncertainty_type in ["epistemic", "aleatoric", "total"]:
            unc_store = scenario_store.get(uncertainty_type, {})
            if not unc_store:
                continue

            def _model_sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, str]:
                payload = item[1]
                fam_order = 0 if payload["model_family"] == "base" else 1
                return fam_order, str(payload["model_name"])

            ordered_items = sorted(unc_store.items(), key=_model_sort_key)
            box_values: List[np.ndarray] = []
            box_positions: List[float] = []
            box_labels: List[str] = []
            csv_rows: List[Dict[str, Any]] = []
            tick_pos: List[float] = []
            tick_labels: List[str] = []
            pos = 1.0
            pair_gap = 1.1

            for _, payload in ordered_items:
                val_arrays = [arr for arr in payload["val"] if arr.size > 0]
                test_arrays = [arr for arr in payload["test"] if arr.size > 0]
                if not val_arrays and not test_arrays:
                    continue

                val_values = np.concatenate(val_arrays, axis=0) if val_arrays else np.asarray([], dtype=float)
                test_values = np.concatenate(test_arrays, axis=0) if test_arrays else np.asarray([], dtype=float)
                model_label = f"{payload['model_family']}::{payload['model_name']}"
                val_pos = pos
                test_pos = pos + 0.35
                tick_pos.append(0.5 * (val_pos + test_pos))
                tick_labels.append(model_label)
                pos = pos + pair_gap

                for split, values, split_pos in [
                    ("val", val_values, val_pos),
                    ("test", test_values, test_pos),
                ]:
                    if values.size == 0:
                        continue
                    box_values.append(values)
                    box_positions.append(split_pos)
                    box_labels.append(split)
                    csv_rows.append(
                        {
                            "scenario": scenario,
                            "uncertainty_type": uncertainty_type,
                            "model_family": payload["model_family"],
                            "model_name": payload["model_name"],
                            "split": split,
                            "count": int(values.size),
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values, ddof=0)),
                            "min": float(np.min(values)),
                            "q1": float(np.quantile(values, 0.25)),
                            "median": float(np.quantile(values, 0.5)),
                            "q3": float(np.quantile(values, 0.75)),
                            "max": float(np.max(values)),
                        }
                    )

            if not box_values:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(max(14, 1.5 * len(tick_labels)), 6))
            bp = ax.boxplot(
                box_values,
                positions=box_positions,
                widths=0.28,
                patch_artist=True,
                showfliers=False,
                whis=(5, 95),
            )
            for patch, split_label in zip(bp["boxes"], box_labels):
                patch.set_facecolor("#4C78A8" if split_label == "val" else "#F58518")
                patch.set_alpha(0.65)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=35, ha="right")
            ax.set_ylabel(f"{uncertainty_type} uncertainty")
            ax.set_title(f"{scenario} | val/test paired boxplots by model")
            ax.grid(axis="y", alpha=0.25)
            from matplotlib.patches import Patch
            ax.legend(
                handles=[
                    Patch(facecolor="#4C78A8", alpha=0.65, label="val"),
                    Patch(facecolor="#F58518", alpha=0.65, label="test"),
                ],
                frameon=False,
                loc="upper right",
            )
            fig.tight_layout()

            stem = f"{scenario}_{uncertainty_type}_distribution"
            fig.savefig(out_dir / f"{stem}.png", dpi=200)
            plt.close(fig)
            pd.DataFrame(csv_rows).to_csv(out_dir / f"{stem}.csv", index=False)


def _export_brier_by_uncertainty_cutoff(
    store: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cutoff_percents = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # User-requested behavior:
    # - include both all-data and conformal scenarios
    # - only epistemic uncertainty cutoff
    # - export cutoff curves for Brier, F1, and MCC
    uncertainty_type = "epistemic"

    def _f1_mcc(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Tuple[float, float]:
        y = np.asarray(y_true, dtype=int).reshape(-1)
        p = np.asarray(prob, dtype=float).reshape(-1)
        pred = (p >= float(threshold)).astype(int)
        tp = float(np.sum((pred == 1) & (y == 1)))
        fp = float(np.sum((pred == 1) & (y == 0)))
        fn = float(np.sum((pred == 0) & (y == 1)))
        tn = float(np.sum((pred == 0) & (y == 0)))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-12))
        mcc_denom = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = float(((tp * tn) - (fp * fn)) / mcc_denom) if mcc_denom > 0 else 0.0
        return f1, mcc

    metric_cfg = {
        "brier": ("Brier", "lower"),
        "f1": ("F1", "higher"),
        "mcc": ("MCC", "higher"),
    }
    for scenario in ["all", "conformal"]:
        unc_store = store.get(scenario, {}).get(uncertainty_type, {})
        if not unc_store:
            continue
        for metric_key, (metric_title, direction) in metric_cfg.items():
            rows: List[Dict[str, Any]] = []
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
            plotted_any = False
            for split, ax in zip(["val", "test"], axes):
                for model_key in sorted(unc_store.keys()):
                    payload = unc_store[model_key]
                    unc_parts = [x for x in payload[f"{split}_unc"] if x.size > 0]
                    prob_parts = [x for x in payload[f"{split}_prob"] if x.size > 0]
                    y_parts = [x for x in payload[f"{split}_y"] if x.size > 0]
                    if not unc_parts or not prob_parts or not y_parts:
                        continue

                    unc = np.concatenate(unc_parts, axis=0)
                    prob = np.concatenate(prob_parts, axis=0)
                    y = np.concatenate(y_parts, axis=0)
                    if len(unc) != len(prob) or len(prob) != len(y) or len(y) == 0:
                        continue

                    order = np.argsort(unc, kind="mergesort")
                    x_vals: List[int] = []
                    y_vals: List[float] = []
                    for pct in cutoff_percents:
                        keep_n = int(np.floor(len(order) * (pct / 100.0)))
                        keep_n = max(1, min(keep_n, len(order)))
                        idx = order[:keep_n]
                        if metric_key == "brier":
                            metric_val = float(np.mean((prob[idx] - y[idx]) ** 2))
                        else:
                            f1_val, mcc_val = _f1_mcc(y_true=y[idx], prob=prob[idx], threshold=0.5)
                            metric_val = float(f1_val if metric_key == "f1" else mcc_val)
                        x_vals.append(pct)
                        y_vals.append(metric_val)
                        rows.append(
                            {
                                "scenario": scenario,
                                "uncertainty_type": uncertainty_type,
                                "metric": metric_key,
                                "split": split,
                                "model_family": payload["model_family"],
                                "model_name": payload["model_name"],
                                "cutoff_percent_lowest_uncertainty": int(pct),
                                "samples_used": int(keep_n),
                                "uncertainty_threshold": float(unc[idx].max()),
                                "metric_value": float(metric_val),
                            }
                        )
                    label = _display_model_name(f"{payload['model_family']}::{payload['model_name']}")
                    ax.plot(x_vals, y_vals, marker="o", linewidth=1.2, label=label)
                    plotted_any = True
                ax.set_title(f"{split} {metric_title} vs low-epistemic-unc coverage")
                ax.set_xlabel("Lowest epistemic uncertainty coverage (%)")
                ax.grid(alpha=0.25)
                ax.set_xticks(cutoff_percents)
            axes[0].set_ylabel(f"{metric_title} ({direction} is better)")
            if plotted_any:
                axes[1].legend(fontsize=8, frameon=False, loc="best")
            fig.suptitle(f"{scenario} | {uncertainty_type} uncertainty cutoff {metric_title}", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            stem = f"{scenario}_{uncertainty_type}_{metric_key}_by_uncertainty_cutoff"
            fig.savefig(out_dir / f"{stem}.png", dpi=200)
            plt.close(fig)
            if rows:
                pd.DataFrame(rows).to_csv(out_dir / f"{stem}.csv", index=False)


def _export_confusion_matrix_grid(
    entries: List[Dict[str, Any]],
    out_png: Path,
    out_csv: Path,
) -> None:
    if not entries:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(entries)
    fig, axes = plt.subplots(
        2, n_models, figsize=(max(14, 3.2 * n_models), 7.2), squeeze=False
    )
    vmax = 100.0

    csv_rows: List[Dict[str, Any]] = []
    for col, entry in enumerate(entries):
        model_name = str(entry["model_name"])
        display_name = _display_model_name(model_name)
        for row, split in enumerate(["val", "test"]):
            cm_counts = np.asarray(entry[f"{split}_cm"], dtype=float)
            row_sums = cm_counts.sum(axis=1, keepdims=True)
            cm = 100.0 * cm_counts / np.clip(row_sums, 1e-12, None)
            ax = axes[row, col]
            ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=vmax)
            for i in range(2):
                for j in range(2):
                    val = float(cm[i, j])
                    ax.text(
                        j,
                        i,
                        f"{val:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black",
                    )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=8)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["True 0", "True 1"], fontsize=8)
            if row == 0:
                ax.set_title(display_name, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{split}\nlabel", fontsize=9)
            ax.set_xlabel("prediction", fontsize=8)

            csv_rows.append(
                {
                    "model_name": model_name,
                    "model_display_name": display_name,
                    "split": split,
                    "tn_percent": float(cm[0, 0]),
                    "fp_percent": float(cm[0, 1]),
                    "fn_percent": float(cm[1, 0]),
                    "tp_percent": float(cm[1, 1]),
                    "negative_recall_percent": float(cm[0, 0]),
                    "positive_recall_percent": float(cm[1, 1]),
                }
            )

    fig.suptitle("Confusion matrices by method (row-normalized by true label)", fontsize=12)
    # Avoid tight_layout warning with colorbar + multi-axes grids.
    fig.subplots_adjust(left=0.06, right=0.93, bottom=0.10, top=0.88, wspace=0.30, hspace=0.28)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)


def _export_radar_grid(
    radar_metrics_store: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]],
    scenario: str,
    out_png: Path,
    out_csv: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    metric_spec = [
        ("AUC", True),
        ("F1", True),
        ("MCC", True),
        ("Accuracy", True),
        ("NLL", False),
        ("Brier", False),
        ("ECE", False),
    ]
    metric_names = [m for m, _ in metric_spec]
    metric_dirs = {m: hb for m, hb in metric_spec}
    splits = ["val", "test"]
    # Keep polygons away from the origin to improve readability.
    radar_floor = 0.40

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), subplot_kw={"polar": True})
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])
    csv_rows: List[Dict[str, Any]] = []
    legend_items: Dict[str, Any] = {}

    for c, split in enumerate(splits):
        ax = axes[c]
        per_model_runs = radar_metrics_store.get(scenario, {}).get(split, {})
        mean_by_model: Dict[str, Dict[str, float]] = {}
        for model_name, rows in per_model_runs.items():
            if not rows:
                continue
            metric_keys = sorted(set().union(*[rr.keys() for rr in rows]))
            mm: Dict[str, float] = {}
            for metric_name in metric_keys:
                sample = rows[0].get(metric_name, np.nan)
                if not isinstance(sample, (int, float, np.number)):
                    continue
                vals = np.asarray([rr.get(metric_name, np.nan) for rr in rows], dtype=float)
                finite = np.isfinite(vals)
                if not np.any(finite):
                    mm[metric_name] = float("nan")
                else:
                    mm[metric_name] = float(np.mean(vals[finite]))
            mean_by_model[model_name] = mm

        if not mean_by_model:
            ax.set_xticks(angles)
            ax.set_xticklabels(metric_names, fontsize=9)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(
                ["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=8
            )
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.25)
            continue

        model_names = sorted(mean_by_model.keys())
        raw = np.full((len(model_names), len(metric_names)), np.nan, dtype=float)
        for i, model_name in enumerate(model_names):
            vals = mean_by_model[model_name]
            for j, metric_name in enumerate(metric_names):
                raw[i, j] = float(vals.get(metric_name, np.nan))

        norm = np.zeros_like(raw, dtype=float)
        for j, metric_name in enumerate(metric_names):
            col = raw[:, j]
            finite = np.isfinite(col)
            if not np.any(finite):
                continue
            cvals = col[finite]
            c_min = float(np.min(cvals))
            c_max = float(np.max(cvals))
            if abs(c_max - c_min) < 1e-12:
                norm[finite, j] = 0.5
            else:
                if metric_dirs[metric_name]:
                    norm[finite, j] = (cvals - c_min) / (c_max - c_min)
                else:
                    norm[finite, j] = (c_max - cvals) / (c_max - c_min)
            norm[~finite, j] = 0.0
        norm_shifted = radar_floor + (1.0 - radar_floor) * norm

        for i, model_name in enumerate(model_names):
            values = np.concatenate([norm_shifted[i], norm_shifted[i, :1]])
            display_name = _display_model_name(model_name)
            line, = ax.plot(angles_closed, values, linewidth=1.2, label=display_name)
            ax.fill(angles_closed, values, alpha=0.06)
            if display_name not in legend_items:
                legend_items[display_name] = line
            row: Dict[str, Any] = {
                "scenario": scenario,
                "split": split,
                "model_name": model_name,
                "model_display_name": display_name,
            }
            for j, metric_name in enumerate(metric_names):
                row[f"{metric_name}_mean_raw"] = float(raw[i, j]) if np.isfinite(raw[i, j]) else float("nan")
                row[f"{metric_name}_radar_norm"] = float(norm[i, j])
                row[f"{metric_name}_radar_plot_radius"] = float(norm_shifted[i, j])
            csv_rows.append(row)

        ax.set_xticks(angles)
        ax.set_xticklabels(metric_names, fontsize=9)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(
            ["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=8
        )
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)

    if legend_items:
        fig.legend(
            handles=list(legend_items.values()),
            labels=list(legend_items.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=min(4, max(1, len(legend_items))),
            frameon=False,
            fontsize=11,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(out_csv, index=False)

def _build_label_conditional_conformal_scores(
    y_val: np.ndarray, prob_pos_val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_val, dtype=int).reshape(-1)
    p = np.clip(np.asarray(prob_pos_val, dtype=float).reshape(-1), 1e-8, 1.0 - 1e-8)
    # Nonconformity for positive label uses p(y=1|x); for negative label uses p(y=0|x)=1-p.
    scores_pos = 1.0 - p[y == 1]
    scores_neg = 1.0 - (1.0 - p[y == 0])  # equals p for y==0 samples
    return np.sort(scores_pos), np.sort(scores_neg)


def _derived_csv_path(base_path: Path, suffix: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def _accumulate_conformal_hist_values(
    store: Dict[str, Dict[str, List[np.ndarray]]],
    *,
    split: str,
    y_true: np.ndarray,
    prob: np.ndarray,
    accepted_mask: np.ndarray,
    pred_label: np.ndarray,
) -> None:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.clip(np.asarray(prob, dtype=float).reshape(-1), 1e-8, 1.0 - 1e-8)
    mask = np.asarray(accepted_mask, dtype=bool).reshape(-1)
    pred = np.asarray(pred_label, dtype=int).reshape(-1)
    if mask.sum() == 0:
        return
    y_acc = y[mask]
    p_acc = p[mask]
    pred_acc = pred[mask]
    correct = pred_acc == y_acc
    nll = -(y_acc * np.log(p_acc) + (1 - y_acc) * np.log(1 - p_acc))
    conf = np.maximum(p_acc, 1.0 - p_acc)

    payload = store.setdefault(
        split,
        {
            "nll_correct": [],
            "nll_wrong": [],
            "conf_correct": [],
            "conf_wrong": [],
        },
    )
    payload["nll_correct"].append(nll[correct])
    payload["nll_wrong"].append(nll[~correct])
    payload["conf_correct"].append(conf[correct])
    payload["conf_wrong"].append(conf[~correct])


def _export_conformal_histograms(
    store: Dict[str, Dict[str, List[np.ndarray]]],
    out_dir: Path,
    bins_nll: int = 60,
    bins_conf: int = 40,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _concat(parts: List[np.ndarray]) -> np.ndarray:
        valid = [x for x in parts if x.size > 0]
        return np.concatenate(valid, axis=0) if valid else np.asarray([], dtype=float)

    # Plot 1: per-sample NLL histogram on accepted samples.
    fig_nll, axes_nll = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for split, ax in zip(["val", "test"], axes_nll):
        payload = store.get(split, {})
        nll_correct = _concat(payload.get("nll_correct", []))
        nll_wrong = _concat(payload.get("nll_wrong", []))
        if nll_correct.size == 0 and nll_wrong.size == 0:
            ax.text(0.5, 0.5, "No accepted samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        all_vals = np.concatenate([x for x in [nll_correct, nll_wrong] if x.size > 0], axis=0)
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        bins = np.linspace(vmin, vmax, bins_nll + 1)
        if nll_correct.size > 0:
            ax.hist(nll_correct, bins=bins, alpha=0.55, label="accepted_correct", color="#4C78A8")
        if nll_wrong.size > 0:
            ax.hist(nll_wrong, bins=bins, alpha=0.55, label="accepted_wrong", color="#E45756")
        ax.set_xlabel("Per-sample NLL contribution")
        ax.set_ylabel("Count")
        ax.set_title(f"{split} accepted samples")
        ax.grid(alpha=0.2)
    axes_nll[1].legend(frameon=False, fontsize=9, loc="best")
    fig_nll.tight_layout()
    fig_nll.savefig(out_dir / "conformal_accepted_nll_hist.png", dpi=220)
    plt.close(fig_nll)

    # Plot 2: confidence histogram on accepted samples.
    fig_conf, axes_conf = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    conf_bins = np.linspace(0.0, 1.0, bins_conf + 1)
    for split, ax in zip(["val", "test"], axes_conf):
        payload = store.get(split, {})
        conf_correct = _concat(payload.get("conf_correct", []))
        conf_wrong = _concat(payload.get("conf_wrong", []))
        if conf_correct.size == 0 and conf_wrong.size == 0:
            ax.text(0.5, 0.5, "No accepted samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        if conf_correct.size > 0:
            ax.hist(conf_correct, bins=conf_bins, alpha=0.55, label="accepted_correct", color="#4C78A8")
        if conf_wrong.size > 0:
            ax.hist(conf_wrong, bins=conf_bins, alpha=0.55, label="accepted_wrong", color="#E45756")
        ax.set_xlabel("Confidence max(p, 1-p)")
        ax.set_ylabel("Count")
        ax.set_title(f"{split} accepted samples")
        ax.grid(alpha=0.2)
    axes_conf[1].legend(frameon=False, fontsize=9, loc="best")
    fig_conf.tight_layout()
    fig_conf.savefig(out_dir / "conformal_accepted_confidence_hist.png", dpi=220)
    plt.close(fig_conf)


def _export_reliability_diagrams(
    store: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    out_dir: Path,
    n_bins: int = 20,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)

    for scenario in ["all", "conformal"]:
        # Reuse one uncertainty bucket to avoid duplicated prob/y payloads.
        scenario_store = store.get(scenario, {})
        rel_store = scenario_store.get("total", {}) or scenario_store.get("epistemic", {})
        if not rel_store:
            continue

        rows: List[Dict[str, Any]] = []
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        plotted_any = False
        for split, ax in zip(["val", "test"], axes):
            ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1.0, label="perfect")
            for model_key in sorted(rel_store.keys()):
                payload = rel_store[model_key]
                prob_parts = [x for x in payload.get(f"{split}_prob", []) if x.size > 0]
                y_parts = [x for x in payload.get(f"{split}_y", []) if x.size > 0]
                if not prob_parts or not y_parts:
                    continue
                prob = np.concatenate(prob_parts, axis=0)
                y = np.asarray(np.concatenate(y_parts, axis=0), dtype=int)
                if len(prob) == 0 or len(prob) != len(y):
                    continue

                pred = (prob >= 0.5).astype(int)
                conf = np.maximum(prob, 1.0 - prob)
                correct = (pred == y).astype(float)

                x_vals: List[float] = []
                y_vals: List[float] = []
                for i in range(n_bins):
                    left, right = bins[i], bins[i + 1]
                    if i == n_bins - 1:
                        mask = (conf >= left) & (conf <= right)
                    else:
                        mask = (conf >= left) & (conf < right)
                    count = int(mask.sum())
                    if count == 0:
                        continue
                    conf_bin = conf[mask]
                    corr_bin = correct[mask]
                    mean_conf = float(np.mean(conf_bin))
                    emp_acc = float(np.mean(corr_bin))
                    x_vals.append(mean_conf)
                    y_vals.append(emp_acc)
                    rows.append(
                        {
                            "scenario": scenario,
                            "split": split,
                            "model_family": payload["model_family"],
                            "model_name": payload["model_name"],
                            "bin_index": int(i),
                            "bin_left": float(left),
                            "bin_right": float(right),
                            "count": count,
                            "mean_confidence": mean_conf,
                            "empirical_accuracy": emp_acc,
                        }
                    )
                if x_vals:
                    label = _display_model_name(f"{payload['model_family']}::{payload['model_name']}")
                    ax.plot(x_vals, y_vals, marker="o", linewidth=1.2, label=label)
                    plotted_any = True

            ax.set_title(f"{split} reliability")
            ax.set_xlabel("Mean confidence max(p, 1-p)")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Empirical accuracy")
        axes[0].set_xlim(0.0, 1.0)
        axes[1].set_xlim(0.0, 1.0)
        axes[0].set_ylim(0.0, 1.0)
        axes[1].set_ylim(0.0, 1.0)
        if plotted_any:
            axes[1].legend(fontsize=8, frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{scenario}_reliability_diagram.png", dpi=220)
        plt.close(fig)
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"{scenario}_reliability_diagram.csv", index=False)


def _conformal_single_label_accept_mask(
    prob_pos_test: np.ndarray,
    sorted_scores_pos: np.ndarray,
    sorted_scores_neg: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    p = np.clip(np.asarray(prob_pos_test, dtype=float).reshape(-1), 1e-8, 1.0 - 1e-8)
    n = len(p)
    if len(sorted_scores_pos) == 0 or len(sorted_scores_neg) == 0:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=int)

    s_pos = 1.0 - p
    s_neg = p

    idx_pos = np.searchsorted(sorted_scores_pos, s_pos, side="left")
    idx_neg = np.searchsorted(sorted_scores_neg, s_neg, side="left")
    count_ge_pos = len(sorted_scores_pos) - idx_pos
    count_ge_neg = len(sorted_scores_neg) - idx_neg
    pval_pos = (count_ge_pos + 1.0) / (len(sorted_scores_pos) + 1.0)
    pval_neg = (count_ge_neg + 1.0) / (len(sorted_scores_neg) + 1.0)

    in_pos = pval_pos > float(alpha)
    in_neg = pval_neg > float(alpha)
    accepted = np.logical_xor(in_pos, in_neg)
    pred_label = np.where(np.logical_and(in_pos, np.logical_not(in_neg)), 1, 0).astype(int)
    return accepted, pred_label


def _evaluate_accepted_subset(
    y_true: np.ndarray,
    prob: np.ndarray,
    uncertainty: np.ndarray,
    accepted_mask: np.ndarray,
    n_bins: int,
    threshold: float,
) -> Dict[str, Any]:
    y = np.asarray(y_true).reshape(-1)
    p = np.asarray(prob).reshape(-1)
    u = np.asarray(uncertainty).reshape(-1)
    mask = np.asarray(accepted_mask, dtype=bool).reshape(-1)
    accepted_count = int(mask.sum())
    total_count = int(len(mask))
    if accepted_count == 0:
        out = {
            "accepted_count": 0.0,
            "acceptance_rate": 0.0,
            "rejected_count": float(total_count),
            "rejection_rate": 1.0,
        }
        for k in [
            "AUC", "F1", "MCC", "Accuracy", "NLL", "Brier", "ECE",
            "ECE_Overconfident_Bin_Count", "ECE_Underconfident_Bin_Count", "ECE_NonEmpty_Bin_Count",
            "Avg_Entropy", "Spearman_Err_Unc",
            "Decision_Threshold", "confusion_tn", "confusion_fp", "confusion_fn", "confusion_tp",
        ]:
            out[k] = float("nan")
        out["ECE_Per_Bin_Details_JSON"] = "[]"
        return out

    metrics = _evaluate_like_main_active(
        y_true=y[mask], prob=p[mask], uncertainty=u[mask], n_bins=n_bins, threshold=threshold
    )
    metrics["accepted_count"] = float(accepted_count)
    metrics["acceptance_rate"] = float(accepted_count / max(total_count, 1))
    metrics["rejected_count"] = float(total_count - accepted_count)
    metrics["rejection_rate"] = float(1.0 - metrics["acceptance_rate"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stack BIi method predictions using validation-set learned weights."
    )
    parser.add_argument("--pred_root", type=str, default="inference_outputs")
    parser.add_argument("--label_dir", type=str, default="cytotoxicity_data")
    parser.add_argument("--val_split", type=str, default="HEK293_test_BM")
    parser.add_argument("--test_split", type=str, default="tox21_all")
    parser.add_argument("--label_column", type=str, default="Outcome")
    parser.add_argument(
        "--methods",
        type=str,
        default="dmpnn_bii_dmpnn_balanced,dmpnn_bii_dmpnn,identity_bii_identity_balanced,identity_bii_identity",
        help="Comma-separated method names."
    )
    parser.add_argument("--run_ids", type=str, default="0,1,2,3,4")
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--uncertainty_tau", type=float, default=1.0)
    parser.add_argument(
        "--uncertainty_weight_source",
        type=str,
        default="total",
        choices=["total", "epistemic"],
        help="Uncertainty type used for isotonic inverse-uncertainty weighting.",
    )
    parser.add_argument("--score_temp", type=float, default=12.0)
    parser.add_argument("--ece_bins", type=int, default=10)
    parser.add_argument(
        "--reliability_bins",
        type=int,
        default=20,
        help="Bin count for reliability diagrams (confidence vs empirical accuracy).",
    )
    parser.add_argument("--target_recall", type=float, default=0.95)
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="inference_outputs/stacking/stacking_summary_all_metrics.csv",
        help="Single consolidated CSV path for all per-run and aggregate metrics.",
    )
    parser.add_argument(
        "--conformal_alpha",
        type=float,
        default=0.1,
        help="Conformal significance level for single-label acceptance filtering.",
    )
    parser.add_argument(
        "--conformal_summary_csv",
        type=str,
        default="inference_outputs/stacking/stacking_summary_conformal_accepted.csv",
        help="Additional CSV with metrics on conformal-accepted test subset.",
    )
    parser.add_argument(
        "--uncertainty_dist_dir",
        type=str,
        default="",
        help="Optional output directory for uncertainty distribution CSV/PNG artifacts. If empty, auto-create one under summary_csv parent.",
    )
    parser.add_argument(
        "--uncertainty_hist_bins",
        type=int,
        default=60,
        help="Histogram bins used for uncertainty distribution exports.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    pred_root = Path(args.pred_root)
    if not pred_root.is_absolute():
        pred_root = repo_root / pred_root

    label_dir = Path(args.label_dir)
    if not label_dir.is_absolute():
        label_dir = repo_root / label_dir
    summary_csv = Path(args.summary_csv)
    if not summary_csv.is_absolute():
        summary_csv = repo_root / summary_csv
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    conformal_summary_csv = Path(args.conformal_summary_csv)
    if not conformal_summary_csv.is_absolute():
        conformal_summary_csv = repo_root / conformal_summary_csv
    conformal_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.uncertainty_dist_dir.strip():
        uncertainty_dist_dir = Path(args.uncertainty_dist_dir)
        if not uncertainty_dist_dir.is_absolute():
            uncertainty_dist_dir = repo_root / uncertainty_dist_dir
    else:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        uncertainty_dist_dir = summary_csv.parent / f"uncertainty_distributions_{run_tag}"

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    run_ids = [int(x.strip()) for x in args.run_ids.split(",") if x.strip()]
    if not methods:
        raise ValueError("No methods provided.")
    if not run_ids:
        raise ValueError("No run_ids provided.")

    y_val = _load_labels(label_dir=label_dir, split_name=args.val_split, label_column=args.label_column)
    y_test = _load_labels(label_dir=label_dir, split_name=args.test_split, label_column=args.label_column)

    all_test_probs: Dict[str, List[np.ndarray]] = {
        "uniform": [],
        "uncertainty_inverse_isotonic": [],
        "softmax_brier_model_score": [],
        "brier_opt": [],
        "logloss_opt": [],
    }
    all_val_probs: Dict[str, List[np.ndarray]] = {k: [] for k in all_test_probs.keys()}
    all_test_uncs: Dict[str, List[np.ndarray]] = {k: [] for k in all_test_probs.keys()}
    all_weights: Dict[str, List[np.ndarray]] = {k: [] for k in all_test_probs.keys()}
    aggregate_metrics: Dict[str, List[Dict[str, Any]]] = {}
    report_rows: List[Dict[str, object]] = []
    conformal_rows: List[Dict[str, object]] = []
    conformal_aggregate: Dict[str, List[Dict[str, Any]]] = {}
    uncertainty_store: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    auc_cutoff_store: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    conformal_hist_store: Dict[str, Dict[str, List[np.ndarray]]] = {
        "val": {"nll_correct": [], "nll_wrong": [], "conf_correct": [], "conf_wrong": []},
        "test": {"nll_correct": [], "nll_wrong": [], "conf_correct": [], "conf_wrong": []},
    }
    confusion_store: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = {
        "all": {"val": {}, "test": {}},
        "conformal": {"val": {}, "test": {}},
    }
    radar_metrics_store: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {
        "all": {"val": {}, "test": {}},
        "conformal": {"val": {}, "test": {}},
    }

    for run_id in run_ids:
        p_val, u_val_epi, u_val_ale, u_val, tags_val = _load_split_predictions(
            pred_root=pred_root, split_name=args.val_split, methods=methods, run_id=run_id
        )
        p_test, u_test_epi, u_test_ale, u_test, tags_test = _load_split_predictions(
            pred_root=pred_root, split_name=args.test_split, methods=methods, run_id=run_id
        )
        if tags_val != tags_test:
            raise RuntimeError(f"Method ordering mismatch for run {run_id}.")

        _ensure_same_length(y_val, p_val, f"val run {run_id}")
        _ensure_same_length(y_test, p_test, f"test run {run_id}")

        print(f"\n=== Run {run_id} ===")
        print("[Base methods]")
        for idx, method in enumerate(methods):
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="all",
                uncertainty_type="epistemic",
                model_family="base",
                model_name=method,
                split="val",
                uncertainty=u_val_epi[idx],
                prob=p_val[idx],
                labels=y_val,
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="all",
                uncertainty_type="aleatoric",
                model_family="base",
                model_name=method,
                split="val",
                uncertainty=u_val_ale[idx],
                prob=p_val[idx],
                labels=y_val,
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="all",
                uncertainty_type="total",
                model_family="base",
                model_name=method,
                split="val",
                uncertainty=u_val[idx],
                prob=p_val[idx],
                labels=y_val,
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="all",
                uncertainty_type="epistemic",
                model_family="base",
                model_name=method,
                split="test",
                uncertainty=u_test_epi[idx],
                prob=p_test[idx],
                labels=y_test,
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="all",
                uncertainty_type="aleatoric",
                model_family="base",
                model_name=method,
                split="test",
                uncertainty=u_test_ale[idx],
                prob=p_test[idx],
                labels=y_test,
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="all",
                uncertainty_type="total",
                model_family="base",
                model_name=method,
                split="test",
                uncertainty=u_test[idx],
                prob=p_test[idx],
                labels=y_test,
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="all",
                uncertainty_type="epistemic",
                model_family="base",
                model_name=method,
                split="val",
                values=u_val_epi[idx],
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="all",
                uncertainty_type="aleatoric",
                model_family="base",
                model_name=method,
                split="val",
                values=u_val_ale[idx],
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="all",
                uncertainty_type="total",
                model_family="base",
                model_name=method,
                split="val",
                values=u_val[idx],
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="all",
                uncertainty_type="epistemic",
                model_family="base",
                model_name=method,
                split="test",
                values=u_test_epi[idx],
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="all",
                uncertainty_type="aleatoric",
                model_family="base",
                model_name=method,
                split="test",
                values=u_test_ale[idx],
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="all",
                uncertainty_type="total",
                model_family="base",
                model_name=method,
                split="test",
                values=u_test[idx],
            )
            thr, val_prec_at_thr, val_rec_at_thr = _select_threshold_by_val_pr(
                y_val=y_val, prob_val=p_val[idx], target_recall=args.target_recall
            )
            val_m = _evaluate_like_main_active(
                y_true=y_val,
                prob=p_val[idx],
                uncertainty=u_val[idx],
                n_bins=args.ece_bins,
                threshold=thr,
            )
            test_m = _evaluate_like_main_active(
                y_true=y_test,
                prob=p_test[idx],
                uncertainty=u_test[idx],
                n_bins=args.ece_bins,
                threshold=thr,
            )
            val_m["Val_Precision_At_Selected_Thr"] = float(val_prec_at_thr)
            val_m["Val_Recall_At_Selected_Thr"] = float(val_rec_at_thr)
            test_m["Val_Precision_At_Selected_Thr"] = float(val_prec_at_thr)
            test_m["Val_Recall_At_Selected_Thr"] = float(val_rec_at_thr)
            print(f"- {method} | val  | {_format_metric_line(val_m)}")
            print(f"- {method} | test | {_format_metric_line(test_m)}")
            base_key = f"base::{method}"
            aggregate_metrics.setdefault(base_key, []).append(test_m)
            radar_metrics_store["all"]["val"].setdefault(base_key, []).append(val_m)
            radar_metrics_store["all"]["test"].setdefault(base_key, []).append(test_m)
            confusion_store["all"]["val"].setdefault(base_key, []).append(_cm_counts_from_metrics(val_m))
            confusion_store["all"]["test"].setdefault(base_key, []).append(_cm_counts_from_metrics(test_m))
            _append_metric_row(
                report_rows,
                row_type="per_run",
                run_id=str(run_id),
                split="val",
                model_family="base",
                model_name=method,
                metrics=val_m,
            )
            _append_metric_row(
                report_rows,
                row_type="per_run",
                run_id=str(run_id),
                split="test",
                model_family="base",
                model_name=method,
                metrics=test_m,
            )

            scores_pos, scores_neg = _build_label_conditional_conformal_scores(
                y_val=y_val, prob_pos_val=p_val[idx]
            )
            accepted_mask_val, pred_label_val = _conformal_single_label_accept_mask(
                prob_pos_test=p_val[idx],
                sorted_scores_pos=scores_pos,
                sorted_scores_neg=scores_neg,
                alpha=args.conformal_alpha,
            )
            accepted_mask, pred_label_test = _conformal_single_label_accept_mask(
                prob_pos_test=p_test[idx],
                sorted_scores_pos=scores_pos,
                sorted_scores_neg=scores_neg,
                alpha=args.conformal_alpha,
            )
            _accumulate_conformal_hist_values(
                conformal_hist_store,
                split="val",
                y_true=y_val,
                prob=p_val[idx],
                accepted_mask=accepted_mask_val,
                pred_label=pred_label_val,
            )
            _accumulate_conformal_hist_values(
                conformal_hist_store,
                split="test",
                y_true=y_test,
                prob=p_test[idx],
                accepted_mask=accepted_mask,
                pred_label=pred_label_test,
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="conformal",
                uncertainty_type="epistemic",
                model_family="base",
                model_name=method,
                split="val",
                values=u_val_epi[idx][accepted_mask_val],
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="conformal",
                uncertainty_type="epistemic",
                model_family="base",
                model_name=method,
                split="val",
                uncertainty=u_val_epi[idx][accepted_mask_val],
                prob=p_val[idx][accepted_mask_val],
                labels=y_val[accepted_mask_val],
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="conformal",
                uncertainty_type="aleatoric",
                model_family="base",
                model_name=method,
                split="val",
                values=u_val_ale[idx][accepted_mask_val],
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="conformal",
                uncertainty_type="aleatoric",
                model_family="base",
                model_name=method,
                split="val",
                uncertainty=u_val_ale[idx][accepted_mask_val],
                prob=p_val[idx][accepted_mask_val],
                labels=y_val[accepted_mask_val],
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="conformal",
                uncertainty_type="total",
                model_family="base",
                model_name=method,
                split="val",
                values=u_val[idx][accepted_mask_val],
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="conformal",
                uncertainty_type="total",
                model_family="base",
                model_name=method,
                split="val",
                uncertainty=u_val[idx][accepted_mask_val],
                prob=p_val[idx][accepted_mask_val],
                labels=y_val[accepted_mask_val],
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="conformal",
                uncertainty_type="epistemic",
                model_family="base",
                model_name=method,
                split="test",
                values=u_test_epi[idx][accepted_mask],
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="conformal",
                uncertainty_type="epistemic",
                model_family="base",
                model_name=method,
                split="test",
                uncertainty=u_test_epi[idx][accepted_mask],
                prob=p_test[idx][accepted_mask],
                labels=y_test[accepted_mask],
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="conformal",
                uncertainty_type="aleatoric",
                model_family="base",
                model_name=method,
                split="test",
                values=u_test_ale[idx][accepted_mask],
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="conformal",
                uncertainty_type="aleatoric",
                model_family="base",
                model_name=method,
                split="test",
                uncertainty=u_test_ale[idx][accepted_mask],
                prob=p_test[idx][accepted_mask],
                labels=y_test[accepted_mask],
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="conformal",
                uncertainty_type="total",
                model_family="base",
                model_name=method,
                split="test",
                values=u_test[idx][accepted_mask],
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="conformal",
                uncertainty_type="total",
                model_family="base",
                model_name=method,
                split="test",
                uncertainty=u_test[idx][accepted_mask],
                prob=p_test[idx][accepted_mask],
                labels=y_test[accepted_mask],
            )
            conf_test_m = _evaluate_accepted_subset(
                y_true=y_test,
                prob=p_test[idx],
                uncertainty=u_test[idx],
                accepted_mask=accepted_mask,
                n_bins=args.ece_bins,
                threshold=thr,
            )
            conf_val_m = _evaluate_accepted_subset(
                y_true=y_val,
                prob=p_val[idx],
                uncertainty=u_val[idx],
                accepted_mask=accepted_mask_val,
                n_bins=args.ece_bins,
                threshold=thr,
            )
            conf_val_m["Val_Precision_At_Selected_Thr"] = float(val_prec_at_thr)
            conf_val_m["Val_Recall_At_Selected_Thr"] = float(val_rec_at_thr)
            conf_test_m["Val_Precision_At_Selected_Thr"] = float(val_prec_at_thr)
            conf_test_m["Val_Recall_At_Selected_Thr"] = float(val_rec_at_thr)
            conformal_aggregate.setdefault(base_key, []).append(conf_test_m)
            radar_metrics_store["conformal"]["val"].setdefault(base_key, []).append(conf_val_m)
            radar_metrics_store["conformal"]["test"].setdefault(base_key, []).append(conf_test_m)
            if int(conf_val_m.get("accepted_count", 0.0)) > 0:
                confusion_store["conformal"]["val"].setdefault(base_key, []).append(
                    _cm_counts_from_metrics(conf_val_m)
                )
            if int(conf_test_m.get("accepted_count", 0.0)) > 0:
                confusion_store["conformal"]["test"].setdefault(base_key, []).append(
                    _cm_counts_from_metrics(conf_test_m)
                )
            _append_metric_row(
                conformal_rows,
                row_type="conformal_per_run",
                run_id=str(run_id),
                split="val_conformal_accepted",
                model_family="base",
                model_name=method,
                metrics=conf_val_m,
                weights=f"alpha={args.conformal_alpha}",
            )
            _append_metric_row(
                conformal_rows,
                row_type="conformal_per_run",
                run_id=str(run_id),
                split="test_conformal_accepted",
                model_family="base",
                model_name=method,
                metrics=conf_test_m,
                weights=f"alpha={args.conformal_alpha}",
            )

        u_val_for_iso = u_val if args.uncertainty_weight_source == "total" else u_val_epi
        u_test_for_iso = u_test if args.uncertainty_weight_source == "total" else u_test_epi
        iso_w, iso_coeff_val, iso_coeff_test = _fit_weights_uncertainty_inverse_isotonic(
            p_val=p_val,
            u_val=u_val_for_iso,
            u_test=u_test_for_iso,
            y_val=y_val,
            target_recall=args.target_recall,
            tau=args.uncertainty_tau,
        )
        weights = {
            "uniform": np.ones(len(methods), dtype=np.float64) / len(methods),
            "uncertainty_inverse_isotonic": iso_w,
            "softmax_brier_model_score": _fit_weights_softmax_brier_per_model(
                p_val=p_val, y_val=y_val, temp=args.score_temp
            ),
            "brier_opt": _fit_weights_objective(
                p_val=p_val, y_val=y_val, objective="brier", l2=args.l2
            ),
            "logloss_opt": _fit_weights_objective(
                p_val=p_val, y_val=y_val, objective="logloss", l2=args.l2
            ),
        }

        print("[Stacking methods]")
        for stack_name, w in weights.items():
            val_prob = _apply_weights(p_val, w)
            test_prob = _apply_weights(p_test, w)
            val_unc_epi = _apply_weights(u_val_epi, w)
            test_unc_epi = _apply_weights(u_test_epi, w)
            val_unc_ale = _apply_weights(u_val_ale, w)
            test_unc_ale = _apply_weights(u_test_ale, w)
            val_unc = _apply_weights(u_val, w)
            test_unc = _apply_weights(u_test, w)
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="all",
                uncertainty_type="epistemic",
                model_family="stack",
                model_name=stack_name,
                split="val",
                values=val_unc_epi,
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="all",
                uncertainty_type="epistemic",
                model_family="stack",
                model_name=stack_name,
                split="val",
                uncertainty=val_unc_epi,
                prob=val_prob,
                labels=y_val,
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="all",
                uncertainty_type="aleatoric",
                model_family="stack",
                model_name=stack_name,
                split="val",
                values=val_unc_ale,
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="all",
                uncertainty_type="aleatoric",
                model_family="stack",
                model_name=stack_name,
                split="val",
                uncertainty=val_unc_ale,
                prob=val_prob,
                labels=y_val,
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="all",
                uncertainty_type="total",
                model_family="stack",
                model_name=stack_name,
                split="val",
                values=val_unc,
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="all",
                uncertainty_type="total",
                model_family="stack",
                model_name=stack_name,
                split="val",
                uncertainty=val_unc,
                prob=val_prob,
                labels=y_val,
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="all",
                uncertainty_type="epistemic",
                model_family="stack",
                model_name=stack_name,
                split="test",
                values=test_unc_epi,
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="all",
                uncertainty_type="epistemic",
                model_family="stack",
                model_name=stack_name,
                split="test",
                uncertainty=test_unc_epi,
                prob=test_prob,
                labels=y_test,
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="all",
                uncertainty_type="aleatoric",
                model_family="stack",
                model_name=stack_name,
                split="test",
                values=test_unc_ale,
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="all",
                uncertainty_type="aleatoric",
                model_family="stack",
                model_name=stack_name,
                split="test",
                uncertainty=test_unc_ale,
                prob=test_prob,
                labels=y_test,
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="all",
                uncertainty_type="total",
                model_family="stack",
                model_name=stack_name,
                split="test",
                values=test_unc,
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="all",
                uncertainty_type="total",
                model_family="stack",
                model_name=stack_name,
                split="test",
                uncertainty=test_unc,
                prob=test_prob,
                labels=y_test,
            )
            thr, val_prec_at_thr, val_rec_at_thr = _select_threshold_by_val_pr(
                y_val=y_val, prob_val=val_prob, target_recall=args.target_recall
            )
            val_metrics = _evaluate_like_main_active(
                y_true=y_val, prob=val_prob, uncertainty=val_unc, n_bins=args.ece_bins, threshold=thr
            )
            test_metrics = _evaluate_like_main_active(
                y_true=y_test, prob=test_prob, uncertainty=test_unc, n_bins=args.ece_bins, threshold=thr
            )
            val_metrics["Val_Precision_At_Selected_Thr"] = float(val_prec_at_thr)
            val_metrics["Val_Recall_At_Selected_Thr"] = float(val_rec_at_thr)
            test_metrics["Val_Precision_At_Selected_Thr"] = float(val_prec_at_thr)
            test_metrics["Val_Recall_At_Selected_Thr"] = float(val_rec_at_thr)

            all_val_probs[stack_name].append(val_prob)
            all_test_probs[stack_name].append(test_prob)
            all_test_uncs[stack_name].append(test_unc)
            all_weights[stack_name].append(w)
            stack_key = f"stack::{stack_name}"
            aggregate_metrics.setdefault(stack_key, []).append(test_metrics)
            radar_metrics_store["all"]["val"].setdefault(stack_key, []).append(val_metrics)
            radar_metrics_store["all"]["test"].setdefault(stack_key, []).append(test_metrics)
            confusion_store["all"]["val"].setdefault(stack_key, []).append(_cm_counts_from_metrics(val_metrics))
            confusion_store["all"]["test"].setdefault(stack_key, []).append(_cm_counts_from_metrics(test_metrics))
            w_str = ", ".join(f"{m}={ww:.4f}" for m, ww in zip(methods, w))
            if stack_name == "uncertainty_inverse_isotonic":
                coef_val_str = ", ".join(f"{m}={cc:.4f}" for m, cc in zip(methods, iso_coeff_val))
                coef_test_str = ", ".join(f"{m}={cc:.4f}" for m, cc in zip(methods, iso_coeff_test))
                print(f"- {stack_name} | isotonic_coef_val: {coef_val_str}")
                print(f"- {stack_name} | isotonic_coef_test: {coef_test_str}")
            print(f"- {stack_name} | weights: {w_str}")
            print(f"  val  | {_format_metric_line(val_metrics)}")
            print(f"  test | {_format_metric_line(test_metrics)}")
            _append_metric_row(
                report_rows,
                row_type="per_run",
                run_id=str(run_id),
                split="val",
                model_family="stack",
                model_name=stack_name,
                metrics=val_metrics,
                weights=(
                    w_str
                    if stack_name != "uncertainty_inverse_isotonic"
                    else (
                        w_str
                        + " | isotonic_coef_val: "
                        + ", ".join(f"{m}={cc:.4f}" for m, cc in zip(methods, iso_coeff_val))
                        + " | isotonic_coef_test: "
                        + ", ".join(f"{m}={cc:.4f}" for m, cc in zip(methods, iso_coeff_test))
                    )
                ),
            )
            _append_metric_row(
                report_rows,
                row_type="per_run",
                run_id=str(run_id),
                split="test",
                model_family="stack",
                model_name=stack_name,
                metrics=test_metrics,
                weights=(
                    w_str
                    if stack_name != "uncertainty_inverse_isotonic"
                    else (
                        w_str
                        + " | isotonic_coef_val: "
                        + ", ".join(f"{m}={cc:.4f}" for m, cc in zip(methods, iso_coeff_val))
                        + " | isotonic_coef_test: "
                        + ", ".join(f"{m}={cc:.4f}" for m, cc in zip(methods, iso_coeff_test))
                    )
                ),
            )

            scores_pos, scores_neg = _build_label_conditional_conformal_scores(
                y_val=y_val, prob_pos_val=val_prob
            )
            accepted_mask_val, pred_label_val = _conformal_single_label_accept_mask(
                prob_pos_test=val_prob,
                sorted_scores_pos=scores_pos,
                sorted_scores_neg=scores_neg,
                alpha=args.conformal_alpha,
            )
            accepted_mask, pred_label_test = _conformal_single_label_accept_mask(
                prob_pos_test=test_prob,
                sorted_scores_pos=scores_pos,
                sorted_scores_neg=scores_neg,
                alpha=args.conformal_alpha,
            )
            _accumulate_conformal_hist_values(
                conformal_hist_store,
                split="val",
                y_true=y_val,
                prob=val_prob,
                accepted_mask=accepted_mask_val,
                pred_label=pred_label_val,
            )
            _accumulate_conformal_hist_values(
                conformal_hist_store,
                split="test",
                y_true=y_test,
                prob=test_prob,
                accepted_mask=accepted_mask,
                pred_label=pred_label_test,
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="conformal",
                uncertainty_type="epistemic",
                model_family="stack",
                model_name=stack_name,
                split="val",
                values=val_unc_epi[accepted_mask_val],
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="conformal",
                uncertainty_type="epistemic",
                model_family="stack",
                model_name=stack_name,
                split="val",
                uncertainty=val_unc_epi[accepted_mask_val],
                prob=val_prob[accepted_mask_val],
                labels=y_val[accepted_mask_val],
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="conformal",
                uncertainty_type="aleatoric",
                model_family="stack",
                model_name=stack_name,
                split="val",
                values=val_unc_ale[accepted_mask_val],
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="conformal",
                uncertainty_type="aleatoric",
                model_family="stack",
                model_name=stack_name,
                split="val",
                uncertainty=val_unc_ale[accepted_mask_val],
                prob=val_prob[accepted_mask_val],
                labels=y_val[accepted_mask_val],
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="conformal",
                uncertainty_type="total",
                model_family="stack",
                model_name=stack_name,
                split="val",
                values=val_unc[accepted_mask_val],
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="conformal",
                uncertainty_type="total",
                model_family="stack",
                model_name=stack_name,
                split="val",
                uncertainty=val_unc[accepted_mask_val],
                prob=val_prob[accepted_mask_val],
                labels=y_val[accepted_mask_val],
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="conformal",
                uncertainty_type="epistemic",
                model_family="stack",
                model_name=stack_name,
                split="test",
                values=test_unc_epi[accepted_mask],
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="conformal",
                uncertainty_type="epistemic",
                model_family="stack",
                model_name=stack_name,
                split="test",
                uncertainty=test_unc_epi[accepted_mask],
                prob=test_prob[accepted_mask],
                labels=y_test[accepted_mask],
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="conformal",
                uncertainty_type="aleatoric",
                model_family="stack",
                model_name=stack_name,
                split="test",
                values=test_unc_ale[accepted_mask],
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="conformal",
                uncertainty_type="aleatoric",
                model_family="stack",
                model_name=stack_name,
                split="test",
                uncertainty=test_unc_ale[accepted_mask],
                prob=test_prob[accepted_mask],
                labels=y_test[accepted_mask],
            )
            _accumulate_uncertainty_values(
                uncertainty_store,
                scenario="conformal",
                uncertainty_type="total",
                model_family="stack",
                model_name=stack_name,
                split="test",
                values=test_unc[accepted_mask],
            )
            _accumulate_auc_cutoff_inputs(
                auc_cutoff_store,
                scenario="conformal",
                uncertainty_type="total",
                model_family="stack",
                model_name=stack_name,
                split="test",
                uncertainty=test_unc[accepted_mask],
                prob=test_prob[accepted_mask],
                labels=y_test[accepted_mask],
            )
            conf_test_m = _evaluate_accepted_subset(
                y_true=y_test,
                prob=test_prob,
                uncertainty=test_unc,
                accepted_mask=accepted_mask,
                n_bins=args.ece_bins,
                threshold=thr,
            )
            conf_val_m = _evaluate_accepted_subset(
                y_true=y_val,
                prob=val_prob,
                uncertainty=val_unc,
                accepted_mask=accepted_mask_val,
                n_bins=args.ece_bins,
                threshold=thr,
            )
            conf_val_m["Val_Precision_At_Selected_Thr"] = float(val_prec_at_thr)
            conf_val_m["Val_Recall_At_Selected_Thr"] = float(val_rec_at_thr)
            conf_test_m["Val_Precision_At_Selected_Thr"] = float(val_prec_at_thr)
            conf_test_m["Val_Recall_At_Selected_Thr"] = float(val_rec_at_thr)
            conformal_aggregate.setdefault(stack_key, []).append(conf_test_m)
            radar_metrics_store["conformal"]["val"].setdefault(stack_key, []).append(conf_val_m)
            radar_metrics_store["conformal"]["test"].setdefault(stack_key, []).append(conf_test_m)
            if int(conf_val_m.get("accepted_count", 0.0)) > 0:
                confusion_store["conformal"]["val"].setdefault(stack_key, []).append(
                    _cm_counts_from_metrics(conf_val_m)
                )
            if int(conf_test_m.get("accepted_count", 0.0)) > 0:
                confusion_store["conformal"]["test"].setdefault(stack_key, []).append(
                    _cm_counts_from_metrics(conf_test_m)
                )
            _append_metric_row(
                conformal_rows,
                row_type="conformal_per_run",
                run_id=str(run_id),
                split="val_conformal_accepted",
                model_family="stack",
                model_name=stack_name,
                metrics=conf_val_m,
                weights=f"{w_str}; alpha={args.conformal_alpha}",
            )
            _append_metric_row(
                conformal_rows,
                row_type="conformal_per_run",
                run_id=str(run_id),
                split="test_conformal_accepted",
                model_family="stack",
                model_name=stack_name,
                metrics=conf_test_m,
                weights=f"{w_str}; alpha={args.conformal_alpha}",
            )

        # Radar-only baselines for visual scale anchoring.
        baseline_defs = {
            "always_pos": (
                np.ones_like(y_val, dtype=np.float64),
                np.ones_like(y_test, dtype=np.float64),
            ),
            "always_neg": (
                np.zeros_like(y_val, dtype=np.float64),
                np.zeros_like(y_test, dtype=np.float64),
            ),
        }
        for baseline_name, (val_prob_b, test_prob_b) in baseline_defs.items():
            key = f"baseline::{baseline_name}"
            val_unc_b = np.full_like(val_prob_b, 1e-7, dtype=np.float64)
            test_unc_b = np.full_like(test_prob_b, 1e-7, dtype=np.float64)
            thr_b = 0.5
            val_metrics_b = _evaluate_like_main_active(
                y_true=y_val,
                prob=val_prob_b,
                uncertainty=val_unc_b,
                n_bins=args.ece_bins,
                threshold=thr_b,
            )
            test_metrics_b = _evaluate_like_main_active(
                y_true=y_test,
                prob=test_prob_b,
                uncertainty=test_unc_b,
                n_bins=args.ece_bins,
                threshold=thr_b,
            )
            radar_metrics_store["all"]["val"].setdefault(key, []).append(val_metrics_b)
            radar_metrics_store["all"]["test"].setdefault(key, []).append(test_metrics_b)

    # Aggregate over runs: average test probabilities and average weights.
    print("\n=== Mean-over-runs (test) ===")
    for stack_name, prob_list in all_test_probs.items():
        if not prob_list:
            continue
        val_list = all_val_probs[stack_name]
        p_mean = np.mean(np.stack(prob_list, axis=0), axis=0)
        p_val_mean = np.mean(np.stack(val_list, axis=0), axis=0)
        u_mean = np.mean(np.stack(all_test_uncs[stack_name], axis=0), axis=0)
        w_mean = np.mean(np.stack(all_weights[stack_name], axis=0), axis=0)
        thr, val_prec_at_thr, val_rec_at_thr = _select_threshold_by_val_pr(
            y_val=y_val, prob_val=p_val_mean, target_recall=args.target_recall
        )
        agg_metrics = _evaluate_like_main_active(
            y_true=y_test, prob=p_mean, uncertainty=u_mean, n_bins=args.ece_bins, threshold=thr
        )
        agg_metrics["Val_Precision_At_Selected_Thr"] = float(val_prec_at_thr)
        agg_metrics["Val_Recall_At_Selected_Thr"] = float(val_rec_at_thr)
        w_str = ", ".join(f"{m}={ww:.4f}" for m, ww in zip(methods, w_mean))
        print(f"- {stack_name} | mean weights: {w_str}")
        print(f"  {_format_metric_line(agg_metrics)}")
        _append_metric_row(
            report_rows,
            row_type="mean_over_runs_prediction",
            run_id="mean_over_runs",
            split="test",
            model_family="stack",
            model_name=stack_name,
            metrics=agg_metrics,
            weights=w_str,
        )

    if aggregate_metrics:
        print("\n=== Per-method mean±std over runs (test) ===")
        ordered = [
            "AUC", "F1", "MCC", "Accuracy", "NLL", "Brier", "ECE",
            "Avg_Entropy", "Spearman_Err_Unc",
            "Decision_Threshold", "Val_Precision_At_Selected_Thr", "Val_Recall_At_Selected_Thr",
            "confusion_tn", "confusion_fp", "confusion_fn", "confusion_tp",
        ]
        for name, rows in aggregate_metrics.items():
            parts = []
            mean_std_metrics: Dict[str, float] = {}
            for k in ordered:
                vals = np.asarray([r[k] for r in rows], dtype=float)
                if "confusion_" in k:
                    parts.append(f"{k}={vals.mean():.1f}±{vals.std(ddof=0):.1f}")
                else:
                    parts.append(f"{k}={vals.mean():.6f}±{vals.std(ddof=0):.6f}")
                mean_std_metrics[f"{k}_mean"] = float(vals.mean())
                mean_std_metrics[f"{k}_std"] = float(vals.std(ddof=0))
            print(f"- {name} | " + " | ".join(parts))
            fam, model = name.split("::", 1)
            _append_metric_row(
                report_rows,
                row_type="mean_std_over_runs",
                run_id="all_runs",
                split="test",
                model_family=fam,
                model_name=model,
                metrics=mean_std_metrics,
            )

    report_df = pd.DataFrame(report_rows)
    report_all_runs_df = report_df[report_df["row_type"] == "per_run"].copy()
    report_summary_df = report_df[report_df["row_type"] != "per_run"].copy()
    summary_mean_csv = _derived_csv_path(summary_csv, "summary_mean")
    report_all_runs_df.to_csv(summary_csv, index=False)
    report_summary_df.to_csv(summary_mean_csv, index=False)
    print(f"\nSaved per-run metrics CSV: {summary_csv}")
    print(f"Saved summary-mean metrics CSV: {summary_mean_csv}")
    _export_uncertainty_distributions(
        store=uncertainty_store,
        out_dir=uncertainty_dist_dir,
        hist_bins=args.uncertainty_hist_bins,
    )
    _export_brier_by_uncertainty_cutoff(
        store=auc_cutoff_store,
        out_dir=uncertainty_dist_dir,
    )
    _export_reliability_diagrams(
        store=auc_cutoff_store,
        out_dir=uncertainty_dist_dir,
        n_bins=args.reliability_bins,
    )
    _export_conformal_histograms(
        store=conformal_hist_store,
        out_dir=uncertainty_dist_dir,
    )
    print(f"Saved uncertainty distribution artifacts under: {uncertainty_dist_dir}")

    # Export confusion matrices without/with conformal filtering.
    def _model_sort_key(name: str) -> Tuple[int, str]:
        return (0 if name.startswith("base::") else 1, name)

    for scenario in ["all", "conformal"]:
        entries: List[Dict[str, Any]] = []
        model_names = sorted(set(confusion_store[scenario]["val"].keys()) | set(confusion_store[scenario]["test"].keys()), key=_model_sort_key)
        for model_name in model_names:
            val_runs = confusion_store[scenario]["val"].get(model_name, [])
            test_runs = confusion_store[scenario]["test"].get(model_name, [])
            if not val_runs or not test_runs:
                continue
            val_sum = np.sum(np.stack(val_runs, axis=0), axis=0)
            test_sum = np.sum(np.stack(test_runs, axis=0), axis=0)
            entries.append(
                {
                    "model_name": model_name,
                    "val_cm": val_sum,
                    "test_cm": test_sum,
                }
            )
        _export_confusion_matrix_grid(
            entries=entries,
            out_png=summary_csv.parent / f"confusion_matrices_val_test_all_methods_{scenario}.png",
            out_csv=summary_csv.parent / f"confusion_matrices_val_test_all_methods_{scenario}.csv",
        )

    # Export radar figures separately for all-data and conformal scenarios.
    _export_radar_grid(
        radar_metrics_store=radar_metrics_store,
        scenario="all",
        out_png=summary_csv.parent / "radar_all.png",
        out_csv=summary_csv.parent / "radar_all.csv",
    )
    _export_radar_grid(
        radar_metrics_store=radar_metrics_store,
        scenario="conformal",
        out_png=summary_csv.parent / "radar_conformal.png",
        out_csv=summary_csv.parent / "radar_conformal.csv",
    )

    if conformal_aggregate:
        for name, rows in conformal_aggregate.items():
            fam, model = name.split("::", 1)
            mean_std_metrics: Dict[str, float] = {}
            metric_keys = sorted(set().union(*[r.keys() for r in rows]))
            for k in metric_keys:
                sample_val = rows[0].get(k, np.nan)
                if not isinstance(sample_val, (int, float, np.number)):
                    continue
                vals = np.asarray([r.get(k, np.nan) for r in rows], dtype=float)
                finite = np.isfinite(vals)
                if not np.any(finite):
                    mean_std_metrics[f"{k}_mean"] = float("nan")
                    mean_std_metrics[f"{k}_std"] = float("nan")
                else:
                    mean_std_metrics[f"{k}_mean"] = float(np.mean(vals[finite]))
                    mean_std_metrics[f"{k}_std"] = float(np.std(vals[finite], ddof=0))
            _append_metric_row(
                conformal_rows,
                row_type="conformal_mean_std_over_runs",
                run_id="all_runs",
                split="test_conformal_accepted",
                model_family=fam,
                model_name=model,
                metrics=mean_std_metrics,
                weights=f"alpha={args.conformal_alpha}",
            )

    conformal_df = pd.DataFrame(conformal_rows)
    conformal_all_runs_df = conformal_df[conformal_df["row_type"] == "conformal_per_run"].copy()
    conformal_summary_df = conformal_df[conformal_df["row_type"] != "conformal_per_run"].copy()
    conformal_summary_mean_csv = _derived_csv_path(conformal_summary_csv, "summary_mean")
    conformal_all_runs_df.to_csv(conformal_summary_csv, index=False)
    conformal_summary_df.to_csv(conformal_summary_mean_csv, index=False)
    print(f"Saved conformal per-run metrics CSV: {conformal_summary_csv}")
    print(f"Saved conformal summary-mean metrics CSV: {conformal_summary_mean_csv}")


if __name__ == "__main__":
    main()
