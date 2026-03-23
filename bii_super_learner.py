import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import deepchem as dc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
import torch
from nn import graphdata_to_batchmolgraph
from nn_baseline import build_model


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
    score_cap: float = 1e6,
    ood_lambda: float = 0.75,
    ood_z_tolerance: float = 1.5,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Calibrate per-method uncertainty via isotonic regression against validation error.

    x: uncertainty score
    y: 1 if prediction is wrong at that method's threshold, else 0

    Weighting:
    - Build empirical CDF quantiles against each method's validation isotonic distribution.
    - Use exponential inverse-uncertainty scores exp(-tau * quantile), normalized per sample.
    """
    y = np.asarray(y_val, dtype=int).reshape(-1)
    calibrated = np.zeros_like(u_val, dtype=np.float64)
    calibrated_test = np.zeros_like(u_test, dtype=np.float64)
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
            calibrated_test[k] = cal_test_k
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
        calibrated_test[k] = cal_test_k
        ratio_test = cal_test_k / np.clip(u_test_k, 1e-12, None)
        iso_coeff_test[k] = float(np.mean(ratio_test))

    _ = ood_lambda
    _ = ood_z_tolerance
    cap = float(max(score_cap, 1e-6))

    # Empirical CDF quantile against validation isotonic-uncertainty distribution.
    quantile_val = np.zeros_like(calibrated, dtype=np.float64)
    quantile_test = np.zeros_like(calibrated_test, dtype=np.float64)
    for k in range(calibrated.shape[0]):
        ref = np.sort(np.asarray(calibrated[k], dtype=np.float64).reshape(-1))
        n_ref = len(ref)
        if n_ref == 0:
            quantile_val[k] = np.full_like(calibrated[k], 0.5, dtype=np.float64)
            quantile_test[k] = np.full_like(calibrated_test[k], 0.5, dtype=np.float64)
            continue
        q_val_idx = np.searchsorted(ref, calibrated[k], side="right")
        q_test_idx = np.searchsorted(ref, calibrated_test[k], side="right")
        quantile_val[k] = (q_val_idx.astype(np.float64) + 1.0) / float(n_ref + 1.0)
        quantile_test[k] = (q_test_idx.astype(np.float64) + 1.0) / float(n_ref + 1.0)
    quantile_val = np.clip(quantile_val, 1e-8, 1.0)
    quantile_test = np.clip(quantile_test, 1e-8, 1.0)

    # Exponential inverse-uncertainty weighting in quantile space.
    # Lower quantile (less uncertain vs val reference) => larger score.
    exp_val = np.exp(-float(tau) * quantile_val)
    exp_test = np.exp(-float(tau) * quantile_test)
    inv_val = np.clip(exp_val, 1e-12, cap)
    inv_test_raw = np.clip(exp_test, 1e-12, cap)
    val_mean = np.mean(calibrated, axis=1, keepdims=True)
    val_std = np.std(calibrated, axis=1, ddof=0, keepdims=True)
    val_std = np.clip(val_std, 1e-6, None)
    z_test = (calibrated_test - val_mean) / val_std
    z_test_abs = np.abs(z_test)
    ood_penalty = np.ones_like(inv_test_raw, dtype=np.float64)
    inv_test = inv_test_raw * ood_penalty

    sum_val = np.clip(inv_val.sum(axis=0, keepdims=True), 1e-12, None)
    sum_test = np.clip(inv_test.sum(axis=0, keepdims=True), 1e-12, None)
    sample_w_val = inv_val / sum_val
    sample_w_test = inv_test / sum_test
    # Keep a compact per-method summary (mean over val samples) for logs/bar plots.
    mean_w = np.mean(sample_w_val, axis=1)
    mean_w = mean_w / np.clip(mean_w.sum(), 1e-12, None)
    return (
        mean_w,
        iso_coeff_val,
        iso_coeff_test,
        calibrated,
        calibrated_test,
        sample_w_val,
        sample_w_test,
        z_test,
        z_test_abs,
        quantile_test,
        inv_test_raw,
        ood_penalty,
    )


def _apply_weights(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    p_arr = np.asarray(p, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64)
    if w_arr.ndim == 1:
        if w_arr.shape[0] != p_arr.shape[0]:
            raise ValueError(f"Weight length mismatch: {w_arr.shape[0]} vs {p_arr.shape[0]}.")
        return (w_arr[:, None] * p_arr).sum(axis=0)
    if w_arr.ndim == 2:
        if w_arr.shape != p_arr.shape:
            raise ValueError(f"Adaptive weight shape mismatch: {w_arr.shape} vs {p_arr.shape}.")
        return (w_arr * p_arr).sum(axis=0)
    raise ValueError(f"Unsupported weight ndim={w_arr.ndim}; expected 1 or 2.")


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
            # Distinguish balanced vs unbalanced MC variants.
            return "base_mc2" if "unb" in lname else "base_mc1"
        if "evd" in lname or "new" in lname:
            # Distinguish balanced vs unbalanced EVD/new variants.
            return "base_evd2" if "unb" in lname else "base_evd1"
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


def _export_isotonic_uncertainty_boxplots(
    *,
    methods: Sequence[str],
    val_store: Dict[str, List[np.ndarray]],
    test_store: Dict[str, List[np.ndarray]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _export_one_split(split: str, store: Dict[str, List[np.ndarray]], out_png: Path) -> None:
        labels: List[str] = []
        values: List[np.ndarray] = []
        for method in methods:
            arrays = [np.asarray(a, dtype=float).reshape(-1) for a in store.get(method, []) if np.asarray(a).size > 0]
            if not arrays:
                continue
            merged = np.concatenate(arrays, axis=0)
            if merged.size == 0:
                continue
            labels.append(_display_model_name(f"base::{method}"))
            values.append(merged)
        if not values:
            return

        fig, ax = plt.subplots(1, 1, figsize=(max(12, 1.4 * len(labels)), 6))
        bp = ax.boxplot(
            values,
            labels=labels,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C78A8")
            patch.set_alpha(0.65)
        ax.set_ylabel("Isotonic-calibrated uncertainty")
        ax.set_xlabel("Base model")
        ax.set_title(f"{split} isotonic-calibrated uncertainty by base model")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)

    _export_one_split(
        split="Validation",
        store=val_store,
        out_png=out_dir / "isotonic_uncertainty_boxplot_val.png",
    )
    _export_one_split(
        split="Test",
        store=test_store,
        out_png=out_dir / "isotonic_uncertainty_boxplot_test.png",
    )

    # Paired val/test boxplots per base model for direct distribution-gap inspection.
    paired_labels: List[str] = []
    paired_values: List[np.ndarray] = []
    paired_positions: List[float] = []
    paired_sides: List[str] = []
    tick_positions: List[float] = []
    pos = 1.0
    pair_gap = 1.15
    for method in methods:
        val_arrays = [np.asarray(a, dtype=float).reshape(-1) for a in val_store.get(method, []) if np.asarray(a).size > 0]
        test_arrays = [np.asarray(a, dtype=float).reshape(-1) for a in test_store.get(method, []) if np.asarray(a).size > 0]
        if not val_arrays and not test_arrays:
            continue
        val_values = np.concatenate(val_arrays, axis=0) if val_arrays else np.asarray([], dtype=float)
        test_values = np.concatenate(test_arrays, axis=0) if test_arrays else np.asarray([], dtype=float)
        if val_values.size == 0 and test_values.size == 0:
            continue
        label = _display_model_name(f"base::{method}")
        left_pos = pos
        right_pos = pos + 0.34
        tick_positions.append(0.5 * (left_pos + right_pos))
        paired_labels.append(label)
        pos += pair_gap
        if val_values.size > 0:
            paired_values.append(val_values)
            paired_positions.append(left_pos)
            paired_sides.append("val")
        if test_values.size > 0:
            paired_values.append(test_values)
            paired_positions.append(right_pos)
            paired_sides.append("test")

    if paired_values:
        fig, ax = plt.subplots(1, 1, figsize=(max(13, 1.6 * len(paired_labels)), 6.2))
        bp = ax.boxplot(
            paired_values,
            positions=paired_positions,
            widths=0.26,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
        )
        for patch, side in zip(bp["boxes"], paired_sides):
            patch.set_facecolor("#4C78A8" if side == "val" else "#F58518")
            patch.set_alpha(0.68)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(paired_labels, rotation=30, ha="right")
        ax.set_ylabel("Isotonic-calibrated uncertainty")
        ax.set_xlabel("Base model")
        ax.set_title("Validation vs test isotonic-calibrated uncertainty by base model")
        ax.grid(axis="y", alpha=0.25)
        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(facecolor="#4C78A8", alpha=0.68, label="val"),
                Patch(facecolor="#F58518", alpha=0.68, label="test"),
            ],
            frameon=False,
            loc="upper right",
        )
        fig.tight_layout()
        fig.savefig(out_dir / "isotonic_uncertainty_boxplot_val_test_paired.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def _export_inverse_isotonic_test_diagnostics(
    *,
    methods: Sequence[str],
    z_score_test_store: Dict[str, List[np.ndarray]],
    inverse_score_test_store: Dict[str, List[np.ndarray]],
    quantile_test_store: Dict[str, List[np.ndarray]],
    resultant_weight_test_store: Dict[str, List[np.ndarray]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _export_boxplot(
        *,
        store: Dict[str, List[np.ndarray]],
        ylabel: str,
        title: str,
        out_png: Path,
    ) -> None:
        labels: List[str] = []
        values: List[np.ndarray] = []
        for method in methods:
            arrays = [np.asarray(a, dtype=float).reshape(-1) for a in store.get(method, []) if np.asarray(a).size > 0]
            if not arrays:
                continue
            merged = np.concatenate(arrays, axis=0)
            if merged.size == 0:
                continue
            labels.append(_display_model_name(f"base::{method}"))
            values.append(merged)
        if not values:
            return
        fig, ax = plt.subplots(1, 1, figsize=(max(12, 1.4 * len(labels)), 6))
        bp = ax.boxplot(
            values,
            labels=labels,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#72B7B2")
            patch.set_alpha(0.70)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Base model")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)

    _export_boxplot(
        store=z_score_test_store,
        ylabel="Z-score vs calibration isotonic uncertainty distribution",
        title="Test isotonic uncertainty z-score by base model",
        out_png=out_dir / "inverse_isotonic_test_zscore_boxplot.png",
    )
    _export_boxplot(
        store=inverse_score_test_store,
        ylabel="Inverse-isotonic score (after tau and cap)",
        title="Test inverse-isotonic score by base model",
        out_png=out_dir / "inverse_isotonic_test_inverse_score_boxplot.png",
    )
    _export_boxplot(
        store=quantile_test_store,
        ylabel="Empirical CDF quantile vs val isotonic uncertainty",
        title="Test isotonic uncertainty quantile by base model",
        out_png=out_dir / "inverse_isotonic_test_quantile_boxplot.png",
    )
    _export_boxplot(
        store=resultant_weight_test_store,
        ylabel="Resultant isotonic-inverse test weight",
        title="Test isotonic-inverse resultant weight by base model",
        out_png=out_dir / "inverse_isotonic_test_resultant_weight_boxplot.png",
    )


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
            fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.2), sharey=False)
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
                    ax.plot(x_vals, y_vals, marker="o", linewidth=1.8, label=label)
                    plotted_any = True
                ax.set_title(f"{split} {metric_title} vs low-epistemic-unc coverage", fontsize=15)
                ax.set_xlabel("Lowest epistemic uncertainty coverage (%)", fontsize=14)
                ax.grid(alpha=0.25)
                ax.set_xticks(cutoff_percents)
                ax.tick_params(axis="both", labelsize=12)
            axes[0].set_ylabel(f"{metric_title} ({direction} is better)", fontsize=14)
            if plotted_any:
                handles, labels = axes[1].get_legend_handles_labels()
                if handles:
                    fig.legend(
                        handles,
                        labels,
                        fontsize=12,
                        frameon=False,
                        loc="lower center",
                        bbox_to_anchor=(0.5, 0.02),
                        ncol=min(4, max(2, len(labels))),
                    )
            fig.tight_layout(rect=[0, 0.12, 1, 1])

            stem = f"{scenario}_{uncertainty_type}_{metric_key}_by_uncertainty_cutoff"
            fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
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
    cm_font_scale = 1.75
    fig, axes = plt.subplots(
        2, n_models, figsize=(max(21, 4.8 * n_models), 11.5), squeeze=False
    )
    vmax = 100.0

    csv_rows: List[Dict[str, Any]] = []
    for col, entry in enumerate(entries):
        model_name = str(entry["model_name"])
        display_name = _display_model_name(model_name)
        title_name = display_name
        if title_name.startswith("base_"):
            title_name = title_name[len("base_"):]
        elif title_name.startswith("ensemble_"):
            title_name = title_name[len("ensemble_"):]
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
                        fontsize=18 * cm_font_scale,
                        color="black",
                    )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=16 * cm_font_scale)
            if col == 0:
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["True 0", "True 1"], fontsize=16 * cm_font_scale)
            else:
                ax.set_yticks([])
            if row == 0:
                ax.set_title(title_name, fontsize=18 * cm_font_scale)
            if col == 0:
                ax.set_ylabel(f"{split}\nlabel", fontsize=17 * cm_font_scale)
            ax.set_xlabel("prediction", fontsize=16 * cm_font_scale)

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

    fig.suptitle(
        "Confusion matrices by method (row-normalized by true label)",
        fontsize=22 * cm_font_scale,
    )
    # Avoid tight_layout warning with colorbar + multi-axes grids.
    fig.subplots_adjust(left=0.06, right=0.93, bottom=0.10, top=0.88, wspace=0.30, hspace=0.28)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)


def _export_ensemble_weight_bars(
    all_weights: Dict[str, List[np.ndarray]],
    base_methods: Sequence[str],
    out_png: Path,
    out_csv: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    stack_names = [k for k in sorted(all_weights.keys()) if len(all_weights.get(k, [])) > 0]
    if not stack_names or len(base_methods) == 0:
        return

    means = np.zeros((len(stack_names), len(base_methods)), dtype=float)
    stds = np.zeros((len(stack_names), len(base_methods)), dtype=float)
    csv_rows: List[Dict[str, Any]] = []
    for i, stack_name in enumerate(stack_names):
        arr = np.asarray(all_weights[stack_name], dtype=float)  # (runs, K)
        if arr.ndim != 2 or arr.shape[1] != len(base_methods):
            continue
        means[i] = arr.mean(axis=0)
        stds[i] = arr.std(axis=0, ddof=0)
        for j, method in enumerate(base_methods):
            csv_rows.append(
                {
                    "ensemble_method": stack_name,
                    "base_method": method,
                    "weight_mean": float(means[i, j]),
                    "weight_std": float(stds[i, j]),
                    "runs": int(arr.shape[0]),
                }
            )

    K = len(base_methods)
    n_cols = len(stack_names)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(max(12, 3.2 * n_cols), 4.6),
        squeeze=False,
        sharey=True,
    )
    cmap = plt.cm.get_cmap("tab10", K)
    x_local = np.arange(K, dtype=float)
    y_max = max(1.0, float(np.max(means + stds) * 1.15))

    for i, stack_name in enumerate(stack_names):
        ax = axes[0, i]
        ax.bar(
            x_local,
            means[i],
            yerr=stds[i],
            capsize=3,
            color=[cmap(j) for j in range(K)],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_title(_display_model_name(f"stack::{stack_name}"), fontsize=11)
        ax.set_xticks(x_local)
        ax.set_xticklabels([_display_model_name(f"base::{m}") for m in base_methods], rotation=45, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0.0, y_max)
        # Make each panel visually square-ish like confusion-matrix blocks.
        try:
            ax.set_box_aspect(1.0)
        except Exception:
            pass
        if i != 0:
            ax.set_yticklabels([])

    handles = [
        plt.Line2D([0], [0], color=cmap(j), lw=8, label=_display_model_name(f"base::{m}"))
        for j, m in enumerate(base_methods)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=min(4, max(1, K)),
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)


def _export_radar_grid(
    radar_metrics_store: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]],
    scenario: str,
    out_png: Path,
    out_csv: Path,
    focus_base_model: str | None = None,
    include_baselines: bool = False,
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
    radar_floor = 0.20

    fig, axes = plt.subplots(1, 2, figsize=(20.0, 9.6), subplot_kw={"polar": True})
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
            ax.set_xticklabels(metric_names, fontsize=15)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(
                ["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=14
            )
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.25)
            continue

        # Focus per selected base model + all ensemble methods.
        candidate_model_names = sorted(mean_by_model.keys())
        model_names = [m for m in candidate_model_names if m.startswith("stack::")]
        if focus_base_model is not None and focus_base_model in candidate_model_names:
            model_names.append(focus_base_model)
        elif "base::ensemble" in candidate_model_names:
            model_names.append("base::ensemble")
        if include_baselines:
            model_names.extend([m for m in candidate_model_names if m.startswith("baseline::")])
        model_names = list(dict.fromkeys(model_names))
        if not model_names:
            model_names = candidate_model_names
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
            if model_name == "baseline::always_pos":
                line, = ax.plot(
                    angles_closed,
                    values,
                    linewidth=2.2,
                    linestyle=":",
                    marker="s",
                    markersize=4.5,
                    color="#E45756",
                    label=display_name,
                )
            elif model_name == "baseline::always_neg":
                line, = ax.plot(
                    angles_closed,
                    values,
                    linewidth=2.2,
                    linestyle="-.",
                    marker="^",
                    markersize=4.5,
                    color="#72B7B2",
                    label=display_name,
                )
            elif model_name.startswith("base::"):
                line, = ax.plot(
                    angles_closed,
                    values,
                    linewidth=2.0,
                    linestyle="--",
                    color="#666666",
                    label=display_name,
                )
            else:
                line, = ax.plot(angles_closed, values, linewidth=1.8, linestyle="-", label=display_name)
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
        ax.set_xticklabels(metric_names, fontsize=16)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(
            ["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=15
        )
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.set_title(split.upper(), fontsize=18, pad=18)

    if legend_items:
        fig.legend(
            handles=list(legend_items.values()),
            labels=list(legend_items.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=min(4, max(1, len(legend_items))),
            frameon=False,
            fontsize=16,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(out_csv, index=False)


def _export_radar_base_vs_baselines(
    radar_metrics_store: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]],
    out_png: Path,
    out_csv: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    scenario = "all"
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
    radar_floor = 0.20

    fig, axes = plt.subplots(1, 2, figsize=(20.0, 9.6), subplot_kw={"polar": True})
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
                mm[metric_name] = float(np.mean(vals[finite])) if np.any(finite) else float("nan")
            mean_by_model[model_name] = mm

        selected_models = sorted(
            [m for m in mean_by_model.keys() if m.startswith("base::")]
        ) + ["baseline::always_pos", "baseline::always_neg"]
        selected_models = [m for m in selected_models if m in mean_by_model]
        if not selected_models:
            continue

        raw = np.full((len(selected_models), len(metric_names)), np.nan, dtype=float)
        for i, model_name in enumerate(selected_models):
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

        for i, model_name in enumerate(selected_models):
            values = np.concatenate([norm_shifted[i], norm_shifted[i, :1]])
            display_name = _display_model_name(model_name)
            if model_name.startswith("baseline::"):
                line, = ax.plot(
                    angles_closed,
                    values,
                    linewidth=2.4,
                    linestyle="--",
                    label=display_name,
                )
            else:
                line, = ax.plot(
                    angles_closed,
                    values,
                    linewidth=2.2,
                    linestyle="-",
                    label=display_name,
                )
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
        ax.set_xticklabels(metric_names, fontsize=16)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=15)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.set_title(split.upper(), fontsize=18, pad=18)

    if legend_items:
        fig.legend(
            handles=list(legend_items.values()),
            labels=list(legend_items.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=min(5, max(1, len(legend_items))),
            frameon=False,
            fontsize=16,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(out_csv, index=False)


def _get_weighted_cp_sample_weights(
    *,
    split_name: str,
    n_samples: int,
) -> np.ndarray:
    """
    Placeholder hook for weighted conformal prediction importance weights w(x).

    Expected semantics under covariate shift:
      w(x) ~ p_test(x) / p_calibration(x)

    Replace this body with your project-specific weighting function once available.
    Until then we return all-ones, which recovers standard (unweighted) conformal.
    """
    _ = split_name
    return np.ones(int(n_samples), dtype=np.float64)


def _build_domain_ecfp_features_from_smiles(
    smiles: np.ndarray,
    ecfp_size: int,
    ecfp_radius: int,
) -> np.ndarray:
    featurizer = dc.feat.CircularFingerprint(size=int(ecfp_size), radius=int(ecfp_radius))
    raw = featurizer.featurize([str(s) for s in smiles.tolist()])
    feats = np.zeros((len(raw), int(ecfp_size)), dtype=np.float32)
    for i, f in enumerate(raw):
        if f is None:
            continue
        arr = np.asarray(f, dtype=np.float32).reshape(-1)
        if arr.size == int(ecfp_size):
            feats[i] = arr
    return feats


def _build_domain_graph_features_from_smiles(smiles: np.ndarray) -> np.ndarray:
    featurizer = dc.feat.DMPNNFeaturizer()
    raw = featurizer.featurize([str(s) for s in smiles.tolist()])
    clean = [f for f in raw if f is not None]
    if not clean:
        raise ValueError("No valid graph features produced by DMPNN featurizer.")
    return np.array(clean, dtype=object)


def _build_domain_split_features(
    *,
    label_dir: Path,
    split_name: str,
    encoder_type: str,
    smiles_column: str,
    ecfp_size: int,
    ecfp_radius: int,
) -> np.ndarray:
    pkl_path = label_dir / f"{split_name}.pkl"
    df = _safe_read_pickle(pkl_path)
    if smiles_column not in df.columns:
        raise ValueError(f"Missing '{smiles_column}' in {pkl_path}")
    smiles = df[smiles_column].astype(str).to_numpy()
    if encoder_type == "dmpnn":
        return _build_domain_graph_features_from_smiles(smiles)
    return _build_domain_ecfp_features_from_smiles(
        smiles,
        ecfp_size=ecfp_size,
        ecfp_radius=ecfp_radius,
    )


def _load_domain_classifier_checkpoint(ckpt_path: Path) -> Tuple[torch.nn.Module, Dict[str, Any], torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(ckpt_path, map_location=device)
    metadata = dict(payload.get("metadata", {}))
    n_features = int(metadata.get("n_features", 0))
    n_tasks = int(metadata.get("n_tasks", 1))
    encoder_type = str(metadata.get("encoder_type", "identity"))
    if n_features <= 0:
        raise ValueError(f"Invalid n_features in domain model checkpoint: {ckpt_path}")
    model = build_model(
        model_type="baseline",
        n_features=n_features,
        n_tasks=n_tasks,
        mode="classification",
        encoder_type=encoder_type,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata, device


@torch.no_grad()
def _predict_domain_positive_prob(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
    encoder_type: str,
    batch_size: int = 1024,
) -> np.ndarray:
    out = np.zeros((len(features),), dtype=np.float64)
    for i in range(0, len(features), int(batch_size)):
        xb_raw = features[i:i + int(batch_size)]
        if encoder_type == "dmpnn":
            xb = graphdata_to_batchmolgraph(list(xb_raw)).to(device)
        else:
            xb = torch.from_numpy(np.asarray(xb_raw, dtype=np.float32)).to(device)
        _, logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        out[i:i + int(batch_size)] = p.astype(np.float64)
    return out


def _compute_weighted_cp_weights_from_domain_model(
    *,
    label_dir: Path,
    val_split: str,
    test_split: str,
    domain_model_path: Path,
    default_label_column: str,
    weight_clip_max: float,
    prob_clip: float,
    ratio_offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    model, metadata, device = _load_domain_classifier_checkpoint(domain_model_path)
    encoder_type = str(metadata.get("encoder_type", "identity"))
    label_column = str(metadata.get("label_column", default_label_column))
    smiles_column = str(metadata.get("smiles_column", "SMILES"))
    ecfp_size = int(metadata.get("ecfp_size", 1024))
    ecfp_radius = int(metadata.get("ecfp_radius", 2))

    x_val = _build_domain_split_features(
        label_dir=label_dir,
        split_name=val_split,
        encoder_type=encoder_type,
        smiles_column=smiles_column,
        ecfp_size=ecfp_size,
        ecfp_radius=ecfp_radius,
    )
    x_test = _build_domain_split_features(
        label_dir=label_dir,
        split_name=test_split,
        encoder_type=encoder_type,
        smiles_column=smiles_column,
        ecfp_size=ecfp_size,
        ecfp_radius=ecfp_radius,
    )
    # Enforce robust clipping for weighted CP ratio stability.
    p_eps = float(max(prob_clip, 1e-2))
    p_val = np.clip(
        _predict_domain_positive_prob(model, x_val, device=device, encoder_type=encoder_type),
        p_eps,
        1.0 - p_eps,
    )
    p_test = np.clip(
        _predict_domain_positive_prob(model, x_test, device=device, encoder_type=encoder_type),
        p_eps,
        1.0 - p_eps,
    )
    off = float(max(ratio_offset, 0.0))
    # Stable weighted CP instantiation:
    # w(x) = (p(x) + off) / (1 - p(x) + off), where p(x)=P(test domain|x).
    # off>0 softens extreme ratios when p(x) is near 1.
    w_val = (p_val + off) / np.clip(1.0 - p_val + off, 1e-12, None)
    w_test = (p_test + off) / np.clip(1.0 - p_test + off, 1e-12, None)
    w_max = float(max(weight_clip_max, 1.0))
    w_val = np.clip(w_val, 1e-8, w_max).astype(np.float64)
    w_test = np.clip(w_test, 1e-8, w_max).astype(np.float64)
    return w_val, w_test


def _compute_weighted_cp_weights_online_rf(
    *,
    label_dir: Path,
    val_split: str,
    test_split: str,
    smiles_column: str,
    ecfp_size: int,
    ecfp_radius: int,
    weight_clip_max: float,
    prob_clip: float,
    ratio_offset: float,
    n_estimators: int,
    max_depth: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Online lightweight domain classifier: val=0, test=1.
    x_val = _build_domain_split_features(
        label_dir=label_dir,
        split_name=val_split,
        encoder_type="identity",
        smiles_column=smiles_column,
        ecfp_size=ecfp_size,
        ecfp_radius=ecfp_radius,
    )
    x_test = _build_domain_split_features(
        label_dir=label_dir,
        split_name=test_split,
        encoder_type="identity",
        smiles_column=smiles_column,
        ecfp_size=ecfp_size,
        ecfp_radius=ecfp_radius,
    )
    x_val = np.asarray(x_val, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_val_domain = np.zeros(len(x_val), dtype=int)
    y_test_domain = np.ones(len(x_test), dtype=int)
    x_domain = np.concatenate([x_val, x_test], axis=0)
    y_domain = np.concatenate([y_val_domain, y_test_domain], axis=0)

    rf = RandomForestClassifier(
        n_estimators=int(max(n_estimators, 1)),
        max_depth=None if int(max_depth) <= 0 else int(max_depth),
        random_state=int(seed),
        n_jobs=-1,
    )
    rf.fit(x_domain, y_domain)
    p_val = rf.predict_proba(x_val)[:, 1].astype(np.float64)
    p_test = rf.predict_proba(x_test)[:, 1].astype(np.float64)

    p_eps = float(max(prob_clip, 1e-2))
    p_val = np.clip(p_val, p_eps, 1.0 - p_eps)
    p_test = np.clip(p_test, p_eps, 1.0 - p_eps)
    off = float(max(ratio_offset, 0.0))
    w_val = (p_val + off) / np.clip(1.0 - p_val + off, 1e-12, None)
    w_test = (p_test + off) / np.clip(1.0 - p_test + off, 1e-12, None)
    w_max = float(max(weight_clip_max, 1.0))
    w_val = np.clip(w_val, 1e-8, w_max).astype(np.float64)
    w_test = np.clip(w_test, 1e-8, w_max).astype(np.float64)
    return w_val, w_test


def _build_label_conditional_conformal_scores(
    y_val: np.ndarray,
    prob_pos_val: np.ndarray,
    sample_weights_val: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_val, dtype=int).reshape(-1)
    p = np.clip(np.asarray(prob_pos_val, dtype=float).reshape(-1), 1e-8, 1.0 - 1e-8)
    if sample_weights_val is None:
        w = np.ones_like(p, dtype=np.float64)
    else:
        w = np.asarray(sample_weights_val, dtype=np.float64).reshape(-1)
        if len(w) != len(p):
            raise ValueError(
                f"sample_weights_val length mismatch: {len(w)} vs {len(p)}."
            )
        w = np.clip(w, 1e-12, None)
    # Nonconformity for positive label uses p(y=1|x); for negative label uses p(y=0|x)=1-p.
    scores_pos = 1.0 - p[y == 1]
    weights_pos = w[y == 1]
    scores_neg = 1.0 - (1.0 - p[y == 0])  # equals p for y==0 samples
    weights_neg = w[y == 0]

    order_pos = np.argsort(scores_pos, kind="mergesort")
    order_neg = np.argsort(scores_neg, kind="mergesort")
    return (
        scores_pos[order_pos],
        scores_neg[order_neg],
        weights_pos[order_pos],
        weights_neg[order_neg],
    )


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
    sorted_weights_pos: np.ndarray | None = None,
    sorted_weights_neg: np.ndarray | None = None,
    test_sample_weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    p = np.clip(np.asarray(prob_pos_test, dtype=float).reshape(-1), 1e-8, 1.0 - 1e-8)
    n = len(p)
    if len(sorted_scores_pos) == 0 or len(sorted_scores_neg) == 0:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=int)
    if test_sample_weights is None:
        w_test = np.ones(n, dtype=np.float64)
    else:
        w_test = np.asarray(test_sample_weights, dtype=np.float64).reshape(-1)
        if len(w_test) != n:
            raise ValueError(f"test_sample_weights length mismatch: {len(w_test)} vs {n}.")
        w_test = np.clip(w_test, 1e-12, None)

    w_pos = (
        np.ones(len(sorted_scores_pos), dtype=np.float64)
        if sorted_weights_pos is None
        else np.clip(np.asarray(sorted_weights_pos, dtype=np.float64).reshape(-1), 1e-12, None)
    )
    w_neg = (
        np.ones(len(sorted_scores_neg), dtype=np.float64)
        if sorted_weights_neg is None
        else np.clip(np.asarray(sorted_weights_neg, dtype=np.float64).reshape(-1), 1e-12, None)
    )
    if len(w_pos) != len(sorted_scores_pos):
        raise ValueError(
            f"sorted_weights_pos length mismatch: {len(w_pos)} vs {len(sorted_scores_pos)}."
        )
    if len(w_neg) != len(sorted_scores_neg):
        raise ValueError(
            f"sorted_weights_neg length mismatch: {len(w_neg)} vs {len(sorted_scores_neg)}."
        )

    s_pos = 1.0 - p
    s_neg = p

    idx_pos = np.searchsorted(sorted_scores_pos, s_pos, side="left")
    idx_neg = np.searchsorted(sorted_scores_neg, s_neg, side="left")
    suffix_pos = np.cumsum(w_pos[::-1], dtype=np.float64)[::-1]
    suffix_neg = np.cumsum(w_neg[::-1], dtype=np.float64)[::-1]
    # Avoid out-of-bounds when searchsorted returns len(sorted_scores_*).
    tail_w_pos = np.zeros_like(s_pos, dtype=np.float64)
    tail_w_neg = np.zeros_like(s_neg, dtype=np.float64)
    valid_pos = idx_pos < len(sorted_scores_pos)
    valid_neg = idx_neg < len(sorted_scores_neg)
    if np.any(valid_pos):
        tail_w_pos[valid_pos] = suffix_pos[idx_pos[valid_pos]]
    if np.any(valid_neg):
        tail_w_neg[valid_neg] = suffix_neg[idx_neg[valid_neg]]
    total_w_pos = float(np.sum(w_pos))
    total_w_neg = float(np.sum(w_neg))
    # Weighted generalization: with unit weights this is the usual smoothed split-CP p-value.
    pval_pos = (tail_w_pos + w_test) / (total_w_pos + w_test)
    pval_neg = (tail_w_neg + w_test) / (total_w_neg + w_test)

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
        "--iso_inverse_score_cap",
        type=float,
        default=1e6,
        help="Upper cap for inverse-isotonic pre-normalization score.",
    )
    parser.add_argument(
        "--iso_ood_lambda",
        type=float,
        default=0.75,
        help="Legacy argument (unused in current quantile-exponential inverse-isotonic weighting).",
    )
    parser.add_argument(
        "--iso_ood_z_tolerance",
        type=float,
        default=1.5,
        help="Legacy argument (unused in current quantile-exponential inverse-isotonic weighting).",
    )
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
    parser.add_argument(
        "--weighted_cp_domain_model",
        type=str,
        default="",
        help="Optional checkpoint path for val-vs-test domain classifier used in weighted CP.",
    )
    parser.add_argument(
        "--weighted_cp_source",
        type=str,
        default="none",
        choices=["none", "checkpoint", "online_rf"],
        help="Source of weighted CP domain weights: none, checkpoint, or online_rf.",
    )
    parser.add_argument(
        "--weighted_cp_weight_clip_max",
        type=float,
        default=1000.0,
        help="Upper clip for weighted CP ratio w(x)=p/(1-p).",
    )
    parser.add_argument(
        "--weighted_cp_prob_clip",
        type=float,
        default=1e-2,
        help="Clip domain p(x) into [eps, 1-eps] before computing weighted CP ratio (minimum enforced eps=0.01).",
    )
    parser.add_argument(
        "--weighted_cp_ratio_offset",
        type=float,
        default=1e-3,
        help="Offset in stable ratio w(x)=(p+offset)/(1-p+offset).",
    )
    parser.add_argument(
        "--weighted_cp_smiles_column",
        type=str,
        default="SMILES",
        help="SMILES column name for online_rf weighted CP source.",
    )
    parser.add_argument(
        "--weighted_cp_ecfp_size",
        type=int,
        default=1024,
        help="ECFP size for online_rf weighted CP source.",
    )
    parser.add_argument(
        "--weighted_cp_ecfp_radius",
        type=int,
        default=2,
        help="ECFP radius for online_rf weighted CP source.",
    )
    parser.add_argument(
        "--weighted_cp_rf_n_estimators",
        type=int,
        default=25,
        help="Number of trees for online_rf weighted CP source.",
    )
    parser.add_argument(
        "--weighted_cp_rf_max_depth",
        type=int,
        default=6,
        help="Tree max depth for online_rf weighted CP source. <=0 means unlimited.",
    )
    parser.add_argument(
        "--weighted_cp_rf_seed",
        type=int,
        default=0,
        help="Random seed for online_rf weighted CP source.",
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
    cp_w_val = _get_weighted_cp_sample_weights(
        split_name=args.val_split, n_samples=len(y_val)
    )
    cp_w_test = _get_weighted_cp_sample_weights(
        split_name=args.test_split, n_samples=len(y_test)
    )
    if args.weighted_cp_source == "online_rf":
        cp_w_val, cp_w_test = _compute_weighted_cp_weights_online_rf(
            label_dir=label_dir,
            val_split=args.val_split,
            test_split=args.test_split,
            smiles_column=args.weighted_cp_smiles_column,
            ecfp_size=args.weighted_cp_ecfp_size,
            ecfp_radius=args.weighted_cp_ecfp_radius,
            weight_clip_max=args.weighted_cp_weight_clip_max,
            prob_clip=args.weighted_cp_prob_clip,
            ratio_offset=args.weighted_cp_ratio_offset,
            n_estimators=args.weighted_cp_rf_n_estimators,
            max_depth=args.weighted_cp_rf_max_depth,
            seed=args.weighted_cp_rf_seed,
        )
    elif args.weighted_cp_source == "checkpoint" or (
        args.weighted_cp_source == "none" and args.weighted_cp_domain_model.strip()
    ):
        if not args.weighted_cp_domain_model.strip():
            raise ValueError(
                "--weighted_cp_source checkpoint requires --weighted_cp_domain_model path."
            )
        domain_model_path = Path(args.weighted_cp_domain_model)
        if not domain_model_path.is_absolute():
            domain_model_path = repo_root / domain_model_path
        cp_w_val, cp_w_test = _compute_weighted_cp_weights_from_domain_model(
            label_dir=label_dir,
            val_split=args.val_split,
            test_split=args.test_split,
            domain_model_path=domain_model_path,
            default_label_column=args.label_column,
            weight_clip_max=args.weighted_cp_weight_clip_max,
            prob_clip=args.weighted_cp_prob_clip,
            ratio_offset=args.weighted_cp_ratio_offset,
        )
    print(
        f"[Weighted CP] val_w mean={cp_w_val.mean():.4f} "
        f"p05={np.quantile(cp_w_val, 0.05):.4f} p95={np.quantile(cp_w_val, 0.95):.4f}"
    )
    print(
        f"[Weighted CP] test_w mean={cp_w_test.mean():.4f} "
        f"p05={np.quantile(cp_w_test, 0.05):.4f} p95={np.quantile(cp_w_test, 0.95):.4f}"
    )

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
    isotonic_uncertainty_val_store: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
    isotonic_uncertainty_test_store: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
    inverse_isotonic_zscore_test_store: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
    inverse_isotonic_score_test_store: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
    inverse_isotonic_quantile_test_store: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
    inverse_isotonic_resultant_weight_test_store: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
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

            scores_pos, scores_neg, scores_pos_w, scores_neg_w = _build_label_conditional_conformal_scores(
                y_val=y_val, prob_pos_val=p_val[idx], sample_weights_val=cp_w_val
            )
            accepted_mask_val, pred_label_val = _conformal_single_label_accept_mask(
                prob_pos_test=p_val[idx],
                sorted_scores_pos=scores_pos,
                sorted_scores_neg=scores_neg,
                alpha=args.conformal_alpha,
                sorted_weights_pos=scores_pos_w,
                sorted_weights_neg=scores_neg_w,
                test_sample_weights=cp_w_val,
            )
            accepted_mask, pred_label_test = _conformal_single_label_accept_mask(
                prob_pos_test=p_test[idx],
                sorted_scores_pos=scores_pos,
                sorted_scores_neg=scores_neg,
                alpha=args.conformal_alpha,
                sorted_weights_pos=scores_pos_w,
                sorted_weights_neg=scores_neg_w,
                test_sample_weights=cp_w_test,
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
        (
            iso_w_mean,
            iso_coeff_val,
            iso_coeff_test,
            iso_calibrated_val,
            iso_calibrated_test,
            iso_sample_w_val,
            iso_sample_w_test,
            iso_z_test,
            iso_z_test_abs,
            iso_quantile_test,
            iso_inv_score_test,
            _iso_unused_penalty,
        ) = _fit_weights_uncertainty_inverse_isotonic(
            p_val=p_val,
            u_val=u_val_for_iso,
            u_test=u_test_for_iso,
            y_val=y_val,
            target_recall=args.target_recall,
            tau=args.uncertainty_tau,
            score_cap=args.iso_inverse_score_cap,
            ood_lambda=args.iso_ood_lambda,
            ood_z_tolerance=args.iso_ood_z_tolerance,
        )
        for idx, method in enumerate(methods):
            isotonic_uncertainty_val_store[method].append(
                np.asarray(iso_calibrated_val[idx], dtype=np.float64).reshape(-1)
            )
            isotonic_uncertainty_test_store[method].append(
                np.asarray(iso_calibrated_test[idx], dtype=np.float64).reshape(-1)
            )
            inverse_isotonic_zscore_test_store[method].append(
                np.asarray(iso_z_test[idx], dtype=np.float64).reshape(-1)
            )
            inverse_isotonic_score_test_store[method].append(
                np.asarray(iso_inv_score_test[idx], dtype=np.float64).reshape(-1)
            )
            inverse_isotonic_quantile_test_store[method].append(
                np.asarray(iso_quantile_test[idx], dtype=np.float64).reshape(-1)
            )
            inverse_isotonic_resultant_weight_test_store[method].append(
                np.asarray(iso_sample_w_test[idx], dtype=np.float64).reshape(-1)
            )
        weights = {
            "uniform": np.ones(len(methods), dtype=np.float64) / len(methods),
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
        stack_order = [
            "uncertainty_inverse_isotonic",
            "uniform",
            "softmax_brier_model_score",
            "brier_opt",
            "logloss_opt",
        ]
        for stack_name in stack_order:
            if stack_name == "uncertainty_inverse_isotonic":
                w_val = iso_sample_w_val
                w_test = iso_sample_w_test
                w_summary = iso_w_mean
                adaptive_note = "sample_adaptive_iso_inverse"
            else:
                w_static = weights[stack_name]
                w_val = w_static
                w_test = w_static
                w_summary = w_static
                adaptive_note = ""

            val_prob = _apply_weights(p_val, w_val)
            test_prob = _apply_weights(p_test, w_test)
            val_unc_epi = _apply_weights(u_val_epi, w_val)
            test_unc_epi = _apply_weights(u_test_epi, w_test)
            val_unc_ale = _apply_weights(u_val_ale, w_val)
            test_unc_ale = _apply_weights(u_test_ale, w_test)
            val_unc = _apply_weights(u_val, w_val)
            test_unc = _apply_weights(u_test, w_test)
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
            all_weights[stack_name].append(w_summary)
            stack_key = f"stack::{stack_name}"
            aggregate_metrics.setdefault(stack_key, []).append(test_metrics)
            radar_metrics_store["all"]["val"].setdefault(stack_key, []).append(val_metrics)
            radar_metrics_store["all"]["test"].setdefault(stack_key, []).append(test_metrics)
            confusion_store["all"]["val"].setdefault(stack_key, []).append(_cm_counts_from_metrics(val_metrics))
            confusion_store["all"]["test"].setdefault(stack_key, []).append(_cm_counts_from_metrics(test_metrics))
            w_str = ", ".join(f"{m}={ww:.4f}" for m, ww in zip(methods, w_summary))
            if stack_name == "uncertainty_inverse_isotonic":
                coef_val_str = ", ".join(f"{m}={cc:.4f}" for m, cc in zip(methods, iso_coeff_val))
                coef_test_str = ", ".join(f"{m}={cc:.4f}" for m, cc in zip(methods, iso_coeff_test))
                z_mean_str = ", ".join(
                    f"{m}={zz:.3f}" for m, zz in zip(methods, np.mean(iso_z_test_abs, axis=1))
                )
                z_p95_str = ", ".join(
                    f"{m}={zz:.3f}" for m, zz in zip(methods, np.quantile(iso_z_test_abs, 0.95, axis=1))
                )
                print(f"- {stack_name} | isotonic_coef_val: {coef_val_str}")
                print(f"- {stack_name} | isotonic_coef_test: {coef_test_str}")
                print(f"- {stack_name} | ood_abs_z_mean(test): {z_mean_str}")
                print(f"- {stack_name} | ood_abs_z_p95(test): {z_p95_str}")
                q_mean_str = ", ".join(
                    f"{m}={qq:.3f}" for m, qq in zip(methods, np.mean(iso_quantile_test, axis=1))
                )
                print(f"- {stack_name} | empirical_cdf_quantile_mean(test): {q_mean_str}")
                print(
                    f"- {stack_name} | weighting: {adaptive_note}; "
                    f"score_cap={args.iso_inverse_score_cap:.1f}; "
                    f"tau={args.uncertainty_tau:.3f}; "
                    f"method=exp(-tau*quantile)"
                )
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
                    (w_str if not adaptive_note else f"{w_str} | {adaptive_note}")
                    if stack_name != "uncertainty_inverse_isotonic"
                    else (
                        w_str
                        + " | isotonic_coef_val: "
                        + ", ".join(f"{m}={cc:.4f}" for m, cc in zip(methods, iso_coeff_val))
                        + " | isotonic_coef_test: "
                        + ", ".join(f"{m}={cc:.4f}" for m, cc in zip(methods, iso_coeff_test))
                        + " | "
                        + adaptive_note
                        + f" | score_cap={args.iso_inverse_score_cap:.1f}"
                        + f" | tau={args.uncertainty_tau:.3f}"
                        + " | method=exp(-tau*quantile)"
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
                    (w_str if not adaptive_note else f"{w_str} | {adaptive_note}")
                    if stack_name != "uncertainty_inverse_isotonic"
                    else (
                        w_str
                        + " | isotonic_coef_val: "
                        + ", ".join(f"{m}={cc:.4f}" for m, cc in zip(methods, iso_coeff_val))
                        + " | isotonic_coef_test: "
                        + ", ".join(f"{m}={cc:.4f}" for m, cc in zip(methods, iso_coeff_test))
                        + " | "
                        + adaptive_note
                        + f" | score_cap={args.iso_inverse_score_cap:.1f}"
                        + f" | tau={args.uncertainty_tau:.3f}"
                        + " | method=exp(-tau*quantile)"
                    )
                ),
            )

            scores_pos, scores_neg, scores_pos_w, scores_neg_w = _build_label_conditional_conformal_scores(
                y_val=y_val, prob_pos_val=val_prob, sample_weights_val=cp_w_val
            )
            accepted_mask_val, pred_label_val = _conformal_single_label_accept_mask(
                prob_pos_test=val_prob,
                sorted_scores_pos=scores_pos,
                sorted_scores_neg=scores_neg,
                alpha=args.conformal_alpha,
                sorted_weights_pos=scores_pos_w,
                sorted_weights_neg=scores_neg_w,
                test_sample_weights=cp_w_val,
            )
            accepted_mask, pred_label_test = _conformal_single_label_accept_mask(
                prob_pos_test=test_prob,
                sorted_scores_pos=scores_pos,
                sorted_scores_neg=scores_neg,
                alpha=args.conformal_alpha,
                sorted_weights_pos=scores_pos_w,
                sorted_weights_neg=scores_neg_w,
                test_sample_weights=cp_w_test,
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
    _export_ensemble_weight_bars(
        all_weights=all_weights,
        base_methods=methods,
        out_png=summary_csv.parent / "ensemble_weight_bars.png",
        out_csv=summary_csv.parent / "ensemble_weight_bars.csv",
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
    _export_isotonic_uncertainty_boxplots(
        methods=methods,
        val_store=isotonic_uncertainty_val_store,
        test_store=isotonic_uncertainty_test_store,
        out_dir=summary_csv.parent,
    )
    _export_inverse_isotonic_test_diagnostics(
        methods=methods,
        z_score_test_store=inverse_isotonic_zscore_test_store,
        inverse_score_test_store=inverse_isotonic_score_test_store,
        quantile_test_store=inverse_isotonic_quantile_test_store,
        resultant_weight_test_store=inverse_isotonic_resultant_weight_test_store,
        out_dir=summary_csv.parent,
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

    # Export 2 * len(methods) radar figures:
    # one base-focused radar for each CLI-provided base method, for both all and conformal scenarios.
    for base_method in methods:
        focus_key = f"base::{base_method}"
        focus_stem = _display_model_name(focus_key)
        _export_radar_grid(
            radar_metrics_store=radar_metrics_store,
            scenario="all",
            out_png=summary_csv.parent / f"radar_all_focus_{focus_stem}.png",
            out_csv=summary_csv.parent / f"radar_all_focus_{focus_stem}.csv",
            focus_base_model=focus_key,
            include_baselines=True,  # include all_pos/all_neg on all-data radar
        )
        _export_radar_grid(
            radar_metrics_store=radar_metrics_store,
            scenario="conformal",
            out_png=summary_csv.parent / f"radar_conformal_focus_{focus_stem}.png",
            out_csv=summary_csv.parent / f"radar_conformal_focus_{focus_stem}.csv",
            focus_base_model=focus_key,
            include_baselines=False,
        )
    _export_radar_base_vs_baselines(
        radar_metrics_store=radar_metrics_store,
        out_png=summary_csv.parent / "radar_all_base_vs_allpos_allneg.png",
        out_csv=summary_csv.parent / "radar_all_base_vs_allpos_allneg.csv",
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
