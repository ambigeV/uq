import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
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


def _auprc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(y_score, dtype=float).reshape(-1)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _classification_metrics(
    y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    y = y_true.reshape(-1).astype(float)
    p = np.clip(prob.reshape(-1).astype(float), 1e-8, 1.0 - 1e-8)
    pred = (p >= float(threshold)).astype(float)
    acc = float((pred == y).mean())
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    auc = _roc_auc_binary(y_true=y.astype(int), y_score=p)
    auprc = _auprc_binary(y_true=y.astype(int), y_score=p)
    return {"acc": acc, "brier": brier, "logloss": logloss, "auc": auc, "auprc": auprc}


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


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    p = np.asarray(probs, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=float).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
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
        acc = float(np.mean((p_bin >= 0.5).astype(float) == y_bin))
        ece += (count / max(n, 1)) * abs(acc - conf)
    return float(ece)


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
) -> Dict[str, float]:
    cls = _classification_metrics(y_true, prob, threshold=threshold)
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.clip(np.asarray(prob, dtype=float).reshape(-1), 1e-8, 1.0 - 1e-8)
    u = np.asarray(uncertainty, dtype=float).reshape(-1)
    sq_err = (y - p) ** 2
    out = {
        "AUC": float(cls["auc"]),
        "AUPRC": float(cls["auprc"]),
        "Accuracy": float(cls["acc"]),
        "NLL": float(cls["logloss"]),
        "Brier": float(cls["brier"]),
        "ECE": float(_compute_ece(p, y, n_bins=n_bins)),
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


def _load_one_prediction(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    df = pd.read_csv(path)
    for col in [PROB_COL, EPI_COL, ALE_COL]:
        if col not in df.columns:
            raise ValueError(f"{path} missing column: {col}")
    prob = df[PROB_COL].to_numpy(dtype=np.float64)
    unc = (df[EPI_COL].to_numpy(dtype=np.float64) + df[ALE_COL].to_numpy(dtype=np.float64))
    return prob, np.clip(unc, 1e-12, None)


def _load_split_predictions(
    pred_root: Path,
    split_name: str,
    methods: Sequence[str],
    run_id: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    probs, uncs, tags = [], [], []
    split_dir = pred_root / split_name
    for method in methods:
        path = split_dir / f"{method}_run_{run_id}.csv"
        p, u = _load_one_prediction(path)
        probs.append(p)
        uncs.append(u)
        tags.append(method)
    p_stack = np.stack(probs, axis=0)  # (K, N)
    u_stack = np.stack(uncs, axis=0)   # (K, N)
    return p_stack, u_stack, tags


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


def _fit_weights_uncertainty_inverse_mean(u_val: np.ndarray, tau: float = 1.0) -> np.ndarray:
    inv = (1.0 / np.clip(u_val, 1e-12, None)).mean(axis=1)
    score = np.maximum(inv, 1e-12) ** float(tau)
    return score / score.sum()


def _fit_weights_softmax_brier_per_model(p_val: np.ndarray, y_val: np.ndarray, temp: float = 12.0) -> np.ndarray:
    y = y_val.reshape(-1)
    brier = ((p_val - y[None, :]) ** 2).mean(axis=1)
    score = -float(temp) * brier
    score = score - np.max(score)
    w = np.exp(score)
    return w / w.sum()


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


def _format_metric_line(metrics: Dict[str, float]) -> str:
    ordered = [
        "AUC", "AUPRC", "Accuracy", "NLL", "Brier", "ECE",
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
    metrics: Dict[str, float],
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
        row[k] = float(v)
    rows.append(row)


def _build_label_conditional_conformal_scores(
    y_val: np.ndarray, prob_pos_val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_val, dtype=int).reshape(-1)
    p = np.clip(np.asarray(prob_pos_val, dtype=float).reshape(-1), 1e-8, 1.0 - 1e-8)
    # Nonconformity for positive label uses p(y=1|x); for negative label uses p(y=0|x)=1-p.
    scores_pos = 1.0 - p[y == 1]
    scores_neg = 1.0 - (1.0 - p[y == 0])  # equals p for y==0 samples
    return np.sort(scores_pos), np.sort(scores_neg)


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
) -> Dict[str, float]:
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
            "AUC", "AUPRC", "Accuracy", "NLL", "Brier", "ECE",
            "Avg_Entropy", "Spearman_Err_Unc",
            "Decision_Threshold", "confusion_tn", "confusion_fp", "confusion_fn", "confusion_tp",
        ]:
            out[k] = float("nan")
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
    parser.add_argument("--score_temp", type=float, default=12.0)
    parser.add_argument("--ece_bins", type=int, default=10)
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
        "uncertainty_inverse_mean": [],
        "softmax_brier_model_score": [],
        "brier_opt": [],
        "logloss_opt": [],
    }
    all_val_probs: Dict[str, List[np.ndarray]] = {k: [] for k in all_test_probs.keys()}
    all_test_uncs: Dict[str, List[np.ndarray]] = {k: [] for k in all_test_probs.keys()}
    all_weights: Dict[str, List[np.ndarray]] = {k: [] for k in all_test_probs.keys()}
    aggregate_metrics: Dict[str, List[Dict[str, float]]] = {}
    report_rows: List[Dict[str, object]] = []
    conformal_rows: List[Dict[str, object]] = []
    conformal_aggregate: Dict[str, List[Dict[str, float]]] = {}

    for run_id in run_ids:
        p_val, u_val, tags_val = _load_split_predictions(
            pred_root=pred_root, split_name=args.val_split, methods=methods, run_id=run_id
        )
        p_test, u_test, tags_test = _load_split_predictions(
            pred_root=pred_root, split_name=args.test_split, methods=methods, run_id=run_id
        )
        if tags_val != tags_test:
            raise RuntimeError(f"Method ordering mismatch for run {run_id}.")

        _ensure_same_length(y_val, p_val, f"val run {run_id}")
        _ensure_same_length(y_test, p_test, f"test run {run_id}")

        print(f"\n=== Run {run_id} ===")
        print("[Base methods]")
        for idx, method in enumerate(methods):
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
            aggregate_metrics.setdefault(f"base::{method}", []).append(test_m)
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
            accepted_mask, _ = _conformal_single_label_accept_mask(
                prob_pos_test=p_test[idx],
                sorted_scores_pos=scores_pos,
                sorted_scores_neg=scores_neg,
                alpha=args.conformal_alpha,
            )
            conf_test_m = _evaluate_accepted_subset(
                y_true=y_test,
                prob=p_test[idx],
                uncertainty=u_test[idx],
                accepted_mask=accepted_mask,
                n_bins=args.ece_bins,
                threshold=thr,
            )
            conf_test_m["Val_Precision_At_Selected_Thr"] = float(val_prec_at_thr)
            conf_test_m["Val_Recall_At_Selected_Thr"] = float(val_rec_at_thr)
            conformal_aggregate.setdefault(f"base::{method}", []).append(conf_test_m)
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

        weights = {
            "uniform": np.ones(len(methods), dtype=np.float64) / len(methods),
            "uncertainty_inverse_mean": _fit_weights_uncertainty_inverse_mean(
                u_val=u_val, tau=args.uncertainty_tau
            ),
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
            val_unc = _apply_weights(u_val, w)
            test_unc = _apply_weights(u_test, w)
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
            aggregate_metrics.setdefault(f"stack::{stack_name}", []).append(test_metrics)
            w_str = ", ".join(f"{m}={ww:.4f}" for m, ww in zip(methods, w))
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
                weights=w_str,
            )
            _append_metric_row(
                report_rows,
                row_type="per_run",
                run_id=str(run_id),
                split="test",
                model_family="stack",
                model_name=stack_name,
                metrics=test_metrics,
                weights=w_str,
            )

            scores_pos, scores_neg = _build_label_conditional_conformal_scores(
                y_val=y_val, prob_pos_val=val_prob
            )
            accepted_mask, _ = _conformal_single_label_accept_mask(
                prob_pos_test=test_prob,
                sorted_scores_pos=scores_pos,
                sorted_scores_neg=scores_neg,
                alpha=args.conformal_alpha,
            )
            conf_test_m = _evaluate_accepted_subset(
                y_true=y_test,
                prob=test_prob,
                uncertainty=test_unc,
                accepted_mask=accepted_mask,
                n_bins=args.ece_bins,
                threshold=thr,
            )
            conf_test_m["Val_Precision_At_Selected_Thr"] = float(val_prec_at_thr)
            conf_test_m["Val_Recall_At_Selected_Thr"] = float(val_rec_at_thr)
            conformal_aggregate.setdefault(f"stack::{stack_name}", []).append(conf_test_m)
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
            "AUC", "AUPRC", "Accuracy", "NLL", "Brier", "ECE",
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

    pd.DataFrame(report_rows).to_csv(summary_csv, index=False)
    print(f"\nSaved consolidated metrics CSV: {summary_csv}")

    if conformal_aggregate:
        for name, rows in conformal_aggregate.items():
            fam, model = name.split("::", 1)
            mean_std_metrics: Dict[str, float] = {}
            metric_keys = sorted(set().union(*[r.keys() for r in rows]))
            for k in metric_keys:
                vals = np.asarray([r.get(k, np.nan) for r in rows], dtype=float)
                mean_std_metrics[f"{k}_mean"] = float(np.nanmean(vals))
                mean_std_metrics[f"{k}_std"] = float(np.nanstd(vals, ddof=0))
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

    pd.DataFrame(conformal_rows).to_csv(conformal_summary_csv, index=False)
    print(f"Saved conformal accepted-subset metrics CSV: {conformal_summary_csv}")


if __name__ == "__main__":
    main()
