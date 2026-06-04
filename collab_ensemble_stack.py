#!/usr/bin/env python3
"""
collab_ensemble_stack.py
=========================
Self-contained ensemble stacking for binary classification uncertainty models.

No project-internal imports — only standard scientific Python.

Requirements:
    pip install torch numpy pandas scikit-learn

Input prediction CSVs are assumed to have the columns produced by
collab_finetune_eval.py (--mode eval) or bii.py (--mode infer):
    prob_class1_positive
    epistemic_uncertainty
    aleatoric_uncertainty

Two stacking methods are supported. Select one (or both) via --strategy
{iso_inverse, iso_inverse_reject, both}; default is "both".

  Method 1 — iso_inverse
    For each base model, an isotonic regressor is fitted on the validation set
    to map raw uncertainty → calibrated error probability.  The inverse of
    the calibrated uncertainty (in quantile space) then determines how much
    weight each model receives *per test sample*.  Lower-uncertainty models
    get higher weight on that sample.

    Output columns:
        iso_inv_prob     — weighted-ensemble probability
        iso_inv_accepted — always 1 (no rejection)

  Method 2 — iso_inverse_reject
    Same weighting as Method 1, plus a label-conditional split conformal
    prediction rule that marks each test sample as accepted or rejected.

    A sample is ACCEPTED if the conformal prediction set contains exactly one
    class label (confident), i.e.  p-value for that label > alpha AND
    p-value for the opposite label ≤ alpha.

    A sample is REJECTED if both labels are plausible (uncertain region) or
    neither is plausible (very unusual prediction).

    Output columns:
        iso_inv_reject_prob     — weighted-ensemble probability
        iso_inv_reject_accepted — 1 if accepted by CP rule, 0 if rejected
        iso_inv_reject_pred_label — CP-predicted label (valid only when accepted=1)

Directory layout expected under --pred_root:
    <pred_root>/
        <val_split>/
            <method>_run_0.csv
            <method>_run_1.csv
            ...
        <test_split>/
            <method>_run_0.csv
            ...

Label files (<label_dir>/<split_name>.pkl or .csv) must have the
--label_column.

Example:
    python collab_ensemble_stack.py \\
        --pred_root inference_outputs \\
        --label_dir cytotoxicity_data \\
        --val_split  HEK293_test_BM \\
        --test_split tox21_all \\
        --methods    dmpnn_bii_dmpnn_balanced,identity_bii_identity_balanced \\
        --run_ids    0,1,2,3,4 \\
        --alpha      0.10 \\
        --output_dir ensemble_results
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Data loading
# ─────────────────────────────────────────────────────────────────────────────

PROB_COL = "prob_class1_positive"
EPI_COL  = "epistemic_uncertainty"
ALE_COL  = "aleatoric_uncertainty"


def _safe_load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if str(path).endswith(".pkl"):
        try:
            obj = pd.read_pickle(path)
        except Exception:
            with open(path, "rb") as f:
                obj = pickle.load(f, encoding="latin1")
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"Expected DataFrame in {path}, got {type(obj)}")
        return obj
    return pd.read_csv(path)


def load_labels(label_dir: Path, split_name: str, label_column: str) -> np.ndarray:
    for ext in (".pkl", ".csv"):
        p = label_dir / f"{split_name}{ext}"
        if p.exists():
            df = _safe_load_dataframe(p)
            if label_column not in df.columns:
                raise ValueError(f"Column '{label_column}' not found in {p}")
            return df[label_column].to_numpy(dtype=np.float64).reshape(-1)
    raise FileNotFoundError(
        f"No label file found for split '{split_name}' in {label_dir} (.pkl or .csv)"
    )


def load_one_prediction(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (prob, epistemic, aleatoric), each shape (N,)."""
    df = _safe_load_dataframe(path)
    for col in [PROB_COL, EPI_COL, ALE_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    prob = df[PROB_COL].to_numpy(dtype=np.float64)
    epi  = np.clip(df[EPI_COL].to_numpy(dtype=np.float64), 1e-12, None)
    ale  = np.clip(df[ALE_COL].to_numpy(dtype=np.float64), 1e-12, None)
    return prob, epi, ale


def load_predictions(
    pred_root: Path,
    split_name: str,
    methods: Sequence[str],
    run_ids: Sequence[int],
) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Returns {method: {run_id: (prob, epi, ale)}} for all methods and runs.
    """
    result: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    split_dir = pred_root / split_name
    for method in methods:
        result[method] = {}
        for run_id in run_ids:
            path = split_dir / f"{method}_run_{run_id}.csv"
            result[method][run_id] = load_one_prediction(path)
    return result


def parse_explicit_file_groups(spec: str) -> List[List[Path]]:
    """
    Parse an explicit prediction-file spec that bypasses the directory/naming
    convention.

    Format: runs are separated by ';', methods within a run by ','.
        "valA_run0.csv,valB_run0.csv ; valA_run1.csv,valB_run1.csv"
      -> run 0: [valA_run0.csv, valB_run0.csv]
         run 1: [valA_run1.csv, valB_run1.csv]

    A single run (no ';') is also valid: "a.csv,b.csv".
    """
    groups: List[List[Path]] = []
    for grp in spec.split(";"):
        grp = grp.strip()
        if not grp:
            continue
        files = [Path(x.strip()) for x in grp.split(",") if x.strip()]
        if files:
            groups.append(files)
    return groups


def load_predictions_explicit(
    file_groups: Sequence[Sequence[Path]],
    methods: Sequence[str],
    run_ids: Sequence[int],
) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Build the {method: {run_id: (prob, epi, ale)}} structure from explicit file
    paths. Within each run group, files map positionally to `methods`.
    """
    result: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        m: {} for m in methods
    }
    for run_id, files in zip(run_ids, file_groups):
        if len(files) != len(methods):
            raise ValueError(
                f"Run {run_id}: got {len(files)} file(s) but {len(methods)} method(s) "
                f"({list(methods)}). Each run group must list one file per method."
            )
        for method, path in zip(methods, files):
            result[method][run_id] = load_one_prediction(path)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Isotonic inverse uncertainty weighting
# ─────────────────────────────────────────────────────────────────────────────

def _select_threshold_by_recall(
    y: np.ndarray, prob: np.ndarray, target_recall: float = 0.95
) -> float:
    """Best-precision threshold that achieves at least target_recall on val."""
    y = y.astype(int)
    thresholds = np.unique(prob)
    best_prec, best_thr = -1.0, 0.5
    for thr in thresholds:
        pred = prob >= thr
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        recall = tp / max(tp + fn, 1)
        prec   = tp / max(tp + fp, 1)
        if recall >= target_recall and prec > best_prec:
            best_prec, best_thr = prec, float(thr)
    return best_thr


def fit_isotonic_inverse_weights(
    p_val: np.ndarray,   # (K, N_val)
    u_val: np.ndarray,   # (K, N_val) uncertainty used for calibration
    u_test: np.ndarray,  # (K, N_test)
    y_val: np.ndarray,   # (N_val,)
    target_recall: float = 0.95,
    tau: float = 1.0,
    score_cap: float = 1e6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit isotonic regression (uncertainty → error probability) per model,
    then compute per-sample adaptive weights via exponential inverse-quantile.

    Returns:
        sample_weights_val  (K, N_val)  — per-sample weights for validation
        sample_weights_test (K, N_test) — per-sample weights for test
    """
    K, N_val  = p_val.shape
    N_test = u_test.shape[1]
    y = y_val.astype(int).reshape(-1)

    cal_val  = np.zeros((K, N_val),  dtype=np.float64)
    cal_test = np.zeros((K, N_test), dtype=np.float64)

    for k in range(K):
        pk = p_val[k]
        uk_val  = u_val[k]
        uk_test = u_test[k]
        thr = _select_threshold_by_recall(y, pk, target_recall)
        err = ((pk >= thr).astype(int) != y).astype(float)

        if np.allclose(uk_val, uk_val[0]):
            cal_val[k]  = np.full(N_val,  np.mean(err))
            cal_test[k] = np.full(N_test, np.mean(err))
        else:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip",
                                     y_min=0.0, y_max=1.0)
            cal_val[k]  = iso.fit_transform(uk_val, err)
            cal_test[k] = iso.predict(uk_test)

    # Empirical-CDF quantile of calibrated uncertainty (reference = val distribution).
    q_val  = np.zeros_like(cal_val,  dtype=np.float64)
    q_test = np.zeros_like(cal_test, dtype=np.float64)
    for k in range(K):
        ref = np.sort(cal_val[k])
        n   = len(ref)
        q_val[k]  = (np.searchsorted(ref, cal_val[k],  side="right") + 1.0) / (n + 1.0)
        q_test[k] = (np.searchsorted(ref, cal_test[k], side="right") + 1.0) / (n + 1.0)

    q_val  = np.clip(q_val,  1e-8, 1.0)
    q_test = np.clip(q_test, 1e-8, 1.0)

    cap = float(max(score_cap, 1e-6))
    inv_val  = np.clip(np.exp(-tau * q_val),  1e-12, cap)
    inv_test = np.clip(np.exp(-tau * q_test), 1e-12, cap)

    w_val  = inv_val  / np.clip(inv_val.sum(axis=0,  keepdims=True), 1e-12, None)
    w_test = inv_test / np.clip(inv_test.sum(axis=0, keepdims=True), 1e-12, None)
    return w_val, w_test


def apply_weights(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted sum of K model probabilities. p, w each (K, N)."""
    return (w * p).sum(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Label-conditional split conformal prediction
# ─────────────────────────────────────────────────────────────────────────────

def fit_conformal_thresholds(
    y_val: np.ndarray,
    prob_val: np.ndarray,
    alpha: float = 0.10,
    cp_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sorted nonconformity scores for each class from val set.

    Nonconformity score:
        y=1 → s = 1 - p   (how wrong is the positive prediction)
        y=0 → s = p       (how wrong is the negative prediction)

    Returns:
        sorted_scores_pos  — sorted scores for positive val samples
        sorted_scores_neg  — sorted scores for negative val samples
        (both with uniform weights baked in if cp_weights=None)
    """
    y    = y_val.reshape(-1).astype(int)
    p    = np.clip(prob_val.reshape(-1), 1e-8, 1 - 1e-8)
    w    = np.ones(len(y)) if cp_weights is None else np.clip(cp_weights.reshape(-1), 1e-12, None)

    pos_mask = (y == 1)
    neg_mask = (y == 0)

    s_pos = 1.0 - p[pos_mask]
    s_neg = p[neg_mask]
    w_pos = w[pos_mask]
    w_neg = w[neg_mask]

    order_pos = np.argsort(s_pos)
    order_neg = np.argsort(s_neg)
    return (
        s_pos[order_pos], w_pos[order_pos],
        s_neg[order_neg], w_neg[order_neg],
    )


def apply_conformal_rejection(
    prob_test: np.ndarray,
    sorted_scores_pos: np.ndarray,
    sorted_weights_pos: np.ndarray,
    sorted_scores_neg: np.ndarray,
    sorted_weights_neg: np.ndarray,
    alpha: float = 0.10,
    test_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Label-conditional weighted split-CP.

    For each test sample compute:
        pval_pos = P(score_pos >= s_pos_test)  under null y=1
        pval_neg = P(score_neg >= s_neg_test)  under null y=0

    Include label k in prediction set if pval_k > alpha.
    ACCEPTED = prediction set has exactly one label (XOR).
    REJECTED = 0 or 2 labels plausible.

    Returns:
        accepted    (N,) bool
        pred_label  (N,) int  — CP-predicted label when accepted, else -1
    """
    p  = np.clip(prob_test.reshape(-1).astype(float), 1e-8, 1 - 1e-8)
    N  = len(p)
    wt = np.ones(N) if test_weights is None else np.clip(test_weights.reshape(-1), 1e-12, None)

    s_pos_test = 1.0 - p  # nonconformity if label=1
    s_neg_test = p         # nonconformity if label=0

    def _pval(s_test_i, sorted_s, sorted_w, w_test_i):
        idx = np.searchsorted(sorted_s, s_test_i, side="left")
        total_w = float(sorted_w.sum())
        if idx >= len(sorted_s):
            tail = 0.0
        else:
            tail = float(sorted_w[idx:].sum())
        return (tail + w_test_i) / (total_w + w_test_i)

    in_pos = np.zeros(N, dtype=bool)
    in_neg = np.zeros(N, dtype=bool)
    for i in range(N):
        pv_pos = _pval(s_pos_test[i], sorted_scores_pos, sorted_weights_pos, wt[i])
        pv_neg = _pval(s_neg_test[i], sorted_scores_neg, sorted_weights_neg, wt[i])
        in_pos[i] = pv_pos > alpha
        in_neg[i] = pv_neg > alpha

    accepted   = np.logical_xor(in_pos, in_neg)
    pred_label = np.where(in_pos & ~in_neg, 1, np.where(in_neg & ~in_pos, 0, -1)).astype(int)
    return accepted, pred_label


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    y, p = y.astype(int).reshape(-1), p.reshape(-1)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float = 0.5,
    mask: Optional[np.ndarray] = None,
    label: str = "",
) -> Dict[str, float]:
    y = y_true.reshape(-1).astype(int)
    p = np.clip(prob.reshape(-1).astype(float), 1e-8, 1 - 1e-8)
    if mask is not None:
        m = mask.reshape(-1).astype(bool)
        y, p = y[m], p[m]
    if len(y) == 0:
        return {}
    pred  = (p >= threshold).astype(int)
    acc   = float((pred == y).mean())
    brier = float(np.mean((p - y) ** 2))
    nll   = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    auc   = _roc_auc(y, p)
    tp    = float(np.sum((pred == 1) & (y == 1)))
    fp    = float(np.sum((pred == 1) & (y == 0)))
    fn    = float(np.sum((pred == 0) & (y == 1)))
    tn    = float(np.sum((pred == 0) & (y == 0)))
    prec  = tp / max(tp + fp, 1)
    rec   = tp / max(tp + fn, 1)
    f1    = 2 * prec * rec / max(prec + rec, 1e-12)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc   = (tp * tn - fp * fn) / denom if denom > 0 else float("nan")
    n_total = len(y)
    n_acc   = int(mask.sum()) if mask is not None else n_total
    return dict(
        label=label, n_total=n_total, n_used=n_acc,
        AUC=auc, F1=f1, MCC=mcc, Accuracy=acc,
        NLL=nll, Brier=brier,
        TP=tp, FP=fp, FN=fn, TN=tn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Per-run processing
# ─────────────────────────────────────────────────────────────────────────────

def process_one_run(
    run_id: int,
    methods: List[str],
    pred_val: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    pred_test: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    y_val: np.ndarray,
    y_test: Optional[np.ndarray],
    alpha: float,
    tau: float,
    score_cap: float,
    target_recall: float,
    uncertainty_source: str,   # "total" | "epistemic"
    strategy: str = "both",    # "iso_inverse" | "iso_inverse_reject" | "both"
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Runs the requested stacking strategy(ies) on a single run and returns:
        metrics dict   (for the summary CSV)
        molecule df    (per-molecule predictions)

    strategy:
        "iso_inverse"        — weighted ensemble only (no rejection).
        "iso_inverse_reject" — weighted ensemble + label-conditional CP rejection.
        "both"               — emit both (default; preserves legacy behavior).
    """
    K = len(methods)
    run_method1 = strategy in ("iso_inverse", "both")
    run_method2 = strategy in ("iso_inverse_reject", "both")

    # Stack predictions: (K, N)
    p_val  = np.stack([pred_val[m][0]  for m in methods], axis=0)
    p_test = np.stack([pred_test[m][0] for m in methods], axis=0)

    if uncertainty_source == "epistemic":
        u_val  = np.stack([pred_val[m][1]  for m in methods], axis=0)
        u_test = np.stack([pred_test[m][1] for m in methods], axis=0)
    else:  # total
        u_val  = np.stack([pred_val[m][1]  + pred_val[m][2]  for m in methods], axis=0)
        u_test = np.stack([pred_test[m][1] + pred_test[m][2] for m in methods], axis=0)

    N_test = p_test.shape[1]

    # Inverse-uncertainty weights are shared by both strategies.
    w_val, w_test = fit_isotonic_inverse_weights(
        p_val, u_val, u_test, y_val,
        target_recall=target_recall, tau=tau, score_cap=score_cap,
    )
    ens_prob_val  = apply_weights(p_val,  w_val)
    ens_prob_test = apply_weights(p_test, w_test)

    # ── Build molecule-level DataFrame ──────────────────────────────────────
    mol_data: Dict[str, Any] = {"run_id": run_id}

    # ── Method 1: iso_inverse (weighted ensemble, no rejection) ─────────────
    if run_method1:
        mol_data["iso_inv_prob"]     = ens_prob_test
        mol_data["iso_inv_accepted"] = np.ones(N_test, dtype=int)

    # ── Method 2: iso_inverse_reject (same weights + CP rejection) ──────────
    accepted: Optional[np.ndarray] = None
    if run_method2:
        cp_data = fit_conformal_thresholds(y_val, ens_prob_val, alpha=alpha)
        ss_pos, sw_pos, ss_neg, sw_neg = cp_data
        accepted, pred_label = apply_conformal_rejection(
            ens_prob_test, ss_pos, sw_pos, ss_neg, sw_neg, alpha=alpha
        )
        mol_data["iso_inv_reject_prob"]       = ens_prob_test
        mol_data["iso_inv_reject_accepted"]   = accepted.astype(int)
        mol_data["iso_inv_reject_pred_label"] = pred_label

    mol_df = pd.DataFrame(mol_data)
    if y_test is not None:
        mol_df["y_true"] = y_test.reshape(-1)

    # ── Compute metrics ──────────────────────────────────────────────────────
    rows: List[Dict[str, Any]] = []
    if y_test is not None:
        metric_specs: List[Tuple[str, np.ndarray, Optional[np.ndarray]]] = []
        if run_method1:
            metric_specs.append(("iso_inv", ens_prob_test, None))
        if run_method2:
            metric_specs.append(("iso_inv_reject", ens_prob_test, accepted))
        for method_label, prob, mask in metric_specs:
            m = compute_metrics(y_test, prob, mask=mask, label=method_label)
            m["run_id"] = run_id
            rows.append(m)

    return rows, mol_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Isotonic-inverse uncertainty ensemble stacking with optional CP rejection."
    )
    p.add_argument("--pred_root",  type=str, default="",
                   help="Root directory containing <split>/<method>_run_<id>.csv files. "
                        "Required unless --val_pred_files/--test_pred_files are given.")
    p.add_argument("--label_dir",  type=str, required=True,
                   help="Directory with label files (<split>.pkl or <split>.csv).")
    p.add_argument(
        "--val_pred_files", type=str, default="",
        help="Explicit validation prediction CSV paths, bypassing pred_root naming. "
             "Runs separated by ';', methods within a run by ',' "
             "(e.g. 'a0.csv,b0.csv;a1.csv,b1.csv'). Requires --test_pred_files too.",
    )
    p.add_argument(
        "--test_pred_files", type=str, default="",
        help="Explicit test prediction CSV paths, same format as --val_pred_files. "
             "Run/method counts must match --val_pred_files.",
    )
    p.add_argument("--val_split",  type=str, default="HEK293_test_BM")
    p.add_argument("--test_split", type=str, default="tox21_all")
    p.add_argument("--label_column", type=str, default="Outcome")
    p.add_argument(
        "--methods", type=str,
        default="dmpnn_bii_dmpnn_balanced,dmpnn_bii_dmpnn,identity_bii_identity_balanced,identity_bii_identity",
        help="Comma-separated base method names.",
    )
    p.add_argument("--run_ids", type=str, default="0,1,2,3,4",
                   help="Comma-separated run IDs to aggregate.")
    p.add_argument("--alpha", type=float, default=0.10,
                   help="Conformal significance level (Method 2 rejection rate).")
    p.add_argument("--tau",   type=float, default=1.0,
                   help="Temperature for exponential inverse-isotonic weighting.")
    p.add_argument("--score_cap", type=float, default=1e6,
                   help="Cap on inverse-isotonic pre-normalisation score.")
    p.add_argument("--target_recall", type=float, default=0.95,
                   help="Target val recall when finding classification threshold.")
    p.add_argument(
        "--uncertainty_source", type=str, default="total",
        choices=["total", "epistemic"],
        help="Which uncertainty column to use for isotonic calibration.",
    )
    p.add_argument(
        "--strategy", type=str, default="both",
        choices=["iso_inverse", "iso_inverse_reject", "both"],
        help=(
            "Stacking strategy to compute/emit: "
            "'iso_inverse' (weighted ensemble, no rejection), "
            "'iso_inverse_reject' (weighted ensemble + label-conditional CP rejection), "
            "or 'both' (default)."
        ),
    )
    p.add_argument("--no_test_labels", action="store_true",
                   help="Skip loading test ground-truth labels (inference-only mode).")
    p.add_argument("--output_dir", type=str, default="ensemble_results",
                   help="Directory for output CSVs.")
    return p.parse_args()


def main():
    args = _parse_args()

    pred_root = Path(args.pred_root)
    label_dir = Path(args.label_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    explicit_mode = bool(args.val_pred_files.strip() or args.test_pred_files.strip())
    val_groups: List[List[Path]] = []
    test_groups: List[List[Path]] = []

    if explicit_mode:
        if not (args.val_pred_files.strip() and args.test_pred_files.strip()):
            raise ValueError(
                "Explicit mode requires BOTH --val_pred_files and --test_pred_files."
            )
        val_groups = parse_explicit_file_groups(args.val_pred_files)
        test_groups = parse_explicit_file_groups(args.test_pred_files)
        if not val_groups:
            raise ValueError("No files parsed from --val_pred_files.")
        if len(val_groups) != len(test_groups):
            raise ValueError(
                f"Run count mismatch: --val_pred_files has {len(val_groups)} run(s) "
                f"but --test_pred_files has {len(test_groups)}."
            )
        K = len(val_groups[0])
        if len(methods) != K:
            methods = [f"model_{i}" for i in range(K)]
            print(f"[Explicit mode] --methods count != {K}; using auto names {methods}")
        run_ids = list(range(len(val_groups)))
    else:
        if not args.pred_root.strip():
            raise ValueError(
                "--pred_root is required unless --val_pred_files/--test_pred_files are given."
            )
        run_ids = [int(x.strip()) for x in args.run_ids.split(",") if x.strip()]
        if not methods:
            raise ValueError("--methods cannot be empty.")
        if not run_ids:
            raise ValueError("--run_ids cannot be empty.")

    print(f"Methods  : {methods}")
    print(f"Run IDs  : {run_ids}")
    print(f"Strategy : {args.strategy}")
    print(f"Alpha    : {args.alpha}")
    if explicit_mode:
        print("Pred src : explicit file lists (--val_pred_files / --test_pred_files)")
    else:
        print(f"Val split: {args.val_split}  |  Test split: {args.test_split}")

    # Load ground-truth labels
    y_val = load_labels(label_dir, args.val_split,  args.label_column)
    y_test: Optional[np.ndarray] = None
    if not args.no_test_labels:
        y_test = load_labels(label_dir, args.test_split, args.label_column)
        print(f"Labels loaded — val: {len(y_val)}  test: {len(y_test)}")
    else:
        print(f"Labels loaded — val: {len(y_val)}  (test labels skipped)")

    # Load all prediction files
    print("\nLoading prediction files …")
    if explicit_mode:
        val_preds  = load_predictions_explicit(val_groups,  methods, run_ids)
        test_preds = load_predictions_explicit(test_groups, methods, run_ids)
    else:
        val_preds  = load_predictions(pred_root, args.val_split,  methods, run_ids)
        test_preds = load_predictions(pred_root, args.test_split, methods, run_ids)

    # Process each run
    all_metric_rows: List[Dict[str, Any]] = []
    all_mol_dfs:     List[pd.DataFrame]   = []

    for run_id in run_ids:
        print(f"\n── Run {run_id} ──────────────────────────────────")
        pred_val_run  = {m: val_preds[m][run_id]  for m in methods}
        pred_test_run = {m: test_preds[m][run_id] for m in methods}

        metric_rows, mol_df = process_one_run(
            run_id=run_id,
            methods=methods,
            pred_val=pred_val_run,
            pred_test=pred_test_run,
            y_val=y_val,
            y_test=y_test,
            alpha=args.alpha,
            tau=args.tau,
            score_cap=args.score_cap,
            target_recall=args.target_recall,
            uncertainty_source=args.uncertainty_source,
            strategy=args.strategy,
        )
        all_metric_rows.extend(metric_rows)
        all_mol_dfs.append(mol_df)

        # Print metrics for this run
        for row in metric_rows:
            parts = [f"  [{row.get('label', '')}]"]
            for k in ["AUC", "F1", "MCC", "Accuracy", "Brier", "n_used", "n_total"]:
                v = row.get(k)
                if v is None:
                    continue
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            print("  " + " | ".join(parts))

    # ── Aggregate metrics across runs ────────────────────────────────────────
    if all_metric_rows:
        metrics_df = pd.DataFrame(all_metric_rows)
        metrics_path = out_dir / "ensemble_metrics_per_run.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nPer-run metrics → {metrics_path}")

        numeric_cols = [c for c in metrics_df.columns
                        if metrics_df[c].dtype in (float, np.float64, np.float32)
                        and c not in ("run_id",)]
        if numeric_cols:
            agg = (
                metrics_df.groupby("label")[numeric_cols]
                .agg(["mean", "std"])
                .round(4)
            )
            agg_path = out_dir / "ensemble_metrics_aggregate.csv"
            agg.to_csv(agg_path)
            print(f"Aggregate metrics → {agg_path}")
            print("\nAggregate summary:")
            for label_val, grp in metrics_df.groupby("label"):
                print(f"\n  [{label_val}]")
                for col in ["AUC", "F1", "MCC", "Accuracy", "Brier"]:
                    if col in grp.columns:
                        vals = grp[col].dropna()
                        if len(vals):
                            print(f"    {col}: mean={vals.mean():.4f}  std={vals.std(ddof=0):.4f}")

    # ── Save per-molecule predictions ────────────────────────────────────────
    mol_df_all = pd.concat(all_mol_dfs, ignore_index=True)
    mol_path = out_dir / "ensemble_predictions_all_runs.csv"
    mol_df_all.to_csv(mol_path, index=False)
    print(f"\nPer-molecule predictions → {mol_path}")

    # Also save a single-run version if only one run requested (convenience)
    if len(run_ids) == 1:
        single_path = out_dir / f"ensemble_predictions_run_{run_ids[0]}.csv"
        mol_df_all.to_csv(single_path, index=False)
        print(f"Single-run predictions   → {single_path}")

    # ── Print rejection statistics ───────────────────────────────────────────
    if "iso_inv_reject_accepted" in mol_df_all.columns:
        for rid in run_ids:
            sub = mol_df_all[mol_df_all["run_id"] == rid]
            n_acc = int(sub["iso_inv_reject_accepted"].sum())
            n_tot = len(sub)
            print(f"  Run {rid}: CP accepted {n_acc}/{n_tot} "
                  f"({100 * n_acc / max(n_tot, 1):.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
