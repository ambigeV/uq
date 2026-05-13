from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import deepchem as dc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import torch

from nn import graphdata_to_batchmolgraph
from nn_baseline import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Display helper
# ─────────────────────────────────────────────────────────────────────────────

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
            "uncertainty_inverse_isotonic_reject_filtered": "ensemble_unc_iso_rf",
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


# ─────────────────────────────────────────────────────────────────────────────
# Weighted conformal prediction — domain model infrastructure
# ─────────────────────────────────────────────────────────────────────────────

def _safe_read_pickle(path: Path) -> pd.DataFrame:
    try:
        obj = pd.read_pickle(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read pickle: {path}") from exc
    if not isinstance(obj, pd.DataFrame):
        raise ValueError(f"Expected DataFrame in {path}, got {type(obj)}")
    return obj


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


# ─────────────────────────────────────────────────────────────────────────────
# Label-conditional weighted split conformal prediction
# ─────────────────────────────────────────────────────────────────────────────

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

                # Reliability x-axis uses raw P(y=1|x), not max(p, 1-p).
                conf = prob

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
                    mean_conf = float(np.mean(conf_bin))
                    emp_pos_rate = float(np.mean(y[mask]))
                    x_vals.append(mean_conf)
                    y_vals.append(emp_pos_rate)
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
                            "mean_probability": mean_conf,
                            "empirical_positive_rate": emp_pos_rate,
                        }
                    )
                if x_vals:
                    label = _display_model_name(f"{payload['model_family']}::{payload['model_name']}")
                    ax.plot(x_vals, y_vals, marker="o", linewidth=1.2, label=label)
                    plotted_any = True

            ax.set_title(f"{split} reliability")
            ax.set_xlabel("Mean predicted probability p(y=1)")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Empirical positive rate")
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


# ─────────────────────────────────────────────────────────────────────────────
# Conformal Score Aggregation (CSA)
# Reference: multivariate conformal via random projections on ensemble scores.
# ─────────────────────────────────────────────────────────────────────────────

def _csa_sample_directions(K: int, M: int, seed: int) -> np.ndarray:
    """Sample M unit vectors uniformly from the positive orthant of S^{K-1}.

    Strategy: draw z_k ~ |N(0,1)| for each coordinate, then L2-normalise.
    Returns array of shape (M, K).
    """
    rng = np.random.default_rng(seed)
    raw = np.abs(rng.standard_normal((M, K))).astype(np.float64)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / np.clip(norms, 1e-12, None)


def _csa_fit(
    p_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float,
    M: int = 128,
    split_ratio: float = 0.5,
    seed: int = 0,
) -> Dict[str, Any]:
    """Fit a Conformal Score Aggregation (CSA) object on validation data.

    For a binary classifier ensemble with K base models, the nonconformity
    score vector for sample (x, y) is:
        s_k(x, y=1) = 1 - p_k(x),   s_k(x, y=0) = p_k(x)

    Two-stage calibration:
      Stage 1 (I1): learn the shape of the multivariate acceptance region via
                    random projections + binary search for beta*.
      Stage 2 (I2): standard split-CP on the induced scalar T scores to set
                    the final conformal threshold t_hat.

    Parameters
    ----------
    p_val       : (K, N) array — K base-model P(y=1|x) on the validation set.
    y_val       : (N,) binary ground-truth labels.
    alpha       : significance level; credal sets have marginal coverage >= 1-alpha.
    M           : number of random projection directions.
    split_ratio : fraction of validation samples used for stage-1 envelope fitting.
    seed        : RNG seed (directions use seed+1 to decouple from split).

    Returns
    -------
    dict with keys: alpha, K, M, U, q_tilde, t_hat, q_bar, T2,
                    beta_star, split_ratio, n1, n2.
    """
    p_arr = np.asarray(p_val, dtype=np.float64)
    if p_arr.ndim == 1:
        p_arr = p_arr[None, :]
    K, N = p_arr.shape
    y = np.asarray(y_val, dtype=int).reshape(-1)
    p = np.clip(p_arr, 1e-8, 1.0 - 1e-8)

    # Score matrix: s_k(x_i, y_i).  Shape (N, K).
    # s_k = 1 - p_k  if y_i = 1,   s_k = p_k  if y_i = 0.
    S = np.where(y[None, :] == 1, 1.0 - p, p).T.astype(np.float64)

    # --- calibration split ---------------------------------------------------
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n1 = max(1, int(split_ratio * N))
    n2 = max(1, N - n1)
    I1, I2 = idx[:n1], idx[n1:n1 + n2]
    S1, S2 = S[I1], S[I2]  # (n1, K), (n2, K)

    # --- projection directions -----------------------------------------------
    U = _csa_sample_directions(K, M, seed=seed + 1)  # (M, K)

    # --- stage 1: learn envelope shape ---------------------------------------
    P1 = S1 @ U.T  # (n1, M): projected calibration scores

    def _cov1(beta: float) -> float:
        q = np.quantile(P1, 1.0 - float(beta), axis=0)  # (M,)
        return float(np.mean(np.all(P1 <= q[None, :], axis=1)))

    # Binary search: smallest beta s.t. stage-1 coverage >= 1-alpha.
    beta_lo, beta_hi = 0.0, 1.0
    for _ in range(60):
        beta_mid = 0.5 * (beta_lo + beta_hi)
        if _cov1(beta_mid) >= 1.0 - float(alpha):
            beta_hi = beta_mid
        else:
            beta_lo = beta_mid
    beta_star = float(beta_hi)
    q_tilde = np.quantile(P1, 1.0 - beta_star, axis=0).astype(np.float64)
    q_tilde = np.clip(q_tilde, 1e-12, None)

    # --- stage 2: scalar conformal calibration on I2 -------------------------
    P2 = S2 @ U.T  # (n2, M)
    T2 = np.max(P2 / q_tilde[None, :], axis=1).astype(np.float64)  # (n2,)

    level = float(np.ceil((n2 + 1) * (1.0 - float(alpha))) / n2)
    level = min(level, 1.0)
    t_hat = float(np.quantile(T2, level)) if n2 > 0 else float("inf")
    q_bar = (t_hat * q_tilde).astype(np.float64)  # final per-direction thresholds

    print(
        f"[CSA] K={K} M={M} n1={n1} n2={n2} beta*={beta_star:.4f} "
        f"t_hat={t_hat:.4f} alpha={alpha:.4f}"
    )
    return {
        "alpha": float(alpha),
        "K": int(K),
        "M": int(M),
        "U": U,           # (M, K) projection directions
        "q_tilde": q_tilde,  # (M,) stage-1 quantiles
        "t_hat": t_hat,
        "q_bar": q_bar,   # (M,) = t_hat * q_tilde, final thresholds
        "T2": T2,         # (n2,) stage-2 scalar T values (for re-thresholding)
        "beta_star": beta_star,
        "split_ratio": float(split_ratio),
        "n1": int(n1),
        "n2": int(n2),
    }


def _csa_thr_at_alpha(csa_obj: Dict[str, Any], alpha: float) -> float:
    """Compute t_hat for a given alpha reusing the stored stage-2 T2 values.

    This keeps the stage-1 envelope shape fixed (beta_star from primary alpha)
    and adjusts only the final conformal threshold level.
    """
    T2 = np.asarray(csa_obj["T2"], dtype=np.float64)
    n2 = len(T2)
    if n2 == 0:
        return float("inf")
    level = float(np.ceil((n2 + 1) * (1.0 - float(alpha))) / n2)
    level = min(level, 1.0)
    return float(np.quantile(T2, level))


def _csa_predict(
    csa_obj: Dict[str, Any],
    p: np.ndarray,
    alpha: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a fitted CSA object to produce credal sets for each test sample.

    For each candidate label y in {0, 1}, computes the K-dimensional score
    vector s(x, y) and checks whether T(s(x,y)) <= t_hat.

    Parameters
    ----------
    csa_obj : fitted dict from _csa_fit.
    p       : (K, N_test) base-model P(y=1|x) for test samples.
    alpha   : override significance level; None uses csa_obj['alpha'].

    Returns
    -------
    credal     : (N_test,) object array: "{None}", "{0}", "{1}", or "{0,1}".
    accepted   : (N_test,) bool — True iff the credal set is a singleton.
    pred_label : (N_test,) int  — 1 if credal={1}, else 0.
    """
    U = np.asarray(csa_obj["U"], dtype=np.float64)          # (M, K)
    q_tilde = np.asarray(csa_obj["q_tilde"], dtype=np.float64)  # (M,)
    if alpha is not None:
        t_hat = _csa_thr_at_alpha(csa_obj, alpha)
    else:
        t_hat = float(csa_obj["t_hat"])

    p_arr = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    if p_arr.ndim == 1:
        p_arr = p_arr[None, :]
    _K, N_test = p_arr.shape

    # Score vectors: s_k(x, y=1) = 1-p_k,  s_k(x, y=0) = p_k
    S_pos = (1.0 - p_arr).T  # (N_test, K)
    S_neg = p_arr.T          # (N_test, K)

    # Projections and induced scalar T scores
    T_pos = np.max((S_pos @ U.T) / q_tilde[None, :], axis=1)  # (N_test,)
    T_neg = np.max((S_neg @ U.T) / q_tilde[None, :], axis=1)  # (N_test,)

    in_pos = T_pos <= float(t_hat)   # include label 1?
    in_neg = T_neg <= float(t_hat)   # include label 0?

    credal = np.full(N_test, "{None}", dtype=object)
    credal[np.logical_and(~in_pos, in_neg)] = "{0}"
    credal[np.logical_and(in_pos, ~in_neg)] = "{1}"
    credal[np.logical_and(in_pos, in_neg)] = "{0,1}"

    accepted = np.logical_xor(in_pos, in_neg)
    pred_label = np.where(np.logical_and(in_pos, ~in_neg), 1, 0).astype(int)
    return credal, accepted, pred_label


def _csa_fit_label_conditional(
    p_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float,
    M: int = 128,
    split_ratio: float = 0.5,
    seed: int = 0,
) -> Dict[str, Any]:
    """Fit two independent CSA objects, one per binary class label.

    For class c, only the calibration samples with y_i = c are used.
    Their score vectors are:
        c=1:  s_k(x_i, y=1) = 1 - p_k(x_i)
        c=0:  s_k(x_i, y=0) = p_k(x_i)

    Each class gets its own projection directions, stage-1 envelope, and
    conformal threshold, giving label-conditional coverage:
        P(y_true in C(x) | y_true = c) >= 1 - alpha   for c in {0, 1}

    Parameters are identical to _csa_fit.
    Returns dict with keys 'pos' (class-1 CSA object) and 'neg' (class-0).
    """
    p_arr = np.asarray(p_val, dtype=np.float64)
    if p_arr.ndim == 1:
        p_arr = p_arr[None, :]
    y = np.asarray(y_val, dtype=int).reshape(-1)

    pos_mask = y == 1
    neg_mask = y == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos < 4:
        raise ValueError(
            f"[CSA-LC] Too few positive calibration samples ({n_pos}). "
            "Need at least 4 for a meaningful two-stage split."
        )
    if n_neg < 4:
        raise ValueError(
            f"[CSA-LC] Too few negative calibration samples ({n_neg}). "
            "Need at least 4 for a meaningful two-stage split."
        )

    # Fit positive-class CSA on s_k = 1 - p_k  (achieved by passing y_i = 1 for all).
    print(f"[CSA-LC] Fitting positive-class CSA (n_pos={n_pos})")
    csa_pos = _csa_fit(
        p_val=p_arr[:, pos_mask],
        y_val=np.ones(n_pos, dtype=int),
        alpha=alpha,
        M=M,
        split_ratio=split_ratio,
        seed=seed,
    )

    # Fit negative-class CSA on s_k = p_k  (achieved by passing y_i = 0 for all).
    print(f"[CSA-LC] Fitting negative-class CSA (n_neg={n_neg})")
    csa_neg = _csa_fit(
        p_val=p_arr[:, neg_mask],
        y_val=np.zeros(n_neg, dtype=int),
        alpha=alpha,
        M=M,
        split_ratio=split_ratio,
        seed=seed + 1000,
    )

    return {"pos": csa_pos, "neg": csa_neg}


def _csa_predict_label_conditional(
    lc_csa_obj: Dict[str, Any],
    p: np.ndarray,
    alpha: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply label-conditional CSA to produce credal sets.

    Candidate label y=1 is tested against the positive-class CSA object
    (calibrated on 1-p_k scores from true positive samples).
    Candidate label y=0 is tested against the negative-class CSA object
    (calibrated on p_k scores from true negative samples).

    Each class has its own projection directions and threshold, so
    label-conditional coverage is maintained even under class imbalance.

    Parameters
    ----------
    lc_csa_obj : dict with keys 'pos' and 'neg' from _csa_fit_label_conditional.
    p          : (K, N_test) base-model P(y=1|x).
    alpha      : override significance level; None uses the fitted alpha.

    Returns
    -------
    credal, accepted, pred_label  (same semantics as _csa_predict)
    """
    csa_pos = lc_csa_obj["pos"]
    csa_neg = lc_csa_obj["neg"]

    t_hat_pos = _csa_thr_at_alpha(csa_pos, alpha) if alpha is not None else float(csa_pos["t_hat"])
    t_hat_neg = _csa_thr_at_alpha(csa_neg, alpha) if alpha is not None else float(csa_neg["t_hat"])

    U_pos = np.asarray(csa_pos["U"], dtype=np.float64)          # (M, K)
    q_pos = np.asarray(csa_pos["q_tilde"], dtype=np.float64)    # (M,)
    U_neg = np.asarray(csa_neg["U"], dtype=np.float64)          # (M, K)
    q_neg = np.asarray(csa_neg["q_tilde"], dtype=np.float64)    # (M,)

    p_arr = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    if p_arr.ndim == 1:
        p_arr = p_arr[None, :]
    _K, N_test = p_arr.shape

    S_pos = (1.0 - p_arr).T   # (N_test, K): s_k(x, y=1) = 1 - p_k
    S_neg = p_arr.T            # (N_test, K): s_k(x, y=0) = p_k

    # Each label uses its own geometry.
    T_pos = np.max((S_pos @ U_pos.T) / q_pos[None, :], axis=1)   # (N_test,)
    T_neg = np.max((S_neg @ U_neg.T) / q_neg[None, :], axis=1)   # (N_test,)

    in_pos = T_pos <= float(t_hat_pos)
    in_neg = T_neg <= float(t_hat_neg)

    credal = np.full(N_test, "{None}", dtype=object)
    credal[np.logical_and(~in_pos, in_neg)] = "{0}"
    credal[np.logical_and(in_pos, ~in_neg)] = "{1}"
    credal[np.logical_and(in_pos, in_neg)] = "{0,1}"

    accepted = np.logical_xor(in_pos, in_neg)
    pred_label = np.where(np.logical_and(in_pos, ~in_neg), 1, 0).astype(int)
    return credal, accepted, pred_label


# ─────────────────────────────────────────────────────────────────────────────
# CP visualization
# ─────────────────────────────────────────────────────────────────────────────

def _export_conformal_trend_confusion_grid(
    entries: List[Dict[str, Any]],
    row_specs: Sequence[Tuple[str, str]],
    out_png: Path,
    out_csv: Path,
) -> None:
    if not entries or not row_specs:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(entries)
    n_rows = len(row_specs)
    cm_font_scale = 1.55
    fig, axes = plt.subplots(
        n_rows,
        n_models,
        figsize=(max(21, 4.8 * n_models), max(8.5, 4.2 * n_rows)),
        squeeze=False,
    )
    vmax = 100.0
    csv_rows: List[Dict[str, Any]] = []
    base_cols: List[int] = []
    ensemble_cols: List[int] = []
    other_cols: List[int] = []

    for col, entry in enumerate(entries):
        model_name = str(entry["model_name"])
        if model_name.startswith("base::"):
            base_cols.append(col)
        elif model_name.startswith("stack::"):
            ensemble_cols.append(col)
        else:
            other_cols.append(col)
        display_name = _display_model_name(model_name)
        title_name = display_name
        if title_name.startswith("base_"):
            title_name = title_name[len("base_"):]
        elif title_name.startswith("ensemble_"):
            title_name = title_name[len("ensemble_"):]

        row_cms = entry.get("row_cms", {})
        for row_idx, (row_key, row_label) in enumerate(row_specs):
            cm_counts = np.asarray(row_cms.get(row_key, np.zeros((2, 2), dtype=float)), dtype=float)
            has_data = bool(np.isfinite(cm_counts).all()) and float(np.sum(cm_counts)) > 0.0
            if has_data:
                row_sums = cm_counts.sum(axis=1, keepdims=True)
                cm = 100.0 * cm_counts / np.clip(row_sums, 1e-12, None)
            else:
                cm = np.zeros((2, 2), dtype=float)

            ax = axes[row_idx, col]
            ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=vmax, alpha=0.88)
            for i in range(2):
                for j in range(2):
                    val = float(cm[i, j])
                    txt = f"{val:.1f}%" if has_data else "N/A"
                    text_color = "white" if has_data and val >= 55.0 else "black"
                    ax.text(
                        j,
                        i,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=16 * cm_font_scale,
                        color=text_color,
                    )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=14 * cm_font_scale)
            if col == 0:
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["True 0", "True 1"], fontsize=14 * cm_font_scale)
            else:
                ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title_name, fontsize=16 * cm_font_scale)
            if col == 0:
                ax.set_ylabel(f"{row_label}\nlabel", fontsize=14 * cm_font_scale)
            ax.set_xlabel("prediction", fontsize=14 * cm_font_scale)

            csv_rows.append(
                {
                    "model_name": model_name,
                    "model_display_name": display_name,
                    "row_key": row_key,
                    "row_label": row_label,
                    "tn_percent": float(cm[0, 0]) if has_data else float("nan"),
                    "fp_percent": float(cm[0, 1]) if has_data else float("nan"),
                    "fn_percent": float(cm[1, 0]) if has_data else float("nan"),
                    "tp_percent": float(cm[1, 1]) if has_data else float("nan"),
                    "negative_recall_percent": float(cm[0, 0]) if has_data else float("nan"),
                    "positive_recall_percent": float(cm[1, 1]) if has_data else float("nan"),
                }
            )

    def _add_group_annotation(cols: List[int], label: str) -> None:
        if not cols:
            return
        left_box = axes[0, min(cols)].get_position()
        right_box = axes[0, max(cols)].get_position()
        x_center = 0.5 * (left_box.x0 + right_box.x1)
        fig.text(
            x_center,
            0.925,
            label,
            ha="center",
            va="center",
            fontsize=11 * cm_font_scale,
            fontweight="bold",
            color="#1f2937",
        )

    _add_group_annotation(base_cols, "Base models")
    _add_group_annotation(ensemble_cols, "Ensembles")
    _add_group_annotation(other_cols, "Other")

    # Intentionally no global title: keep figure compact for reports.
    fig.subplots_adjust(left=0.06, right=0.93, bottom=0.08, top=0.90, wspace=0.30, hspace=0.30)

    if base_cols and ensemble_cols:
        # Place separator between the true subplot groups.
        # Must be computed after subplots_adjust so get_position() returns final coordinates.
        base_right = axes[0, max(base_cols)].get_position().x1
        ens_left = axes[0, min(ensemble_cols)].get_position().x0
        split_x = base_right + 0.5 * (ens_left - base_right)
        sep_line = plt.Line2D(
            [split_x, split_x],
            [0.08, 0.90],
            transform=fig.transFigure,
            color="#374151",
            linestyle="--",
            linewidth=2.8,
            alpha=0.85,
            zorder=3,
        )
        fig.add_artist(sep_line)

    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)


def _export_csa_decision_space_scatter(
    stack_name: str,
    p_test_runs: List[np.ndarray],
    u_test_epi_runs: List[np.ndarray],
    credal_ac_accept_store: Dict[str, List[np.ndarray]],
    conformal_trend_alphas: List[float],
    category_specs: List[Tuple[str, str, np.ndarray]],
    methods: List[str],
    out_dir: Path,
) -> None:
    """
    Visualise where AC / non-AC test samples sit in the K-dimensional ensemble
    decision space, one figure per confidence level (alpha).

    Each figure has two side-by-side panels:
      Left  — CP non-conformity score space  (s_k = 1 − p_k per base model k)
      Right — Epistemic uncertainty space     (u_k per base model k)

    For K=2 the axes are 2-D; for K=3 they are 3-D (Axes3D).
    Other K values are skipped.

    Point colour encodes ac_label category (AC / non-AC / non-AC no MMP).
    Filled circles = accepted by CSA; hollow × markers = rejected.
    Predictions are averaged across runs before plotting.
    """
    if not p_test_runs or not u_test_epi_runs or not category_specs:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    K = p_test_runs[0].shape[0]
    if K not in (2, 3):
        print(
            f"[CSA scatter] K={K} base models — scatter plot supports only K=2 or K=3. Skipping."
        )
        return

    use_3d = K == 3

    # Average across runs.
    p_avg = np.mean(np.stack(p_test_runs, axis=0), axis=0)      # (K, N)
    u_avg = np.mean(np.stack(u_test_epi_runs, axis=0), axis=0)  # (K, N)

    # CP non-conformity score for candidate label y=1: s_k = 1 − p_k
    cp_scores = 1.0 - np.clip(p_avg, 1e-8, 1.0 - 1e-8)  # (K, N)

    # Display names for axes (strip path-like prefixes if any)
    axis_labels = [_display_model_name(f"base::{m}") for m in methods]

    cat_colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # AC, non-AC (no MMP), non-AC
    marker_accept = "o"
    marker_reject = "X"

    for alpha in conformal_trend_alphas:
        alpha_key = f"alpha_{alpha:.4f}"
        run_masks = credal_ac_accept_store.get(alpha_key, [])
        conf_level_pct = int(round((1.0 - float(alpha)) * 100.0))

        # Majority-vote acceptance across runs.
        N = cp_scores.shape[1]
        if run_masks:
            valid = [np.asarray(m, dtype=float) for m in run_masks if len(m) == N]
            accepted = np.mean(np.stack(valid, axis=0), axis=0) >= 0.5 if valid else np.ones(N, dtype=bool)
        else:
            accepted = np.ones(N, dtype=bool)

        out_png = out_dir / f"csa_decision_space_{stack_name}_cp{conf_level_pct}.png"

        if use_3d:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure(figsize=(17, 7))
            ax_cp  = fig.add_subplot(1, 2, 1, projection="3d")
            ax_epi = fig.add_subplot(1, 2, 2, projection="3d")
            panel_pairs = [(ax_cp, cp_scores, "CP non-conformity score (1−p_k)"),
                           (ax_epi, u_avg,    "Epistemic uncertainty")]
        else:
            fig, (ax_cp, ax_epi) = plt.subplots(1, 2, figsize=(14, 6))
            panel_pairs = [(ax_cp, cp_scores, "CP non-conformity score (1−p_k)"),
                           (ax_epi, u_avg,    "Epistemic uncertainty")]

        for ax, coords, panel_label in panel_pairs:
            for (cat_name, _cat_slug, cat_mask), color in zip(category_specs, cat_colors):
                for acc_flag, marker, pt_alpha, zorder in [
                    (True,  marker_accept, 0.80, 3),
                    (False, marker_reject, 0.35, 2),
                ]:
                    pts = cat_mask & (accepted == acc_flag)
                    if not pts.any():
                        continue
                    label = f"{cat_name} ({'accepted' if acc_flag else 'rejected'})"
                    scatter_kw = dict(
                        c=color, marker=marker,
                        s=28 if acc_flag else 35,
                        alpha=pt_alpha, zorder=zorder, label=label,
                        edgecolors="none" if acc_flag else color,
                        linewidths=0.0 if acc_flag else 0.7,
                    )
                    if use_3d:
                        ax.scatter(
                            coords[0, pts], coords[1, pts], coords[2, pts],
                            **scatter_kw,
                        )
                        ax.set_xlabel(axis_labels[0], fontsize=9, labelpad=4)
                        ax.set_ylabel(axis_labels[1], fontsize=9, labelpad=4)
                        ax.set_zlabel(axis_labels[2], fontsize=9, labelpad=4)
                    else:
                        ax.scatter(coords[0, pts], coords[1, pts], **scatter_kw)
                        ax.set_xlabel(axis_labels[0], fontsize=10)
                        ax.set_ylabel(axis_labels[1], fontsize=10)

            ax.set_title(f"{panel_label}", fontsize=11)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, loc="best", frameon=False)

        fig.suptitle(
            f"CSA ensemble decision space — {_display_model_name('stack::' + stack_name)}"
            f" — {conf_level_pct}% confidence",
            fontsize=12,
            y=1.01,
        )
        fig.tight_layout()
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[CSA scatter] saved: {out_png}")


def _export_credal_ac_label_grid(
    stack_name: str,
    credal_ac_accept_store: Dict[str, List[np.ndarray]],
    conformal_trend_alphas: List[float],
    ac_labels_test: np.ndarray,
    category_specs: List[Tuple[str, str, np.ndarray]],
    out_png: Path,
    out_csv: Path,
    norm: str = "column",
) -> None:
    """
    For one iso-stack method, plot a 1-row × len(conformal_trend_alphas) figure.

    Each column corresponds to one confidence level (alpha).  Inside each column
    we show a 2 × C imshow table where:
      rows = [Accepted, Rejected]
      cols = ac_label categories (C = len(category_specs))

    norm="column" (default): column-normalised percentages — within each
      ac_label category, what fraction was accepted vs rejected.
    norm="row": row-normalised percentages — within Accepted/Rejected,
      what fraction of samples belong to each ac_label category.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    n_cats = len(category_specs)
    n_alphas = len(conformal_trend_alphas)
    if n_cats == 0 or n_alphas == 0:
        return

    cat_names = [name for name, _, _ in category_specs]
    cat_masks = [mask for _, _, mask in category_specs]  # each (N_test,) bool
    row_labels = ["Accepted", "Rejected"]
    font_scale = 1.5

    fig, axes = plt.subplots(
        1, n_alphas,
        figsize=(max(10, 3.8 * n_alphas), 5.0),
        squeeze=False,
    )
    csv_rows: List[Dict[str, Any]] = []

    for col_idx, alpha in enumerate(conformal_trend_alphas):
        ax = axes[0, col_idx]
        alpha_key = f"alpha_{alpha:.4f}"
        run_masks = credal_ac_accept_store.get(alpha_key, [])
        conf_level_pct = int(round((1.0 - float(alpha)) * 100.0))

        # Build 2×C count matrix (rows=accepted/rejected, cols=categories)
        # Aggregate accepted_mask across runs: treat each (sample, run) as one observation.
        table = np.zeros((2, n_cats), dtype=float)
        if run_masks:
            for mask_arr in run_masks:
                mask = np.asarray(mask_arr, dtype=bool).reshape(-1)
                for c_idx, cat_mask in enumerate(cat_masks):
                    cat = np.asarray(cat_mask, dtype=bool).reshape(-1)
                    in_cat = cat & (len(mask) == len(cat))
                    if len(mask) != len(cat):
                        continue
                    table[0, c_idx] += float(np.sum(mask & cat))   # accepted
                    table[1, c_idx] += float(np.sum(~mask & cat))  # rejected

        # Normalise: column-wise (per ac_label category) or row-wise (per decision row)
        if norm == "row":
            sums = np.clip(table.sum(axis=1, keepdims=True), 1e-12, None)
        else:
            sums = np.clip(table.sum(axis=0, keepdims=True), 1e-12, None)
        table_pct = 100.0 * table / sums

        ax.imshow(table_pct, cmap="Blues", vmin=0.0, vmax=100.0, alpha=0.88)
        for r in range(2):
            for c in range(n_cats):
                val = float(table_pct[r, c])
                count = int(round(table[r, c]))
                text_color = "white" if val >= 55.0 else "black"
                ax.text(
                    c, r,
                    f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=10 * font_scale,
                    color=text_color,
                )
                csv_rows.append({
                    "stack_name": stack_name,
                    "confidence_level_pct": conf_level_pct,
                    "alpha": float(alpha),
                    "row": row_labels[r],
                    "ac_label_category": cat_names[c],
                    "count": count,
                    f"{'row' if norm == 'row' else 'column'}_normalized_pct": float(val),
                })

        ax.set_xticks(np.arange(n_cats))
        ax.set_xticklabels(cat_names, rotation=25, ha="right", fontsize=9 * font_scale)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(row_labels, fontsize=9 * font_scale)
        ax.set_title(f"CP {conf_level_pct}% confidence", fontsize=11 * font_scale)

    norm_desc = "row-normalised (% of accepted/rejected per category)" if norm == "row" else "column-normalised (% accepted/rejected within category)"
    fig.suptitle(
        f"Conformal acceptance vs ac_label — {_display_model_name('stack::' + stack_name)}\n{norm_desc}",
        fontsize=12 * font_scale,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
