"""
Active Learning Module for Neural Network Uncertainty Quantification

This module implements active learning functionality for neural network models,
allowing iterative query selection based on uncertainty estimates.
"""

import numpy as np
import deepchem as dc
from typing import List, Optional, Tuple, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.stats import norm

# Import functions from nn_baseline.py to maintain uniformity
from nn_baseline import (
    build_model, get_n_features, mse_from_mean_prediction,
    DeepEnsembleRegressor, DeepEnsembleClassifier,
    MCDropoutRegressorRefined, MCDropoutClassifierWrapper,
    UnifiedTorchModel, HeteroscedasticL2Loss, HeteroscedasticClassificationLoss,
    EvidentialRegressionLoss, EvidentialClassificationLoss, GradientClippingCallback
)
import nn_baseline
from data_utils import (
    evaluate_uq_metrics_from_interval, evaluate_uq_metrics_classification,
    calculate_cutoff_error_data, calculate_cutoff_classification_data,
    auc_from_probs, compute_confusion_matrix_binary
)
import deepchem as dc
import torch


# ============================================================
# Helper Functions: Data Manipulation
# ============================================================

def combine_datasets(ds1: dc.data.NumpyDataset, ds2: dc.data.NumpyDataset) -> dc.data.NumpyDataset:
    """
    Combine two DeepChem NumpyDatasets into one.
    
    Args:
        ds1: First dataset
        ds2: Second dataset
        
    Returns:
        Combined dataset
    """
    X_combined = np.concatenate([ds1.X, ds2.X], axis=0)
    y_combined = np.concatenate([ds1.y, ds2.y], axis=0)
    
    w_combined = None
    if ds1.w is not None and ds2.w is not None:
        w_combined = np.concatenate([ds1.w, ds2.w], axis=0)
    elif ds1.w is not None:
        w_combined = np.concatenate([ds1.w, np.ones_like(ds2.y)], axis=0)
    elif ds2.w is not None:
        w_combined = np.concatenate([np.ones_like(ds1.y), ds2.w], axis=0)
    
    ids_combined = None
    try:
        if ds1.ids is not None and ds2.ids is not None:
            ids_combined = list(ds1.ids) + list(ds2.ids)
        elif ds1.ids is not None:
            ids_combined = list(ds1.ids) + [None] * len(ds2)
        elif ds2.ids is not None:
            ids_combined = [None] * len(ds1) + list(ds2.ids)
    except:
        ids_combined = None
    
    return dc.data.NumpyDataset(
        X=X_combined,
        y=y_combined,
        w=w_combined,
        ids=ids_combined
    )


def _subset_numpy_dataset(ds: dc.data.NumpyDataset, idx: np.ndarray) -> dc.data.NumpyDataset:
    """Return a DeepChem NumpyDataset subsetted by idx."""
    X = ds.X[idx]
    y = ds.y[idx]
    w = None if ds.w is None else ds.w[idx]
    try:
        ids = ds.ids[idx]
    except Exception:
        try:
            ids = [ds.ids[int(i)] for i in idx]
        except Exception:
            ids = None
    return dc.data.NumpyDataset(X=X, y=y, w=w, ids=ids)


def copy_numpy_dataset(ds: dc.data.NumpyDataset) -> dc.data.NumpyDataset:
    """Create a copy of a DeepChem NumpyDataset."""
    return dc.data.NumpyDataset(
        X=ds.X.copy() if hasattr(ds.X, 'copy') else ds.X,
        y=ds.y.copy() if hasattr(ds.y, 'copy') else ds.y,
        w=ds.w.copy() if ds.w is not None and hasattr(ds.w, 'copy') else ds.w,
        ids=list(ds.ids) if ds.ids is not None else None
    )


def create_stratified_initial_split(
    train_dc: dc.data.NumpyDataset,
    initial_ratio: float = 0.1,
    mode: str = "regression",
    random_state: int = 0
) -> Tuple[dc.data.NumpyDataset, dc.data.NumpyDataset]:
    """
    Create initial training set with stratified sampling for classification.
    Uses deterministic selection (sorted indices) to ensure same initial dataset across methods.
    
    Args:
        train_dc: Training dataset
        initial_ratio: Fraction of data to use for initial training (0.1 = 10%)
        mode: "regression" or "classification"
        random_state: Random seed (for reproducibility, but uses deterministic selection)
        
    Returns:
        Tuple of (initial_train_dc, pool_dc)
    """
    # Use only training dataset (ignore validation)
    N_total = len(train_dc)
    n_initial = max(1, int(N_total * initial_ratio))
    
    if mode == "classification":
        # Stratified sampling for classification (deterministic)
        y = train_dc.y
        
        # Handle multitask: ensure class balance per task
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        n_tasks = y.shape[1]
        
        # For each task, perform stratified sampling deterministically
        selected_indices_list = []
        for task_idx in range(n_tasks):
            y_task = y[:, task_idx]
            
            # Get unique classes for this task
            unique_classes = np.unique(y_task)
            
            # Ensure we have at least one sample from each class
            task_indices = []
            for cls in unique_classes:
                cls_indices = np.where(y_task == cls)[0]
                n_per_class = max(1, int(len(cls_indices) * initial_ratio))
                if n_per_class > len(cls_indices):
                    n_per_class = len(cls_indices)
                
                # Deterministic selection: sort indices and take first n_per_class
                # This ensures same selection regardless of method
                sorted_cls_indices = np.sort(cls_indices)
                selected = sorted_cls_indices[:n_per_class]
                task_indices.extend(selected.tolist())
            
            selected_indices_list.append(set(task_indices))
        
        # Take union for multitask to ensure we have enough samples
        if n_tasks > 1:
            final_indices = set()
            for idx_set in selected_indices_list:
                final_indices.update(idx_set)
            final_indices = np.array(list(final_indices))
        else:
            final_indices = np.array(list(selected_indices_list[0]))
        
        # If we need more samples, add deterministically (sorted remaining indices)
        if len(final_indices) < n_initial:
            remaining = np.setdiff1d(np.arange(N_total), final_indices)
            n_needed = n_initial - len(final_indices)
            # Take first n_needed from sorted remaining indices (deterministic)
            additional = np.sort(remaining)[:min(n_needed, len(remaining))]
            final_indices = np.concatenate([final_indices, additional])
        
        # Limit to n_initial deterministically (take first n_initial from sorted)
        if len(final_indices) > n_initial:
            final_indices = np.sort(final_indices)[:n_initial]
        
        selected_indices = np.sort(final_indices)  # Ensure sorted for determinism
    else:
        # Deterministic sampling for regression: take first n_initial indices
        # This ensures same initial dataset across all methods
        selected_indices = np.arange(n_initial)
    
    # Create initial training set and pool
    initial_train_dc = _subset_numpy_dataset(train_dc, selected_indices)
    pool_indices = np.setdiff1d(np.arange(N_total), selected_indices)
    pool_dc = _subset_numpy_dataset(train_dc, pool_indices)
    
    return initial_train_dc, pool_dc


# ============================================================
# Uncertainty Extraction
# ============================================================

def aggregate_multitask_uncertainty(uncertainty: np.ndarray, strategy: str = "mean") -> np.ndarray:
    """
    Aggregate uncertainty across multiple tasks.
    
    Args:
        uncertainty: Uncertainty array of shape (N, T) where T is number of tasks
        strategy: Aggregation strategy ("mean", "max", "sum")
        
    Returns:
        Aggregated uncertainty of shape (N,)
    """
    if uncertainty.ndim == 1:
        return uncertainty
    
    if strategy == "mean":
        return np.mean(uncertainty, axis=1)
    elif strategy == "max":
        return np.max(uncertainty, axis=1)
    elif strategy == "sum":
        return np.sum(uncertainty, axis=1)
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")


def extract_uncertainty(
    model,
    dataset: dc.data.NumpyDataset,
    method_name: str,
    mode: str,
    encoder_type: str = "identity"
) -> np.ndarray:
    """
    Extract uncertainty scores from a trained model.
    
    Args:
        model: Trained model (varies by method)
        dataset: Dataset to evaluate
        method_name: One of "nn_baseline", "nn_deep_ensemble", "nn_mc_dropout", "nn_evd"
        mode: "regression" or "classification"
        encoder_type: "identity" or "dmpnn"
        
    Returns:
        Uncertainty scores array of shape (N,) or (N, T) for multitask
    """
    if method_name == "nn_baseline":
        # Baseline has no uncertainty, return zeros (or skip in query)
        # For now, return small random values to enable selection
        n_samples = len(dataset)
        n_tasks = dataset.y.shape[1] if dataset.y.ndim > 1 else 1
        return np.random.rand(n_samples, n_tasks) * 1e-6 if n_tasks > 1 else np.random.rand(n_samples) * 1e-6
    
    elif method_name == "nn_deep_ensemble":
        if mode == "regression":
            ensemble = DeepEnsembleRegressor(model)
            mean, lower, upper = ensemble.predict_interval(dataset, alpha=0.05)
            # Extract std from intervals: std = (upper - lower) / (2 * z)
            z = norm.ppf(0.975)  # 95% CI
            std = (upper - lower) / (2 * z)
            # Handle shape: std might be (N, T) or (N,)
            if std.ndim == 1:
                std = std.reshape(-1, 1)
            return std
        else:
            ensemble = DeepEnsembleClassifier(model)
            _, _, _, MI = ensemble.predict_uncertainty(dataset)
            # Epistemic (MI) for classification AL
            return MI
    
    elif method_name == "nn_mc_dropout":
        if mode == "regression":
            mc_wrapper = MCDropoutRegressorRefined(model, n_samples=100)
            _, std = mc_wrapper.predict_uncertainty(dataset)
            # std is (N, 1) or (N, T)
            if std.ndim == 2 and std.shape[1] == 1:
                std = std.squeeze(1)
            return std
        else:
            mc_wrapper = MCDropoutClassifierWrapper(model, n_samples=50)
            _, _, _, MI = mc_wrapper.predict_uncertainty(dataset)
            # Epistemic (MI) for classification AL
            if MI.ndim == 2 and MI.shape[1] == 1:
                MI = MI.squeeze(1)
            return MI
    
    elif method_name == "nn_evd":
        device = next(model.model.parameters()).device
        
        # Convert data to tensor
        if encoder_type == "dmpnn":
            from nn import graphdata_to_batchmolgraph
            if isinstance(dataset.X, np.ndarray) and dataset.X.dtype == np.object_:
                X_list = dataset.X.tolist()
            else:
                X_list = dataset.X
            X_tensor = graphdata_to_batchmolgraph(X_list)
            X_tensor = X_tensor.to(device)
        else:
            X_tensor = torch.from_numpy(dataset.X).float().to(device)
        
        with torch.no_grad():
            outputs = model.model(X_tensor)
        
        if mode == "regression":
            # outputs: (mu, params, aleatoric, epistemic)
            _, _, aleatoric, epistemic = outputs
            # Total uncertainty = aleatoric + epistemic
            total_unc = (aleatoric + epistemic).cpu().numpy()
            if total_unc.ndim == 1:
                total_unc = total_unc.reshape(-1, 1)
            return total_unc
        else:
            # Classification: extract epistemic uncertainty (vacuity)
            _, _, _, epistemic = outputs
            epistemic_np = epistemic.cpu().numpy()
            if epistemic_np.ndim == 1:
                epistemic_np = epistemic_np.reshape(-1, 1)
            return epistemic_np
    
    else:
        raise ValueError(f"Unknown method: {method_name}")


# ============================================================

# ============================================================
# Conformal Prediction (nn_conformal) — binary classification only
# ============================================================

def _get_probs_from_model(model, dataset: dc.data.NumpyDataset, encoder_type: str = "identity") -> np.ndarray:
    """Get predicted class-1 probabilities (N, T) from a baseline-style classifier."""
    out = model.predict(dataset)
    out = np.asarray(out)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    return out


def _standard_cp_quantile(scores: np.ndarray, q: float) -> float:
    """Standard finite-sample conformal quantile: Q_hat = S_(k) with k = ceil((n+1)*q)."""
    scores = np.asarray(scores).flatten()
    n = len(scores)
    if n == 0:
        return np.nan
    k = min(int(np.ceil((n + 1) * q)), n)
    idx = max(0, k - 1)
    return float(np.sort(scores)[idx])


def get_conformal_thresholds(
    model,
    valid_data: dc.data.NumpyDataset,
    alpha: float = 0.05,
    encoder_type: str = "identity"
) -> np.ndarray:
    """Class-conditional conformal thresholds per task from validation set. Non-conformity = 1 - p_y (true class)."""
    probs = _get_probs_from_model(model, valid_data, encoder_type)
    y = np.asarray(valid_data.y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n_tasks = probs.shape[1]
    thresholds = np.zeros((n_tasks, 2))
    q_level = 1.0 - alpha
    for t in range(n_tasks):
        y_t = y[:, t]
        p1_t = probs[:, t]
        scores_y0 = p1_t[y_t == 0]
        scores_y1 = 1.0 - p1_t[y_t == 1]
        thresholds[t, 0] = _standard_cp_quantile(scores_y0, q_level)
        thresholds[t, 1] = _standard_cp_quantile(scores_y1, q_level)
    return thresholds


# Set-type codes for conformal prediction sets (per sample, per task)
SET_EMPTY = 0      # prediction set = {}
SET_SINGLETON_0 = 1   # prediction set = {0}
SET_SINGLETON_1 = 2   # prediction set = {1}
SET_BOTH = 3       # prediction set = {0, 1}


def get_prediction_sets(
    model,
    dataset: dc.data.NumpyDataset,
    thresholds: np.ndarray,
    encoder_type: str = "identity"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Conformal prediction sets per (sample, task). Returns exactly four set types per (n, t).

    Set membership: include 0 if (1-p0) <= q_t0, include 1 if (1-p1) <= q_t1.

    Returns:
        n_uncertain: (N,) int — count of tasks where set is not a singleton (empty or {0,1}).
        mean_nc: (N,) float — mean over tasks of min(p1, 1-p1); higher = more uncertain.
        is_uncertain_nt: (N, T) bool — True where set is empty or {0,1}.
        set_types: (N, T) int — for each (sample, task) one of:
            SET_EMPTY = 0       -> {}
            SET_SINGLETON_0 = 1 -> {0}
            SET_SINGLETON_1 = 2 -> {1}
            SET_BOTH = 3        -> {0, 1}
    """
    probs = _get_probs_from_model(model, dataset, encoder_type)
    N, T = probs.shape
    p1 = probs
    q0 = thresholds[:, 0]
    q1 = thresholds[:, 1]
    in0 = (p1 <= q0[np.newaxis, :])   # include label 0
    in1 = ((1.0 - p1) <= q1[np.newaxis, :])   # include label 1
    # Encode exactly four types: 0=empty, 1={0}, 2={1}, 3={0,1}  ->  set_type = in0 + 2*in1
    set_types = (in0.astype(np.int32) + 2 * in1.astype(np.int32))
    is_uncertain_nt = (set_types == SET_EMPTY) | (set_types == SET_BOTH)
    n_uncertain = np.sum(is_uncertain_nt, axis=1).astype(np.int64)
    nc_per_task = np.minimum(p1, 1.0 - p1)
    mean_nc = np.mean(nc_per_task, axis=1)
    return n_uncertain, mean_nc, is_uncertain_nt, set_types


def _pareto_front_indices(objectives: np.ndarray, maximize: bool = True) -> np.ndarray:
    N, D = objectives.shape
    if N == 0:
        return np.array([], dtype=bool)
    front = np.ones(N, dtype=bool)
    for i in range(N):
        if not front[i]:
            continue
        for j in range(N):
            if i == j or not front[j]:
                continue
            if maximize:
                dom = np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i])
            else:
                dom = np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i])
            if dom:
                front[i] = False
                break
    return front


def _non_dominated_sort(objectives: np.ndarray, maximize: bool = True) -> List[np.ndarray]:
    N = objectives.shape[0]
    remaining = np.arange(N)
    fronts = []
    while len(remaining) > 0:
        obj_rem = objectives[remaining]
        front_mask = _pareto_front_indices(obj_rem, maximize=maximize)
        front_indices = remaining[front_mask]
        fronts.append(front_indices)
        remaining = remaining[~front_mask]
    return fronts


def select_conformal_acquisition(
    n_uncertain: np.ndarray,
    mean_nc: np.ndarray,
    is_uncertain_nt: np.ndarray,
    n_query: int,
    tie_break: str = "mean_nc",
    random_state: Optional[int] = None
) -> np.ndarray:
    N = len(n_uncertain)
    n_select = min(n_query, N)
    if n_select <= 0:
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(random_state)
    if tie_break == "mean_nc":
        order = np.lexsort((-mean_nc, -n_uncertain))
        return order[:n_select].astype(np.int64)
    fronts = _non_dominated_sort(is_uncertain_nt.astype(np.float64), maximize=True)
    selected = []
    for front in fronts:
        if len(selected) >= n_select:
            break
        need = n_select - len(selected)
        if len(front) <= need:
            selected.extend(front.tolist())
        else:
            chosen = rng.choice(front, size=need, replace=False)
            selected.extend(chosen.tolist())
    return np.array(selected[:n_select], dtype=np.int64)


def select_instances_conformal(
    model,
    pool_dc: dc.data.NumpyDataset,
    valid_dc: dc.data.NumpyDataset,
    thresholds: Optional[np.ndarray],
    n_query: int,
    encoder_type: str = "identity",
    tie_break: str = "mean_nc",
    use_cluster: bool = False,
    latent_vectors: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> np.ndarray:
    if thresholds is None:
        thresholds = get_conformal_thresholds(model, valid_dc, alpha=0.05, encoder_type=encoder_type)
    n_uncertain, mean_nc, is_uncertain_nt, set_types = get_prediction_sets(model, pool_dc, thresholds, encoder_type)
    if use_cluster and latent_vectors is not None:
        n_pool = len(pool_dc)
        n_clusters = min(n_query, n_pool)
        if n_clusters <= 0:
            return np.array([], dtype=np.int64)
        if n_clusters == 1:
            idx = select_conformal_acquisition(n_uncertain, mean_nc, is_uncertain_nt, 1, tie_break, random_state)
            return idx
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=5, max_iter=50)
        labels = kmeans.fit_predict(latent_vectors)
        selected = []
        for c in range(n_clusters):
            mask = labels == c
            if not np.any(mask):
                continue
            indices_in_c = np.where(mask)[0]
            n_unc_c = n_uncertain[indices_in_c]
            mean_nc_c = mean_nc[indices_in_c]
            is_unc_c = is_uncertain_nt[indices_in_c]
            best_local = select_conformal_acquisition(n_unc_c, mean_nc_c, is_unc_c, 1, tie_break, random_state)
            global_idx = indices_in_c[best_local[0]]
            selected.append(global_idx)
        return np.array(selected, dtype=np.int64)
    return select_conformal_acquisition(n_uncertain, mean_nc, is_uncertain_nt, n_query, tie_break, random_state)


# ============================================================
# Query Selection
# ============================================================

def select_instances_by_uncertainty(
    pool_dc: dc.data.NumpyDataset,
    uncertainty_scores: np.ndarray,
    query_ratio: float = 0.05
) -> np.ndarray:
    """
    Select instances from pool based on uncertainty scores.
    
    Args:
        pool_dc: Pool dataset
        uncertainty_scores: Uncertainty scores, shape (N,) or (N, T)
        query_ratio: Fraction of pool to query (0.05 = 5%)
        
    Returns:
        Indices of selected instances
    """
    # Aggregate multitask uncertainty if needed
    if uncertainty_scores.ndim > 1 and uncertainty_scores.shape[1] > 1:
        uncertainty_scores = aggregate_multitask_uncertainty(uncertainty_scores, strategy="mean")
    
    # Ensure 1D
    uncertainty_scores = uncertainty_scores.flatten()
    
    # Number to query: same as pool-based fraction every time
    n_pool = len(pool_dc)
    n_query = max(1, int(n_pool * query_ratio))
    
    # Select top-k most uncertain
    top_indices = np.argsort(uncertainty_scores)[::-1][:n_query]
    
    return top_indices


def extract_latent_vectors(
    model_or_models,
    pool_dc: dc.data.NumpyDataset,
    method_name: str,
    encoder_type: str = "identity"
) -> np.ndarray:
    """
    Extract latent vectors for pool samples using UnifiedModel.latent (no gradients).
    For ensemble (nn_deep_ensemble), uses the first model's latent.
    
    Returns:
        latent_vectors: (N, D) numpy array
    """
    if method_name == "nn_deep_ensemble":
        model = model_or_models[0]
    else:
        model = model_or_models
    unified = model.model  # UnifiedModel
    device = next(unified.parameters()).device

    if encoder_type == "dmpnn":
        from nn import graphdata_to_batchmolgraph
        if isinstance(pool_dc.X, np.ndarray) and pool_dc.X.dtype == np.object_:
            X_list = pool_dc.X.tolist()
        else:
            X_list = pool_dc.X
        X_tensor = graphdata_to_batchmolgraph(X_list)
        X_tensor = X_tensor.to(device)
    else:
        X_tensor = torch.from_numpy(pool_dc.X).float().to(device)

    latent = unified.latent(X_tensor)
    return latent.cpu().numpy()


def select_instances_by_uncertainty_clustered(
    uncertainty_scores: np.ndarray,
    latent_vectors: Optional[np.ndarray],
    n_query: int,
    method_name: str = "nn_baseline"
) -> np.ndarray:
    """
    Cluster candidates into n_query clusters by latent; in each cluster select one sample.
    Used when use_cluster is set. n_query is the same as for non-clustered selection (from query_ratio).
    
    For nn_baseline we do not cluster: return n_query random indices (latent_vectors can be None).
    For other methods we cluster by latent and pick the most uncertain sample in each cluster.
    
    Args:
        uncertainty_scores: (N,) or (N, T) - used for pool size when nn_baseline; else for uncertainty
        latent_vectors: (N, D) from UnifiedModel.latent, or None for nn_baseline
        n_query: number of samples to select (from query_ratio)
        method_name: "nn_baseline" -> random sample without clustering; else -> cluster + argmax uncertainty per cluster
        
    Returns:
        Indices of selected instances
    """
    if method_name == "nn_baseline":
        n_pool = np.asarray(uncertainty_scores).shape[0]
        n_select = min(n_query, n_pool)
        if n_select <= 0:
            return np.array([], dtype=np.int64)
        return np.random.choice(n_pool, size=n_select, replace=False).astype(np.int64)

    if uncertainty_scores.ndim > 1:
        uncertainty_scores = aggregate_multitask_uncertainty(uncertainty_scores, strategy="mean")
    uncertainty_scores = np.asarray(uncertainty_scores).flatten()
    n_pool = len(uncertainty_scores)
    n_clusters = min(n_query, n_pool)

    if n_clusters <= 0:
        return np.array([], dtype=np.int64)
    if n_clusters == 1:
        return np.array([np.argmax(uncertainty_scores)], dtype=np.int64)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=5, max_iter=50)
    labels = kmeans.fit_predict(latent_vectors)
    selected = []
    for c in range(n_clusters):
        mask = labels == c
        if not np.any(mask):
            continue
        indices_in_c = np.where(mask)[0]
        best_idx = indices_in_c[np.argmax(uncertainty_scores[indices_in_c])]
        selected.append(best_idx)
    return np.array(selected, dtype=np.int64)


def query_instances(
    pool_dc: dc.data.NumpyDataset,
    selected_indices: np.ndarray
) -> Tuple[dc.data.NumpyDataset, dc.data.NumpyDataset]:
    """
    Extract queried instances from pool and return remaining pool.
    
    Args:
        pool_dc: Pool dataset
        selected_indices: Indices to query
        
    Returns:
        Tuple of (queried_dc, remaining_pool_dc)
    """
    queried_dc = _subset_numpy_dataset(pool_dc, selected_indices)
    remaining_indices = np.setdiff1d(np.arange(len(pool_dc)), selected_indices)
    remaining_pool_dc = _subset_numpy_dataset(pool_dc, remaining_indices)
    
    return queried_dc, remaining_pool_dc


# ============================================================
# Active Learning Step Functions
# ============================================================

def evaluate_model_on_test(
    model,
    test_dc: dc.data.NumpyDataset,
    method_name: str,
    mode: str,
    use_weights: bool,
    encoder_type: str = "identity"
) -> Dict:
    """
    Evaluate model on test set and return metrics.
    
    Returns:
        Dictionary with metrics (MSE/AUC, UQ metrics)
    """
    if method_name == "nn_baseline":
        if mode == "regression":
            metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
            test_scores = model.evaluate(test_dc, [metric], use_sample_weights=use_weights, per_task_metrics=True)
            _, test_detailed = test_scores
            test_mse = test_detailed[metric.name]
            
            return {
                "MSE": test_mse,
                "empirical_coverage": None,
                "avg_pred_std": None,
                "nll": None,
                "ce": None,
                "spearman_err_unc": None,
            }
        else:
            metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
            test_results = model.evaluate(test_dc, [metric], use_sample_weights=use_weights, per_task_metrics=True)
            _, test_detailed = test_results
            test_auc = test_detailed[metric.name]
            probs = _get_probs_from_model(model, test_dc, encoder_type)
            probs_positive = probs.reshape(-1, 1) if probs.ndim == 1 else probs
            cm = compute_confusion_matrix_binary(
                test_dc.y, probs_positive, weights=test_dc.w, use_weights=use_weights
            )
            return {
                "AUC": test_auc,
                "NLL": None,
                "Brier": None,
                "ECE": None,
                "Avg_Entropy": None,
                "Spearman_Err_Unc": None,
                **cm,
            }
    
    elif method_name == "nn_deep_ensemble":
        if mode == "regression":
            ensemble = DeepEnsembleRegressor(model)
            mean_test, lower_test, upper_test = ensemble.predict_interval(test_dc)
            test_mse = mse_from_mean_prediction(mean_test, test_dc, use_weights=use_weights)
            
            uq_metrics = evaluate_uq_metrics_from_interval(
                y_true=test_dc.y,
                mean=mean_test,
                lower=lower_test,
                upper=upper_test,
                weights=test_dc.w,
                use_weights=use_weights,
                alpha=0.05,
                test_error=test_mse,
            )
            uq_metrics["MSE"] = test_mse
            return uq_metrics
        else:
            ensemble = DeepEnsembleClassifier(model)
            mean_probs, H_total, _, _ = ensemble.predict_uncertainty(test_dc)
            
            n_tasks = test_dc.y.shape[1]
            if n_tasks == 1 and mean_probs.shape[1] == 2:
                probs_positive = mean_probs[:, 1].reshape(-1, 1)
            else:
                probs_positive = mean_probs
            
            test_auc = auc_from_probs(test_dc.y, probs_positive, test_dc.w, use_weights=use_weights)
            
            uq_metrics = evaluate_uq_metrics_classification(
                y_true=test_dc.y,
                probs=probs_positive,
                auc=test_auc,
                uncertainty=H_total,
                weights=test_dc.w,
                use_weights=use_weights,
                n_bins=20
            )
            cm = compute_confusion_matrix_binary(
                test_dc.y, probs_positive, weights=test_dc.w, use_weights=use_weights
            )
            uq_metrics.update(cm)
            return uq_metrics
    
    elif method_name == "nn_mc_dropout":
        if mode == "regression":
            mc_wrapper = MCDropoutRegressorRefined(model, n_samples=100)
            mean_test, std_test = mc_wrapper.predict_uncertainty(test_dc)
            
            if mean_test.ndim == 1:
                mean_test = mean_test.reshape(-1, 1)
            if std_test.ndim == 1:
                std_test = std_test.reshape(-1, 1)
            
            test_mse = mse_from_mean_prediction(mean_test, test_dc, use_weights=use_weights)
            
            alpha = 0.05
            z = norm.ppf(1 - alpha / 2.0)
            lower = mean_test - z * std_test
            upper = mean_test + z * std_test
            
            uq_metrics = evaluate_uq_metrics_from_interval(
                y_true=test_dc.y,
                mean=mean_test,
                lower=lower,
                upper=upper,
                alpha=alpha,
                test_error=test_mse,
                weights=test_dc.w,
                use_weights=use_weights
            )
            uq_metrics["MSE"] = test_mse
            return uq_metrics
        else:
            mc_wrapper = MCDropoutClassifierWrapper(model, n_samples=50)
            mean_probs, H_total, _, _ = mc_wrapper.predict_uncertainty(test_dc)
            
            n_tasks = test_dc.y.shape[1]
            if n_tasks == 1 and mean_probs.shape[1] == 2:
                probs_positive = mean_probs[:, 1].reshape(-1, 1)
                H_total = H_total.reshape(-1, 1)
            else:
                probs_positive = mean_probs
            
            test_auc = auc_from_probs(test_dc.y, probs_positive, test_dc.w, use_weights=use_weights)
            
            uq_metrics = evaluate_uq_metrics_classification(
                y_true=test_dc.y,
                probs=probs_positive,
                auc=test_auc,
                uncertainty=H_total,
                weights=test_dc.w,
                use_weights=use_weights,
                n_bins=20
            )
            cm = compute_confusion_matrix_binary(
                test_dc.y, probs_positive, weights=test_dc.w, use_weights=use_weights
            )
            uq_metrics.update(cm)
            return uq_metrics
    
    elif method_name == "nn_evd":
        device = next(model.model.parameters()).device
        
        if encoder_type == "dmpnn":
            from nn import graphdata_to_batchmolgraph
            if isinstance(test_dc.X, np.ndarray) and test_dc.X.dtype == np.object_:
                X_list = test_dc.X.tolist()
            else:
                X_list = test_dc.X
            test_X_tensor = graphdata_to_batchmolgraph(X_list)
            test_X_tensor = test_X_tensor.to(device)
        else:
            test_X_tensor = torch.from_numpy(test_dc.X).float().to(device)
        
        with torch.no_grad():
            mu_test, params_test, aleatoric_test, epistemic_test = model.model(test_X_tensor)
        
        if mode == "regression":
            total_var_test = (aleatoric_test + epistemic_test).cpu().numpy()
            std_test = np.sqrt(total_var_test)
            mu_test = mu_test.cpu().numpy()
            
            if mu_test.ndim == 1:
                mu_test = mu_test.reshape(-1, 1)
            if std_test.ndim == 1:
                std_test = std_test.reshape(-1, 1)
            
            test_mse = mse_from_mean_prediction(mu_test, test_dc, use_weights=use_weights)
            
            alpha = 0.05
            z = norm.ppf(1 - alpha / 2.0)
            lower = mu_test - z * std_test
            upper = mu_test + z * std_test
            
            uq_metrics = evaluate_uq_metrics_from_interval(
                y_true=test_dc.y,
                mean=mu_test,
                lower=lower,
                upper=upper,
                alpha=alpha,
                test_error=test_mse,
                weights=test_dc.w,
                use_weights=use_weights
            )
            uq_metrics["MSE"] = test_mse
            return uq_metrics
        else:
            if isinstance(mu_test, torch.Tensor):
                mu_test = mu_test.cpu().numpy()
            
            if mu_test.ndim > 1 and mu_test.shape[1] > 1:
                mu_test = mu_test[:, 1::2]
            
            if mu_test.ndim == 1:
                mu_test = mu_test.reshape(-1, 1)
            
            total_var_test = (aleatoric_test + epistemic_test).cpu().numpy()
            test_auc = auc_from_probs(test_dc.y, mu_test, test_dc.w, use_weights=use_weights)
            
            uq_metrics = evaluate_uq_metrics_classification(
                y_true=test_dc.y,
                probs=mu_test,
                auc=test_auc,
                uncertainty=total_var_test,
                weights=test_dc.w,
                use_weights=use_weights,
                n_bins=20
            )
            cm = compute_confusion_matrix_binary(
                test_dc.y, mu_test, weights=test_dc.w, use_weights=use_weights
            )
            uq_metrics.update(cm)
            return uq_metrics
    
    elif method_name == "nn_conformal":
        if mode != "classification":
            raise ValueError("nn_conformal is only supported for classification")
        probs = _get_probs_from_model(model, test_dc, encoder_type)
        probs_positive = probs.reshape(-1, 1) if probs.ndim == 1 else probs
        test_auc = auc_from_probs(test_dc.y, probs_positive, test_dc.w, use_weights=use_weights)
        cm = compute_confusion_matrix_binary(
            test_dc.y, probs_positive, weights=test_dc.w, use_weights=use_weights
        )
        return {
            "AUC": test_auc,
            "NLL": None,
            "Brier": None,
            "ECE": None,
            "Avg_Entropy": None,
            "Spearman_Err_Unc": None,
            **cm,
        }
    
    else:
        raise ValueError(f"Unknown method: {method_name}")


# ============================================================
# Method-Specific Active Learning Functions
# ============================================================

def train_nn_baseline_active_learning(
    initial_train_dc: dc.data.NumpyDataset,
    pool_dc: dc.data.NumpyDataset,
    test_dc: dc.data.NumpyDataset,
    n_steps: int = 10,
    query_ratio: float = 0.05,
    use_cluster: bool = False,
    run_id: int = 0,
    use_weights: bool = False,
    mode: str = "regression",
    encoder_type: str = "identity"
) -> List[Dict]:
    """
    Active learning for nn_baseline method.
    
    Returns:
        List of dictionaries, one per step, containing metrics
    """
    n_tasks = initial_train_dc.y.shape[1] if initial_train_dc.y.ndim > 1 else 1
    n_features = get_n_features(initial_train_dc, encoder_type=encoder_type)
    
    # Set epochs based on encoder type
    if encoder_type == "dmpnn":
        initial_epochs = 50
        fine_tune_epochs = 30
    else:
        initial_epochs = 100
        fine_tune_epochs = 50
    
    # Build and train initial model
    model = build_model("baseline", n_features, n_tasks, mode=mode, encoder_type=encoder_type)
    
    if mode == "regression":
        loss = dc.models.losses.L2Loss()
        dc_model = UnifiedTorchModel(
            model=model,
            loss=loss,
            output_types=['prediction'],
            batch_size=64,
            learning_rate=1e-3,
            mode='regression',
            encoder_type=encoder_type,
        )
    else:
        loss = dc.models.losses.SigmoidCrossEntropy()
        dc_model = UnifiedTorchModel(
            model=model,
            loss=loss,
            output_types=['prediction', 'loss'],
            batch_size=64,
            learning_rate=1e-3,
            mode='classification',
            encoder_type=encoder_type,
        )
    
    # Initial training
    print(f"[AL nn_baseline] Initial training with {len(initial_train_dc)} samples ({initial_epochs} epochs)")
    dc_model.fit(initial_train_dc, nb_epoch=initial_epochs)
    
    # Evaluate step 0
    current_train_dc = initial_train_dc
    step_results = []
    
    step_0_metrics = evaluate_model_on_test(dc_model, test_dc, "nn_baseline", mode, use_weights, encoder_type)
    step_0_metrics["step"] = 0
    step_0_metrics["train_size"] = len(current_train_dc)
    step_results.append(step_0_metrics)
    
    # Active learning loop
    for step in range(1, n_steps + 1):
        print(f"[AL nn_baseline] Step {step}/{n_steps}, Pool size: {len(pool_dc)}")
        
        # Extract uncertainty (baseline has minimal uncertainty, but we'll still query)
        uncertainty = extract_uncertainty(dc_model, pool_dc, "nn_baseline", mode, encoder_type)
        
        # Select instances: when use_cluster, nn_baseline returns random indices (no cluster); else by uncertainty only
        if use_cluster:
            n_query = max(1, int(len(pool_dc) * query_ratio))
            selected_indices = select_instances_by_uncertainty_clustered(uncertainty, None, n_query, method_name="nn_baseline")
        else:
            selected_indices = select_instances_by_uncertainty(pool_dc, uncertainty, query_ratio)
        
        # Query instances
        queried_dc, pool_dc = query_instances(pool_dc, selected_indices)
        
        # Update training set
        current_train_dc = combine_datasets(current_train_dc, queried_dc)
        
        # Fine-tune model
        print(f"[AL nn_baseline] Fine-tuning with {len(current_train_dc)} samples ({fine_tune_epochs} epochs)")
        dc_model.fit(current_train_dc, nb_epoch=fine_tune_epochs)
        
        # Evaluate
        step_metrics = evaluate_model_on_test(dc_model, test_dc, "nn_baseline", mode, use_weights, encoder_type)
        step_metrics["step"] = step
        step_metrics["train_size"] = len(current_train_dc)
        step_results.append(step_metrics)
    
    return step_results


def train_nn_deep_ensemble_active_learning(
    initial_train_dc: dc.data.NumpyDataset,
    pool_dc: dc.data.NumpyDataset,
    test_dc: dc.data.NumpyDataset,
    n_steps: int = 10,
    query_ratio: float = 0.05,
    use_cluster: bool = False,
    M: int = 5,
    run_id: int = 0,
    use_weights: bool = False,
    mode: str = "regression",
    encoder_type: str = "identity"
) -> List[Dict]:
    """Active learning for nn_deep_ensemble method."""
    n_tasks = initial_train_dc.y.shape[1] if initial_train_dc.y.ndim > 1 else 1
    n_features = get_n_features(initial_train_dc, encoder_type=encoder_type)
    
    # Set epochs based on encoder type
    if encoder_type == "dmpnn":
        initial_epochs = 50
        fine_tune_epochs = 30
    else:
        initial_epochs = 100
        fine_tune_epochs = 50
    
    # Build and train initial ensemble
    models = []
    for i in range(M):
        ensemble_seed = run_id * 1000 + i
        import random
        random.seed(ensemble_seed)
        np.random.seed(ensemble_seed)
        torch.manual_seed(ensemble_seed)
        
        model = build_model("baseline", n_features, n_tasks, mode=mode, encoder_type=encoder_type)
        
        if mode == "regression":
            loss = dc.models.losses.L2Loss()
            dc_model = UnifiedTorchModel(
                model=model,
                loss=loss,
                output_types=['prediction'],
                batch_size=64,
                learning_rate=1e-3,
                mode=mode,
                encoder_type=encoder_type,
            )
        else:
            loss = dc.models.losses.SigmoidCrossEntropy()
            dc_model = UnifiedTorchModel(
                model=model,
                loss=loss,
                output_types=['prediction', 'loss'],
                batch_size=64,
                learning_rate=1e-3,
                mode=mode,
                encoder_type=encoder_type,
            )
        
        dc_model.fit(initial_train_dc, nb_epoch=initial_epochs)
        models.append(dc_model)
    
    print(f"[AL nn_deep_ensemble] Initial training with {len(initial_train_dc)} samples ({initial_epochs} epochs)")
    
    # Evaluate step 0
    current_train_dc = initial_train_dc
    step_results = []
    
    step_0_metrics = evaluate_model_on_test(models, test_dc, "nn_deep_ensemble", mode, use_weights, encoder_type)
    step_0_metrics["step"] = 0
    step_0_metrics["train_size"] = len(current_train_dc)
    step_results.append(step_0_metrics)
    
    # Active learning loop
    for step in range(1, n_steps + 1):
        print(f"[AL nn_deep_ensemble] Step {step}/{n_steps}, Pool size: {len(pool_dc)}")
        
        # Extract uncertainty
        uncertainty = extract_uncertainty(models, pool_dc, "nn_deep_ensemble", mode, encoder_type)
        
        # Select instances: clustered by latent when use_cluster, else by uncertainty only
        if use_cluster:
            n_query = max(1, int(len(pool_dc) * query_ratio))
            latent_vectors = extract_latent_vectors(models, pool_dc, "nn_deep_ensemble", encoder_type)
            selected_indices = select_instances_by_uncertainty_clustered(uncertainty, latent_vectors, n_query, method_name="nn_deep_ensemble")
        else:
            selected_indices = select_instances_by_uncertainty(pool_dc, uncertainty, query_ratio)
        
        # Query instances
        queried_dc, pool_dc = query_instances(pool_dc, selected_indices)
        
        # Update training set
        current_train_dc = combine_datasets(current_train_dc, queried_dc)
        
        # Fine-tune all ensemble members
        print(f"[AL nn_deep_ensemble] Fine-tuning with {len(current_train_dc)} samples ({fine_tune_epochs} epochs each)")
        for model in models:
            model.fit(current_train_dc, nb_epoch=fine_tune_epochs)
        
        # Evaluate
        step_metrics = evaluate_model_on_test(models, test_dc, "nn_deep_ensemble", mode, use_weights, encoder_type)
        step_metrics["step"] = step
        step_metrics["train_size"] = len(current_train_dc)
        step_results.append(step_metrics)
    
    return step_results


def train_nn_mc_dropout_active_learning(
    initial_train_dc: dc.data.NumpyDataset,
    pool_dc: dc.data.NumpyDataset,
    test_dc: dc.data.NumpyDataset,
    n_steps: int = 10,
    query_ratio: float = 0.05,
    use_cluster: bool = False,
    run_id: int = 0,
    use_weights: bool = False,
    mode: str = "regression",
    encoder_type: str = "identity"
) -> List[Dict]:
    """Active learning for nn_mc_dropout method."""
    n_tasks = initial_train_dc.y.shape[1] if initial_train_dc.y.ndim > 1 else 1
    n_features = get_n_features(initial_train_dc, encoder_type=encoder_type)
    
    # Set epochs based on encoder type
    if encoder_type == "dmpnn":
        initial_epochs = 50
        fine_tune_epochs = 30
    else:
        initial_epochs = 100
        fine_tune_epochs = 50
    
    # Build and train initial model
    if mode == "regression":
        model = build_model("mc_dropout", n_features, n_tasks, mode=mode, dropout_rate=0.1, encoder_type=encoder_type)
        loss = HeteroscedasticL2Loss()
        
        if encoder_type == "dmpnn":
            dc_model = UnifiedTorchModel(
                model=model,
                loss=loss,
                output_types=['prediction', 'variance', 'loss'],
                batch_size=64,
                learning_rate=1e-3,
                mode='regression',
                encoder_type=encoder_type
            )
        else:
            dc_model = dc.models.TorchModel(
                model=model,
                loss=loss,
                output_types=['prediction', 'variance', 'loss'],
                batch_size=64,
                learning_rate=1e-3,
                mode='regression'
            )
    else:
        y = np.asarray(initial_train_dc.y)
        if y.ndim == 2 and y.shape[1] > 1 and set(np.unique(y)).issubset({0, 1}):
            n_classes = y.shape[1]
        else:
            n_classes = 2
        
        model = build_model("mc_dropout", n_features, n_tasks, mode="classification", dropout_rate=0.2, encoder_type=encoder_type)
        loss = HeteroscedasticClassificationLoss(n_samples=20)
        
        if encoder_type == "dmpnn":
            dc_model = UnifiedTorchModel(
                model=model,
                loss=loss,
                output_types=['prediction'],
                batch_size=64,
                learning_rate=1e-3,
                mode='classification',
                encoder_type=encoder_type
            )
        else:
            dc_model = dc.models.TorchModel(
                model=model,
                loss=loss,
                output_types=['prediction'],
                batch_size=64,
                learning_rate=1e-3,
                mode='classification'
            )
    
    print(f"[AL nn_mc_dropout] Initial training with {len(initial_train_dc)} samples ({initial_epochs} epochs)")
    dc_model.fit(initial_train_dc, nb_epoch=initial_epochs)
    
    # Evaluate step 0
    current_train_dc = initial_train_dc
    step_results = []
    
    step_0_metrics = evaluate_model_on_test(dc_model, test_dc, "nn_mc_dropout", mode, use_weights, encoder_type)
    step_0_metrics["step"] = 0
    step_0_metrics["train_size"] = len(current_train_dc)
    step_results.append(step_0_metrics)
    
    # Active learning loop
    for step in range(1, n_steps + 1):
        print(f"[AL nn_mc_dropout] Step {step}/{n_steps}, Pool size: {len(pool_dc)}")
        
        # Extract uncertainty
        uncertainty = extract_uncertainty(dc_model, pool_dc, "nn_mc_dropout", mode, encoder_type)
        
        # Select instances: clustered by latent when use_cluster, else by uncertainty only
        if use_cluster:
            n_query = max(1, int(len(pool_dc) * query_ratio))
            latent_vectors = extract_latent_vectors(dc_model, pool_dc, "nn_mc_dropout", encoder_type)
            selected_indices = select_instances_by_uncertainty_clustered(uncertainty, latent_vectors, n_query, method_name="nn_mc_dropout")
        else:
            selected_indices = select_instances_by_uncertainty(pool_dc, uncertainty, query_ratio)
        
        # Query instances
        queried_dc, pool_dc = query_instances(pool_dc, selected_indices)
        
        # Update training set
        current_train_dc = combine_datasets(current_train_dc, queried_dc)
        
        # Fine-tune model
        print(f"[AL nn_mc_dropout] Fine-tuning with {len(current_train_dc)} samples ({fine_tune_epochs} epochs)")
        dc_model.fit(current_train_dc, nb_epoch=fine_tune_epochs)
        
        # Evaluate
        step_metrics = evaluate_model_on_test(dc_model, test_dc, "nn_mc_dropout", mode, use_weights, encoder_type)
        step_metrics["step"] = step
        step_metrics["train_size"] = len(current_train_dc)
        step_results.append(step_metrics)
    
    return step_results


def train_evd_baseline_active_learning(
    initial_train_dc: dc.data.NumpyDataset,
    pool_dc: dc.data.NumpyDataset,
    test_dc: dc.data.NumpyDataset,
    n_steps: int = 10,
    query_ratio: float = 0.05,
    use_cluster: bool = False,
    reg_coeff: float = 1.0,
    run_id: int = 0,
    use_weights: bool = False,
    mode: str = "regression",
    encoder_type: str = "identity"
) -> List[Dict]:
    """Active learning for nn_evd method."""
    n_tasks = initial_train_dc.y.shape[1] if initial_train_dc.y.ndim > 1 else 1
    n_features = get_n_features(initial_train_dc, encoder_type=encoder_type)
    
    # Set epochs based on encoder type
    if encoder_type == "dmpnn":
        initial_epochs = 50
        fine_tune_epochs = 30
    else:
        initial_epochs = 100
        fine_tune_epochs = 50
    
    # Build and train initial model
    if mode == "regression":
        model = build_model("evidential", n_features, n_tasks, mode="regression", encoder_type=encoder_type)
        loss = EvidentialRegressionLoss(reg_coeff=reg_coeff)
    else:
        model = build_model("evidential", n_features, n_tasks, mode="classification", encoder_type=encoder_type)
        loss = EvidentialClassificationLoss()
    
    gradientClip = GradientClippingCallback()
    
    dc_model = UnifiedTorchModel(
        model=model,
        loss=loss,
        output_types=['prediction', 'loss', 'var1', 'var2'],
        batch_size=128,
        learning_rate=1e-4,
        log_frequency=40,
        mode=mode,
        encoder_type=encoder_type
    )
    
    print(f"[AL nn_evd] Initial training with {len(initial_train_dc)} samples ({initial_epochs} epochs)")
    dc_model.fit(initial_train_dc, nb_epoch=initial_epochs, callbacks=[gradientClip])
    
    # Evaluate step 0
    current_train_dc = initial_train_dc
    step_results = []
    
    step_0_metrics = evaluate_model_on_test(dc_model, test_dc, "nn_evd", mode, use_weights, encoder_type)
    step_0_metrics["step"] = 0
    step_0_metrics["train_size"] = len(current_train_dc)
    step_results.append(step_0_metrics)
    
    # Active learning loop
    for step in range(1, n_steps + 1):
        print(f"[AL nn_evd] Step {step}/{n_steps}, Pool size: {len(pool_dc)}")
        
        # Extract uncertainty
        uncertainty = extract_uncertainty(dc_model, pool_dc, "nn_evd", mode, encoder_type)
        
        # Select instances: clustered by latent when use_cluster, else by uncertainty only
        if use_cluster:
            n_query = max(1, int(len(pool_dc) * query_ratio))
            latent_vectors = extract_latent_vectors(dc_model, pool_dc, "nn_evd", encoder_type)
            selected_indices = select_instances_by_uncertainty_clustered(uncertainty, latent_vectors, n_query, method_name="nn_evd")
        else:
            selected_indices = select_instances_by_uncertainty(pool_dc, uncertainty, query_ratio)
        
        # Query instances
        queried_dc, pool_dc = query_instances(pool_dc, selected_indices)
        
        # Update training set
        current_train_dc = combine_datasets(current_train_dc, queried_dc)
        
        # Fine-tune model
        print(f"[AL nn_evd] Fine-tuning with {len(current_train_dc)} samples ({fine_tune_epochs} epochs)")
        dc_model.fit(current_train_dc, nb_epoch=fine_tune_epochs, callbacks=[gradientClip])
        
        # Evaluate
        step_metrics = evaluate_model_on_test(dc_model, test_dc, "nn_evd", mode, use_weights, encoder_type)
        step_metrics["step"] = step
        step_metrics["train_size"] = len(current_train_dc)
        step_results.append(step_metrics)
    
    return step_results


def train_conformal_active_learning(
    initial_train_dc: dc.data.NumpyDataset,
    pool_dc: dc.data.NumpyDataset,
    valid_dc: dc.data.NumpyDataset,
    test_dc: dc.data.NumpyDataset,
    n_steps: int = 10,
    query_ratio: float = 0.05,
    use_cluster: bool = False,
    tie_break: str = "mean_nc",
    run_id: int = 0,
    use_weights: bool = False,
    encoder_type: str = "identity"
) -> List[Dict]:
    """Active learning for nn_conformal (binary classification only). Uses valid_dc for conformal calibration each step."""
    n_tasks = initial_train_dc.y.shape[1] if initial_train_dc.y.ndim > 1 else 1
    n_features = get_n_features(initial_train_dc, encoder_type=encoder_type)
    if encoder_type == "dmpnn":
        initial_epochs = 50
        fine_tune_epochs = 30
    else:
        initial_epochs = 100
        fine_tune_epochs = 50
    model = build_model("baseline", n_features, n_tasks, mode="classification", encoder_type=encoder_type)
    loss = dc.models.losses.SigmoidCrossEntropy()
    dc_model = UnifiedTorchModel(
        model=model,
        loss=loss,
        output_types=['prediction', 'loss'],
        batch_size=64,
        learning_rate=1e-3,
        mode='classification',
        encoder_type=encoder_type,
    )
    print(f"[AL nn_conformal] Initial training with {len(initial_train_dc)} samples ({initial_epochs} epochs)")
    dc_model.fit(initial_train_dc, nb_epoch=initial_epochs)
    current_train_dc = initial_train_dc
    step_results = []
    thresholds = None
    step_0_metrics = evaluate_model_on_test(dc_model, test_dc, "nn_conformal", "classification", use_weights, encoder_type)
    step_0_metrics["step"] = 0
    step_0_metrics["train_size"] = len(current_train_dc)
    step_results.append(step_0_metrics)
    for step in range(1, n_steps + 1):
        print(f"[AL nn_conformal] Step {step}/{n_steps}, Pool size: {len(pool_dc)}")
        thresholds = get_conformal_thresholds(dc_model, valid_dc, alpha=0.05, encoder_type=encoder_type)
        n_query = max(1, int(len(pool_dc) * query_ratio))
        latent_vectors = extract_latent_vectors(dc_model, pool_dc, "nn_conformal", encoder_type) if use_cluster else None
        selected_indices = select_instances_conformal(
            dc_model, pool_dc, valid_dc, thresholds=thresholds, n_query=n_query,
            encoder_type=encoder_type, tie_break=tie_break, use_cluster=use_cluster,
            latent_vectors=latent_vectors, random_state=run_id + step,
        )
        queried_dc, pool_dc = query_instances(pool_dc, selected_indices)
        current_train_dc = combine_datasets(current_train_dc, queried_dc)
        print(f"[AL nn_conformal] Fine-tuning with {len(current_train_dc)} samples ({fine_tune_epochs} epochs)")
        dc_model.fit(current_train_dc, nb_epoch=fine_tune_epochs)
        step_metrics = evaluate_model_on_test(dc_model, test_dc, "nn_conformal", "classification", use_weights, encoder_type)
        step_metrics["step"] = step
        step_metrics["train_size"] = len(current_train_dc)
        step_results.append(step_metrics)
    return step_results


# ============================================================
# Main Active Learning Orchestrator
# ============================================================

def run_active_learning_nn(
    dataset_name: str,
    seed: int = 0,
    run_id: int = 0,
    split: str = "random",
    mode: str = "regression",
    use_weights: bool = False,
    task_indices: Optional[List[int]] = None,
    encoder_type: str = "identity",
    use_graph: bool = False,
    initial_ratio: Optional[float] = None,
    query_ratio: float = 0.05,
    use_cluster: bool = False,
    n_steps: int = 10,
    conformal_tie_break: str = "mean_nc",
) -> Dict[str, List[Dict]]:
    """
    Main active learning orchestrator.
    
    Returns:
        Dictionary mapping method names to lists of step results
    """
    from main import load_dataset, set_global_seed
    
    set_global_seed(seed)
    
    # Load dataset
    tasks, train_dc, valid_dc, test_dc, transformers = load_dataset(
        dataset_name=dataset_name,
        split=split,
        task_indices=task_indices,
        use_graph=use_graph
    )
    
    # Set initial ratio based on dataset
    # clintox uses 0.2, all others use 0.1
    if dataset_name == "clintox":
        actual_initial_ratio = 0.2
    else:
        actual_initial_ratio = 0.1
    
    # Override with provided initial_ratio if explicitly set (for flexibility)
    if initial_ratio is not None:
        actual_initial_ratio = initial_ratio
    
    # Create initial split (ignore validation dataset)
    initial_train_dc, pool_dc = create_stratified_initial_split(
        train_dc,
        initial_ratio=actual_initial_ratio,
        mode=mode,
        random_state=seed
    )
    
    print(f"\n=== Active Learning Run {run_id} on {dataset_name} ===")
    print(f"Initial training size: {len(initial_train_dc)}")
    print(f"Pool size: {len(pool_dc)}")
    print(f"Test size: {len(test_dc)}")
    
    # Determine use_weights
    if dataset_name in ["tox21", "toxcast", "sider", "clintox"]:
        use_weights = True
    
    # Run active learning for each method (nn_conformal only when classification)
    all_results = {}
    # methods = ["nn_baseline", "nn_deep_ensemble", "nn_mc_dropout", "nn_evd"]
    # if mode == "classification":
    #     methods = methods + ["nn_conformal"]
    
    methods = ["nn_conformal"]

    for method in methods:
        print(f"\n--- Running Active Learning for {method} ---")
        
        try:
            if method == "nn_baseline":
                step_results = train_nn_baseline_active_learning(
                    initial_train_dc, copy_numpy_dataset(pool_dc), test_dc,
                    n_steps=n_steps, query_ratio=query_ratio, use_cluster=use_cluster,
                    run_id=run_id, use_weights=use_weights,
                    mode=mode, encoder_type=encoder_type
                )
            elif method == "nn_deep_ensemble":
                step_results = train_nn_deep_ensemble_active_learning(
                    initial_train_dc, copy_numpy_dataset(pool_dc), test_dc,
                    n_steps=n_steps, query_ratio=query_ratio, use_cluster=use_cluster,
                    M=5, run_id=run_id, use_weights=use_weights,
                    mode=mode, encoder_type=encoder_type
                )
            elif method == "nn_mc_dropout":
                step_results = train_nn_mc_dropout_active_learning(
                    initial_train_dc, copy_numpy_dataset(pool_dc), test_dc,
                    n_steps=n_steps, query_ratio=query_ratio, use_cluster=use_cluster,
                    run_id=run_id, use_weights=use_weights,
                    mode=mode, encoder_type=encoder_type
                )
            elif method == "nn_evd":
                step_results = train_evd_baseline_active_learning(
                    initial_train_dc, copy_numpy_dataset(pool_dc), test_dc,
                    n_steps=n_steps, query_ratio=query_ratio, use_cluster=use_cluster,
                    reg_coeff=1.0, run_id=run_id, use_weights=use_weights,
                    mode=mode, encoder_type=encoder_type
                )
            elif method == "nn_conformal":
                step_results = train_conformal_active_learning(
                    initial_train_dc, copy_numpy_dataset(pool_dc), valid_dc, test_dc,
                    n_steps=n_steps, query_ratio=query_ratio, use_cluster=use_cluster,
                    tie_break=conformal_tie_break, run_id=run_id, use_weights=use_weights,
                    encoder_type=encoder_type
                )
            else:
                continue
            
            # When use_cluster (al_batch), store under method_batch
            storage_name = f"{method}_batch" if use_cluster else method
            all_results[storage_name] = step_results
            
            # Save results to CSV (with task splitting)
            save_active_learning_results(
                storage_name, step_results, dataset_name, mode, run_id, split,
                task_indices, encoder_type, use_graph
            )
            
        except Exception as e:
            print(f"Error in {method}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results


def save_active_learning_results(
    method_name: str,
    step_results: List[Dict],
    dataset_name: str,
    mode: str,
    run_id: int,
    split: str,
    task_indices: Optional[List[int]] = None,
    encoder_type: str = "identity",
    use_graph: bool = False
):
    """
    Save active learning results to CSV, splitting multitask results per task.
    Matches naming convention from main.py.
    """
    import os
    
    # Create output directory
    output_dir = f"./cdata_{mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create split name
    split_name = f"{split}_graph" if use_graph else split
    
    # Create task suffix (matches main.py format)
    if task_indices is None:
        global_task_suffix = ""
    else:
        global_task_suffix = "_tasks_" + "_".join(map(str, task_indices))
    
    # Check if results are multitask by examining first step
    if not step_results:
        return
    
    first_step = step_results[0]
    # Find first non-None value to check if multitask
    first_val = None
    for v in first_step.values():
        if v is not None and v != "step" and v != "train_size":
            first_val = v
            break
    
    is_multitask = isinstance(first_val, (list, np.ndarray)) and np.size(first_val) > 1
    
    if not is_multitask:
        # Single task: save one file with task_id=0 (matches main.py format)
        df = pd.DataFrame(step_results)
        
        # Create filename matching main.py format: AL_{method}_{split}_{dataset}{task_suffix}_id_0_run_{run_id}.csv
        # Always include _id_0 for single task (matches main.py behavior)
        final_suffix = f"{global_task_suffix}_id_0"
        filename = f"{output_dir}/AL_{method_name}_{split_name}_{dataset_name}{final_suffix}_run_{run_id}.csv"
        
        df.to_csv(filename, index=False)
        print(f"Saved results to {filename}")
    else:
        # Multitask: split results per task and save separate files (matches main.py format)
        n_tasks_detected = len(first_val)
        
        for t in range(n_tasks_detected):
            # Map index t to real task ID (matches main.py logic)
            if task_indices is not None and len(task_indices) > t:
                real_task_id = task_indices[t]
            else:
                real_task_id = t
            
            # Extract per-task results
            task_step_results = []
            for step_dict in step_results:
                task_dict = {}
                for k, v in step_dict.items():
                    if k in ["step", "train_size"]:
                        task_dict[k] = v
                    elif v is not None:
                        # Extract value for this task
                        if isinstance(v, (list, np.ndarray)) and len(v) > t:
                            task_dict[k] = v[t]
                        else:
                            task_dict[k] = v
                    else:
                        task_dict[k] = v
                task_step_results.append(task_dict)
            
            # Create DataFrame for this task
            df = pd.DataFrame(task_step_results)
            
            # Create filename matching main.py format: AL_{method}_{split}_{dataset}{task_suffix}_id_{task_id}_run_{run_id}.csv
            final_suffix = f"{global_task_suffix}_id_{real_task_id}"
            filename = f"{output_dir}/AL_{method_name}_{split_name}_{dataset_name}{final_suffix}_run_{run_id}.csv"
            
            df.to_csv(filename, index=False)
            print(f"Saved results for task {real_task_id} to {filename}")
