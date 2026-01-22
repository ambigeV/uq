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
    auc_from_probs
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
            _, H_total, _, _ = ensemble.predict_uncertainty(dataset)
            # H_total is (N, T) for multitask
            return H_total
    
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
            _, entropy = mc_wrapper.predict_uncertainty(dataset)
            # entropy is (N, 1) or (N, T)
            if entropy.ndim == 2 and entropy.shape[1] == 1:
                entropy = entropy.squeeze(1)
            return entropy
    
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
    
    # Calculate number to query
    n_pool = len(pool_dc)
    n_query = max(1, int(n_pool * query_ratio))
    
    # Select top-k most uncertain
    top_indices = np.argsort(uncertainty_scores)[::-1][:n_query]
    
    return top_indices


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
            
            return {
                "AUC": test_auc,
                "NLL": None,
                "Brier": None,
                "ECE": None,
                "Avg_Entropy": None,
                "Spearman_Err_Unc": None,
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
            mean_probs, entropy = mc_wrapper.predict_uncertainty(test_dc)
            
            n_tasks = test_dc.y.shape[1]
            if n_tasks == 1 and mean_probs.shape[1] == 2:
                probs_positive = mean_probs[:, 1].reshape(-1, 1)
                entropy = entropy.reshape(-1, 1)
            else:
                probs_positive = mean_probs
            
            test_auc = auc_from_probs(test_dc.y, probs_positive, test_dc.w, use_weights=use_weights)
            
            uq_metrics = evaluate_uq_metrics_classification(
                y_true=test_dc.y,
                probs=probs_positive,
                auc=test_auc,
                uncertainty=entropy,
                weights=test_dc.w,
                use_weights=use_weights,
                n_bins=20
            )
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
            return uq_metrics
    
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
    
    # Initial training (50 epochs)
    print(f"[AL nn_baseline] Initial training with {len(initial_train_dc)} samples")
    dc_model.fit(initial_train_dc, nb_epoch=50)
    
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
        
        # Select instances
        selected_indices = select_instances_by_uncertainty(pool_dc, uncertainty, query_ratio)
        
        # Query instances
        queried_dc, pool_dc = query_instances(pool_dc, selected_indices)
        
        # Update training set
        current_train_dc = combine_datasets(current_train_dc, queried_dc)
        
        # Fine-tune model (30 epochs)
        print(f"[AL nn_baseline] Fine-tuning with {len(current_train_dc)} samples")
        dc_model.fit(current_train_dc, nb_epoch=20)
        
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
    M: int = 5,
    run_id: int = 0,
    use_weights: bool = False,
    mode: str = "regression",
    encoder_type: str = "identity"
) -> List[Dict]:
    """Active learning for nn_deep_ensemble method."""
    n_tasks = initial_train_dc.y.shape[1] if initial_train_dc.y.ndim > 1 else 1
    n_features = get_n_features(initial_train_dc, encoder_type=encoder_type)
    
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
        
        dc_model.fit(initial_train_dc, nb_epoch=50)
        models.append(dc_model)
    
    print(f"[AL nn_deep_ensemble] Initial training with {len(initial_train_dc)} samples")
    
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
        
        # Select instances
        selected_indices = select_instances_by_uncertainty(pool_dc, uncertainty, query_ratio)
        
        # Query instances
        queried_dc, pool_dc = query_instances(pool_dc, selected_indices)
        
        # Update training set
        current_train_dc = combine_datasets(current_train_dc, queried_dc)
        
        # Fine-tune all ensemble members (20 epochs each)
        print(f"[AL nn_deep_ensemble] Fine-tuning with {len(current_train_dc)} samples")
        for model in models:
            model.fit(current_train_dc, nb_epoch=20)
        
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
    run_id: int = 0,
    use_weights: bool = False,
    mode: str = "regression",
    encoder_type: str = "identity"
) -> List[Dict]:
    """Active learning for nn_mc_dropout method."""
    n_tasks = initial_train_dc.y.shape[1] if initial_train_dc.y.ndim > 1 else 1
    n_features = get_n_features(initial_train_dc, encoder_type=encoder_type)
    
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
    
    print(f"[AL nn_mc_dropout] Initial training with {len(initial_train_dc)} samples")
    dc_model.fit(initial_train_dc, nb_epoch=50)
    
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
        
        # Select instances
        selected_indices = select_instances_by_uncertainty(pool_dc, uncertainty, query_ratio)
        
        # Query instances
        queried_dc, pool_dc = query_instances(pool_dc, selected_indices)
        
        # Update training set
        current_train_dc = combine_datasets(current_train_dc, queried_dc)
        
        # Fine-tune model (30 epochs)
        print(f"[AL nn_mc_dropout] Fine-tuning with {len(current_train_dc)} samples")
        dc_model.fit(current_train_dc, nb_epoch=20)
        
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
    reg_coeff: float = 1.0,
    run_id: int = 0,
    use_weights: bool = False,
    mode: str = "regression",
    encoder_type: str = "identity"
) -> List[Dict]:
    """Active learning for nn_evd method."""
    n_tasks = initial_train_dc.y.shape[1] if initial_train_dc.y.ndim > 1 else 1
    n_features = get_n_features(initial_train_dc, encoder_type=encoder_type)
    
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
    
    print(f"[AL nn_evd] Initial training with {len(initial_train_dc)} samples")
    dc_model.fit(initial_train_dc, nb_epoch=50, callbacks=[gradientClip])
    
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
        
        # Select instances
        selected_indices = select_instances_by_uncertainty(pool_dc, uncertainty, query_ratio)
        
        # Query instances
        queried_dc, pool_dc = query_instances(pool_dc, selected_indices)
        
        # Update training set
        current_train_dc = combine_datasets(current_train_dc, queried_dc)
        
        # Fine-tune model (30 epochs)
        print(f"[AL nn_evd] Fine-tuning with {len(current_train_dc)} samples")
        dc_model.fit(current_train_dc, nb_epoch=20, callbacks=[gradientClip])
        
        # Evaluate
        step_metrics = evaluate_model_on_test(dc_model, test_dc, "nn_evd", mode, use_weights, encoder_type)
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
    initial_ratio: float = 0.1,
    query_ratio: float = 0.05,
    n_steps: int = 10
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
    
    # Create initial split (ignore validation dataset)
    initial_train_dc, pool_dc = create_stratified_initial_split(
        train_dc,
        initial_ratio=initial_ratio,
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
    
    # Run active learning for each method
    all_results = {}
    methods = ["nn_baseline", "nn_deep_ensemble", "nn_mc_dropout", "nn_evd"]
    
    for method in methods:
        print(f"\n--- Running Active Learning for {method} ---")
        
        try:
            if method == "nn_baseline":
                step_results = train_nn_baseline_active_learning(
                    initial_train_dc, copy_numpy_dataset(pool_dc), test_dc,
                    n_steps=n_steps, query_ratio=query_ratio,
                    run_id=run_id, use_weights=use_weights,
                    mode=mode, encoder_type=encoder_type
                )
            elif method == "nn_deep_ensemble":
                step_results = train_nn_deep_ensemble_active_learning(
                    initial_train_dc, copy_numpy_dataset(pool_dc), test_dc,
                    n_steps=n_steps, query_ratio=query_ratio,
                    M=5, run_id=run_id, use_weights=use_weights,
                    mode=mode, encoder_type=encoder_type
                )
            elif method == "nn_mc_dropout":
                step_results = train_nn_mc_dropout_active_learning(
                    initial_train_dc, copy_numpy_dataset(pool_dc), test_dc,
                    n_steps=n_steps, query_ratio=query_ratio,
                    run_id=run_id, use_weights=use_weights,
                    mode=mode, encoder_type=encoder_type
                )
            elif method == "nn_evd":
                step_results = train_evd_baseline_active_learning(
                    initial_train_dc, copy_numpy_dataset(pool_dc), test_dc,
                    n_steps=n_steps, query_ratio=query_ratio,
                    reg_coeff=1.0, run_id=run_id, use_weights=use_weights,
                    mode=mode, encoder_type=encoder_type
                )
            
            all_results[method] = step_results
            
            # Save results to CSV
            save_active_learning_results(
                method, step_results, dataset_name, mode, run_id, split, encoder_type, use_graph
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
    encoder_type: str = "identity",
    use_graph: bool = False
):
    """Save active learning results to CSV."""
    import os
    
    # Create DataFrame from step results
    df = pd.DataFrame(step_results)
    
    # Create output directory
    output_dir = f"./cdata_{mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    split_name = f"{split}_graph" if use_graph else split
    filename = f"{output_dir}/AL_{method_name}_{split_name}_{dataset_name}_run_{run_id}.csv"
    
    # Save
    df.to_csv(filename, index=False)
    print(f"Saved results to {filename}")
