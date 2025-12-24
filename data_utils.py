# data_utils.py
import deepchem as dc
from typing import Tuple
import math
import numpy as np
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score


def acc_from_probs(y_true, probs, weights=None, use_weights=False):
    """
    Calculates Accuracy.
    - If Single-Task: Returns float.
    - If Multi-Task: Returns list of floats.
    """
    # Standardize shapes
    if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
    if probs.ndim == 1:  probs = probs.reshape(-1, 1)
    
    n_tasks = y_true.shape[1]
    
    if use_weights and weights is not None:
        if weights.ndim == 1: weights = weights.reshape(-1, 1)
    else:
        weights = None

    # Convert probabilities to binary class predictions (0 or 1)
    preds = (probs > 0.5).astype(int)

    acc_list = []
    for t in range(n_tasks):
        y_t = y_true[:, t]
        pred_t = preds[:, t]
        w_t = weights[:, t] if weights is not None else None
        
        try:
            score = accuracy_score(y_t, pred_t, sample_weight=w_t)
        except ValueError:
            score = 0.0  # Fallback if something unexpected happens
            
        acc_list.append(score)

    if n_tasks == 1:
        return acc_list[0]
    else:
        return acc_list


def auc_from_probs(y_true, probs, weights=None, use_weights=False):
    """
    Calculates AUC.
    - If Single-Task: Returns float.
    - If Multi-Task: Returns list of floats.
    """
    # Standardize shapes
    if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
    if probs.ndim == 1:  probs = probs.reshape(-1, 1)
    
    n_tasks = y_true.shape[1]
    
    if use_weights and weights is not None:
        if weights.ndim == 1: weights = weights.reshape(-1, 1)
    else:
        weights = None

    auc_list = []
    for t in range(n_tasks):
        y_t = y_true[:, t]
        p_t = probs[:, t]
        w_t = weights[:, t] if weights is not None else None
        
        try:
            # Handle edge case: Batch has only 0s or only 1s
            if len(np.unique(y_t)) > 1:
                score = roc_auc_score(y_t, p_t, sample_weight=w_t)
            else:
                score = 0.5 
        except ValueError:
            score = 0.5
            
        auc_list.append(score)

    if n_tasks == 1:
        return auc_list[0]
    else:
        return auc_list


def compute_ece(probs, labels, weights=None, n_bins=10):
    """
    Calculates Weighted Expected Calibration Error (ECE) for binary classification.

    Args:
        probs (np.array): Predicted probabilities (0 to 1).
        labels (np.array): True binary labels (0 or 1).
        weights (np.array, optional): Sample weights. Defaults to uniform weights.
        n_bins (int): Number of bins for calibration.

    Returns:
        float: The weighted ECE score.
    """
    probs = np.array(probs).flatten()
    labels = np.array(labels).flatten()

    # Handle Weights
    if weights is not None:
        weights = np.array(weights).flatten()
        # Basic safety: ensure no negative weights and consistent shape
        assert weights.shape == probs.shape, "Weights shape must match probs"
    else:
        weights = np.ones_like(probs)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    total_weight = np.sum(weights)

    # Avoid division by zero if total_weight is 0 (edge case)
    if total_weight == 0:
        return 0.0

    for i in range(n_bins):
        # 1. Mask: Find points in this bin
        # Note: We use > and <= to handle the edges,
        # but usually the first bin should include 0.0 explicitly if needed.
        if i == 0:
            bin_mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        else:
            bin_mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])

        # If bin is empty, skip
        if not np.any(bin_mask):
            continue

        # 2. Extract data for this bin
        bin_weights = weights[bin_mask]
        bin_probs = probs[bin_mask]
        bin_labels = labels[bin_mask]

        bin_weight_sum = np.sum(bin_weights)

        if bin_weight_sum > 0:
            # 3. Weighted Stats
            # Avg Confidence: Weighted Mean of predicted probabilities
            avg_confidence = np.average(bin_probs, weights=bin_weights)

            # Avg Accuracy: Weighted Mean of actual labels (fraction of positives)
            avg_accuracy = np.average(bin_labels, weights=bin_weights)

            # 4. Weighted ECE contribution
            # Contribution = (Bin_Weight / Total_Weight) * |Acc - Conf|
            ece += (bin_weight_sum / total_weight) * np.abs(avg_accuracy - avg_confidence)

    return ece


def _flatten_if_needed(dataset: dc.data.NumpyDataset) -> dc.data.NumpyDataset:
    """
    If X is (num_samples, A, A), flatten to (num_samples, A*A).
    Otherwise, leave it as is.
    """
    X = dataset.X
    if X.ndim == 3:
        n, a1, a2 = X.shape
        assert a1 == a2, "Expect square matrices for CoulombMatrix features"
        X_flat = X.reshape(n, a1 * a2)
        return dc.data.NumpyDataset(
            X=X_flat,
            y=dataset.y,
            w=dataset.w,
            ids=dataset.ids,
        )
    elif X.ndim == 2:
        # Already flat (e.g., ECFP or already processed)
        return dataset
    else:
        raise ValueError(f"Unexpected X shape: {X.shape}")


def prepare_datasets(
    datasets: Tuple[dc.data.NumpyDataset, dc.data.NumpyDataset, dc.data.NumpyDataset],
    featurizer_name: str,
):
    """
    If featurizer_name != 'ECFP', try to flatten (assume CoulombMatrix or similar).
    Otherwise, return as-is.
    """
    featurizer_name = featurizer_name.lower()
    train, valid, test = datasets

    if featurizer_name == "ecfp":
        return train, valid, test
    else:
        train = _flatten_if_needed(train)
        valid = _flatten_if_needed(valid)
        test = _flatten_if_needed(test)
        return train, valid, test


def calculate_entropy(probs):
    """
    Binary Entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
    Max uncertainty at p=0.5 (Entropy ~ 0.693). Min at 0.0/1.0.
    """
    # Clip to avoid log(0)
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
    return entropy


def calculate_cutoff_classification_data(probs, y_true, weights=None, use_weights=False):
    """
    Calculates Weighted AUC and Brier Score for subsets sorted by confidence.
    Supports BOTH Single-Task and Multi-Task (returns a 'Task' column).
    """
    # 1. Standardize Shapes to (N, n_tasks)
    if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
    if probs.ndim == 1:  probs = probs.reshape(-1, 1)

    n_tasks = y_true.shape[1]

    # Handle Weights
    if use_weights and weights is not None:
        if weights.ndim == 1: weights = weights.reshape(-1, 1)
        if weights.shape != y_true.shape: weights = np.ones_like(y_true)
    else:
        weights = np.ones_like(y_true)

    all_results = []
    cutoffs = np.arange(0.05, 1.05, 0.05)

    # 2. Iterate Per Task (The Multi-Task Fix)
    for t in range(n_tasks):
        y_t = y_true[:, t]
        p_t = probs[:, t]
        w_t = weights[:, t]

        # Calculate Uncertainty (Binary Entropy)
        # Low Entropy = High Confidence
        uncertainty = calculate_entropy(p_t)

        df_temp = pd.DataFrame({
            'y_true': y_t,
            'prob': p_t,
            'uncertainty': uncertainty,
            'weight': w_t
        })

        # Sort by Uncertainty (Lowest first = Most Confident)
        df_temp = df_temp.sort_values(by='uncertainty', ascending=True).reset_index(drop=True)
        N = len(df_temp)

        for cutoff in cutoffs:
            n_keep = int(np.ceil(N * cutoff))
            subset = df_temp.iloc[:n_keep]

            sub_y = subset['y_true']
            sub_probs = subset['prob']
            sub_weights = subset['weight']

            # [METRIC 1] Weighted AUC
            # Handle edge case where subset has only 1 class (AUC undefined)
            if len(np.unique(sub_y)) > 1:
                try:
                    auc = roc_auc_score(sub_y, sub_probs, sample_weight=sub_weights)
                except ValueError:
                    auc = np.nan
            else:
                auc = np.nan

            # [METRIC 2] Weighted Brier Score
            sq_err = (sub_y - sub_probs) ** 2
            denom = sub_weights.sum()
            
            if denom > 0:
                brier = (sq_err * sub_weights).sum() / denom
            else:
                brier = sq_err.mean()

            all_results.append({
                'Confidence_Cutoff': cutoff,
                'AUC': auc,
                'Brier_Score': brier,
                'N_Samples': n_keep,
                'Task': f"Task_{t}"  # Distinguish tasks
            })

    return pd.DataFrame(all_results)

# def calculate_cutoff_error_data(mean_test, var_test, y_true, weights=None, use_weights=False):
#     """
#     Calculates the Root Mean Squared Error (RMSE) for subsets of the data
#     based on confidence percentile cutoffs (lowest variance first).

#     Args:
#         mean_test (np.ndarray): The model's mean predictions (gamma).
#         var_test (np.ndarray): The total predictive variance (aleatoric + epistemic).
#         y_true (np.ndarray): The true ground truth labels.

#     Returns:
#         pd.DataFrame: A DataFrame with columns 'Confidence_Cutoff'
#                       (fraction of data kept, sorted by confidence) and 'RMSE'.
#     """
#     # 1. Combine data into a DataFrame for easy handling
#     # Flattens all arrays to 1D, assuming aggregation over tasks if multi-task.
#     mean_test = mean_test.flatten()
#     var_test = var_test.flatten()
#     y_true = y_true.flatten()

#     if use_weights and weights is not None:
#         weights = weights.flatten()
#     else:
#         # If no weights provided or flag is off, use uniform weights (1.0)
#         # This simplifies the logic inside the loop
#         weights = np.ones_like(y_true)
    
#     data = pd.DataFrame({
#         'y_true': y_true,
#         'y_pred': mean_test,
#         'uncertainty': var_test,
#         'weight': weights
#     })

#     # 2. Sort by uncertainty (Total Variance) in ascending order (Highest Confidence first)
#     data = data.sort_values(by='uncertainty', ascending=True).reset_index(drop=True)
    
#     N = len(data)
#     cutoffs = np.arange(0.05, 1.05, 0.05)
#     results = []
    
#     # 4. Iterate through cutoffs and calculate error
#     for cutoff_fraction in cutoffs:
#         # Determine the number of top confident samples to keep
#         n_keep = int(np.ceil(N * cutoff_fraction))
        
#         # Select the top n_keep most confident samples
#         subset = data.iloc[:n_keep]

#         # Squared Errors for this subset
#         sq_errors = (subset['y_true'] - subset['y_pred']) ** 2

#         # [MODIFIED] Calculate Metric
#         if use_weights:
#             # Weighted MSE: sum(w * error^2) / sum(w)
#             mse = np.average(sq_errors, weights=subset['weight'])
#         else:
#             # Standard MSE
#             mse = np.mean(sq_errors)

#         rmse = np.sqrt(mse)
        
#         results.append({
#             'Confidence_Cutoff': cutoff_fraction,
#             'RMSE': rmse,
#             'MSE': mse,
#             'N_Samples': n_keep
#         })

#     return pd.DataFrame(results)

import pandas as pd
import numpy as np

def calculate_cutoff_error_data(mean_test, var_test, y_true, weights=None, use_weights=False):
    """
    Calculates RMSE vs. Confidence Retention for BOTH individual tasks and the global aggregate.

    Returns:
        pd.DataFrame: Columns ['Confidence_Cutoff', 'RMSE', 'MSE', 'N_Samples', 'Task']
                      'Task' will contain "Task_0", "Task_1", ..., and "Global".
    """
    # 1. Standardize Shapes to (N, T)
    # If inputs are 1D (N,), reshape to (N, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if mean_test.ndim == 1:
        mean_test = mean_test.reshape(-1, 1)
    if var_test.ndim == 1:
        var_test = var_test.reshape(-1, 1)
    
    # Handle Weights
    if use_weights and weights is not None:
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)
    else:
        # Create dummy weights of ones (N, T)
        weights = np.ones_like(y_true)

    n_tasks = y_true.shape[1]
    all_results = []

    # --- Helper: Core Logic for One Curve ---
    def _compute_curve(y, pred, unc, w, task_label):
        # Create a temp DF for sorting
        df_temp = pd.DataFrame({
            'y_true': y,
            'y_pred': pred,
            'uncertainty': unc,
            'weight': w
        })
        
        # Sort by Uncertainty (Lowest Variance = Highest Confidence first)
        df_temp = df_temp.sort_values(by='uncertainty', ascending=True).reset_index(drop=True)
        
        N = len(df_temp)
        cutoffs = np.arange(0.05, 1.05, 0.05) # 5% to 100% retention
        
        curve_stats = []
        for cutoff_fraction in cutoffs:
            n_keep = int(np.ceil(N * cutoff_fraction))
            subset = df_temp.iloc[:n_keep]

            sq_errors = (subset['y_true'] - subset['y_pred']) ** 2
            
            # Compute Metric
            if use_weights:
                # Weighted Average
                denom = np.sum(subset['weight'])
                if denom == 0: denom = 1.0 # Avoid div/0
                mse = np.sum(sq_errors * subset['weight']) / denom
            else:
                mse = np.mean(sq_errors)

            rmse = np.sqrt(mse)
            
            curve_stats.append({
                'Confidence_Cutoff': cutoff_fraction,
                'RMSE': rmse,
                'MSE': mse,
                'N_Samples': n_keep,
                'Task': task_label
            })
        return curve_stats

    # --- 2. Iterate Per Task ---
    for t in range(n_tasks):
        stats = _compute_curve(
            y=y_true[:, t], 
            pred=mean_test[:, t], 
            unc=var_test[:, t], 
            w=weights[:, t], 
            task_label=f"Task_{t}"
        )
        all_results.extend(stats)

    # # --- 3. Compute Global (Flattened) Curve ---
    # # Only useful if n_tasks > 1 to see the overall trend
    # if n_tasks > 1:
    #     stats_global = _compute_curve(
    #         y=y_true.flatten(),
    #         pred=mean_test.flatten(),
    #         unc=var_test.flatten(),
    #         w=weights.flatten(),
    #         task_label="Global"
    #     )
    #     all_results.extend(stats_global)

    # 4. Final DataFrame
    return pd.DataFrame(all_results)


def _to_1d(a):
    """Helper: convert to 1D numpy array."""
    return np.asarray(a, dtype=float).reshape(-1)


def _standard_normal_cdf(z):
    """Standard normal CDF using erf, no SciPy needed."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _spearman_rank_corr(x, y):
    """
    Spearman rank correlation ρ between x and y,
    implemented via Pearson on ranks (with tie-handling).
    """
    x = _to_1d(x)
    y = _to_1d(y)
    assert x.shape == y.shape

    def _rank(v):
        # rank with average for ties
        v = np.asarray(v)
        order = v.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(v), dtype=float)

        # tie-averaging
        vals, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        cumsum = np.cumsum(counts)
        start = cumsum - counts
        avg_rank = (start + cumsum - 1) / 2.0
        return avg_rank[inv]

    rx = _rank(x)
    ry = _rank(y)

    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)

    return float(np.mean(rx * ry))


def evaluate_uq_metrics_from_interval(
        y_true,
        mean,
        lower,
        upper,
        alpha: float = 0.05,
        test_error = 0, # Can be float (1 task) or list (N tasks)
        weights=None,
        use_weights: bool = False,
        ce_p_grid: np.ndarray | None = None,
):
    """
    Compute UQ metrics.
    - If Multi-Task (N columns): Returns LISTS of metrics (length N).
    - If Single-Task (1 column): Returns SCALARS (floats).
    """

    # --- 1. Standardize Shapes to (N, n_tasks) ---
    # We DO NOT flatten to 1D here. We preserve columns.
    if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
    if mean.ndim == 1:   mean = mean.reshape(-1, 1)
    if lower.ndim == 1:  lower = lower.reshape(-1, 1)
    if upper.ndim == 1:  upper = upper.reshape(-1, 1)

    n_tasks = y_true.shape[1]

    # Handle Weights
    if use_weights and weights is not None:
        if weights.ndim == 1: 
            weights = weights.reshape(-1, 1)
        # Safety check for shape mismatch
        if weights.shape != y_true.shape:
             weights = np.ones_like(y_true)
    else:
        weights = np.ones_like(y_true)

    # --- 2. Prepare Storage ---
    results = {
        "empirical_coverage": [],
        "avg_pred_std": [],
        "nll": [],
        "ce": [],
        "spearman_err_unc": []
    }

    # Z-score for sigma reconstruction from intervals
    z_table = {0.1: 1.644854, 0.05: 1.959964, 0.01: 2.575829}
    z_val = z_table.get(alpha, 1.96)

    # --- 3. Iterate Per Task ---
    for t in range(n_tasks):
        # Slice Data for Task t (flattens to 1D array of size N)
        y_t = y_true[:, t]
        mean_t = mean[:, t]
        lower_t = lower[:, t]
        upper_t = upper[:, t]
        w_t = weights[:, t]

        # Metric A: Empirical Coverage
        covered = (y_t >= lower_t) & (y_t <= upper_t)
        cov = float(covered.mean()) 
        results["empirical_coverage"].append(cov)

        # Metric B: Average Std (Sharpness)
        widths = upper_t - lower_t
        sigma_t = widths / (2.0 * z_val)
        sigma_t = np.clip(sigma_t, 1e-8, None)
        
        avg_std = np.average(sigma_t, weights=w_t)
        results["avg_pred_std"].append(avg_std)

        # Metric C: NLL (Gaussian Assumption)
        sq_err = (y_t - mean_t) ** 2
        nll_per_point = 0.5 * np.log(2.0 * math.pi * sigma_t ** 2) + sq_err / (2.0 * sigma_t ** 2)
        nll = np.average(nll_per_point, weights=w_t)
        results["nll"].append(nll)

        # Metric D: Calibration Error (CE)
        if ce_p_grid is None:
            grid = np.linspace(0.05, 0.95, 20)
        else:
            grid = np.asarray(ce_p_grid, dtype=float)
        
        # PIT / CDF
        z_norm = (y_t - mean_t) / sigma_t
        u = norm.cdf(z_norm)
        
        hat_p = []
        for p in grid:
            hat_p.append((u <= p).mean())
        hat_p = np.array(hat_p)
        
        grid_w = np.ones_like(grid) / grid.size
        ce = float(np.sum(grid_w * (grid - hat_p) ** 2))
        results["ce"].append(ce)

        # Metric E: Spearman (Error vs Uncertainty)
        abs_err = np.abs(y_t - mean_t)
        
        # Check if _spearman_rank_corr exists, else fallback
        try:
            sp = _spearman_rank_corr(abs_err, sigma_t)
        except NameError:
            from scipy.stats import spearmanr
            sp = spearmanr(abs_err, sigma_t).correlation
            
        results["spearman_err_unc"].append(sp)

    # --- 4. Final Formatting (Scalar vs List) ---
    
    # Store test_error directly (it should already be correct format from previous function)
    final_dict = {"MSE": test_error}

    if n_tasks == 1:
        # SINGLE TASK: Extract the single value from the lists
        for k, v in results.items():
            final_dict[k] = v[0]
    else:
        # MULTI TASK: Keep the lists
        for k, v in results.items():
            final_dict[k] = v

    return final_dict


def evaluate_uq_metrics_classification(
        y_true,
        probs,
        auc, # scalar or list
        uncertainty=None,
        weights=None,
        use_weights=False,
        n_bins=10
):
    # 1. Standardize Shapes (N, T)
    if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
    if probs.ndim == 1:  probs = probs.reshape(-1, 1)
    
    # Handle Uncertainty (Entropy) input
    if uncertainty is not None:
        if uncertainty.ndim == 1: uncertainty = uncertainty.reshape(-1, 1)
    
    n_tasks = y_true.shape[1]

    # Handle Weights
    if use_weights and weights is not None:
        if weights.ndim == 1: weights = weights.reshape(-1, 1)
        if weights.shape != y_true.shape: weights = np.ones_like(y_true)
    else:
        weights = np.ones_like(y_true)

    # 2. Prepare Storage
    results = {
        "AUC": auc, # Pass through the input AUC
        "NLL": [],
        "Brier": [],
        "ECE": [],
        "Avg_Entropy": [],     # Or Avg_Uncertainty
        "Spearman_Err_Unc": []
    }

    # 3. Iterate Per Task
    for t in range(n_tasks):
        y_t = y_true[:, t]
        p_t = probs[:, t] # Prob(y=1)
        w_t = weights[:, t]
        u_t = uncertainty[:, t] if uncertainty is not None else None

        # --- Metric A: NLL ---

        # 2. Define Safe Epsilon
        epsilon = 1e-7 

        # [ASSERTION A] Check Machine Precision (Prevent Divide by Zero)
        # Ensures 1.0 - epsilon is actually different from 1.0
        assert np.float32(1.0) - np.float32(epsilon) < 1.0, f"CRASH: Epsilon {epsilon} is too small for float32 precision!"
        p_safe = np.clip(p_t, epsilon, 1 - epsilon)
        nll_per_point = -(y_t * np.log(p_safe) + (1 - y_t) * np.log(1 - p_safe))
        # [ASSERTION B] Check for NaNs (Prevent Invalid Value propagation)
        assert not np.isnan(nll_per_point).any(), f"CRASH: Found {np.sum(np.isnan(nll_per_point))} NaNs in NLL calculation!"
        nll = np.average(nll_per_point, weights=w_t)
        results["NLL"].append(nll)

        # --- Metric B: Brier Score ---
        sq_err = (y_t - p_t) ** 2
        brier = np.average(sq_err, weights=w_t)
        results["Brier"].append(brier)

        # --- Metric C: ECE ---
        # (Assuming compute_ece is defined elsewhere and handles 1D inputs)
        ece = compute_ece(p_t, y_t, w_t, n_bins)
        results["ECE"].append(ece)

        # --- Metric D: Uncertainty & Spearman ---
        if u_t is not None:
            # Average Uncertainty (Sharpness)
            avg_uq = np.average(u_t, weights=w_t)
            results["Avg_Entropy"].append(avg_uq)

            # Spearman (Correlation between Error and Uncertainty)
            # For classification, "Error" is usually Brier Score (sq_err) or NLL
            try:
                sp = _spearman_rank_corr(sq_err, u_t)
            except NameError:
                from scipy.stats import spearmanr
                sp = spearmanr(sq_err, u_t).correlation
            results["Spearman_Err_Unc"].append(sp)
        else:
            results["Avg_Entropy"].append(None)
            results["Spearman_Err_Unc"].append(None)

    # 4. Final Formatting (Scalar vs List)
    if n_tasks == 1:
        # Return Scalars
        final = {}
        for k, v in results.items():
            # AUC is already scalar if n_tasks=1 (passed from input)
            final[k] = v[0] if isinstance(v, list) else v
        return final
    else:
        # Return Lists
        return results
