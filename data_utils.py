# data_utils.py
import deepchem as dc
from typing import Tuple
import math
import numpy as np
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, roc_auc_score



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
    Calculates Weighted AUC and Brier Score for subsets of data sorted by confidence.
    """
    # 1. Flatten
    probs = probs.flatten()
    y_true = y_true.flatten()

    # 2. Define Uncertainty (Entropy)
    # Assuming calculate_entropy is defined elsewhere or imported
    # If not, use: uncertainty = - (probs * np.log(probs + 1e-9) + (1-probs) * np.log(1-probs + 1e-9))
    uncertainty = calculate_entropy(probs) 

    # 3. Prepare DataFrame
    if use_weights and weights is not None:
        w = weights.flatten()
    else:
        w = np.ones_like(y_true)

    data = pd.DataFrame({
        'y_true': y_true,
        'prob': probs,
        'uncertainty': uncertainty,
        'weight': w
    })

    # 4. Sort by Uncertainty (Lowest Entropy first = Most Confident)
    data = data.sort_values(by='uncertainty', ascending=True).reset_index(drop=True)

    N = len(data)
    cutoffs = np.arange(0.05, 1.05, 0.05)
    results = []

    for cutoff in cutoffs:
        n_keep = int(np.ceil(N * cutoff))
        subset = data.iloc[:n_keep]

        # Extract columns for clarity
        sub_y = subset['y_true']
        sub_probs = subset['prob']
        sub_weights = subset['weight']

        # [METRIC 1] Weighted AUC (Replaces Accuracy)
        # Safety Check: AUC is undefined if only 1 class is present in the subset
        if len(np.unique(sub_y)) > 1:
            auc = roc_auc_score(sub_y, sub_probs, sample_weight=sub_weights)
        else:
            auc = np.nan  # Handle single-class subsets gracefully

        # [METRIC 2] Weighted Brier Score (MSE in prob space)
        sq_err = (sub_y - sub_probs) ** 2
        brier = np.average(sq_err, weights=sub_weights)

        results.append({
            'Confidence_Cutoff': cutoff,
            'AUC': auc,           # Changed from 'Accuracy'
            'Brier_Score': brier,
            'N_Samples': n_keep
        })

    return pd.DataFrame(results)


def calculate_cutoff_error_data(mean_test, var_test, y_true, weights=None, use_weights=False):
    """
    Calculates the Root Mean Squared Error (RMSE) for subsets of the data
    based on confidence percentile cutoffs (lowest variance first).

    Args:
        mean_test (np.ndarray): The model's mean predictions (gamma).
        var_test (np.ndarray): The total predictive variance (aleatoric + epistemic).
        y_true (np.ndarray): The true ground truth labels.

    Returns:
        pd.DataFrame: A DataFrame with columns 'Confidence_Cutoff'
                      (fraction of data kept, sorted by confidence) and 'RMSE'.
    """
    # 1. Combine data into a DataFrame for easy handling
    # Flattens all arrays to 1D, assuming aggregation over tasks if multi-task.
    mean_test = mean_test.flatten()
    var_test = var_test.flatten()
    y_true = y_true.flatten()

    if use_weights and weights is not None:
        weights = weights.flatten()
    else:
        # If no weights provided or flag is off, use uniform weights (1.0)
        # This simplifies the logic inside the loop
        weights = np.ones_like(y_true)
    
    data = pd.DataFrame({
        'y_true': y_true,
        'y_pred': mean_test,
        'uncertainty': var_test,
        'weight': weights
    })

    # 2. Sort by uncertainty (Total Variance) in ascending order (Highest Confidence first)
    data = data.sort_values(by='uncertainty', ascending=True).reset_index(drop=True)
    
    N = len(data)
    cutoffs = np.arange(0.05, 1.05, 0.05)
    results = []
    
    # 4. Iterate through cutoffs and calculate error
    for cutoff_fraction in cutoffs:
        # Determine the number of top confident samples to keep
        n_keep = int(np.ceil(N * cutoff_fraction))
        
        # Select the top n_keep most confident samples
        subset = data.iloc[:n_keep]

        # Squared Errors for this subset
        sq_errors = (subset['y_true'] - subset['y_pred']) ** 2

        # [MODIFIED] Calculate Metric
        if use_weights:
            # Weighted MSE: sum(w * error^2) / sum(w)
            mse = np.average(sq_errors, weights=subset['weight'])
        else:
            # Standard MSE
            mse = np.mean(sq_errors)

        rmse = np.sqrt(mse)
        
        results.append({
            'Confidence_Cutoff': cutoff_fraction,
            'RMSE': rmse,
            'MSE': mse,
            'N_Samples': n_keep
        })

    return pd.DataFrame(results)


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
        test_error: float = 0,
        weights=None,
        use_weights: bool = False,
        ce_p_grid: np.ndarray | None = None,
):
    """
    Compute UQ metrics.

    NOTE on Weights:
    - NLL and Avg_Std ARE weighted (if use_weights=True) because they represent
      optimization objectives/magnitudes.
    - Coverage and Calibration Error are NOT weighted (per user request)
      to check global statistical validity.
    """

    # --- 0. Make everything 1D ---
    y_true = _to_1d(y_true)
    mean = _to_1d(mean)
    lower = _to_1d(lower)
    upper = _to_1d(upper)

    assert y_true.shape == mean.shape == lower.shape == upper.shape

    # Prepare Weights for NLL calculation
    if use_weights and weights is not None:
        w = _to_1d(weights)
    else:
        w = np.ones_like(y_true)

    # --- 1. Empirical coverage + sigma reconstruction ---

    covered = (y_true >= lower) & (y_true <= upper)

    # [UNWEIGHTED] Coverage
    # We want to know the raw percentage of data captured.
    empirical_coverage = float(covered.mean())

    # Reconstruct predictive std
    widths = upper - lower
    z_table = {0.1: 1.644854, 0.05: 1.959964, 0.01: 2.575829}
    z = z_table.get(alpha, 1.96)
    sigma = widths / (2.0 * z)
    sigma = np.clip(sigma, 1e-8, None)

    # [WEIGHTED] Avg Std
    # We care more about the sharpness on important samples.
    avg_pred_std = np.average(sigma, weights=w)

    # --- 2. NLL under Gaussian assumption ---

    sq_err = (y_true - mean) ** 2
    nll_per_point = 0.5 * np.log(2.0 * math.pi * sigma ** 2) + sq_err / (2.0 * sigma ** 2)

    # [WEIGHTED] NLL
    # This aligns with the training loss objective.
    nll = np.average(nll_per_point, weights=w)

    # --- 3. Calibration error in probability space ---

    if ce_p_grid is None:
        ce_p_grid = np.linspace(0.05, 0.95, 20)
    ce_p_grid = np.asarray(ce_p_grid, dtype=float)

    # u_i = F_i(y_i) under Gaussian
    z_norm = (y_true - mean) / sigma
    u = norm.cdf(z_norm)

    # [UNWEIGHTED] Calibration Calculation
    hat_p = []
    for p in ce_p_grid:
        # Standard mean: What fraction of ALL data falls below this quantile?
        fraction_below = (u <= p).mean()
        hat_p.append(fraction_below)
    hat_p = np.array(hat_p)

    # L2 Calibration Error
    grid_w = np.ones_like(ce_p_grid) / ce_p_grid.size
    ce = float(np.sum(grid_w * (ce_p_grid - hat_p) ** 2))

    # --- 4. Spearman(error, uncertainty) ---

    abs_err = np.abs(y_true - mean)
    spearman_err_unc = _spearman_rank_corr(abs_err, sigma)

    return {
        "alpha": float(alpha),
        "MSE": test_error,
        "empirical_coverage": empirical_coverage,
        "avg_pred_std": avg_pred_std,
        "nll": nll,
        "ce": ce,
        "spearman_err_unc": spearman_err_unc,
    }


def evaluate_uq_metrics_classification(
        y_true,
        probs,
        weights=None,
        use_weights=False,
        n_bins=10
):
    """
    Compute UQ metrics for Classification: ECE, NLL, Brier, Spearman.
    """
    y_true = y_true.flatten()
    probs = probs.flatten()

    if use_weights and weights is not None:
        w = weights.flatten()
    else:
        w = np.ones_like(y_true)

    # --- 1. Sharpness (Avg Entropy) ---
    # Equivalent to "Avg Pred Std" in regression
    # Indicates how decisive the model is.
    entropy = calculate_entropy(probs)
    avg_entropy = np.average(entropy, weights=w)

    # --- 2. NLL (Log Loss) ---
    # Equivalent to Gaussian NLL
    epsilon = 1e-15
    p_safe = np.clip(probs, epsilon, 1 - epsilon)
    nll_per_point = -(y_true * np.log(p_safe) + (1 - y_true) * np.log(1 - p_safe))
    nll = np.average(nll_per_point, weights=w)

    # --- 3. Brier Score ---
    # Equivalent to MSE in Regression
    sq_err = (y_true - probs) ** 2
    brier = np.average(sq_err, weights=w)

    # --- 4. Calibration Error (ECE) ---
    # Equivalent to "Coverage" (Global statistical check)
    # We can perform a Weighted ECE if desired.
    ece = compute_ece(probs, y_true, w, n_bins)


    # --- 5. Spearman (Uncertainty vs Error) ---
    # Do uncertain points actually have higher errors?
    # Correlate Entropy with Brier Score Error
    spearman_corr = _spearman_rank_corr(sq_err, entropy)

    return {
        "NLL": nll,
        "Brier": brier,
        "ECE": ece,
        "Avg_Entropy": avg_entropy,
        "Spearman_Err_Unc": spearman_corr
    }