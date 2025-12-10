# data_utils.py
import deepchem as dc
from typing import Tuple
import math
import numpy as np
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


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


def calculate_cutoff_error_data(mean_test, var_test, y_true):
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
    
    data = pd.DataFrame({
        'y_true': y_true,
        'y_pred': mean_test,
        'uncertainty': var_test
    })

    # 2. Sort by uncertainty (Total Variance) in ascending order (Highest Confidence first)
    data = data.sort_values(by='uncertainty', ascending=True).reset_index(drop=True)
    
    N = len(data)
    
    # 3. Define cutoff points (fractions of the data to include: 0.05, 0.10, ..., 1.00)
    cutoffs = np.arange(0.05, 1.05, 0.05)
    
    results = []
    
    # 4. Iterate through cutoffs and calculate error
    for cutoff_fraction in cutoffs:
        # Determine the number of top confident samples to keep
        n_keep = int(np.ceil(N * cutoff_fraction))
        
        # Select the top n_keep most confident samples
        subset = data.iloc[:n_keep]
        
        # Calculate RMSE for the subset
        mse = mean_squared_error(subset['y_true'], subset['y_pred'])
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
    ce_p_grid: np.ndarray | None = None,
):
    """
    Compute UQ metrics from a single (1 - alpha) predictive interval.

    Inputs
    ------
    y_true : array-like, shape (N,) or (N,1)
    mean   : array-like, predictive mean
    lower  : array-like, lower bound of (1 - alpha) interval
    upper  : array-like, upper bound of (1 - alpha) interval
    alpha  : miscoverage level (1 - alpha is the nominal coverage)
    ce_p_grid : np.ndarray or None
        Grid of probability levels for the calibration error.
        If None, defaults to 20 points in [0.05, 0.95].

    Returns
    -------
    metrics : dict with keys:
        - n_points
        - alpha
        - empirical_coverage          (w.r.t. the given interval)
        - avg_pred_std                (from reconstructed sigma)
        - nll                         (Gaussian)
        - ce                          (calibration error in probability space)
        - spearman_err_unc            (Spearman(|error|, sigma))
    """

    # --- 0. Make everything 1D ---
    y_true = _to_1d(y_true)
    mean   = _to_1d(mean)
    lower  = _to_1d(lower)
    upper  = _to_1d(upper)

    assert y_true.shape == mean.shape == lower.shape == upper.shape, \
        f"Shape mismatch: y={y_true.shape}, mean={mean.shape}, " \
        f"lower={lower.shape}, upper={upper.shape}"

    N = y_true.shape[0]

    # --- 1. Empirical coverage + sigma reconstruction ---

    covered = (y_true >= lower) & (y_true <= upper)
    empirical_coverage = float(covered.mean())

    # Reconstruct predictive std from interval width assuming Gaussian
    widths = upper - lower
    z_table = {0.1: 1.644854, 0.05: 1.959964, 0.01: 2.575829}
    z = z_table.get(alpha, 1.96)
    sigma = widths / (2.0 * z)
    sigma = np.clip(sigma, 1e-8, None)
    avg_pred_std = float(sigma.mean())

    # --- 2. NLL under Gaussian assumption ---

    sq_err = (y_true - mean) ** 2
    nll_per_point = 0.5 * np.log(2.0 * math.pi * sigma**2) + sq_err / (2.0 * sigma**2)
    nll = float(nll_per_point.mean())

    # --- 3. Calibration error in probability space ---

    if ce_p_grid is None:
        ce_p_grid = np.linspace(0.05, 0.95, 20)
    ce_p_grid = np.asarray(ce_p_grid, dtype=float)

    # u_i = F_i(y_i) under Gaussian
    z_norm = (y_true - mean) / sigma
    u = np.array([_standard_normal_cdf(zi) for zi in z_norm])

    hat_p = np.array([(u <= p).mean() for p in ce_p_grid])
    w = np.ones_like(ce_p_grid) / ce_p_grid.size
    ce = float(np.sum(w * (ce_p_grid - hat_p) ** 2))

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