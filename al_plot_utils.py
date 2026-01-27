"""
Active Learning Plotting Utilities

This module provides plotting and analysis functions for active learning results,
following the style and conventions of plot_utils.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from typing import List, Optional

# --- Configuration ---
# Default values (can be overridden via command-line arguments)
DATASETS = ['tox21']
SPLITS = ['scaffold']
MODE = 'classification'
TASK_INDICES = [0, 1]  # None for single task, or list like [0, 1] for multitask
ENCODER_TYPE = 'identity'
USE_GRAPH = False
N_RUNS = 5


# ============================================================
# Active Learning Plotting and Analysis
# ============================================================

def load_active_learning_results(
    dataset_name: str,
    split: str,
    mode: str = "classification",
    task_indices: Optional[List[int]] = None,
    encoder_type: str = "identity",
    use_graph: bool = False,
    n_runs: int = 5
) -> pd.DataFrame:
    """
    Load and aggregate active learning results across all methods and runs.
    
    Args:
        dataset_name: Name of the dataset
        split: Split type ("random" or "scaffold")
        mode: "regression" or "classification"
        task_indices: Optional list of task indices
        encoder_type: "identity" or "dmpnn"
        use_graph: Whether graph featurizer was used
        n_runs: Number of runs to aggregate
        
    Returns:
        DataFrame with columns: method, step, train_size, and all metrics (aggregated across runs)
    """
    output_dir = f"./cdata_{mode}"
    split_name = f"{split}_graph" if use_graph else split
    
    # Create task suffix
    if task_indices is None:
        global_task_suffix = ""
    else:
        global_task_suffix = "_tasks_" + "_".join(map(str, task_indices))
    
    # Method names to look for
    methods = ["nn_baseline", "nn_deep_ensemble", "nn_mc_dropout", "nn_evd"]
    
    all_data = []
    
    # Determine task IDs to process and filename construction
    # Based on active_learning.py saving logic:
    # - If task_indices is None: saves as _id_0 (single task, no task specified)
    # - If task_indices = [2] (single specified task): saves as _tasks_2_id_0 (always id_0 for single tasks)
    # - If task_indices = [0, 2] (multitask): saves as _tasks_0_2_id_0 and _tasks_0_2_id_2
    if task_indices is None:
        # Single task (no task specified): use id_0
        task_ids = [0]
        filename_suffixes = [("_id_0", 0)]
    elif len(task_indices) == 1:
        # Single task but specified (e.g., task_indices=[2]): use _tasks_2_id_0
        # The actual task_id is from task_indices, but filename uses id_0
        single_task_id = task_indices[0]
        task_ids = [single_task_id]
        filename_suffixes = [(f"{global_task_suffix}_id_0", single_task_id)]
    else:
        # Multitask (e.g., task_indices=[0, 2]): use actual task IDs in filename
        task_ids = task_indices
        filename_suffixes = [(f"{global_task_suffix}_id_{tid}", tid) for tid in task_indices]
    
    for method in methods:
        for final_suffix, actual_task_id in filename_suffixes:
            
            # Load all runs for this method and task
            method_data = []
            for run_id in range(n_runs):
                # Directly format filename: AL_{method}_{split}_{dataset}_tasks_{task_ids}_id_{task_id}_run_{run_id}.csv
                fname = f"AL_{method}_{split_name}_{dataset_name}{final_suffix}_run_{run_id}.csv"
                fpath = os.path.join(output_dir, fname)
                
                if os.path.exists(fpath):
                    try:
                        df = pd.read_csv(fpath)
                        df['method'] = method
                        df['task_id'] = actual_task_id  # Use actual task_id from task_indices, not filename
                        df['run_id'] = run_id
                        method_data.append(df)
                    except Exception as e:
                        print(f"Error loading {fpath}: {e}")
                        continue
                else:
                    print(f"Warning: File not found: {fpath}")
            
            if method_data:
                all_data.extend(method_data)
    
    if not all_data:
        raise ValueError(f"No active learning data found for {dataset_name} with split {split}")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return combined_df


def aggregate_active_learning_results(
    df: pd.DataFrame,
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Aggregate active learning results across runs for each method and step.
    
    Args:
        df: DataFrame from load_active_learning_results
        metrics: List of metric names to aggregate (if None, auto-detect)
        
    Returns:
        DataFrame with mean and std for each metric, grouped by method and step
    """
    if metrics is None:
        # Auto-detect metric columns (exclude non-metric columns)
        exclude_cols = ['method', 'step', 'train_size', 'task_id', 'run_id']
        metrics = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
    
    # Group by method, step, and task_id (if present)
    group_cols = ['method', 'step']
    if 'task_id' in df.columns:
        group_cols.append('task_id')
    
    aggregated = []
    
    # Handle unpacking based on whether task_id is in group_cols
    for group_key, group in df.groupby(group_cols):
        # Unpack group_key based on number of grouping columns
        if len(group_cols) == 3:  # method, step, task_id
            method, step, task_id = group_key
        else:  # method, step
            method, step = group_key
            task_id = None
        
        row = {
            'method': method,
            'step': step,
        }
        
        if task_id is not None:
            row['task_id'] = task_id
        
        # Get train_size (should be same across runs for same step)
        if 'train_size' in group.columns:
            row['train_size'] = group['train_size'].iloc[0]
        
        # Aggregate each metric
        for metric in metrics:
            if metric in group.columns:
                values = group[metric].dropna()
                if len(values) > 0:
                    row[f"{metric}_mean"] = values.mean()
                    row[f"{metric}_std"] = values.std(ddof=0)
                else:
                    row[f"{metric}_mean"] = np.nan
                    row[f"{metric}_std"] = np.nan
        
        aggregated.append(row)
    
    return pd.DataFrame(aggregated)


def plot_active_learning_curves(
    df_agg: pd.DataFrame,
    dataset_name: str,
    split: str,
    mode: str = "classification",
    task_id: Optional[int] = None,
    task_indices: Optional[List[int]] = None,
    output_dir: Optional[str] = None
):
    """
    Plot active learning curves showing how metrics change over steps.
    Similar style to plot_utils.py.
    
    Args:
        df_agg: Aggregated DataFrame from aggregate_active_learning_results
        dataset_name: Dataset name for title/filename
        split: Split type for filename
        mode: "regression" or "classification"
        task_id: Optional task ID (for multitask)
        task_indices: Optional list of task indices (for filename construction)
        output_dir: Output directory (default: ./cdata_{mode}/{dataset_name})
    """
    if output_dir is None:
        # Create subdirectory based on dataset name
        output_dir = f"./cdata_{mode}/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter by task_id if specified
    if task_id is not None and 'task_id' in df_agg.columns:
        df_agg = df_agg[df_agg['task_id'] == task_id].copy()
    
    # Method mapping (same as plot_utils.py)
    METHOD_MAP = {
        'nn_baseline': 'NN',
        'nn_mc_dropout': 'MC',
        'nn_deep_ensemble': 'Deep-Ens',
        'nn_evd': 'EVD',
    }
    
    # Define metrics to plot based on mode
    if mode == "classification":
        METRICS_TO_PLOT = {
            'AUC': 'AUC (Larger is Better)',
            'Brier': 'Brier (Lower is Better)',
            'ECE': 'ECE (Lower is Better)',
            'NLL': 'NLL (Lower is Better)',
            'Spearman_Err_Unc': 'Spearman Cor (Larger is Better)',
        }
    else:
        METRICS_TO_PLOT = {
            'MSE': 'MSE (Lower is Better)',
            'ce': 'CE Error (Lower is Better)',
            'nll': 'NLL (Lower is Better)',
            'empirical_coverage': 'Coverage (Closer to 0.95 is Ideal)',
            'spearman_err_unc': 'Spearman Rho (Higher is Better)',
        }
    
    # Filter to available metrics
    available_metrics = {k: v for k, v in METRICS_TO_PLOT.items() 
                        if f"{k}_mean" in df_agg.columns}
    
    if not available_metrics:
        print(f"Warning: No metrics found to plot for {dataset_name}")
        return
    
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    task_suffix = f"_task_{task_id}" if task_id is not None else ""
    fig.suptitle(f'Active Learning Curves: {dataset_name.upper()}{task_suffix}', fontsize=20, y=1.03)
    
    # Get unique methods and create color map
    methods = df_agg['method'].unique()
    method_labels = [METHOD_MAP.get(m, m) for m in methods]
    n_methods = len(methods)
    palette = sns.color_palette("tab10", n_methods)
    color_map = {method: palette[i] for i, method in enumerate(methods)}
    
    for i, (metric, title) in enumerate(available_metrics.items()):
        ax = axes[i]
        
        # Plot each method
        for method in methods:
            method_df = df_agg[df_agg['method'] == method].sort_values('step')
            
            if len(method_df) == 0:
                continue
            
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            
            if mean_col not in method_df.columns:
                continue
            
            steps = method_df['step'].values
            means = method_df[mean_col].values
            stds = method_df[std_col].values if std_col in method_df.columns else np.zeros_like(means)
            
            # Remove NaN values
            valid_mask = ~np.isnan(means)
            steps = steps[valid_mask]
            means = means[valid_mask]
            stds = stds[valid_mask]
            
            if len(steps) == 0:
                continue
            
            method_label = METHOD_MAP.get(method, method)
            ax.plot(steps, means, label=method_label, color=color_map[method], linewidth=2, marker='o')
            ax.fill_between(steps, means - stds, means + stds, alpha=0.2, color=color_map[method])
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Active Learning Step", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1.0, 0.98])
    
    # Construct filename: {task_indices_str}_task_{task_id}.png
    # Example: 0_3_task_0.png, 0_3_task_3.png, or just task_0.png for single task
    if task_indices is not None and len(task_indices) > 1:
        # Multitask: use task indices in filename (e.g., 0_3_task_0.png)
        task_indices_str = "_".join(map(str, task_indices))
        filename = f"{task_indices_str}_task_{task_id}.png"
    elif task_indices is not None and len(task_indices) == 1:
        # Single specified task: use task index (e.g., 2_task_2.png)
        filename = f"{task_indices[0]}_task_{task_id}.png"
    else:
        # Single task (no task specified): just use task_0.png
        filename = f"task_{task_id}.png" if task_id is not None else "task_0.png"
    
    output_file = os.path.join(output_dir, filename)
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"\nFigure saved to: {output_file}")
    plt.close(fig)


def print_active_learning_table(
    df_agg: pd.DataFrame,
    dataset_name: str,
    task_id: Optional[int] = None
):
    """
    Print a formatted table comparing methods across active learning steps.
    
    Args:
        df_agg: Aggregated DataFrame from aggregate_active_learning_results
        dataset_name: Dataset name
        task_id: Optional task ID (for multitask)
    """
    if task_id is not None and 'task_id' in df_agg.columns:
        df_agg = df_agg[df_agg['task_id'] == task_id].copy()
    
    # Method mapping
    METHOD_MAP = {
        'nn_baseline': 'NN',
        'nn_mc_dropout': 'MC',
        'nn_deep_ensemble': 'Deep-Ens',
        'nn_evd': 'EVD',
    }
    
    # Focus on AUC for classification
    metric = 'AUC'
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    
    if mean_col not in df_agg.columns:
        print(f"Warning: {mean_col} not found in data. Available columns: {df_agg.columns.tolist()}")
        return
    
    print(f"\n{'='*80}")
    task_suffix = f" (Task {task_id})" if task_id is not None else ""
    print(f"Active Learning Performance: {dataset_name.upper()}{task_suffix}")
    print(f"{'='*80}")
    print(f"\n{metric.upper()} Performance Across Steps:\n")
    
    # Get unique steps and methods
    steps = sorted(df_agg['step'].unique())
    methods = sorted(df_agg['method'].unique())
    
    # Print header
    header = f"{'Step':<8} {'Train Size':<12}"
    for method in methods:
        method_label = METHOD_MAP.get(method, method)
        header += f" {method_label:<15}"
    print(header)
    print("-" * 80)
    
    # Print each step
    for step in steps:
        step_df = df_agg[df_agg['step'] == step]
        if len(step_df) == 0:
            continue
        
        train_size = step_df['train_size'].iloc[0] if 'train_size' in step_df.columns else "N/A"
        row = f"{step:<8} {train_size:<12}"
        
        for method in methods:
            method_df = step_df[step_df['method'] == method]
            if len(method_df) > 0 and mean_col in method_df.columns:
                mean_val = method_df[mean_col].iloc[0]
                std_val = method_df[std_col].iloc[0] if std_col in method_df.columns else 0.0
                if not np.isnan(mean_val):
                    row += f" {mean_val:.4f}±{std_val:.4f}"
                else:
                    row += f" {'N/A':<15}"
            else:
                row += f" {'N/A':<15}"
        
        print(row)
    
    print(f"\n{'='*80}\n")


def plot_active_learning_results(
    dataset_name: str,
    split: str,
    mode: str = "classification",
    task_indices: Optional[List[int]] = None,
    encoder_type: str = "identity",
    use_graph: bool = False,
    n_runs: int = 5
):
    """
    Main function to load, aggregate, plot, and print active learning results.
    Matches the style of plot_utils.py.
    
    Args:
        dataset_name: Name of the dataset
        split: Split type ("random" or "scaffold")
        mode: "regression" or "classification"
        task_indices: Optional list of task indices
        encoder_type: "identity" or "dmpnn"
        use_graph: Whether graph featurizer was used
        n_runs: Number of runs to aggregate
    """
    print(f"\n{'='*80}")
    print(f"Processing Active Learning Results: {dataset_name} ({split})")
    print(f"{'='*80}\n")
    
    # Load data
    df = load_active_learning_results(
        dataset_name=dataset_name,
        split=split,
        mode=mode,
        task_indices=task_indices,
        encoder_type=encoder_type,
        use_graph=use_graph,
        n_runs=n_runs
    )
    
    print(f"Loaded {len(df)} rows from active learning files")
    
    # Aggregate across runs
    df_agg = aggregate_active_learning_results(df)
    print(f"Aggregated to {len(df_agg)} method-step combinations\n")
    
    # Determine task IDs to process
    if task_indices is None:
        task_ids = [None]  # Single task
    else:
        task_ids = task_indices
    
    # Plot and print for each task
    for task_id in task_ids:
        plot_active_learning_curves(
            df_agg=df_agg,
            dataset_name=dataset_name,
            split=split,
            mode=mode,
            task_id=task_id,
            task_indices=task_indices
        )
        
        print_active_learning_table(
            df_agg=df_agg,
            dataset_name=dataset_name,
            task_id=task_id
        )


# ============================================================
# Main Function
# ============================================================

def main():
    """
    Main function to run active learning plotting from command line or with default config.
    """
    parser = argparse.ArgumentParser(description='Plot active learning results')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['qm7', 'qm8', 'delaney', 'lipo', 'tox21', 'toxcast', 'sider', 'clintox'],
                        help='Dataset name (default: from config)')
    parser.add_argument('--split', type=str, default=None,
                        choices=['random', 'scaffold'],
                        help='Split type (default: from config)')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['regression', 'classification'],
                        help='Task mode (default: from config)')
    parser.add_argument('--tasks', type=int, nargs='+', default=None,
                        help='Task indices (e.g., --tasks 0 1) (default: from config)')
    parser.add_argument('--encoder_type', type=str, default=None,
                        choices=['identity', 'dmpnn'],
                        help='Encoder type (default: from config)')
    parser.add_argument('--use_graph', action='store_true',
                        help='Use graph featurizer (default: False)')
    parser.add_argument('--n_runs', type=int, default=None,
                        help='Number of runs to aggregate (default: from config)')
    
    args = parser.parse_args()
    
    # Use command-line args if provided, otherwise use config defaults
    datasets = [args.dataset] if args.dataset else DATASETS
    splits = [args.split] if args.split else SPLITS
    mode = args.mode if args.mode else MODE
    task_indices = args.tasks if args.tasks is not None else TASK_INDICES
    encoder_type = args.encoder_type if args.encoder_type else ENCODER_TYPE
    use_graph = args.use_graph if args.use_graph else USE_GRAPH
    n_runs = args.n_runs if args.n_runs else N_RUNS
    
    # Handle None for task_indices (single task)
    if task_indices == []:
        task_indices = None
    
    try:
        # Process each dataset and split combination
        for dataset in datasets:
            for split in splits:
                print(f"\n{'='*80}")
                print(f"Processing: {dataset} ({split})")
                print(f"{'='*80}")
                
                plot_active_learning_results(
                    dataset_name=dataset,
                    split=split,
                    mode=mode,
                    task_indices=task_indices,
                    encoder_type=encoder_type,
                    use_graph=use_graph,
                    n_runs=n_runs
                )
        
        print(f"\n{'='*80}")
        print("All plots generated successfully!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\nAn error occurred during plot generation: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check:")
        print("  - Active learning result files exist in ./cdata_{mode}/")
        print("  - File naming follows pattern: AL_{method}_{split}_{dataset}_tasks_{task_ids}_id_{task_id}_run_{run_id}.csv")
        print("  - Dataset name, split type, and task indices are correct")


if __name__ == "__main__":
    main()
