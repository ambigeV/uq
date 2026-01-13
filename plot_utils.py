import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Configuration (Unchanged) ---
# DATASETS = ["delaney", "lipo", 'qm7', 'qm8', 'tox21']
DATASETS = ['tox21']
DATA_DIR = "./cdata_classification"
SPLIT = "scaffold"

import pandas as pd
import glob
import os
import re

# --- Configuration ---
os.makedirs(DATA_DIR, exist_ok=True)

DATASETS = ['tox21']
SPLITS = [SPLIT]
MODEL_TYPES = ['NN', 'GP']
TARGET_IDS = [0, 3]  # K = 2
TASK_STR = "tasks_0_3"
SOURCE_DIR = "./cdata_classification"
DATA_DIR = f"./cdata_classification_aggregated/{TASK_STR}"

METRICS_TO_PLOT = {
    'MSE_mean': 'MSE (Lower is Better)',
    'ce_mean': 'CE Error (Lower is Better)',
    'nll_mean': 'NLL (Lower is Better)',
    'empirical_coverage_mean': 'Coverage (Closer to 0.95 is Ideal)',
    'spearman_err_unc_mean': 'Spearman Rho (Higher is Better)',
}
METRICS_TO_PLOT = {
    'AUC_mean': 'AUC (Larger is Better)',
    'Brier_mean': 'Brier (Lower is Better)',
    'ECE_mean': 'ECE (Lower is Better)',
    'NLL_mean': 'NLL (Lower is Better)',
    'Spearman_Err_Unc_mean': 'Spearman Cor (Larger is Better)',
}

# --- Fixed Order and Short Names for Plotting ---
# This list defines the order of the bars on the X-axis for all plots.
METHOD_ORDER_KEYS = [
    'nn_baseline',
    'nn_mc_dropout',
    'nn_deep_ensemble',
    'nn_evd',
    'gp_exact',
    'gp_nngp',
    'gp_svgp',
    'gp_nnsvgp',
    'svgp_ensemble_uniform',
    'svgp_ensemble_precision',
    'svgp_ensemble_mse',
    'svgp_ensemble_nll',
    'nngp_ensemble_uniform',
    'nngp_ensemble_precision',
    'nngp_ensemble_mse',
    'nngp_ensemble_nll',
    'nnsvgp_ensemble_uniform',
    'nnsvgp_ensemble_precision',
    'nnsvgp_ensemble_mse',
    'nnsvgp_ensemble_nll',
]

# This map uses the keys for the internal data column and the values for the plot labels.
METHOD_MAP = {
    'nn_baseline': 'NN',
    'nn_mc_dropout': 'MC',
    'nn_deep_ensemble': 'Deep-Ens',
    'nn_evd': 'EVD',
    'gp_exact': 'GP',
    'gp_nngp': 'NNGP',
    'gp_svgp': 'SVGP',
    'gp_nnsvgp': 'NNGP_SVGP',
    'svgp_ensemble_uniform': 'SVGP-Ens_Avg',
    'svgp_ensemble_precision': 'SVGP-Ens_Conf',
    'svgp_ensemble_mse': 'SVGP-Ens_Prec',
    'svgp_ensemble_nll': 'SVGP-Ens_NLL',
    'nngp_ensemble_uniform': 'NNGP-Ens_Avg',
    'nngp_ensemble_precision': 'NNGP-Ens_Conf',
    'nngp_ensemble_mse': 'NNGP-Ens_Prec',
    'nngp_ensemble_nll': 'NNGP-Ens_NLL',
    'nnsvgp_ensemble_uniform': 'NN-SVGP-Ens_Avg',
    'nnsvgp_ensemble_precision': 'NN-SVGP-Ens_Conf',
    'nnsvgp_ensemble_mse': 'NN-SVGP-Ens_Prec',
    'nnsvgp_ensemble_nll': 'NN-SVGP-Ens_NLL',
}

# The final list of labels in the correct order
FINAL_LABEL_ORDER = [METHOD_MAP[key] for key in METHOD_ORDER_KEYS]


# --- Data Loading and Cleaning (Modified method_map) ---

def load_and_clean_data(datasets, split):
    """Loads all NN and GP summary files, combines them, and cleans method names."""
    all_dfs = []

    for dataset in datasets:
        nn_file = os.path.join(DATA_DIR, f"NN_{split}_{dataset}_c.csv")
        gp_file = os.path.join(DATA_DIR, f"GP_{split}_{dataset}_c.csv")

        for file_path in [nn_file, gp_file]:
            try:
                df = pd.read_csv(file_path)
                df['Dataset'] = dataset
                all_dfs.append(df)
            except FileNotFoundError:
                print(f"File not found: {file_path}. Skipping.")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if not all_dfs:
        raise ValueError("No data files were loaded. Check paths and dataset names.")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 1. **IMPROVED ROBUSTNESS FOR METHOD COLUMN IDENTIFICATION**
    potential_method_cols = ['Method', 'method', combined_df.columns[-1]]
    method_col = None

    # Use the keys of the new METHOD_MAP for robust column detection
    for col in potential_method_cols:
        if col in combined_df.columns and combined_df[col].astype(str).str.contains('|'.join(METHOD_MAP.keys())).any():
            method_col = col
            break

    if method_col is None:
        method_col = combined_df.columns[-1]
        print(f"Warning: Could not definitively identify 'Method' column. Assuming it is '{method_col}'.")

    # 2. Apply the new Mapping
    combined_df['Method_Name_Cleaned'] = combined_df[method_col].astype(str).apply(
        lambda x: METHOD_MAP.get(x, x)
    )

    # 3. Filter out any rows that still have an unrecognized method name
    known_methods = set(METHOD_MAP.values())
    combined_df = combined_df[combined_df['Method_Name_Cleaned'].isin(known_methods)]

    # Fill Missing/Invalid UQ Data with NaN
    for col in METRICS_TO_PLOT.keys():
        combined_df[col] = combined_df[col].replace([np.inf, -np.inf, 1.685e14, 0.000e0], np.nan)

    combined_df.dropna(subset=list(METRICS_TO_PLOT.keys()), how='all', inplace=True)

    return combined_df


# --- Plotting Function (Modified for x-axis order) ---

def plot_metrics_per_dataset(df):
    """
    Generates one figure per dataset, with 5 subplots comparing methods
    across 5 metrics, showing the mean value, distinct colors, and a fixed X-axis order.
    """
    n_metrics = len(METRICS_TO_PLOT)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    sns.set_theme(style="whitegrid", font_scale=1.1)

    for dataset in df['Dataset'].unique():
        df_dataset = df[df['Dataset'] == dataset].copy()

        if df_dataset.empty:
            print(f"No valid data remaining for dataset: {dataset.upper()}. Skipping plot.")
            continue

        # Compute the MEAN of the metrics across runs for the bar height.
        df_mean = df_dataset.groupby('Method_Name_Cleaned')[list(METRICS_TO_PLOT.keys())].mean().reset_index()

        if 'Method_Name_Cleaned' not in df_mean.columns or df_mean.empty:
            print(f"Error: Grouping by 'Method_Name_Cleaned' failed for {dataset.upper()}. Check data content.")
            continue

        # Get the list of methods present in the data, filtered by the desired order
        present_methods = [method for method in FINAL_LABEL_ORDER if method in df_mean['Method_Name_Cleaned'].values]

        # 2. Setup figure for 5 metrics
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
        axes = axes.flatten()
        fig.suptitle(f'Method Comparison for Dataset: {dataset.upper()}', fontsize=20, y=1.03)

        # Determine the number of unique methods for the color palette
        n_methods = len(present_methods)
        palette = sns.color_palette("tab20", n_methods)
        # Create a color map based on the order of present_methods
        color_map = {method: palette[i] for i, method in enumerate(present_methods)}

        for i, (metric_col, title) in enumerate(METRICS_TO_PLOT.items()):
            ax = axes[i]

            # 3. Plot Bar Plot with HUE and explicit X-AXIS ORDER
            sns.barplot(
                data=df_mean,
                x='Method_Name_Cleaned',
                y=metric_col,
                ax=ax,
                hue='Method_Name_Cleaned',
                palette=color_map,
                errorbar=None,
                legend=False,
                order=present_methods  # <-- THIS ENSURES THE FIXED ORDER
            )

            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Method", fontsize=12)
            ax.set_ylabel(metric_col, fontsize=12)

            # Rotate x-axis labels to 90 degrees
            ax.tick_params(axis='x', rotation=90)

            # Specific Annotations for Clarity
            if metric_col == 'empirical_coverage_mean':
                ax.axhline(0.95, color='r', linestyle='--', linewidth=1, label='Ideal Coverage (95%)')
                ax.legend(loc='lower left', fontsize=10)

            LOWER_IS_BETTER_METRICS = ['Brier_mean', 'ECE_mean', 'NLL_mean', 'MSE_mean', 'ce_mean']

            if not df_mean[metric_col].isnull().all():
                # Check if the current metric is in our 'Lower' list
                is_lower_better = metric_col in LOWER_IS_BETTER_METRICS

                if is_lower_better:
                    best_method = df_mean.loc[df_mean[metric_col].idxmin(), 'Method_Name_Cleaned']
                    color = 'darkgreen'
                else:
                    best_method = df_mean.loc[df_mean[metric_col].idxmax(), 'Method_Name_Cleaned']
                    color = 'darkred'

                ax.text(0.02, 0.98, f'Best: {best_method}',
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', color=color)

            # # Add a small text annotation showing the minimum/maximum value
            # if not df_mean[metric_col].isnull().all():
            #     is_lower_better = 'Lower is Better' in title
            #
            #     if is_lower_better:
            #         best_method = df_mean.loc[df_mean[metric_col].idxmin(), 'Method_Name_Cleaned']
            #         color = 'darkgreen'
            #     else:
            #         best_method = df_mean.loc[df_mean[metric_col].idxmax(), 'Method_Name_Cleaned']
            #         color = 'darkred'
            #
            #     ax.text(0.02, 0.98, f'Best: {best_method}',
            #             transform=ax.transAxes, fontsize=10,
            #             verticalalignment='top', color=color)

        # 4. Hide any unused subplots (the 6th slot)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # 5. Add a single legend for the entire figure
        # Create legend handles/labels in the correct order
        legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=color_map[method]) for method in present_methods]

        fig.legend(legend_handles, present_methods,
                   title="Method",
                   loc='lower center',
                   bbox_to_anchor=(0.5, -0.05),
                   ncol=5,  # Adjust number of columns for better display
                   fontsize=10)

        # Save the figure
        plt.tight_layout(rect=[0, 0.05, 1.0, 1.0])
        output_plot_file = os.path.join(DATA_DIR, f"Metrics_Comparison_{SPLIT}_{dataset.upper()}_Ordered.png")
        plt.savefig(output_plot_file, bbox_inches='tight')
        print(f"\nFigure saved for {dataset.upper()} to: {output_plot_file}")
        plt.close(fig)


# ... (Keep all your existing imports, configurations, and functions above) ...

# --- NEW FUNCTIONality 1: Per Task Plotting ---

# ... (Keep all your existing configuration and mapping code here) ...

def process_per_task_plots():
    """
    Functionality 1: Generates individual plots for each Task ID (id_0, id_1, etc.)
    by combining NN and GP results for that specific task.
    """
    print("\n--- Generating Per-Task Plots ---")
    for dataset in DATASETS:
        for split in SPLITS:
            for tid in TARGET_IDS:
                task_label = f"{dataset}_task_id_{tid}"
                task_dfs = []

                for m_type in MODEL_TYPES:
                    fname = f"{m_type}_{split}_{dataset}_{TASK_STR}_id_{tid}_c.csv"
                    fpath = os.path.join(SOURCE_DIR, fname)

                    if os.path.exists(fpath):
                        df = pd.read_csv(fpath)
                        # Identify and clean using your existing logic
                        method_col = 'Method' if 'Method' in df.columns else df.columns[-1]
                        df['Method_Name_Cleaned'] = df[method_col].astype(str).apply(lambda x: METHOD_MAP.get(x, x))
                        df['Dataset'] = task_label
                        task_dfs.append(df)

                if task_dfs:
                    combined = pd.concat(task_dfs, ignore_index=True)
                    plot_metrics_per_dataset(combined)


def process_average_task_plots():
    """
    Functionality 2: Averages the results across all TARGET_IDS
    and generates a single summary plot for the multi-task configuration.
    """
    print("\n--- Generating Average-Task Plots ---")
    for dataset in DATASETS:
        for split in SPLITS:
            all_models_avg = []

            for m_type in MODEL_TYPES:
                model_task_dfs = []
                for tid in TARGET_IDS:
                    fname = f"{m_type}_{split}_{dataset}_{TASK_STR}_id_{tid}_c.csv"
                    fpath = os.path.join(SOURCE_DIR, fname)
                    if os.path.exists(fpath):
                        model_task_dfs.append(pd.read_csv(fpath))

                if model_task_dfs:
                    # Merge id_0 and id_1 then average
                    temp_combined = pd.concat(model_task_dfs, ignore_index=True)
                    method_col = 'Method' if 'Method' in temp_combined.columns else temp_combined.columns[-1]

                    avg_df = temp_combined.groupby(method_col, as_index=False).mean(numeric_only=True)
                    avg_df['Method_Name_Cleaned'] = avg_df[method_col].astype(str).apply(lambda x: METHOD_MAP.get(x, x))
                    avg_df['Dataset'] = f"{dataset}_AVERAGE"
                    all_models_avg.append(avg_df)

            if all_models_avg:
                final_df = pd.concat(all_models_avg, ignore_index=True)
                plot_metrics_per_dataset(final_df)


# --- Logic Correction for the Best Method Annotation ---
# (Note: Apply this logic inside your plot_metrics_per_dataset function
# where it handles the 'Best' text annotation)

"""
REPLACEMENT LOGIC FOR THE 'BEST' TEXT IN plot_metrics_per_dataset:

# Define which metrics follow 'Lower is Better'
LOWER_IS_BETTER_METRICS = ['Brier_mean', 'ECE_mean', 'NLL_mean', 'MSE_mean', 'ce_mean']

if not df_mean[metric_col].isnull().all():
    # Check if the current metric is in our 'Lower' list
    is_lower_better = metric_col in LOWER_IS_BETTER_METRICS

    if is_lower_better:
        best_method = df_mean.loc[df_mean[metric_col].idxmin(), 'Method_Name_Cleaned']
        color = 'darkgreen'
    else:
        best_method = df_mean.loc[df_mean[metric_col].idxmax(), 'Method_Name_Cleaned']
        color = 'darkred'

    ax.text(0.02, 0.98, f'Best: {best_method}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color=color)
"""

if __name__ == "__main__":
    try:
        # Run both functionalities
        process_per_task_plots()
        process_average_task_plots()

    except Exception as e:
        print(f"\nAn error occurred: {e}")

# if __name__ == "__main__":
#     try:
#         final_df = load_and_clean_data(DATASETS, SPLIT)
#         plot_metrics_per_dataset(final_df)
#
#     except Exception as e:
#         print(f"\nAn error occurred during plot generation: {e}")
#         print("Please check your input CSV files, column names, and the DATASETS list.")