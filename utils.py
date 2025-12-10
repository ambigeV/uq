import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Configuration (Keep this section the same) ---
DATASET_NAME = "lipo"  
SPLIT_TYPE = "scaffold"     
RUN_IDS = list(range(2)) # List of run indices (e.g., [0, 1, 2, ..., 9] for 10 trials)
ERROR_METRICS = ['RMSE', 'MAE'] 
BASE_DIR = "./data/figure" 

# --- Data Loading (Keep this function the same) ---
def load_and_combine_data_multi_run():
    """Loads all NN and GP data files for all specified RUN_IDS and combines them."""
    
    all_dfs = []
    
    # ... (Loading logic from previous response) ...
    for run_id in RUN_IDS:
        nn_file = os.path.join(BASE_DIR, f"{SPLIT_TYPE}_{DATASET_NAME}_NN_cutoff_run_{run_id}.csv")
        gp_file = os.path.join(BASE_DIR, f"{SPLIT_TYPE}_{DATASET_NAME}_GP_cutoff_run_{run_id}.csv")

        for file_path in [nn_file]:
            try:
                df = pd.read_csv(file_path)
                df['Run_ID'] = run_id
                all_dfs.append(df)
            except FileNotFoundError:
                pass
            except pd.errors.EmptyDataError:
                pass

    if not all_dfs:
        raise ValueError("Could not load data from any cutoff file across all runs.")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

# --- Plotting Function (Modified for External Legend) ---
def plot_confidence_with_bands(df, error_metric, dataset_name, run_id):
    """Generates a Confidence vs. Error plot with uncertainty bands and a legend below the plot."""

    if error_metric not in df.columns:
        print(f"Skipping plot for {error_metric}: Column not found in combined data.")
        return

    sns.set_theme(style="whitegrid")
    
    # Calculate Mean and Standard Deviation for plotting
    summary_df = df.groupby(['Method', 'Confidence_Cutoff'])[error_metric].agg(
        mean=np.mean, 
        std=np.std
    ).reset_index()
    
    # Calculate the confidence band (mean +/- 1 standard deviation)
    summary_df['Upper'] = summary_df['mean'] + summary_df['std']
    summary_df['Lower'] = summary_df['mean'] - summary_df['std']

    # 1. Initialize the plot
    # We use a large figure size and then adjust the subplot to make space for the legend.
    fig, ax = plt.subplots(figsize=(10, 6)) 

    # 2. Plotting loop
    methods = summary_df['Method'].unique()
    
    # Use the color cycle from the current style
    palette = sns.color_palette(n_colors=len(methods))
    color_map = dict(zip(methods, palette))

    for method in methods:
        subset = summary_df[summary_df['Method'] == method]
        color = color_map[method]
        
        # Plot the mean line
        ax.plot(subset['Confidence_Cutoff'], subset['mean'], label=method, color=color, linewidth=2)
        
        # Plot the uncertainty band (std dev)
        ax.fill_between(
            subset['Confidence_Cutoff'], 
            subset['Lower'], 
            subset['Upper'], 
            color=color, 
            alpha=0.2, 
            label=None
        )

    # 3. Enhance the plot aesthetics
    num_runs = len(RUN_IDS)
    plt.title(f'Confidence vs. {error_metric} on {dataset_name} (n={num_runs} runs)', fontsize=16)
    plt.xlabel('Confidence Cutoff (Fraction of Most Confident Data Included)', fontsize=12)
    plt.ylabel(f'Prediction Error ({error_metric})', fontsize=12)
    
    plt.xlim(0.0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 4. Move the Legend to a Single Row at the Bottom (Mimicking the example)
    
    # Set the legend location (loc) and anchor point (bbox_to_anchor)
    # The anchor (0.5, -0.2) places the center of the legend box 20% below the plot

    MAX_LEGEND_COLUMNS = 4 

    # Set the legend location (loc) and anchor point (bbox_to_anchor)
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15), # Positioned below the X-axis label
        fancybox=True, 
        shadow=True, 
        ncol=MAX_LEGEND_COLUMNS, # <-- CRUCIAL: Sets a fixed column limit for wrapping
        title="Method"
    )

    # 5. Save the plot
    # Use bbox_inches='tight' to ensure the external legend is fully included in the file.
    output_plot_file = os.path.join(BASE_DIR, f"{SPLIT_TYPE}_{DATASET_NAME}_Confidence_{error_metric}_BANDS.png")
    plt.savefig(output_plot_file, bbox_inches='tight') # <-- Essential for external legend
    print(f"\nPlot saved successfully to: {output_plot_file}")
    plt.close()


# --- Main Execution ---
if __name__ == "__main__":
    try:
        if len(RUN_IDS) < 2:
            print("Warning: RUN_IDS list has less than 2 runs. Uncertainty bands will be based on 1 run and may be inaccurate.")

        final_df = load_and_combine_data_multi_run()
        
        # Plot each requested error metric separately
        for metric in ERROR_METRICS:
            plot_confidence_with_bands(final_df, metric, DATASET_NAME, RUN_IDS[0])
            
    except Exception as e:
        print(f"\nAn error occurred during plot generation: {e}")
        print("Ensure the CSV files exist for ALL specified RUN_IDS.")