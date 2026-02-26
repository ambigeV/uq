# Model Reload and Evaluation Guide

## Overview
This document describes the functionality for reloading saved neural network models and evaluating them on test data without retraining.

## Key Features

### 1. Model Saving with Naming Convention
Models are saved with a consistent naming convention:
- **Single Models**: `{model_type}_{mode}_run_{run_id}.pt`
  - Example: `nn_baseline_regression_run_0.pt`
  - Example: `nn_evd_classification_run_0.pt`
  
- **Ensemble Models**: 
  - Metadata: `nn_deep_ensemble_{mode}_run_{run_id}_metadata.pt`
  - Members: `nn_deep_ensemble_{mode}_run_{run_id}_member_{i}.pt`
  - Example: `nn_deep_ensemble_regression_run_0_metadata.pt`
  - Example: `nn_deep_ensemble_regression_run_0_member_0.pt`

### 2. Ensemble Saving
Ensembles save complete reconstruction information:
- All member models (individual `.pt` files)
- Metadata file containing:
  - Number of members
  - Paths to all member models
  - Model configuration (batch_size, learning_rate, mode)
  - Ensemble type and mode

### 3. GP Models
**Note**: GP models are NOT saved/reloaded due to memory constraints. GPyTorch models require training data for inference, which would be too expensive to save. If you need to evaluate GP models, you should retrain them.

## Usage

### Saving Models During Training

```python
from main import run_once_nn

# Train and save models
results = run_once_nn(
    dataset_name="delaney",
    seed=0,
    run_id=0,
    split="random",
    mode="regression",
    # Models will be saved if save_model=True in training functions
)
```

To enable saving, modify the training calls in `run_once_nn()` to pass `save_model=True`:

```python
# In run_once_nn() or when calling training functions directly:
train_nn_baseline(
    train_dc, valid_dc, test_dc,
    run_id=0,
    save_model=True,
    save_path="./saved_models"
)
```

### Reloading and Evaluating Models

```python
from main import evaluate_once_nn

# Evaluate saved models
results = evaluate_once_nn(
    dataset_name="delaney",
    run_id=0,
    split="random",
    mode="regression",
    use_weights=False,
    model_base_path="./saved_models"
)
```

The function will:
1. Load test data for the specified dataset
2. Search for saved models using the naming convention
3. Load each model type (nn_evd, nn_baseline, nn_mc_dropout, nn_deep_ensemble)
4. Evaluate them on test data
5. Return results in the same format as `run_once_nn()`
6. Save cutoff data to CSV files

## Function Reference

### Evaluation Functions (in `nn_baseline.py`)

#### `evaluate_nn_baseline(dc_model, test_dc, use_weights=False, mode="regression")`
Evaluates a baseline neural network model.
- Returns: UQ metrics dictionary

#### `evaluate_nn_evd(dc_model, test_dc, use_weights=False, mode="regression")`
Evaluates an evidential neural network model.
- Returns: `(uq_metrics, cutoff_error_df)`

#### `evaluate_nn_mc_dropout(dc_model, test_dc, n_samples=100, use_weights=False, mode="regression")`
Evaluates an MC-Dropout neural network model.
- Returns: `(uq_metrics, cutoff_error_df)`

#### `evaluate_nn_deep_ensemble(models, test_dc, use_weights=False, mode="regression")`
Evaluates a deep ensemble.
- Args: `models` - list of trained DeepChem TorchModel instances
- Returns: `(uq_metrics, cutoff_error_df)`

### Reload Functions (in `model_utils.py`)

#### `load_neural_network_model(model_instance, load_path, dc_model_kwargs=None)`
Loads a single saved neural network model.
- Args: `model_instance` - A PyTorch model instance (e.g., `MyTorchRegressor(n_features, n_tasks)`)
- Returns: Loaded DeepChem TorchModel

#### `load_neural_network_ensemble(ensemble_metadata_path, model_class, n_features, n_tasks, mode)`
Loads a saved ensemble with all members.
- Returns: List of loaded DeepChem TorchModel instances

### Main Evaluation Function (in `main.py`)

#### `evaluate_once_nn(dataset_name, run_id, split, mode, use_weights, task_indices, model_base_path, model_prefix)`
Main function to reload and evaluate all saved models for a given run.
- Automatically discovers and loads models based on naming convention
- Returns results dictionary matching `run_once_nn()` format

## Model Path Structure

```
saved_models/
├── nn_baseline_regression_run_0.pt
├── nn_baseline_classification_run_0.pt
├── nn_evd_regression_run_0.pt
├── nn_evd_classification_run_0.pt
├── nn_mc_dropout_regression_run_0.pt
├── nn_mc_dropout_classification_run_0.pt
├── nn_deep_ensemble_regression_run_0_metadata.pt
├── nn_deep_ensemble_regression_run_0_member_0.pt
├── nn_deep_ensemble_regression_run_0_member_1.pt
├── nn_deep_ensemble_regression_run_0_member_2.pt
├── nn_deep_ensemble_regression_run_0_member_3.pt
└── nn_deep_ensemble_regression_run_0_member_4.pt
```

## Example Workflow

### Step 1: Train and Save Models
```python
from main import run_once_nn

# Train models (with saving enabled in training functions)
results = run_once_nn(
    dataset_name="delaney",
    seed=0,
    run_id=0,
    split="random",
    mode="regression"
)
# Models saved to ./saved_models/
```

### Step 2: Evaluate Saved Models
```python
from main import evaluate_once_nn

# Reload and evaluate
results = evaluate_once_nn(
    dataset_name="delaney",
    run_id=0,
    split="random",
    mode="regression",
    model_base_path="./saved_models"
)

# Results will have same format as run_once_nn()
print(results["nn_baseline"])
print(results["nn_evd"])
```

## Notes

1. **Model Reconstruction**: Models need to be reconstructed with the same architecture. The evaluation functions handle this automatically based on saved metadata.

2. **Missing Models**: If a model file is not found, the function will print a warning and skip that model type.

3. **Ensemble Loading**: Ensembles require all member models to be present. The metadata file contains paths to all members.

4. **GP Models**: GP models are not supported for reload/evaluation due to memory constraints. They require training data which is too expensive to save.

5. **Data Consistency**: The evaluation uses the same test data loading logic as training, ensuring consistent evaluation.

## Future Enhancements

- Add support for custom model paths/configurations
- Add model validation (check if model architecture matches saved data)
- Add support for partial ensemble evaluation
- Add caching of loaded models for multiple evaluations





