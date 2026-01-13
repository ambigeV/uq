"""
Helper utilities for saving and loading model parameters.
Supports both Neural Network models (DeepChem TorchModel) and GPyTorch models.
"""
import torch
import os
from typing import Optional, Union
import deepchem as dc


def save_neural_network_ensemble(
    models: list,
    save_path: str,
    model_name: str = "ensemble",
    mode: str = "regression",
    create_dir: bool = True
) -> str:
    """
    Save a deep ensemble (list of models) with full metadata for reconstruction.
    
    Args:
        models: List of DeepChem TorchModel instances
        save_path: Directory path where the ensemble should be saved
        model_name: Base name for the ensemble (without extension)
        mode: "regression" or "classification"
        create_dir: If True, create the directory if it doesn't exist
        
    Returns:
        str: Full path to the saved ensemble metadata file
        
    Example:
        >>> models = [dc_model1, dc_model2, ...]
        >>> save_neural_network_ensemble(models, "./saved_models", "nn_deep_ensemble_regression_run_0")
    """
    if create_dir:
        os.makedirs(save_path, exist_ok=True)
    
    # Save each member model
    member_paths = []
    for i, dc_model in enumerate(models):
        member_path = save_neural_network_model(
            dc_model,
            save_path,
            model_name=f"{model_name}_member_{i}",
            create_dir=False  # Already created
        )
        member_paths.append(member_path)
    
    # Save ensemble metadata
    ensemble_metadata_path = os.path.join(save_path, f"{model_name}_metadata.pt")
    ensemble_dict = {
        'ensemble_type': 'neural_network',
        'mode': mode,
        'num_members': len(models),
        'member_paths': member_paths,
        'model_name': model_name,
    }
    
    # Save model configuration from first member (assuming all have same config)
    if len(models) > 0:
        first_model = models[0]
        if hasattr(first_model, 'mode'):
            ensemble_dict['model_mode'] = first_model.mode
        if hasattr(first_model, 'batch_size'):
            ensemble_dict['batch_size'] = first_model.batch_size
        if hasattr(first_model, 'learning_rate'):
            ensemble_dict['learning_rate'] = first_model.learning_rate
    
    torch.save(ensemble_dict, ensemble_metadata_path)
    print(f"Ensemble metadata saved to: {ensemble_metadata_path}")
    
    return ensemble_metadata_path


def load_neural_network_ensemble(
    ensemble_metadata_path: str,
    model_class: torch.nn.Module,
    n_features: int,
    n_tasks: int,
    mode: str = "regression"
) -> list:
    """
    Load a saved neural network ensemble from disk.
    
    Args:
        ensemble_metadata_path: Path to the ensemble metadata file
        model_class: The PyTorch model class (e.g., MyTorchRegressor)
        n_features: Number of input features (needed to reconstruct model)
        n_tasks: Number of tasks (needed to reconstruct model)
        mode: "regression" or "classification" (needed for loss function)
        
    Returns:
        list: List of loaded DeepChem TorchModel instances
    """
    if not os.path.exists(ensemble_metadata_path):
        raise FileNotFoundError(f"Ensemble metadata file not found: {ensemble_metadata_path}")
    
    ensemble_dict = torch.load(ensemble_metadata_path, map_location='cpu')
    member_paths = ensemble_dict['member_paths']
    
    # Reconstruct dc_model_kwargs from metadata
    if mode == "regression":
        loss = dc.models.losses.L2Loss()
        output_types = ['prediction']
    else:
        loss = dc.models.losses.SigmoidCrossEntropy()
        output_types = ['prediction']
    
    dc_model_kwargs = {
        'loss': loss,
        'output_types': output_types,
        'batch_size': ensemble_dict.get('batch_size', 64),
        'learning_rate': ensemble_dict.get('learning_rate', 1e-3),
        'mode': mode
    }
    
    models = []
    for member_path in member_paths:
        # Create a new model instance
        model_instance = model_class(n_features, n_tasks)
        model = load_neural_network_model(model_instance, member_path, dc_model_kwargs)
        models.append(model)
    
    print(f"Loaded ensemble with {len(models)} members from: {ensemble_metadata_path}")
    return models


def save_neural_network_model(
    dc_model: dc.models.TorchModel,
    save_path: str,
    model_name: str = "model",
    create_dir: bool = True
) -> str:
    """
    Save a DeepChem TorchModel (neural network) to disk.
    
    Args:
        dc_model: The DeepChem TorchModel instance to save
        save_path: Directory path where the model should be saved
        model_name: Name for the saved model file (without extension)
        create_dir: If True, create the directory if it doesn't exist
        
    Returns:
        str: Full path to the saved model file
        
    Example:
        >>> dc_model = dc.models.TorchModel(model=my_model, loss=loss, ...)
        >>> dc_model.fit(train_dc, nb_epoch=100)
        >>> save_neural_network_model(dc_model, "./saved_models", "nn_baseline_run_0")
    """
    if create_dir:
        os.makedirs(save_path, exist_ok=True)
    
    # Full path to the model file
    full_path = os.path.join(save_path, f"{model_name}.pt")
    
    # Save the underlying PyTorch model's state dict
    # DeepChem TorchModel wraps the model in dc_model.model
    model_state = dc_model.model.state_dict()
    
    # Also save optimizer state if available (optional but useful for resuming training)
    optimizer_state = None
    if hasattr(dc_model, '_pytorch_optimizer') and dc_model._pytorch_optimizer is not None:
        optimizer_state = dc_model._pytorch_optimizer.state_dict()
    
    # Infer n_features and n_tasks from model architecture
    n_features = None
    n_tasks = None
    if hasattr(dc_model.model, 'net'):
        # MyTorchRegressor/MyTorchClassifier structure
        first_layer = dc_model.model.net[0]
        if isinstance(first_layer, torch.nn.Linear):
            n_features = first_layer.in_features
        last_layer = dc_model.model.net[-1]
        if isinstance(last_layer, torch.nn.Linear):
            n_tasks = last_layer.out_features
    
    # Create a dictionary with all relevant information
    save_dict = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'model_type': 'neural_network',
        'n_features': n_features,
        'n_tasks': n_tasks,
    }
    
    # Save additional metadata if available
    if hasattr(dc_model, 'mode'):
        save_dict['mode'] = dc_model.mode
    if hasattr(dc_model, 'batch_size'):
        save_dict['batch_size'] = dc_model.batch_size
    if hasattr(dc_model, 'learning_rate'):
        save_dict['learning_rate'] = dc_model.learning_rate
    
    # Save to disk
    torch.save(save_dict, full_path)
    print(f"Neural network model saved to: {full_path}")
    
    return full_path


def load_neural_network_model(
    model_instance: torch.nn.Module,
    load_path: str,
    dc_model_kwargs: Optional[dict] = None
) -> dc.models.TorchModel:
    """
    Load a saved DeepChem TorchModel from disk.
    
    Args:
        model_instance: A PyTorch model instance (e.g., MyTorchRegressor(n_features, n_tasks))
        load_path: Full path to the saved model file (.pt)
        dc_model_kwargs: Optional dictionary of kwargs to pass to TorchModel constructor
                        (e.g., {'loss': loss, 'output_types': ['prediction'], ...})
                        If None, will try to reconstruct from saved metadata
        
    Returns:
        dc.models.TorchModel: The loaded DeepChem model
        
    Example:
        >>> model = MyTorchRegressor(n_features=1024, n_tasks=1)
        >>> dc_model = load_neural_network_model(
        ...     model, "./saved_models/nn_baseline_run_0.pt",
        ...     dc_model_kwargs={'loss': dc.models.losses.L2Loss(), ...}
        ... )
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found: {load_path}")
    
    # Load the saved dictionary
    checkpoint = torch.load(load_path, map_location='cpu')
    
    # Load the state dict into the provided model instance
    model_instance.load_state_dict(checkpoint['model_state_dict'])
    
    # Recreate the DeepChem model wrapper
    if dc_model_kwargs is None:
        # Try to reconstruct from saved metadata
        mode = checkpoint.get('mode', 'regression')
        if mode == 'regression':
            loss = dc.models.losses.L2Loss()
            output_types = ['prediction']
        else:
            loss = dc.models.losses.SigmoidCrossEntropy()
            output_types = ['prediction']
        
        dc_model_kwargs = {
            'loss': loss,
            'output_types': output_types,
            'batch_size': checkpoint.get('batch_size', 64),
            'learning_rate': checkpoint.get('learning_rate', 1e-3),
            'mode': mode
        }
    
    dc_model = dc.models.TorchModel(model=model_instance, **dc_model_kwargs)
    
    # Restore optimizer state if available
    if checkpoint.get('optimizer_state_dict') is not None:
        if hasattr(dc_model, '_pytorch_optimizer') and dc_model._pytorch_optimizer is not None:
            dc_model._pytorch_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Neural network model loaded from: {load_path}")
    
    return dc_model


def save_gpytorch_model(
    gp_model: Union['GPyTorchRegressor', 'GPyTorchClassifier'],
    trainer: Optional[Union['GPTrainer', 'GPClassificationTrainer']] = None,
    save_path: str = "./saved_models",
    model_name: str = "gp_model",
    create_dir: bool = True
) -> str:
    """
    Save a GPyTorch model (GPyTorchRegressor or GPyTorchClassifier) to disk.
    
    Args:
        gp_model: The GPyTorchRegressor or GPyTorchClassifier instance to save
        trainer: Optional trainer instance (to save optimizer states)
        save_path: Directory path where the model should be saved
        model_name: Name for the saved model file (without extension)
        create_dir: If True, create the directory if it doesn't exist
        
    Returns:
        str: Full path to the saved model file
        
    Example:
        >>> gp_model = GPyTorchRegressor(train_x=X, train_y=y, ...)
        >>> trainer = GPTrainer(model=gp_model, train_dataset=train_dc, ...)
        >>> trainer.train()
        >>> save_gpytorch_model(gp_model, trainer, "./saved_models", "gp_exact_run_0")
    """
    if create_dir:
        os.makedirs(save_path, exist_ok=True)
    
    # Full path to the model file
    full_path = os.path.join(save_path, f"{model_name}.pt")
    
    # Save the GP model's state dict
    model_state = gp_model.state_dict()
    
    # Save optimizer states if trainer is provided
    optimizer_states = {}
    if trainer is not None:
        # Save Adam optimizer state if available
        if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
            optimizer_states['adam'] = trainer.optimizer.state_dict()
        if hasattr(trainer, 'opt_adam') and trainer.opt_adam is not None:
            optimizer_states['adam'] = trainer.opt_adam.state_dict()
        # Save NGD optimizer state if available (for SVGP)
        if hasattr(trainer, 'opt_ngd') and trainer.opt_ngd is not None:
            optimizer_states['ngd'] = trainer.opt_ngd.state_dict()
    
    # Create a comprehensive save dictionary
    save_dict = {
        'model_state_dict': model_state,
        'optimizer_states': optimizer_states if optimizer_states else None,
        'model_type': 'gpytorch',
        'model_class': type(gp_model).__name__,  # 'GPyTorchRegressor' or 'GPyTorchClassifier'
    }
    
    # Save model configuration
    if hasattr(gp_model, 'num_tasks'):
        save_dict['num_tasks'] = gp_model.num_tasks
    if hasattr(gp_model, 'normalize_x'):
        save_dict['normalize_x'] = gp_model.normalize_x
    if hasattr(gp_model, 'use_weights'):
        save_dict['use_weights'] = gp_model.use_weights
    
    # Save normalization statistics (important for inference)
    if hasattr(gp_model, 'x_mean'):
        save_dict['x_mean'] = gp_model.x_mean
    if hasattr(gp_model, 'x_std'):
        save_dict['x_std'] = gp_model.x_std
    if hasattr(gp_model, 'y_mean'):
        save_dict['y_mean'] = gp_model.y_mean
    if hasattr(gp_model, 'y_std'):
        save_dict['y_std'] = gp_model.y_std
    
    # Save training data references (optional - can be large, so you might want to skip this)
    # save_dict['train_x'] = gp_model.train_x_raw if hasattr(gp_model, 'train_x_raw') else None
    # save_dict['train_y'] = gp_model.train_y_raw if hasattr(gp_model, 'train_y_raw') else None
    
    # Save to disk
    torch.save(save_dict, full_path)
    print(f"GPyTorch model saved to: {full_path}")
    
    return full_path


def load_gpytorch_model(
    load_path: str,
    gp_model: Optional[Union['GPyTorchRegressor', 'GPyTorchClassifier']] = None
) -> Union['GPyTorchRegressor', 'GPyTorchClassifier']:
    """
    Load a saved GPyTorch model from disk.
    
    Args:
        load_path: Full path to the saved model file (.pt)
        gp_model: Optional pre-initialized model instance to load state into.
                 If None, you'll need to manually reconstruct the model.
        
    Returns:
        The loaded GPyTorch model (or the model with loaded state if gp_model was provided)
        
    Note:
        For full reconstruction, you typically need to:
        1. Create a new GPyTorchRegressor/GPyTorchClassifier with the same parameters
        2. Call this function with that model instance
        
    Example:
        >>> # Option 1: Load into existing model
        >>> gp_model = GPyTorchRegressor(train_x=X, train_y=y, ...)
        >>> gp_model = load_gpytorch_model("./saved_models/gp_exact_run_0.pt", gp_model)
        
        >>> # Option 2: Load checkpoint and manually reconstruct
        >>> checkpoint = torch.load("./saved_models/gp_exact_run_0.pt")
        >>> # Use checkpoint['x_mean'], checkpoint['y_mean'], etc. to reconstruct
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found: {load_path}")
    
    # Load the saved dictionary
    checkpoint = torch.load(load_path, map_location='cpu')
    
    if checkpoint.get('model_type') != 'gpytorch':
        raise ValueError(f"File {load_path} does not appear to be a GPyTorch model")
    
    # If a model instance is provided, load state into it
    if gp_model is not None:
        gp_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"GPyTorch model state loaded from: {load_path}")
        return gp_model
    else:
        # Return the checkpoint for manual reconstruction
        print(f"GPyTorch model checkpoint loaded from: {load_path}")
        print("Note: Model instance not provided. Use checkpoint data to reconstruct the model.")
        return checkpoint

