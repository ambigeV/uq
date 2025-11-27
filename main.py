import deepchem as dc
from data_utils import prepare_datasets, evaluate_uq_metrics_from_interval
from nn_baseline import train_nn_baseline
from deepchem.molnet import load_qm7, load_delaney, load_qm8, load_qm9, load_lipo, load_freesolv

import torch
import gpytorch
from gp_single import GPyTorchRegressor, SVGPModel
from gp_trainer import GPTrainer, EnsembleGPTrainer
import numpy as np


def load_dataset():
    FEATURIZER = "coulomb"  # not ECFP -> will flatten (N, A, A) -> (N, A*A)
    tasks, datasets, transformers = load_qm7()
    # tasks, datasets, transformers = load_qm8()
    #
    # FEATURIZER = "ecfp"  # not ECFP -> will flatten (N, A, A) -> (N, A*A)
    # # tasks, datasets, transformers = load_delaney()
    # tasks, datasets, transformers = load_lipo()

    train_dc, valid_dc, test_dc = prepare_datasets(datasets, featurizer_name=FEATURIZER)

    if train_dc.y.ndim == 2 and train_dc.y.shape[1] > 1:
        first_task_name = tasks[0]
        tasks = [first_task_name]

        def _keep_first_task(ds) -> dc.data.NumpyDataset:
            # y: (N, T) -> (N, 1) taking the first column
            return dc.data.NumpyDataset(
                X=ds.X,
                y=ds.y[:, 0:1],
                w=ds.w[:, 0:1],
                ids=ds.ids,
                n_tasks=1
            )

        train_dc = _keep_first_task(train_dc)
        valid_dc = _keep_first_task(valid_dc)
        test_dc = _keep_first_task(test_dc)

    return tasks, train_dc, valid_dc, test_dc, transformers


def main_gp():
    # 1. Load single-task data (QM7 + Coulomb matrices flattened)
    tasks, train_dc, valid_dc, test_dc, transformers = load_dataset()

    print("Train X shape:", train_dc.X.shape)  # (N, D) after flattening
    print("Train y shape:", train_dc.y.shape)  # (N, 1) or (N,)

    # 2. Build GP model
    X_train_torch = torch.from_numpy(train_dc.X).float()
    # single-task, ensure (N,) for train_y
    y_np = train_dc.y
    if y_np.ndim == 2 and y_np.shape[1] == 1:
        y_np = y_np[:, 0]
    y_train_torch = torch.from_numpy(y_np).float()

    gp_model = GPyTorchRegressor(X_train_torch, y_train_torch)

    # 3. Train GP
    trainer = GPTrainer(
        model=gp_model,
        train_dataset=train_dc,
        lr=0.05,
        # num_iters=500,
        num_iters=200,
        device="cpu",
        log_interval=10,
    )
    trainer.train()

    # 4. Evaluate
    valid_mse = trainer.evaluate_mse(valid_dc)
    test_mse = trainer.evaluate_mse(test_dc)
    print("[GP Single-task] Validation MSE:", valid_mse)
    print("[GP Single-task] Test MSE:",      test_mse)

    # 5. Get uncertainty intervals on test set
    mean_test, lower_test, upper_test = trainer.predict_interval(test_dc, alpha=0.05)

    uq_metrics = evaluate_uq_metrics_from_interval(
        y_true=test_dc.y,
        mean=mean_test,
        lower=lower_test,
        upper=upper_test,
        alpha=0.05,
    )
    print("UQ:", uq_metrics)


def main_svgp():
    # 1. Load single-task data (same as main_gp)
    tasks, train_dc, valid_dc, test_dc, transformers = load_dataset()

    print("Train X shape:", train_dc.X.shape)  # (N, D)
    N, D = train_dc.X.shape
    print("Train y shape:", train_dc.y.shape)  # (N, 1) or (N,)

    # 2. Build SVGP model
    X_train_torch = torch.from_numpy(train_dc.X).float()
    y_np = train_dc.y
    if y_np.ndim == 2 and y_np.shape[1] == 1:
        y_np = y_np[:, 0]
    y_train_torch = torch.from_numpy(y_np).float()

    gp_model = GPyTorchRegressor(
        train_x=X_train_torch,
        train_y=y_train_torch,
        gp_model_cls=SVGPModel,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        num_inducing=max(128, N//20),
        kmeans_iters=15,    # tune if you like
    )

    # 3. Train SVGP
    trainer = GPTrainer(
        model=gp_model,
        train_dataset=train_dc,
        lr=0.01,           # often a bit smaller LR works better for SVGP
        num_iters=500,     # more iters than exact GP is common; adjust as needed
        device="cpu",
        log_interval=10,
    )
    trainer.train()

    # 4. Evaluate
    valid_mse = trainer.evaluate_mse(valid_dc)
    test_mse = trainer.evaluate_mse(test_dc)
    print("[SVGP] Validation MSE:", valid_mse)
    print("[SVGP] Test MSE:",      test_mse)

    # 5. Get uncertainty intervals on test set
    mean_test, lower_test, upper_test = trainer.predict_interval(test_dc, alpha=0.05)

    uq_metrics = evaluate_uq_metrics_from_interval(
        y_true=test_dc.y,
        mean=mean_test,
        lower=lower_test,
        upper=upper_test,
        alpha=0.05,
    )
    print("UQ (SVGP):", uq_metrics)


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


def main_svgp_ensemble():
    tasks, train_dc, valid_dc, test_dc, transformers = load_dataset()

    print("Train X shape:", train_dc.X.shape)
    N, D = train_dc.X.shape
    print("Train y shape:", train_dc.y.shape)

    folds = np.array_split(np.arange(N), 5)

    items = []
    for i in range(5):
        # indices for training this base model: all folds except i
        train_idx_i = np.concatenate([folds[j] for j in range(5) if j != i])
        ds_i = _subset_numpy_dataset(train_dc, train_idx_i)

        # tensors for model init (wrapper learns x/y stats here)
        X_train_torch = torch.from_numpy(ds_i.X).float()
        y_np = ds_i.y
        if y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np[:, 0]
        y_train_torch = torch.from_numpy(y_np).float()

        Ni = X_train_torch.shape[0]
        gp_model_i = GPyTorchRegressor(
            train_x=X_train_torch,
            train_y=y_train_torch,
            gp_model_cls=SVGPModel,
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            num_inducing=max(128, Ni // 20),  # same params across models
            kmeans_iters=15,
        )

        items.append({
            "model": gp_model_i,
            "train_dataset": ds_i,   # trainer will use normalized y internally
            "lr": 0.01,
            "num_iters": 500,
            "log_interval": 10,
        })

    # 4) Train ensemble
    ens = EnsembleGPTrainer(items, device="cpu")
    ens.train()

    # 5) Evaluate (mixture mean via moment matching)
    valid_mse = ens.evaluate_mse(valid_dc)
    test_mse  = ens.evaluate_mse(test_dc)
    print("[SVGP-Ens] Validation MSE:", valid_mse)
    print("[SVGP-Ens] Test MSE:",      test_mse)

    # 6) UQ from mixture (Gaussian approx via matched moments)
    mean_test, lower_test, upper_test = ens.predict_interval(test_dc, alpha=0.05)

    uq_metrics = evaluate_uq_metrics_from_interval(
        y_true=test_dc.y,
        mean=mean_test,
        lower=lower_test,
        upper=upper_test,
        alpha=0.05,
    )
    print("UQ (SVGP-Ens):", uq_metrics)


def main_nn():
    tasks, train_dc, valid_dc, test_dc, transformers = load_dataset()

    train_nn_baseline(train_dc, valid_dc, test_dc)


if __name__ == "__main__":
    # main_nn()
    # main_gp()
    main_svgp_ensemble()
