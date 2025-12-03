import deepchem as dc
from data_utils import prepare_datasets, evaluate_uq_metrics_from_interval
from nn_baseline import train_nn_baseline, train_nn_deep_ensemble, train_nn_mc_dropout
from deepchem.molnet import load_qm7, load_delaney, load_qm8, load_qm9, load_lipo, load_freesolv

import torch
import gpytorch
from gp_single import GPyTorchRegressor, SVGPModel, FeatureNet, \
    DeepFeatureKernel, NNGPExactGPModel, NNSVGPLearnedInducing
from gp_trainer import GPTrainer, EnsembleGPTrainer
import numpy as np


def load_dataset():
    FEATURIZER = "coulomb"  # not ECFP -> will flatten (N, A, A) -> (N, A*A)
    # tasks, datasets, transformers = load_qm7()
    tasks, datasets, transformers = load_qm8()
    #
    # FEATURIZER = "ecfp"  # not ECFP -> will flatten (N, A, A) -> (N, A*A)
    # tasks, datasets, transformers = load_delaney()
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


def _to_torch_xy(dc_dataset):
    X_t = torch.from_numpy(dc_dataset.X).float()
    y_np = dc_dataset.y
    if y_np.ndim == 2 and y_np.shape[1] == 1:
        y_np = y_np[:, 0]
    y_t = torch.from_numpy(y_np).float()
    return X_t, y_t


def main_nngp_exact(device: str = None):
    # 1) Load
    tasks, train_dc, valid_dc, test_dc, transformers = load_dataset()
    print("Train X shape:", train_dc.X.shape)
    print("Train y shape:", train_dc.y.shape)

    # 2) Torch tensors
    X_train_t, y_train_t = _to_torch_xy(train_dc)

    # 3) Build NNGP (Exact)
    in_dim = X_train_t.shape[1]
    feat = FeatureNet(in_dim=in_dim, feat_dim=64, hidden=(128, 64), dropout=0.0)

    nngp_exact = GPyTorchRegressor(
        train_x=X_train_t,
        train_y=y_train_t,
        normalize_x=False,                        # BatchNorm handles scale
        gp_model_cls=NNGPExactGPModel,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        feature_extractor=feat,
        kernel="matern52",
    )

    # 4) Train
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer = GPTrainer(
        model=nngp_exact,
        train_dataset=train_dc,
        lr=5e-3,               # smaller LR tends to work better for deep kernels (exact GP uses single Adam)
        num_iters=300,
        device=dev,
        log_interval=40,
    )
    trainer.train()

    # 5) Eval
    valid_mse = trainer.evaluate_mse(valid_dc)
    test_mse  = trainer.evaluate_mse(test_dc)
    print("[NNGP-Exact] Validation MSE:", valid_mse)
    print("[NNGP-Exact] Test MSE:",      test_mse)

    # 6) UQ (intervals on test)
    mean_t, lo_t, hi_t = trainer.predict_interval(test_dc, alpha=0.05)
    uq = evaluate_uq_metrics_from_interval(
        y_true=test_dc.y, mean=mean_t, lower=lo_t, upper=hi_t, alpha=0.05
    )
    print("UQ (NNGP-Exact):", uq)


def main_nngp_exact_ensemble_all():
    # ---------------- 1) Load data ----------------
    tasks, train_dc, valid_dc, test_dc, transformers = load_dataset()

    print("Train X shape:", train_dc.X.shape)
    N, D = train_dc.X.shape
    print("Train y shape:", train_dc.y.shape)
    # 2) Torch tensors
    X_train_t, y_train_t = _to_torch_xy(train_dc)

    # 3) Build NNGP (Exact)
    in_dim = X_train_t.shape[1]

    device = "cpu"
    # 4) Train
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- 2) Build K base models ----------------
    folds = np.array_split(np.arange(N), 5)

    items = []
    for i in range(5):
        # (keeps your current "two-folds per model" split)
        train_idx_i = np.concatenate([folds[j] for j in [i, (i + 1) % 5]])
        ds_i = _subset_numpy_dataset(train_dc, train_idx_i)

        X_train_torch = torch.from_numpy(ds_i.X).float()
        y_np = ds_i.y
        if y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np[:, 0]
        y_train_torch = torch.from_numpy(y_np).float()

        # Ni = X_train_torch.shape[0]
        feat = FeatureNet(in_dim=in_dim, feat_dim=64, hidden=(128, 64), dropout=0.0)
        gp_model_i = GPyTorchRegressor(
            train_x=X_train_torch,
            train_y=y_train_torch,
            normalize_x=False,                        # BatchNorm handles scale
            gp_model_cls=NNGPExactGPModel,
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            feature_extractor=feat,
            kernel="matern52")

        items.append({
            "model": gp_model_i,
            "train_dataset": ds_i,
            "lr": 5e-3,
            "num_iters": 300,
            "log_interval": 40,
        })

    # ---------------- 3) Train ensemble once ----------------
    ens = EnsembleGPTrainer(items, device="cpu")
    ens.train()
    K = len(ens.trainers)

    # ---------------- 4) Loop over weighting strategies ----------------
    strategies = [
        # name, kwargs, needs_labels?
        ("uniform", {}, False),
        ("precision", {"tau": 1.0}, False),
        ("mse",       {"l2": 1e-3}, True),
        ("nll",       {"l2": 1e-3, "dirichlet_alpha": 1.05}, True),
    ]

    results = []
    for name, kwargs, _needs_y in strategies:
        # (a) get weights
        if name == "uniform":
            w = np.ones(K, dtype=float) / K
            print(f"\n[Calib:{name}] Using uniform weights.")
        else:
            w = ens.calibrate_weights(valid_dc, method=name, **kwargs)
            print(f"\n[Calib:{name}] Weights: {np.round(w, 4)}")

        # (b) evaluate on valid/test with these weights
        valid_mse = ens.evaluate_mse_w(valid_dc, w=w)
        test_mse  = ens.evaluate_mse_w(test_dc,  w=w)
        print(f"[{name}] Validation MSE: {valid_mse:.6f}")
        print(f"[{name}] Test MSE:       {test_mse:.6f}")

        # (c) UQ on test via weighted moment-matched mixture
        mean_test, lower_test, upper_test = ens.predict_interval_w(test_dc, alpha=0.05, w=w)
        uq_metrics = evaluate_uq_metrics_from_interval(
            y_true=test_dc.y,
            mean=mean_test,
            lower=lower_test,
            upper=upper_test,
            alpha=0.05,
        )
        print(f"[{name}] UQ metrics: {uq_metrics}")

        results.append({
            "name": name,
            "weights": w,
            "valid_mse": valid_mse,
            "test_mse": test_mse,
            "uq_metrics": uq_metrics,
        })

    # ---------------- 5) Pretty summary ----------------
    print("\n===== Summary over weighting strategies =====")
    for r in results:
        w_str = " ".join([f"{x:.3f}" for x in r["weights"]])
        print(f"{r['name']:>9} | valid MSE: {r['valid_mse']:.6f} | test MSE: {r['test_mse']:.6f} | w: [{w_str}]")

    return results


def main_nngp_svgp(device: str = None):
    # 1) Load
    tasks, train_dc, valid_dc, test_dc, transformers = load_dataset()
    print("Train X shape:", train_dc.X.shape)
    N, D = train_dc.X.shape
    print("Train y shape:", train_dc.y.shape)

    # 2) Torch tensors
    X_train_t, y_train_t = _to_torch_xy(train_dc)

    # 3) Build NNGP (SVGP with learned inducing inputs)
    in_dim = D
    feat = FeatureNet(in_dim=in_dim, feat_dim=64, hidden=(128, 64), dropout=0.0)

    M = max(256, N // 20)      # e.g., ~5% of data or at least 256 (tune per dataset/capacity)
    nngp_svgp = GPyTorchRegressor(
        train_x=X_train_t,
        train_y=y_train_t,
        normalize_x=False,                        # BatchNorm handles scale
        gp_model_cls=NNSVGPLearnedInducing,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        feature_extractor=feat,
        num_inducing=M,
        kernel="matern52",
        # inducing_idx can be omitted to random-init inside the class
    )

    # 4) Train (SVGP branch uses NGD(q) + Adam(others) in your GPTrainer)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer = GPTrainer(
        model=nngp_svgp,
        train_dataset=train_dc,
        lr=5e-3,               # Adam LR for kernel/likelihood params
        nn_lr=1e-3,            # Adam LR for feature extractor
        ngd_lr=0.02,           # NGD LR for q(u) (un-whitened); try 0.01–0.05 if needed
        warmup_iters=5,        # freeze feature net briefly for stability
        clip_grad=1.0,
        num_iters=500,         # SVGP typically needs more iters
        device=dev,
        log_interval=40,
    )
    trainer.train()

    # 5) Eval
    valid_mse = trainer.evaluate_mse(valid_dc)
    test_mse  = trainer.evaluate_mse(test_dc)
    print("[NNGP-SVGP] Validation MSE:", valid_mse)
    print("[NNGP-SVGP] Test MSE:",      test_mse)

    # 6) UQ (intervals on test)
    mean_t, lo_t, hi_t = trainer.predict_interval(test_dc, alpha=0.05)
    uq = evaluate_uq_metrics_from_interval(
        y_true=test_dc.y, mean=mean_t, lower=lo_t, upper=hi_t, alpha=0.05
    )
    print("UQ (NNGP-SVGP):", uq)


def main_nngp_svgp_exact_ensemble_all():
    # ---------------- 1) Load data ----------------
    tasks, train_dc, valid_dc, test_dc, transformers = load_dataset()

    print("Train X shape:", train_dc.X.shape)
    N, D = train_dc.X.shape
    print("Train y shape:", train_dc.y.shape)
    # 2) Torch tensors
    X_train_t, y_train_t = _to_torch_xy(train_dc)

    # 3) Build NNGP (Exact)
    in_dim = X_train_t.shape[1]

    device = "cpu"
    # 4) Train
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- 2) Build K base models ----------------
    folds = np.array_split(np.arange(N), 5)
    M = max(256, N // 20)      # e.g., ~5% of data or at least 256 (tune per dataset/capacity)
    items = []
    for i in range(5):
        # (keeps your current "two-folds per model" split)
        train_idx_i = np.concatenate([folds[j] for j in [i, (i + 1) % 5]])
        ds_i = _subset_numpy_dataset(train_dc, train_idx_i)

        X_train_torch = torch.from_numpy(ds_i.X).float()
        y_np = ds_i.y
        if y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np[:, 0]
        y_train_torch = torch.from_numpy(y_np).float()

        # Ni = X_train_torch.shape[0]
        feat = FeatureNet(in_dim=in_dim, feat_dim=64, hidden=(128, 64), dropout=0.0)
        gp_model_i = GPyTorchRegressor(
            train_x=X_train_torch,
            train_y=y_train_torch,
            normalize_x=False,                        # BatchNorm handles scale
            gp_model_cls=NNSVGPLearnedInducing,
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            feature_extractor=feat,
            num_inducing=M,
            kernel="matern52")

        items.append({
            "model": gp_model_i,
            "train_dataset": ds_i,
            "lr": 5e-3,
            "nn_lr": 1e-3,
            "ngd_lr": 0.02,
            "warmup_iters": 5,
            "clip_grad": 1.0,
            "num_iters": 500,
            "log_interval": 40,
        })

    # ---------------- 3) Train ensemble once ----------------
    ens = EnsembleGPTrainer(items, device="cpu")
    ens.train()
    K = len(ens.trainers)

    # ---------------- 4) Loop over weighting strategies ----------------
    strategies = [
        # name, kwargs, needs_labels?
        ("uniform", {}, False),
        ("precision", {"tau": 1.0}, False),
        ("mse",       {"l2": 1e-3}, True),
        ("nll",       {"l2": 1e-3, "dirichlet_alpha": 1.05}, True),
    ]

    results = []
    for name, kwargs, _needs_y in strategies:
        # (a) get weights
        if name == "uniform":
            w = np.ones(K, dtype=float) / K
            print(f"\n[Calib:{name}] Using uniform weights.")
        else:
            w = ens.calibrate_weights(valid_dc, method=name, **kwargs)
            print(f"\n[Calib:{name}] Weights: {np.round(w, 4)}")

        # (b) evaluate on valid/test with these weights
        valid_mse = ens.evaluate_mse_w(valid_dc, w=w)
        test_mse  = ens.evaluate_mse_w(test_dc,  w=w)
        print(f"[{name}] Validation MSE: {valid_mse:.6f}")
        print(f"[{name}] Test MSE:       {test_mse:.6f}")

        # (c) UQ on test via weighted moment-matched mixture
        mean_test, lower_test, upper_test = ens.predict_interval_w(test_dc, alpha=0.05, w=w)
        uq_metrics = evaluate_uq_metrics_from_interval(
            y_true=test_dc.y,
            mean=mean_test,
            lower=lower_test,
            upper=upper_test,
            alpha=0.05,
        )
        print(f"[{name}] UQ metrics: {uq_metrics}")

        results.append({
            "name": name,
            "weights": w,
            "valid_mse": valid_mse,
            "test_mse": test_mse,
            "uq_metrics": uq_metrics,
        })

    # ---------------- 5) Pretty summary ----------------
    print("\n===== Summary over weighting strategies =====")
    for r in results:
        w_str = " ".join([f"{x:.3f}" for x in r["weights"]])
        print(f"{r['name']:>9} | valid MSE: {r['valid_mse']:.6f} | test MSE: {r['test_mse']:.6f} | w: [{w_str}]")

    return results


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
        train_idx_i = np.concatenate([folds[j] for j in [i, (i+1)%5]])
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


def main_svgp_ensemble_all():
    # ---------------- 1) Load data ----------------
    tasks, train_dc, valid_dc, test_dc, transformers = load_dataset()

    print("Train X shape:", train_dc.X.shape)
    N, D = train_dc.X.shape
    print("Train y shape:", train_dc.y.shape)

    # ---------------- 2) Build K base models ----------------
    folds = np.array_split(np.arange(N), 5)

    items = []
    for i in range(5):
        # (keeps your current "two-folds per model" split)
        train_idx_i = np.concatenate([folds[j] for j in [i, (i + 1) % 5]])
        ds_i = _subset_numpy_dataset(train_dc, train_idx_i)

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
            num_inducing=max(128, Ni // 20),
            kmeans_iters=15,
        )

        items.append({
            "model": gp_model_i,
            "train_dataset": ds_i,
            "lr": 0.01,
            "num_iters": 500,
            "log_interval": 10,
        })

    # ---------------- 3) Train ensemble once ----------------
    ens = EnsembleGPTrainer(items, device="cpu")
    ens.train()
    K = len(ens.trainers)

    # ---------------- 4) Loop over weighting strategies ----------------
    strategies = [
        # name, kwargs, needs_labels?
        ("precision", {"tau": 1.0}, False),
        ("mse",       {"l2": 1e-3}, True),
        ("nll",       {"l2": 1e-3, "dirichlet_alpha": 1.05}, True),
    ]

    results = []
    for name, kwargs, _needs_y in strategies:
        # (a) get weights
        if name == "uniform":
            w = np.ones(K, dtype=float) / K
            print(f"\n[Calib:{name}] Using uniform weights.")
        else:
            w = ens.calibrate_weights(valid_dc, method=name, **kwargs)
            print(f"\n[Calib:{name}] Weights: {np.round(w, 4)}")

        # (b) evaluate on valid/test with these weights
        valid_mse = ens.evaluate_mse_w(valid_dc, w=w)
        test_mse  = ens.evaluate_mse_w(test_dc,  w=w)
        print(f"[{name}] Validation MSE: {valid_mse:.6f}")
        print(f"[{name}] Test MSE:       {test_mse:.6f}")

        # (c) UQ on test via weighted moment-matched mixture
        mean_test, lower_test, upper_test = ens.predict_interval_w(test_dc, alpha=0.05, w=w)
        uq_metrics = evaluate_uq_metrics_from_interval(
            y_true=test_dc.y,
            mean=mean_test,
            lower=lower_test,
            upper=upper_test,
            alpha=0.05,
        )
        print(f"[{name}] UQ metrics: {uq_metrics}")

        results.append({
            "name": name,
            "weights": w,
            "valid_mse": valid_mse,
            "test_mse": test_mse,
            "uq_metrics": uq_metrics,
        })

    # ---------------- 5) Pretty summary ----------------
    print("\n===== Summary over weighting strategies =====")
    for r in results:
        w_str = " ".join([f"{x:.3f}" for x in r["weights"]])
        print(f"{r['name']:>9} | valid MSE: {r['valid_mse']:.6f} | test MSE: {r['test_mse']:.6f} | w: [{w_str}]")

    return results


def main_nn():
    tasks, train_dc, valid_dc, test_dc, transformers = load_dataset()

    # train_nn_baseline(train_dc, valid_dc, test_dc)
    train_nn_mc_dropout(train_dc, valid_dc, test_dc)
    train_nn_deep_ensemble(train_dc, valid_dc, test_dc)


if __name__ == "__main__":
    # main_svgp_ensemble_all()
    # main_nngp_exact()
    # main_nngp_exact_ensemble_all()
    main_nngp_svgp_exact_ensemble_all()
    # main_nn()
    # main_gp()
    # main_svgp()
