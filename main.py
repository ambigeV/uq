from typing import List, Optional, Tuple
import gc
import deepchem as dc
from data_utils import prepare_datasets, evaluate_uq_metrics_from_interval, \
    compute_ece, calculate_cutoff_classification_data, evaluate_uq_metrics_classification
from nn_baseline import train_nn_baseline, train_nn_deep_ensemble, train_nn_mc_dropout, train_evd_baseline, \
    MyTorchRegressor, MyTorchClassifier, MyTorchRegressorMC, MyTorchClassifierHeteroscedastic, \
    DenseNormalGamma, DenseDirichlet
from model_utils import load_neural_network_model, load_neural_network_ensemble
from deepchem.molnet import load_qm7, load_delaney, load_qm8, load_qm9, load_lipo, load_freesolv, load_tox21, load_toxcast, load_sider, load_clintox

import torch
import gpytorch
from gp_single import GPyTorchRegressor, SVGPModel, FeatureNet, \
    DeepFeatureKernel, NNGPExactGPModel, NNSVGPLearnedInducing, \
    GPyTorchClassifier, SVGPClassificationModel, MultitaskSVGPClassificationModel
from gp_trainer import GPTrainer, EnsembleGPTrainer, GPClassificationTrainer
from data_utils import calculate_cutoff_error_data, acc_from_probs, auc_from_probs
import numpy as np
import pandas as pd
import csv
import random
import os
import collections


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)


def save_summary_to_csv(all_results, n_runs, out_path: str):
    rows = []
    for method, uq_list in all_results.items():
        if not uq_list:
            continue
        metric_names = uq_list[0].keys()
        row = {"method": method}
        for m in metric_names:
            vals = [d[m] for d in uq_list if d[m] is not None]
            if not vals:
                continue
            vals = np.asarray(vals, dtype=float)
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"]  = float(vals.std(ddof=0))
        rows.append(row)

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_dataset(
    dataset_name: str = "delaney", 
    split: str = "random", 
    task_indices: Optional[List[int]] = None,
    use_graph: bool = False
):
    """
    dataset_name: {"qm7", "qm8", "delaney", "lipo", "tox21", "toxcast", "sider", "clintox"}
    task_indices: List of indices to keep. e.g., [0] for first task, 
                  [0, 2] for 1st and 3rd. If None, keeps ALL tasks.
    use_graph: If True, use DMPNN graph featurizer instead of vector featurizer
    """
    # 1. Load Raw Data
    # Create DMPNN featurizer if needed
    if use_graph:
        graph_featurizer = dc.feat.DMPNNFeaturizer()
    
    if dataset_name == "qm7":
        FEATURIZER = "coulomb" if not use_graph else "dmpnn"
        if use_graph:
            tasks, datasets, transformers = dc.molnet.load_qm7(splitter=split, featurizer=graph_featurizer)
        else:
            tasks, datasets, transformers = dc.molnet.load_qm7(splitter=split)
    elif dataset_name == "qm8":
        FEATURIZER = "coulomb" if not use_graph else "dmpnn"
        if use_graph:
            tasks, datasets, transformers = dc.molnet.load_qm8(splitter=split, featurizer=graph_featurizer)
        else:
            tasks, datasets, transformers = dc.molnet.load_qm8(splitter=split)
    elif dataset_name == "delaney":
        FEATURIZER = "ecfp" if not use_graph else "dmpnn"
        if use_graph:
            tasks, datasets, transformers = dc.molnet.load_delaney(splitter=split, featurizer=graph_featurizer)
        else:
            tasks, datasets, transformers = dc.molnet.load_delaney(splitter=split)
    elif dataset_name == "lipo":
        FEATURIZER = "ecfp" if not use_graph else "dmpnn"
        if use_graph:
            tasks, datasets, transformers = dc.molnet.load_lipo(splitter=split, featurizer=graph_featurizer)
        else:
            tasks, datasets, transformers = dc.molnet.load_lipo(splitter=split)
    elif dataset_name == "tox21":
        FEATURIZER = "ecfp" if not use_graph else "dmpnn"
        if use_graph:
            tasks, datasets, transformers = dc.molnet.load_tox21(splitter=split, featurizer=graph_featurizer)
        else:
            tasks, datasets, transformers = dc.molnet.load_tox21(splitter=split)
    elif dataset_name == "toxcast":
        FEATURIZER = "ecfp" if not use_graph else "dmpnn"
        if use_graph:
            tasks, datasets, transformers = dc.molnet.load_toxcast(splitter=split, featurizer=graph_featurizer)
        else:
            tasks, datasets, transformers = dc.molnet.load_toxcast(splitter=split)
    elif dataset_name == "sider":
        FEATURIZER = "ecfp" if not use_graph else "dmpnn"
        if use_graph:
            tasks, datasets, transformers = dc.molnet.load_sider(splitter=split, featurizer=graph_featurizer)
        else:
            tasks, datasets, transformers = dc.molnet.load_sider(splitter=split)
    elif dataset_name == "clintox":
        FEATURIZER = "ecfp" if not use_graph else "dmpnn"
        if use_graph:
            tasks, datasets, transformers = dc.molnet.load_clintox(splitter=split, featurizer=graph_featurizer)
        else:
            tasks, datasets, transformers = dc.molnet.load_clintox(splitter=split)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    # 2. Flatten/Preprocess (only for vector featurizers)
    if not use_graph:
        train_dc, valid_dc, test_dc = prepare_datasets(
            datasets,
            featurizer_name=FEATURIZER
        )
    else:
        # For graph featurizer, datasets already contain GraphData objects
        # We'll handle conversion to BatchMolGraph during training
        train_dc, valid_dc, test_dc = datasets

    # 3. Apply Task Mask (Slicing)
    if task_indices is not None:
        # Validate indices
        n_available = len(tasks)
        if any(i >= n_available for i in task_indices):
            raise ValueError(f"task_indices {task_indices} out of range for {n_available} tasks.")
        
        # Slice task names
        print(f"Original tasks: {len(tasks)}. Selecting indices: {task_indices}")
        tasks = [tasks[i] for i in task_indices]

        # Helper to slice DeepChem datasets
        def _slice_tasks(ds: dc.data.NumpyDataset) -> dc.data.NumpyDataset:
            # Handle Y
            if ds.y.ndim == 2:
                new_y = ds.y[:, task_indices]
            else:
                # If y is 1D (N,), promote to (N,1) then slice, or error
                new_y = ds.y.reshape(-1, 1)[:, task_indices]

            # Handle W (weights)
            if ds.w.ndim == 2:
                new_w = ds.w[:, task_indices]
            else:
                new_w = ds.w.reshape(-1, 1)[:, task_indices]

            return dc.data.NumpyDataset(
                X=ds.X,
                y=new_y,
                w=new_w,
                ids=ds.ids,
                n_tasks=len(task_indices) # Update metadata
            )

        train_dc = _slice_tasks(train_dc)
        valid_dc = _slice_tasks(valid_dc)
        test_dc = _slice_tasks(test_dc)

    print(f"Final Active Tasks: {len(tasks)}")
    print(f"Train Y Shape: {train_dc.y.shape}")

    return tasks, train_dc, valid_dc, test_dc, transformers


def _to_torch_xy(dc_dataset):
    X_t = torch.from_numpy(dc_dataset.X).float()
    y_np = dc_dataset.y
    if y_np.ndim == 2 and y_np.shape[1] == 1:
        y_np = y_np[:, 0]
    y_t = torch.from_numpy(y_np).float()
    return X_t, y_t


def main_nngp_exact(train_dc, valid_dc, test_dc, device: str = "cpu", run_id=0, use_weights=False, mode="regression"):
    # 1) Load
    print("Train X shape:", train_dc.X.shape)
    print("Train y shape:", train_dc.y.shape)
    N, D = train_dc.X.shape

    # 2) Torch tensors
    X_train_t, y_train_t = _to_torch_xy(train_dc)

    y_np = train_dc.y
    if y_np.ndim == 2 and y_np.shape[1] == 1:
        y_np = y_np[:, 0]
    y_train_t = torch.from_numpy(y_np).float()

    train_w_torch = None
    if use_weights and train_dc.w is not None:
        print("--> Using per-point precision weights via FixedNoiseGaussianLikelihood")
        w_np = train_dc.w
        if w_np.ndim == 2 and w_np.shape[1] == 1 and y_np.ndim == 1:
             w_np = w_np[:, 0]
        train_w_torch = torch.from_numpy(w_np).float()

    # 3) Build NNGP (Exact)
    in_dim = X_train_t.shape[1]
    feat = FeatureNet(in_dim=in_dim, feat_dim=64, hidden=(128, 64), dropout=0.0)

    if mode == "regression":
        nngp_exact = GPyTorchRegressor(
            train_x=X_train_t,
            train_y=y_train_t,
            train_w=train_w_torch,  # <-- pass weights, not noise
            use_weights=use_weights,  # <-- just a switch; safe if w missing
            w_min=1e-5,
            noise_cap=1e6,
            normalize_x=False,                        # BatchNorm handles scale
            gp_model_cls=NNGPExactGPModel,
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            feature_extractor=feat,
            kernel="matern52",
        )

        # 4) Train
        dev = "cpu"
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
        valid_mse = trainer.evaluate_mse(valid_dc, use_weights=use_weights)
        test_mse  = trainer.evaluate_mse(test_dc, use_weights=use_weights)
        print("[NNGP-Exact] Validation MSE:", valid_mse)
        print("[NNGP-Exact] Test MSE:",      test_mse)

        # 6) UQ (intervals on test)
        mean_t, lo_t, hi_t = trainer.predict_interval(test_dc, alpha=0.05, use_weights=use_weights)
        cutoff_error_df = calculate_cutoff_error_data(mean_t,
                                                      hi_t-lo_t,
                                                      test_dc.y,
                                                      weights=test_dc.w,
                                                      use_weights=use_weights)
        uq = evaluate_uq_metrics_from_interval(
            y_true=test_dc.y,
            mean=mean_t,
            lower=lo_t,
            upper=hi_t,
            alpha=0.05,
            test_error=test_mse,
            weights=test_dc.w,
            use_weights=use_weights
        )

        print("UQ (NNGP-Exact):", uq)
        uq_metrics = uq
    else:
        num_tasks = 1
        if y_np.ndim == 2:
            num_tasks = y_np.shape[1]
        M = max(512, N//10)
        nngp_classifier = GPyTorchClassifier(
            train_x=X_train_t,
            train_y=y_train_t,
            normalize_x=False, 
            gp_model_cls=NNSVGPLearnedInducing, 
            feature_extractor=feat,  
            num_inducing=M,  
            kernel="matern52",
            num_tasks = num_tasks
        )

        # 4) Train
        dev = "cpu"
        print(f"DEBUG CHECK: Variable 'dev' is {dev}")
        print(f"DEBUG CHECK: Variable 'device' is {device}") # I suspect this will print 'cuda:X'
        trainer = GPClassificationTrainer(
            model=nngp_classifier,
            train_dataset=train_dc,
            lr=5e-3,  
            nn_lr=1e-3,  
            ngd_lr=0.02, 
            warmup_iters=5,
            clip_grad=1.0,
            num_iters=500,
            device=dev,
            log_interval=40,
            use_weights=False,
        )

        trainer.train()

        # 4. Evaluate (AUC/Accuracy)
        valid_metrics = trainer.evaluate(valid_dc, use_weights=use_weights)
        test_metrics = trainer.evaluate(test_dc, use_weights=use_weights)

        print(f"[NN-SVGP Classif] Valid AUC: {valid_metrics['auc']}")
        print(f"[NN-SVGP Classif] Test AUC:  {test_metrics['auc']}")

        # 5. UQ Analysis
        test_probs = test_metrics["probs"]

        # Calculate Cutoff Data for Classification
        cutoff_error_df = calculate_cutoff_classification_data(
            test_probs, 
            test_dc.y, 
            weights=test_dc.w, 
            use_weights=use_weights
        )

        uq_metrics = evaluate_uq_metrics_classification(
            y_true=test_dc.y, 
            probs=test_probs, 
            auc=test_metrics['auc'],
            weights=test_dc.w,
            use_weights=use_weights, 
            n_bins=20
        )
        print("UQ (NN-SVGP Classif):", uq_metrics)

    return uq_metrics, cutoff_error_df


def main_nngp_exact_ensemble_all(train_dc, valid_dc, test_dc, M=5, run_id=0, use_weights=False, mode="regression"):
    # ---------------- 1) Load data ----------------

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
    folds = np.array_split(np.arange(N), M)

    items = []
    for i in range(M):
        # (keeps your current "two-folds per model" split)
        train_idx_i = np.concatenate([folds[j] for j in [i, (i + 1) % M]])
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
    cut_offs = []
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
        cutoff_error_df = calculate_cutoff_error_data(mean_test, upper_test - lower_test, test_dc.y)
        uq_metrics = evaluate_uq_metrics_from_interval(
            y_true=test_dc.y,
            mean=mean_test,
            lower=lower_test,
            upper=upper_test,
            alpha=0.05,
            test_error=test_mse,
        )
        cut_offs.append(cutoff_error_df)
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

    return results, cut_offs


def main_nngp_svgp(train_dc, valid_dc, test_dc, device: str = "cpu", run_id=0):
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
    dev = "cpu"
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
    cutoff_error_df = calculate_cutoff_error_data(mean_t, hi_t-lo_t, test_dc.y)
    uq = evaluate_uq_metrics_from_interval(
        y_true=test_dc.y, mean=mean_t, lower=lo_t, upper=hi_t, alpha=0.05, test_error=test_mse
    )
    print("UQ (NNGP-SVGP):", uq)
    return uq, cutoff_error_df


def main_nngp_svgp_exact_ensemble_all(train_dc, valid_dc, test_dc, E=5, run_id=0, use_weights=False, mode="regression"):
    print("Train X shape:", train_dc.X.shape)
    N, D = train_dc.X.shape
    print("Train y shape:", train_dc.y.shape)
    # 2) Torch tensors
    X_train_t, y_train_t = _to_torch_xy(train_dc)

    # 3) Build NNGP (Exact)
    in_dim = X_train_t.shape[1]

    device = "cpu"
    # 4) Train
    dev = device

    # ---------------- 2) Build K base models ----------------
    folds = np.array_split(np.arange(N), E)
    M = max(256, N // 10)      # e.g., ~5% of data or at least 256 (tune per dataset/capacity)
    items = []
    for i in range(E):
        # (keeps your current "two-folds per model" split)
        train_idx_i = np.concatenate([folds[j] for j in [i, (i + 1) % E]])
        ds_i = _subset_numpy_dataset(train_dc, train_idx_i)

        X_train_torch = torch.from_numpy(ds_i.X).float()
        y_np = ds_i.y
        if y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np[:, 0]
        
        num_tasks = 1
        if y_np.ndim == 2:
            num_tasks = y_np.shape[1]

        y_train_torch = torch.from_numpy(y_np).float()

        train_w_torch = None
        if use_weights and train_dc.w is not None:
            print("--> Using per-point precision weights via NN-SVGP")
            w_np = ds_i.w
            if w_np.ndim == 2 and w_np.shape[1] == 1:
                w_np = w_np[:, 0]
            train_w_torch = torch.from_numpy(w_np).float()
        
        Ni = X_train_torch.shape[0]

        # Ni = X_train_torch.shape[0]
        if mode == "regression":
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
        else:
            feat = FeatureNet(in_dim=in_dim, feat_dim=64, hidden=(128, 64), dropout=0.0)
            gp_model_i = GPyTorchClassifier(
                train_x=X_train_torch,
                train_y=y_train_torch,
                normalize_x=False,
                gp_model_cls=NNSVGPLearnedInducing,
                feature_extractor=feat,
                num_inducing = max(min(512, Ni//5), Ni//10),
                kernel="matern52",
                num_tasks=num_tasks
            )
            items.append({
                "model": gp_model_i,
                "train_dataset": ds_i,
                "lr": 0.005,
                "nn_lr": 1e-3,
                "ngd_lr": 0.02,
                "warmup_iters": 5,
                "clip_grad": 1.0,
                "num_iters": 300,
                "log_interval": 40,
            })

    if mode == "regression":
        # ---------------- 3) Train ensemble once ----------------
        ens = EnsembleGPTrainer(items, device="cpu", use_weights=use_weights)
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
        cut_offs = []
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
            cutoff_error_df = calculate_cutoff_error_data(mean_test, upper_test-lower_test, test_dc.y, test_dc.w, use_weights)
            cut_offs.append(cutoff_error_df)
            uq_metrics = evaluate_uq_metrics_from_interval(
                y_true=test_dc.y,
                mean=mean_test,
                lower=lower_test,
                upper=upper_test,
                alpha=0.05,
                test_error=test_mse,
                weights=test_dc.w,
                use_weights=use_weights
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
    else:
        # ---------------- 3) Train ensemble once ----------------
        ens = EnsembleGPTrainer(items, device="cpu", mode=mode, use_weights=use_weights)
        ens.train()
        K = len(ens.trainers)

        # ---------------- 4) Loop over weighting strategies ----------------
        strategies = [
            # name, kwargs, needs_labels?
            ("uniform", {}, False),
            ("mse",       {"l2": 1e-3}, True),
            ("nll",       {"l2": 1e-3, "dirichlet_alpha": 1.05}, True),
        ]

        results = []
        cut_offs = []

        for name, kwargs, _needs_y in strategies:
            # (a) get weights
            w = ens.calibrate_class_weights(valid_dc, method=name, **kwargs)
            print(f"\n[Calib:{name}] Weights: {w}")

            # (b) evaluate on valid/test with these weights
            valid_metrics = ens.evaluate_auc_w(valid_dc, w=w)
            test_metrics  = ens.evaluate_auc_w(test_dc,  w=w)
            print(f"\n[GP Classification] Validation AUC: {valid_metrics['auc']} | Acc: {valid_metrics['acc']}")
            print(f"[GP Classification] Test AUC:       {test_metrics['auc']} | Acc: {test_metrics['acc']}")

            # 5. UQ Analysis: Calibration (ECE)
            test_probs = test_metrics["probs"]
            test_y = test_metrics["y_true"]

            # mean_test, lower_test, upper_test = trainer.predict_interval(test_dc, alpha=0.05, use_weights=use_weights)
            cutoff_error_df = calculate_cutoff_classification_data(test_probs,
                                                                   test_dc.y,
                                                                   weights=test_dc.w,
                                                                   use_weights=use_weights)
            cut_offs.append(cutoff_error_df)
            uq_metrics = evaluate_uq_metrics_classification(
                y_true=test_dc.y,
                probs=test_probs,
                auc=test_metrics['auc'],
                weights=test_dc.w,
                use_weights=use_weights,
                n_bins=20
            )
            print(f"[{name}] UQ metrics: {uq_metrics}")
            

            results.append({
                "name": name,
                "weights": w,
                "valid_auc": valid_metrics['auc'],
                "test_auc": test_metrics['auc'],
                "uq_metrics": uq_metrics,
            })

    return results, cut_offs


def main_gp(train_dc, valid_dc, test_dc, run_id=0, use_weights=False, mode="regression"):
    print("Train X shape:", train_dc.X.shape)  # (N, D)
    N, D = train_dc.X.shape
    print("Train y shape:", train_dc.y.shape)  # (N, 1) or (N,)

    # 2. Build GP model
    X_train_torch = torch.from_numpy(train_dc.X).float()
    # single-task, ensure (N,) for train_y
    y_np = train_dc.y
    if y_np.ndim == 2 and y_np.shape[1] == 1:
        y_np = y_np[:, 0]
    y_train_torch = torch.from_numpy(y_np).float()

    train_w_torch = None
    if use_weights and train_dc.w is not None:
        print("--> Using per-point precision weights via FixedNoiseGaussianLikelihood")
        w_np = train_dc.w
        if w_np.ndim == 2 and w_np.shape[1] == 1 and y_np.ndim == 1:
             w_np = w_np[:, 0]
        train_w_torch = torch.from_numpy(w_np).float()

    if mode == "regression":
        gp_model = GPyTorchRegressor(
            train_x=X_train_torch,
            train_y=y_train_torch,
            train_w=train_w_torch,  # <-- pass weights, not noise
            use_weights=use_weights,  # <-- just a switch; safe if w missing
            w_min=1e-5,
            noise_cap=1e6,
            normalize_noise_by_median=True,  # recommended to make weights “relative”
        )

        trainer = GPTrainer(
            model=gp_model,
            train_dataset=train_dc,
            lr=0.05,
            num_iters=200,
            device="cpu",
            log_interval=40,
        )
        trainer.train()

        # 4. Evaluate
        valid_mse = trainer.evaluate_mse(valid_dc, use_weights=use_weights)
        test_mse = trainer.evaluate_mse(test_dc, use_weights=use_weights)
        task_str = "Multitask" if (y_np.ndim > 1) else "Single-task"
        print(f"[GP {task_str}] Validation MSE: {valid_mse}")
        print(f"[GP {task_str}] Test MSE:       {test_mse}")

        # 5. Get uncertainty intervals on test set
        mean_test, lower_test, upper_test = trainer.predict_interval(test_dc, alpha=0.05, use_weights=use_weights)
        cutoff_error_df = calculate_cutoff_error_data(mean_test,
                                                      upper_test-lower_test,
                                                      test_dc.y,
                                                      test_dc.w,
                                                      use_weights=use_weights)

        uq_metrics = evaluate_uq_metrics_from_interval(
            y_true=test_dc.y,
            mean=mean_test,
            lower=lower_test,
            upper=upper_test,
            alpha=0.05,
            test_error=test_mse,
            weights=test_dc.w,
            use_weights=use_weights
        )
        print("UQ:", uq_metrics)
    else:
        num_tasks = 1
        if y_np.ndim == 2:
            num_tasks = y_np.shape[1]
        cls = SVGPClassificationModel
        if num_tasks > 1:
            cls = MultitaskSVGPClassificationModel

        # 1. Instantiate Classifier
        gp_model = GPyTorchClassifier(
            train_x=X_train_torch,
            train_y=y_train_torch,
            normalize_x="auto",  # Set to True if not using ECFP/Binary features
            # SVGP Specific Args
            gp_model_cls=cls,
            num_inducing=max(512, N//10),  # 128 is a good balance for speed/accuracy
            kernel="matern52",  # usually better than RBF for molecular data
            num_tasks = num_tasks
        )

        # 2. Instantiate Trainer
        trainer = GPClassificationTrainer(
            model=gp_model,
            train_dataset=train_dc,
            lr=0.01,  # Adam LR (Kernel/Feature Extractor)
            ngd_lr=0.1,  # Natural Gradient LR (Variational Parameters)
            num_iters=400,
            device="cpu",
            log_interval=40,
            warmup_iters=5,
            use_weights=False
        )

        # 3. Train
        print("--- Starting GP Classification Training ---")
        trainer.train()

        # 4. Evaluate (AUC & Accuracy)
        # The trainer.evaluate() method handles 'use_weights' internally for AUC calculation
        valid_metrics = trainer.evaluate(valid_dc, use_weights=use_weights)
        test_metrics = trainer.evaluate(test_dc, use_weights=use_weights)

        print(f"\n[GP Classification] Validation AUC: {valid_metrics['auc']} | Acc: {valid_metrics['acc']}")
        print(f"[GP Classification] Test AUC:       {test_metrics['auc']} | Acc: {test_metrics['acc']}")

        # 5. UQ Analysis: Calibration (ECE)
        test_probs = test_metrics["probs"]
        test_y = test_metrics["y_true"]


        # mean_test, lower_test, upper_test = trainer.predict_interval(test_dc, alpha=0.05, use_weights=use_weights)
        cutoff_error_df = calculate_cutoff_classification_data(test_probs,
                                                               test_dc.y,
                                                               weights=test_dc.w,
                                                               use_weights=use_weights)
        uq_metrics = evaluate_uq_metrics_classification(
            y_true=test_dc.y,
            probs=test_probs,
            auc=test_metrics['auc'],
            weights=test_dc.w,
            use_weights=use_weights,
            n_bins=20
        )
        print("UQ:", uq_metrics)

    return uq_metrics, cutoff_error_df


def main_svgp(train_dc, valid_dc, test_dc, run_id=0):
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
        log_interval=40,
    )
    trainer.train()

    # 4. Evaluate
    valid_mse = trainer.evaluate_mse(valid_dc)
    test_mse = trainer.evaluate_mse(test_dc)
    print("[SVGP] Validation MSE:", valid_mse)
    print("[SVGP] Test MSE:",      test_mse)

    # 5. Get uncertainty intervals on test set
    mean_test, lower_test, upper_test = trainer.predict_interval(test_dc, alpha=0.05)
    cutoff_error_df = calculate_cutoff_error_data(mean_test, upper_test-lower_test, test_dc.y)

    uq_metrics = evaluate_uq_metrics_from_interval(
        y_true=test_dc.y,
        mean=mean_test,
        lower=lower_test,
        upper=upper_test,
        test_error=test_mse,
    )
    print("UQ:", uq_metrics)
    return uq_metrics, cutoff_error_df


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


def main_svgp_ensemble(train_dc, valid_dc, test_dc, run_id=0):
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
            "log_interval": 40,
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


def main_svgp_ensemble_all(train_dc, valid_dc, test_dc, run_id=0, use_weights=False, mode="regression"):
    # ---------------- 1) Load data ----------------
    print("Train X shape:", train_dc.X.shape)
    N, D = train_dc.X.shape
    print("Train y shape:", train_dc.y.shape)

    K_fold = 5
    # ---------------- 2) Build K base models ----------------
    folds = np.array_split(np.arange(N), K_fold)
    trained_models = []

    for i in range(K_fold):
        # (keeps your current "two-folds per model" split)
        train_idx_i = np.concatenate([folds[j] for j in [i, (i + 1) % K_fold]])
        ds_i = _subset_numpy_dataset(train_dc, train_idx_i)

        X_train_torch = torch.from_numpy(ds_i.X).float()
        y_np = ds_i.y
        if y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np[:, 0]
        
        num_tasks = 1
        if y_np.ndim == 2:
            num_tasks = y_np.shape[1]

        y_train_torch = torch.from_numpy(y_np).float()

        train_w_torch = None
        if use_weights and train_dc.w is not None:
            print("--> Using per-point precision weights via NN-SVGP")
            w_np = ds_i.w
            if w_np.ndim == 2 and w_np.shape[1] == 1:
                w_np = w_np[:, 0]
            train_w_torch = torch.from_numpy(w_np).float()

        Ni = X_train_torch.shape[0]

        if mode == "regression":
            gp_model_i = GPyTorchRegressor(
                train_x=X_train_torch,
                train_y=y_train_torch,
                train_w=train_w_torch,
                use_weights=use_weights,
                gp_model_cls=SVGPModel,
                likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                num_inducing=max(128, Ni // 20),
                kmeans_iters=15,
            )
            
            item = {
                "model": gp_model_i,
                "train_dataset": ds_i,
                "lr": 0.01,
                "num_iters": 500,
                "log_interval": 40,
            }
            trained_models.append(item)
        else:
            num_tasks = 1
            if y_np.ndim == 2:
                num_tasks = y_np.shape[1]
            cls = SVGPClassificationModel
            if num_tasks > 1:
                cls = MultitaskSVGPClassificationModel

            gp_model_i = GPyTorchClassifier(
                train_x=X_train_torch,
                train_y=y_train_torch,
                normalize_x="auto",
                gp_model_cls=cls,
                num_inducing=max(512, Ni // 10),
                kernel="matern52",
                num_tasks = num_tasks,
            )

            item = {
                "model": gp_model_i,
                "train_dataset": ds_i,
                "lr": 0.01,
                "num_iters": 500,
                "log_interval": 40,
            }
            trained_models.append(item)


    if mode == "regression":
        # ---------------- 3) Train ensemble once ----------------
        ens = EnsembleGPTrainer(trained_models, device="cpu", use_weights=use_weights)
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
        cut_offs = []
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
            cutoff_error_df = calculate_cutoff_error_data(mean_test, upper_test-lower_test, test_dc.y, test_dc.w, use_weights)
            cut_offs.append(cutoff_error_df)
            uq_metrics = evaluate_uq_metrics_from_interval(
                y_true=test_dc.y,
                mean=mean_test,
                lower=lower_test,
                upper=upper_test,
                alpha=0.05,
                test_error=test_mse,
                weights=test_dc.w,
                use_weights=use_weights
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
    else:
         # ---------------- 3) Train ensemble once ----------------
        ens = EnsembleGPTrainer(trained_models, device="cpu", mode="classification", use_weights=use_weights)
        ens.train()
        K = len(ens.trainers)

        # ---------------- 4) Loop over weighting strategies ----------------
        strategies = [
            # name, kwargs, needs_labels?
            ("uniform", {}, False),
            ("mse",       {"l2": 1e-3}, True),
            ("nll",       {"l2": 1e-3, "dirichlet_alpha": 1.05}, True),
        ]

        results = []
        cut_offs = []
        for name, kwargs, _needs_y in strategies:
            # (a) get weights
            w = ens.calibrate_class_weights(valid_dc, method=name, **kwargs)
            print(f"\n[Calib:{name}] Weights: {w}")

            # 4. Evaluate (AUC & Accuracy)
            # The trainer.evaluate() method handles 'use_weights' internally for AUC calculation
            valid_metrics = ens.evaluate_auc_w(valid_dc, w=w)
            test_metrics = ens.evaluate_auc_w(test_dc, w=w)

            print(f"\n[GP Classification] Validation AUC: {valid_metrics['auc']} | Acc: {valid_metrics['acc']}")
            print(f"[GP Classification] Test AUC:       {test_metrics['auc']} | Acc: {test_metrics['acc']}")

            # 5. UQ Analysis: Calibration (ECE)
            test_probs = test_metrics["probs"]
            test_y = test_metrics["y_true"]

            # mean_test, lower_test, upper_test = trainer.predict_interval(test_dc, alpha=0.05, use_weights=use_weights)
            cutoff_error_df = calculate_cutoff_classification_data(test_probs,
                                                                   test_y,
                                                                   weights=test_dc.w,
                                                                   use_weights=use_weights)
            cut_offs.append(cutoff_error_df)
            uq_metrics = evaluate_uq_metrics_classification(
                y_true=test_dc.y,
                probs=test_probs,
                auc=test_metrics['auc'],
                weights=test_dc.w,
                use_weights=use_weights,
                n_bins=20
            )
            print(f"[{name}] UQ metrics: {uq_metrics}")
            

            results.append({
                "name": name,
                "weights": w,
                "valid_auc": valid_metrics['auc'],
                "test_auc": test_metrics['auc'],
                "uq_metrics": uq_metrics,
            })

    return results, cut_offs


def run_once_nn(dataset_name: str,
                seed: int = 0,
                run_id: int = 0,
                split: str = "random",
                mode: str = "regression",
                use_weights: bool = False,
                task_indices: Optional[List[int]] = None,
                encoder_type: str = "identity",
                use_graph: bool = False): 
    """
    One full run on a given dataset & featurizer with a fixed seed.
    
    Args:
        dataset_name: Name of the dataset
        seed: Random seed
        run_id: Run identifier
        split: Data split type ("random" or "scaffold")
        mode: Task mode ("regression" or "classification")
        use_weights: Whether to use class weights for classification
        task_indices: Optional list of task indices to use
        encoder_type: Type of encoder ("identity" or "dmpnn")
        use_graph: If True, use DMPNN graph featurizer instead of vector featurizer
    """
    set_global_seed(seed)

    tasks, train_dc, valid_dc, test_dc, transformers = load_dataset(
        dataset_name=dataset_name,
        split=split,
        task_indices=task_indices,
        use_graph=use_graph
    )

    use_weights = False
    if dataset_name in ["tox21", "toxcast", "sider", "clintox"]:
        use_weights = True

    print(f"\n=== Run with seed={seed} on dataset={dataset_name}, use_graph={use_graph}, encoder_type={encoder_type}")
    results = {}
    all_cutoff_dfs = []
    
    # Determine encoder type based on data format (if not explicitly provided)
    if encoder_type == "identity" and use_graph:
        encoder_type = "dmpnn"
    elif encoder_type == "dmpnn" and not use_graph:
        raise ValueError("encoder_type='dmpnn' requires use_graph=True")
    
    results["nn_evd"], cut_off_evd = train_evd_baseline(train_dc, valid_dc, test_dc,
                                                        run_id=run_id, use_weights=use_weights, mode=mode, encoder_type=encoder_type)

    cut_off_evd['Method'] = "nn_evd"
    all_cutoff_dfs.append(cut_off_evd)

    if dataset_name not in ["qm8"]:
        results["nn_baseline"] = train_nn_baseline(train_dc, valid_dc, test_dc,
                                                   run_id=run_id, use_weights=use_weights, mode=mode, encoder_type=encoder_type)

    results["nn_mc_dropout"], cut_off_dropout = train_nn_mc_dropout(train_dc, valid_dc, test_dc,
                                                                    run_id=run_id, use_weights=use_weights, mode=mode, encoder_type=encoder_type)
    cut_off_dropout['Method'] = "nn_mc_dropout"
    all_cutoff_dfs.append(cut_off_dropout)

    results["nn_deep_ensemble"], cut_off_ensemble = train_nn_deep_ensemble(train_dc, valid_dc, test_dc,
                                                                           run_id=run_id, use_weights=use_weights, mode=mode, encoder_type=encoder_type)
    cut_off_ensemble['Method'] = "nn_deep_ensemble"
    all_cutoff_dfs.append(cut_off_ensemble)

    import pandas as pd

    # Concatenate all DataFrames into a single one
    combined_cutoff_df = pd.concat(all_cutoff_dfs, ignore_index=True)

    # Define a consistent filename
    # 1. Modify split name to include "_graph" if use_graph is True
    split_name = f"{split}_graph" if use_graph else split
    
    # 2. Generate a suffix for the filename based on task_indices
    if task_indices is None:
        task_suffix = ""
    else:
        # e.g., turns [0, 2] into "_tasks_0_2"
        task_suffix = "_tasks_" + "_".join(map(str, task_indices))

    # 3. Define the consistent filename with the suffix
    output_filename = f"./cdata_{mode}/figure/{split_name}_{dataset_name}{task_suffix}_NN_cutoff_run_{run_id}.csv"

    import os
    # 3. Create the directory if it does not exist
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Store the final combined data
    combined_cutoff_df.to_csv(output_filename, index=False)

    del train_dc, valid_dc, test_dc, tasks, transformers
    del all_cutoff_dfs
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main_nn(dataset_name: str = "delaney",
            n_runs: int = 5,
            split: str = "random",
            base_seed: int = 0,
            mode: str = "regression",
            use_weights: bool = False,
            task_indices: Optional[List[int]] = None,
            encoder_type: str = "identity",
            use_graph: bool = False,
            active_learning: bool = False,
            use_cluster: bool = False): 

    if task_indices is None:
        global_task_suffix = ""
    else:
        global_task_suffix = "_tasks_" + "_".join(map(str, task_indices))

    # Check if active learning is enabled
    if active_learning:
        from active_learning import run_active_learning_nn
        
        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            print(f"\n=== Active Learning Run {run_idx} ===")
            run_active_learning_nn(
                dataset_name=dataset_name,
                seed=seed,
                run_id=run_idx,
                split=split,
                mode=mode,
                use_weights=use_weights,
                task_indices=task_indices,
                encoder_type=encoder_type,
                use_graph=use_graph,
                initial_ratio=None,  # Will be set conditionally in active_learning.py
                query_ratio=0.05,
                use_cluster=use_cluster,
                n_steps=10
            )
        return
    
    tasks_results_container = collections.defaultdict(dict)

    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        
        # Run the model
        run_res = run_once_nn(dataset_name=dataset_name, seed=seed, run_id=run_idx,
                            split=split, mode=mode, use_weights=use_weights, task_indices=task_indices,
                            encoder_type=encoder_type, use_graph=use_graph)
        
        for method, uq_dict in run_res.items():
            # Check if output is multitask (array) or single task (scalar)
            first_val = next(iter(uq_dict.values()))
            is_multitask = isinstance(first_val, (list, np.ndarray)) and np.size(first_val) > 1

            if not is_multitask:
                # Single Task logic (default to task_id 0)
                tasks_results_container[0].setdefault(method, []).append(uq_dict)
            else:
                # Multi Task logic: Split into separate dicts per task
                n_tasks_detected = len(first_val)
                for t in range(n_tasks_detected):
                    # Extract values for just this task
                    task_dict = {k: v[t] for k, v in uq_dict.items() if v is not None}
                    
                    # Correctly map the index t to the real task ID
                    # If indices=[0, 2]: t=0 -> ID=0, t=1 -> ID=2
                    real_task_id = task_indices[t] if (task_indices is not None and len(task_indices) > t) else t
                    
                    tasks_results_container[real_task_id].setdefault(method, []).append(task_dict)

    # 3. Save Results using the Combined Naming Convention
    print(f"\n===== Aggregated results over {n_runs} runs =====")

    output_dir = f"./cdata_{mode}"
    os.makedirs(output_dir, exist_ok=True)

    # Modify split name to include "_graph" if use_graph is True
    split_name = f"{split}_graph" if use_graph else split
    
    for task_id, method_results in sorted(tasks_results_container.items()):
        
        # --- COMBINED SUFFIX LOGIC ---
        # Global context + Specific ID
        # Example: "_tasks_0_2" + "_id_0"
        final_suffix = f"{global_task_suffix}_id_{task_id}"
        
        # Construct filename
        # Result: ./cdata_test/NN_random_chem_tasks_0_2_id_0_c.csv
        csv_filename = f"{output_dir}/NN_{split_name}_{dataset_name}{final_suffix}_c.csv"

        print(f"Saving Summary for TASK {task_id} to: {csv_filename}")
        
        # Call your ORIGINAL save function
        save_summary_to_csv(method_results, n_runs, csv_filename)
        
    # for run_idx in range(n_runs):
    #     seed = base_seed + run_idx
    #     run_res = run_once_nn(dataset_name=dataset_name, seed=seed, run_id=run_idx,
    #                           split=split, mode=mode, use_weights=use_weights, task_indices=task_indices)
    #     for method, uq_dict in run_res.items():
    #         all_results.setdefault(method, []).append(uq_dict)

    # # Aggregate: per method, per metric → mean & std
    # print("\n===== Aggregated results over", n_runs, "runs =====")
    # for method, uq_list in all_results.items():
    #     print(f"\n### Method: {method}")
    #     if not uq_list:
    #         print("  (no results)")
    #         continue

    #     # assume all dicts have same keys
    #     metric_names = uq_list[0].keys()

    #     for m in metric_names:
    #         # collect values, ignoring None
    #         vals = [d[m] for d in uq_list if d[m] is not None]
    #         if not vals:
    #             print(f"  {m}: no valid values")
    #             continue
    #         vals = np.asarray(vals, dtype=float)
    #         mean = float(vals.mean())
    #         std = float(vals.std(ddof=0))
    #         print(f"  {m}: mean={mean:.5g}, std={std:.5g}")

    # save_summary_to_csv(all_results, 
    #                     n_runs, 
    #                     "./cdata_{}/NN_{}_{}_c.csv".format(mode, split, dataset_name))


def run_once_gp(dataset_name: str,
                seed: int = 0,
                run_id: int = 0,
                split: str = "random",
                mode: str = "regression",
                use_weights: bool = False,
                task_indices: Optional[List[int]] = None):
    """
    One full run on a given dataset with a fixed seed,
    running both Exact GP and SVGP in a single shot.
    """
    set_global_seed(seed)

    # Updated to accept task_indices
    tasks, train_dc, valid_dc, test_dc, transformers = load_dataset(
        dataset_name=dataset_name,
        split=split,
        task_indices=task_indices
    )

    print(f"\n=== [GP] Run with seed={seed} on dataset={dataset_name}")
    results = {}
    all_cutoff_dfs = []

    use_weights = False
    if dataset_name in ["tox21", "toxcast", "sider", "clintox"]:
        use_weights = True

    if dataset_name not in ["qm8"]:
        results["gp_exact"], gp_cut_off = main_gp(train_dc, valid_dc, test_dc, run_id=run_id,
                                                  use_weights=use_weights, mode=mode)
        gp_cut_off['Method'] = "gp_exact"
        all_cutoff_dfs.append(gp_cut_off)
        
        results["gp_nngp"], nngp_cut_off = main_nngp_exact(train_dc, valid_dc, test_dc, run_id=run_id,
                                                           use_weights=use_weights, mode=mode)
        nngp_cut_off['Method'] = "gp_nngp"
        all_cutoff_dfs.append(nngp_cut_off)

    if mode == "regression":
        results["gp_svgp"], svgp_cut_off = main_svgp(train_dc, valid_dc, test_dc, run_id=run_id)
        svgp_cut_off['Method'] = "gp_svgp"
        all_cutoff_dfs.append(svgp_cut_off)

        results["gp_nnsvgp"], nnsvgp_cut_off = main_nngp_svgp(train_dc, valid_dc, test_dc, run_id=run_id)
        nnsvgp_cut_off['Method'] = "gp_nnsvgp"
        all_cutoff_dfs.append(nnsvgp_cut_off)

        result, cut_offs = main_nngp_exact_ensemble_all(train_dc, valid_dc, test_dc, run_id=run_id)
        for idx, cur in enumerate(result):
            results["nngp_ensemble_{}".format(cur["name"])] = cur["uq_metrics"]
            cut_offs[idx]['Method'] = "nngp_ensemble_{}".format(cur["name"])
            all_cutoff_dfs.append(cut_offs[idx])

    result, cut_offs = main_nngp_svgp_exact_ensemble_all(train_dc, valid_dc, test_dc,
                                                         run_id=run_id, use_weights=use_weights, mode=mode)
    for idx, cur in enumerate(result):
        results["nnsvgp_ensemble_{}".format(cur["name"])] = cur["uq_metrics"]
        cut_offs[idx]['Method'] = "nnsvgp_ensemble_{}".format(cur["name"])
        all_cutoff_dfs.append(cut_offs[idx])

    result, cut_offs = main_svgp_ensemble_all(train_dc, valid_dc, test_dc,
                                              run_id=run_id, use_weights=use_weights, mode=mode)
    for idx, cur in enumerate(result):
        results["svgp_ensemble_{}".format(cur["name"])] = cur["uq_metrics"]
        cut_offs[idx]['Method'] = "svgp_ensemble_{}".format(cur["name"])
        all_cutoff_dfs.append(cut_offs[idx])    

    # Concatenate all DataFrames into a single one
    combined_cutoff_df = pd.concat(all_cutoff_dfs, ignore_index=True)

    # --- Refined Filename Logic ---
    # 1. Generate a suffix for the filename based on task_indices
    if task_indices is None:
        task_suffix = ""
    else:
        task_suffix = "_tasks_" + "_".join(map(str, task_indices))

    # 2. Define the consistent filename with the suffix
    output_filename = f"./cdata_{mode}/figure/{split}_{dataset_name}{task_suffix}_GP_cutoff_run_{run_id}.csv"

    import os
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Store the final combined data
    combined_cutoff_df.to_csv(output_filename, index=False)

    del train_dc, valid_dc, test_dc, tasks, transformers
    del all_cutoff_dfs
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main_gp_all(dataset_name: str = "delaney",
                n_runs: int = 5,
                split: str = "random",
                base_seed: int = 0,
                mode: str = "regression",
                use_weights: bool = False,
                task_indices: Optional[List[int]] = None):
    """
    Multi-run driver for GP + SVGP.
    - Calls run_once_gp(...) n_runs times with different seeds.
    - Aggregates per-method, per-metric mean/std PER TASK.
    - Saves a CSV via save_summary_to_csv for each task.
    """
    
    # Define global suffix for file naming based on the requested tasks
    if task_indices is None:
        global_task_suffix = ""
    else:
        global_task_suffix = "_tasks_" + "_".join(map(str, task_indices))

    tasks_results_container = collections.defaultdict(dict)

    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        # Pass task_indices to the run_once function
        run_res = run_once_gp(dataset_name=dataset_name, seed=seed, run_id=run_idx, 
                              split=split, mode=mode, use_weights=use_weights, task_indices=task_indices)

        for method, uq_dict in run_res.items():
            # Check if output is multitask (array) or single task (scalar)
            # We assume the first metric in the dict represents the shape of all metrics
            first_val = next(iter(uq_dict.values()))
            is_multitask = isinstance(first_val, (list, np.ndarray)) and np.size(first_val) > 1

            if not is_multitask:
                # Single Task logic (default to task_id 0)
                tasks_results_container[0].setdefault(method, []).append(uq_dict)
            else:
                # Multi Task logic: Split into separate dicts per task
                n_tasks_detected = len(first_val)
                for t in range(n_tasks_detected):
                    # Extract values for just this task
                    task_dict = {k: v[t] for k, v in uq_dict.items() if v is not None}
                    
                    # Correctly map the index t to the real task ID
                    # If indices=[0, 2]: t=0 -> ID=0, t=1 -> ID=2
                    real_task_id = task_indices[t] if (task_indices is not None and len(task_indices) > t) else t
                    
                    tasks_results_container[real_task_id].setdefault(method, []).append(task_dict)

    # 3. Save Results using the Combined Naming Convention
    print(f"\n===== [GP] Aggregated results over {n_runs} runs =====")

    output_dir = f"./cdata_{mode}"
    os.makedirs(output_dir, exist_ok=True)

    for task_id, method_results in sorted(tasks_results_container.items()):
        
        # --- COMBINED SUFFIX LOGIC ---
        # Global context + Specific ID
        # Example: "_tasks_0_2" + "_id_0"
        final_suffix = f"{global_task_suffix}_id_{task_id}"
        
        # Construct filename
        # Result: ./cdata_regression/GP_random_delaney_tasks_0_2_id_0_c.csv
        csv_filename = f"{output_dir}/GP_{split}_{dataset_name}{final_suffix}_c.csv"

        print(f"Saving Summary for TASK {task_id} to: {csv_filename}")
        
        # Print aggregation to console for verification (optional, mirroring original behavior)
        print(f"\n--- Task {task_id} Summary ---")
        for method, uq_list in method_results.items():
            if not uq_list: continue
            metric_names = uq_list[0].keys()
            for m in metric_names:
                vals = [d[m] for d in uq_list if d[m] is not None]
                if not vals: continue
                vals = np.asarray(vals, dtype=float)
                mean = float(vals.mean())
                std = float(vals.std(ddof=0))
                # print(f"  {method} - {m}: mean={mean:.5g}, std={std:.5g}")

        # Call save function
        save_summary_to_csv(method_results, n_runs, csv_filename)


# Note: GP models are not saved/reloaded due to memory constraints (training data required)
# If you need GP evaluation, you should retrain or use a different approach


# if __name__ == "__main__":
#     # main_svgp_ensemble_all()
#     # main_nngp_exact()
#     # main_nngp_exact_ensemble_all()
#     # main_nngp_svgp_exact_ensemble_all()
#     # main_nn()
#     tasks, train_dc, valid_dc, test_dc, transformers = load_dataset(
#         dataset_name="tox21",
#         # dataset_name="qm8",
#         split="random",
#         task_indices=[0,2]
#     )

#     res1, res2 = main_nngp_svgp_exact_ensemble_all(valid_dc, valid_dc, test_dc, run_id=0, use_weights=True, mode="classification")
#     print(res1)
#     print(res2)

#     # train_nn_baseline(train_dc, valid_dc, test_dc,
#     #                   run_id=0, use_weights=False)
#     # train_nn_baseline(train_dc, valid_dc, test_dc,
#     #                   run_id=0, use_weights=True, mode="classification")
#     # train_nn_mc_dropout(train_dc, valid_dc, test_dc,
#     #                   run_id=0, use_weights=False)
#     # train_nn_mc_dropout(train_dc, valid_dc, test_dc,
#     #                   run_id=0, use_weights=True, mode="classification")
#     # train_evd_baseline(train_dc, valid_dc, test_dc,
#     #                   run_id=0, use_weights=False, mode="classification")
#     # train_evd_baseline(train_dc, valid_dc, test_dc,
#     #                   run_id=0, use_weights=True, mode="classification")
#     # train_nn_deep_ensemble(train_dc, valid_dc, test_dc,
#     #                   run_id=0, use_weights=False, mode="classification")
#     # train_nn_deep_ensemble(train_dc, valid_dc, test_dc,
#     #                   run_id=0, use_weights=True, mode="classification")
    
#     # main_svgp_ensemble_all(train_dc=train_dc, valid_dc=valid_dc, test_dc=test_dc,
#     #                 run_id=0, use_weights=False, mode="classification")
#     # main_svgp_ensemble_all(train_dc=train_dc, valid_dc=valid_dc, test_dc=test_dc,
#     #                 run_id=0, use_weights=True, mode="classification")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="delaney",
                        choices=["qm7", "qm8", "delaney", "lipo", "tox21", "toxcast", "sider", "clintox"])
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="random",
                        choices=["random", "scaffold"])
    parser.add_argument("--mode", type=str, default="regression",
                        choices=["regression", "classification"])
    parser.add_argument("--tasks", type=int, nargs='+', default=None,
                    help="List of task indices to use (e.g., --tasks 0 2)")
    parser.add_argument("--encoder_type", type=str, default="identity",
                        choices=["identity", "dmpnn"],
                        help="Type of encoder: 'identity' for vector features, 'dmpnn' for graph features")
    parser.add_argument("--use_graph", action="store_true",
                        help="Use DMPNN graph featurizer instead of vector featurizer (sets encoder_type='dmpnn')")
    parser.add_argument("--al", action="store_true",
                        help="Enable active learning mode")
    parser.add_argument("--batch", action="store_true",
                        help="Active learning: use cluster-based selection (latent + uncertainty; nn_baseline uses random per cluster)")
    args = parser.parse_args()
    
    # If --use_graph is specified, override encoder_type
    if args.use_graph:
        args.encoder_type = "dmpnn"

    # main_gp_all(dataset_name=args.dataset,
    #             n_runs=args.n_runs,
    #             split=args.split,
    #             base_seed=args.base_seed,
    #             mode = args.mode,
    #             use_weights = (args.mode == "classification"),
    #             task_indices=args.tasks)

    main_nn(dataset_name=args.dataset,
            n_runs=args.n_runs,
            split=args.split,
            base_seed=args.base_seed,
            mode = args.mode,
            use_weights = (args.mode == "classification"),
            task_indices=args.tasks,
            encoder_type=args.encoder_type,
            use_graph=args.use_graph,
            active_learning=args.al,
            use_cluster=args.batch)
