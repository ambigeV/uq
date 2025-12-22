import gc
import deepchem as dc
from data_utils import prepare_datasets, evaluate_uq_metrics_from_interval, \
    compute_ece, calculate_cutoff_classification_data, evaluate_uq_metrics_classification
from nn_baseline import train_nn_baseline, train_nn_deep_ensemble, train_nn_mc_dropout, train_evd_baseline
from deepchem.molnet import load_qm7, load_delaney, load_qm8, load_qm9, load_lipo, load_freesolv, load_tox21

import torch
import gpytorch
from gp_single import GPyTorchRegressor, SVGPModel, FeatureNet, \
    DeepFeatureKernel, NNGPExactGPModel, NNSVGPLearnedInducing, \
    GPyTorchClassifier, SVGPClassificationModel
from gp_trainer import GPTrainer, EnsembleGPTrainer, GPClassificationTrainer
from data_utils import calculate_cutoff_error_data
import numpy as np
import csv
import random


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


def load_dataset(dataset_name: str = "delaney", split: str = "random"):
    """
    dataset_name: one of {"qm7", "qm8", "delaney", "lipo"}
    featurizer_name: "ecfp" or "coulomb" (depending on dataset)
    """
    if dataset_name == "qm7":
        FEATURIZER = "coulomb"
        tasks, datasets, transformers = load_qm7(splitter=split)
    elif dataset_name == "qm8":
        FEATURIZER = "coulomb"
        tasks, datasets, transformers = load_qm8(splitter=split)
    elif dataset_name == "delaney":
        FEATURIZER = "ecfp"
        tasks, datasets, transformers = load_delaney(splitter=split)
    elif dataset_name == "lipo":
        FEATURIZER = "ecfp"
        tasks, datasets, transformers = load_lipo(splitter=split)
    elif dataset_name == "tox21":
        FEATURIZER = "ecfp"
        tasks, datasets, transformers = load_tox21(splitter=split)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    train_dc, valid_dc, test_dc = prepare_datasets(
        datasets,
        featurizer_name=FEATURIZER
    )

    print("train_dc.w is", train_dc.w.shape)
    print("train_dc.w is", train_dc.w.shape)
    print("train_dc.w is", train_dc.w.shape)

    # Keep only first task if multi-task
    if train_dc.y.ndim == 2 and train_dc.y.shape[1] > 1:
        first_task_name = tasks[0]
        tasks = [first_task_name]

        def _keep_first_task(ds) -> dc.data.NumpyDataset:
            return dc.data.NumpyDataset(
                X=ds.X,
                y=ds.y[:, 0:1],
                w=ds.w[:, 0:1],
                ids=ds.ids,
                n_tasks=1,
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


def main_nngp_exact(train_dc, valid_dc, test_dc, device: str = "cpu", run_id=0, use_weights=False, mode="regression"):
    # 1) Load
    print("Train X shape:", train_dc.X.shape)
    print("Train y shape:", train_dc.y.shape)
    N, D = train_dc.X.shape

    # 2) Torch tensors
    X_train_t, y_train_t = _to_torch_xy(train_dc)

    train_w_torch = None
    if use_weights and train_dc.w is not None:
        print("--> Using per-point precision weights via FixedNoiseGaussianLikelihood")
        w_np = train_dc.w
        if w_np.ndim == 2:
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
        M = max(512, N//10)
        nngp_classifier = GPyTorchClassifier(
            train_x=X_train_t,
            train_y=y_train_t,
            normalize_x=False, 
            gp_model_cls=NNSVGPLearnedInducing, 
            feature_extractor=feat,  
            num_inducing=M,  
            kernel="matern52",
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
            num_iters=50,
            device=dev,
            log_interval=40,
            use_weights=use_weights  
        )

        trainer.train()

        # 4. Evaluate (AUC/Accuracy)
        valid_metrics = trainer.evaluate(valid_dc, use_weights=use_weights)
        test_metrics = trainer.evaluate(test_dc, use_weights=use_weights)

        print(f"[NN-SVGP Classif] Valid AUC: {valid_metrics['auc']:.4f}")
        print(f"[NN-SVGP Classif] Test AUC:  {test_metrics['auc']:.4f}")

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
    M = max(256, N // 20)      # e.g., ~5% of data or at least 256 (tune per dataset/capacity)
    items = []
    for i in range(E):
        # (keeps your current "two-folds per model" split)
        train_idx_i = np.concatenate([folds[j] for j in [i, (i + 1) % E]])
        ds_i = _subset_numpy_dataset(train_dc, train_idx_i)

        X_train_torch = torch.from_numpy(ds_i.X).float()
        y_np = ds_i.y
        if y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np[:, 0]
        y_train_torch = torch.from_numpy(y_np).float()

        train_w_torch = None
        if use_weights and train_dc.w is not None:
            print("--> Using per-point precision weights via NN-SVGP")
            w_np = ds_i.w
            if w_np.ndim == 2:
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
                num_inducing = max(512, Ni//10),
                kernel="matern52"
            )
            items.append({
                "model": gp_model_i,
                "train_dataset": ds_i,
                "lr": 0.01,
                "nn_lr": 1e-3,
                "ngd_lr": 0.02,
                "warmup_iters": 5,
                "clip_grad": 1.0,
                "num_iters": 500,
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
            if name == "uniform":
                w = np.ones(K, dtype=float) / K
                print(f"\n[Calib:{name}] Using uniform weights.")
            else:
                w = ens.calibrate_class_weights(valid_dc, method=name, **kwargs)
                print(f"\n[Calib:{name}] Weights: {np.round(w, 4)}")

            # (b) evaluate on valid/test with these weights
            valid_metrics = ens.evaluate_auc_w(valid_dc, w=w)
            test_metrics  = ens.evaluate_auc_w(test_dc,  w=w)
            print(f"\n[GP Classification] Validation AUC: {valid_metrics['auc']:.4f} | Acc: {valid_metrics['acc']:.4f}")
            print(f"[GP Classification] Test AUC:       {test_metrics['auc']:.4f} | Acc: {test_metrics['acc']:.4f}")

            # 5. UQ Analysis: Calibration (ECE)
            test_probs = test_metrics["probs"]
            test_y = test_metrics["y_true"]

            ece_score = compute_ece(test_probs, test_y, n_bins=20)
            print(f"[GP Classification] Test ECE (UQ):  {ece_score:.4f}")

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

        # ---------------- 5) Pretty summary ----------------
        print("\n===== Summary over weighting strategies =====")
        for r in results:
            w_str = " ".join([f"{x:.3f}" for x in r["weights"]])
            print(f"{r['name']:>9} | valid MSE: {r['valid_auc']:.6f} | test MSE: {r['test_auc']:.6f} | w: [{w_str}]")

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
        if w_np.ndim == 2:
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
        print("[GP Single-task] Validation MSE:", valid_mse)
        print("[GP Single-task] Test MSE:",      test_mse)

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
        # 1. Instantiate Classifier
        # Note: We do NOT pass train_w here. In classification, weights are used
        # for evaluation metrics (AUC), not for fixing likelihood noise variance.
        gp_model = GPyTorchClassifier(
            train_x=X_train_torch,
            train_y=y_train_torch,
            normalize_x="auto",  # Set to True if not using ECFP/Binary features
            # SVGP Specific Args
            gp_model_cls=SVGPClassificationModel,
            num_inducing=max(512, N//10),  # 128 is a good balance for speed/accuracy
            kernel="matern52"  # usually better than RBF for molecular data
        )

        # 2. Instantiate Trainer
        trainer = GPClassificationTrainer(
            model=gp_model,
            train_dataset=train_dc,
            lr=0.01,  # Adam LR (Kernel/Feature Extractor)
            ngd_lr=0.1,  # Natural Gradient LR (Variational Parameters)
            num_iters=200,
            device="cpu",
            log_interval=40,
            warmup_iters=5,
            use_weights=use_weights
        )

        # 3. Train
        print("--- Starting GP Classification Training ---")
        trainer.train()

        # 4. Evaluate (AUC & Accuracy)
        # The trainer.evaluate() method handles 'use_weights' internally for AUC calculation
        valid_metrics = trainer.evaluate(valid_dc, use_weights=use_weights)
        test_metrics = trainer.evaluate(test_dc, use_weights=use_weights)

        print(f"\n[GP Classification] Validation AUC: {valid_metrics['auc']:.4f} | Acc: {valid_metrics['acc']:.4f}")
        print(f"[GP Classification] Test AUC:       {test_metrics['auc']:.4f} | Acc: {test_metrics['acc']:.4f}")

        # 5. UQ Analysis: Calibration (ECE)
        test_probs = test_metrics["probs"]
        test_y = test_metrics["y_true"]

        ece_score = compute_ece(test_probs, test_y, n_bins=20)
        print(f"[GP Classification] Test ECE (UQ):  {ece_score:.4f}")

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
        train_w_torch = None

        if use_weights and train_dc.w is not None:
            print("--> Using per-point precision weights via FixedNoiseGaussianLikelihood")
            w_np = ds_i.w
            if w_np.ndim == 2:
                w_np = w_np[:, 0]
            train_w_torch = torch.from_numpy(w_np).float()

        if y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np[:, 0]
        y_train_torch = torch.from_numpy(y_np).float()

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
            gp_model_i = GPyTorchClassifier(
                train_x=X_train_torch,
                train_y=y_train_torch,
                normalize_x="auto",
                gp_model_cls=SVGPClassificationModel,
                num_inducing=max(512, Ni // 10),
                kernel="matern52"
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
            if name == "uniform":
                w = np.ones(K, dtype=float) / K
                print(f"\n[Calib:{name}] Using uniform weights.")
            else:
                w = ens.calibrate_class_weights(valid_dc, method=name, **kwargs)
                print(f"\n[Calib:{name}] Weights: {np.round(w, 4)}")

            # 4. Evaluate (AUC & Accuracy)
            # The trainer.evaluate() method handles 'use_weights' internally for AUC calculation
            valid_metrics = ens.evaluate_auc_w(valid_dc, w=w)
            test_metrics = ens.evaluate_auc_w(test_dc, w=w)

            print(f"\n[GP Classification] Validation AUC: {valid_metrics['auc']:.4f} | Acc: {valid_metrics['acc']:.4f}")
            print(f"[GP Classification] Test AUC:       {test_metrics['auc']:.4f} | Acc: {test_metrics['acc']:.4f}")

            # 5. UQ Analysis: Calibration (ECE)
            test_probs = test_metrics["probs"]
            test_y = test_metrics["y_true"]

            ece_score = compute_ece(test_probs, test_y, n_bins=20)
            print(f"[GP Classification] Test ECE (UQ):  {ece_score:.4f}")

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

        # ---------------- 5) Pretty summary ----------------
        print("\n===== Summary over weighting strategies =====")
        for r in results:
            w_str = " ".join([f"{x:.3f}" for x in r["weights"]])
            print(f"{r['name']:>9} | valid MSE: {r['valid_auc']:.6f} | test MSE: {r['test_auc']:.6f} | w: [{w_str}]")

    return results, cut_offs


def run_once_nn(dataset_name: str,
                seed: int = 0,
                run_id: int = 0,
                split: str = "random",
                mode: str = "regression",
                use_weights: bool = False):
    """
    One full run on a given dataset & featurizer with a fixed seed.
    """
    set_global_seed(seed)

    tasks, train_dc, valid_dc, test_dc, transformers = load_dataset(
        dataset_name=dataset_name,
        split=split
    )

    use_weights = False
    if dataset_name in ["tox21"]:
        use_weights = True

    print(f"\n=== Run with seed={seed} on dataset={dataset_name}")
    results = {}
    all_cutoff_dfs = []
    results["nn_evd"], cut_off_evd = train_evd_baseline(train_dc, valid_dc, test_dc,
                                                        run_id=run_id, use_weights=use_weights, mode=mode)

    cut_off_evd['Method'] = "nn_evd"
    all_cutoff_dfs.append(cut_off_evd)

    if dataset_name not in ["qm8"]:
        results["nn_baseline"] = train_nn_baseline(train_dc, valid_dc, test_dc,
                                                   run_id=run_id, use_weights=use_weights, mode=mode)

    results["nn_mc_dropout"], cut_off_dropout = train_nn_mc_dropout(train_dc, valid_dc, test_dc,
                                                                    run_id=run_id, use_weights=use_weights, mode=mode)
    cut_off_dropout['Method'] = "nn_mc_dropout"
    all_cutoff_dfs.append(cut_off_dropout)

    results["nn_deep_ensemble"], cut_off_ensemble = train_nn_deep_ensemble(train_dc, valid_dc, test_dc,
                                                                           run_id=run_id, use_weights=use_weights, mode=mode)
    cut_off_ensemble['Method'] = "nn_deep_ensemble"
    all_cutoff_dfs.append(cut_off_ensemble)

    import pandas as pd

    # Concatenate all DataFrames into a single one
    combined_cutoff_df = pd.concat(all_cutoff_dfs, ignore_index=True)

    # Define a consistent filename
    output_filename = f"./cdata_{mode}/figure/{split}_{dataset_name}_NN_cutoff_run_{run_id}.csv"

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
            use_weights: bool = False):

    all_results = {}

    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        run_res = run_once_nn(dataset_name=dataset_name, seed=seed, run_id=run_idx,
                              split=split, mode=mode, use_weights=use_weights)
        for method, uq_dict in run_res.items():
            all_results.setdefault(method, []).append(uq_dict)

    # Aggregate: per method, per metric → mean & std
    print("\n===== Aggregated results over", n_runs, "runs =====")
    for method, uq_list in all_results.items():
        print(f"\n### Method: {method}")
        if not uq_list:
            print("  (no results)")
            continue

        # assume all dicts have same keys
        metric_names = uq_list[0].keys()

        for m in metric_names:
            # collect values, ignoring None
            vals = [d[m] for d in uq_list if d[m] is not None]
            if not vals:
                print(f"  {m}: no valid values")
                continue
            vals = np.asarray(vals, dtype=float)
            mean = float(vals.mean())
            std = float(vals.std(ddof=0))
            print(f"  {m}: mean={mean:.5g}, std={std:.5g}")

    save_summary_to_csv(all_results, 
                        n_runs, 
                        "./cdata_{}/NN_{}_{}_c.csv".format(mode, split, dataset_name))


def run_once_gp(dataset_name: str,
                seed: int = 0,
                run_id: int = 0,
                split: str = "random",
                mode: str = "regression",
                use_weights: bool = False):
    """
    One full run on a given dataset with a fixed seed,
    running both Exact GP and SVGP in a single shot.

    Analogous to run_once_nn().
    """
    set_global_seed(seed)

    tasks, train_dc, valid_dc, test_dc, transformers = load_dataset(
        dataset_name=dataset_name,
        split=split
    )

    print(f"\n=== [GP] Run with seed={seed} on dataset={dataset_name}")
    results = {}
    all_cutoff_dfs = []

    use_weights = False
    if dataset_name in ["tox21"]:
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

    result, cut_offs = main_svgp_ensemble_all(train_dc, valid_dc, test_dc,
                                              run_id=run_id, use_weights=use_weights, mode=mode)
    for idx, cur in enumerate(result):
        results["svgp_ensemble_{}".format(cur["name"])] = cur["uq_metrics"]
        cut_offs[idx]['Method'] = "svgp_ensemble_{}".format(cur["name"])
        all_cutoff_dfs.append(cut_offs[idx])    

    result, cut_offs = main_nngp_svgp_exact_ensemble_all(train_dc, valid_dc, test_dc,
                                                         run_id=run_id, use_weights=use_weights, mode=mode)
    for idx, cur in enumerate(result):
        results["nnsvgp_ensemble_{}".format(cur["name"])] = cur["uq_metrics"]
        cut_offs[idx]['Method'] = "nnsvgp_ensemble_{}".format(cur["name"])
        all_cutoff_dfs.append(cut_offs[idx])
    
    import pandas as pd

    # Concatenate all DataFrames into a single one
    combined_cutoff_df = pd.concat(all_cutoff_dfs, ignore_index=True)

    # Define a consistent filename
    output_filename = f"./cdata_{mode}/figure/{split}_{dataset_name}_GP_cutoff_run_{run_id}.csv"

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
                use_weights: bool = False):
    """
    Multi-run driver for GP + SVGP, analogous to main_nn().

    - Calls run_once_gp(...) n_runs times with different seeds.
    - Aggregates per-method, per-metric mean/std.
    - Saves a CSV via save_summary_to_csv.
    """
    all_results = {}  # method_name -> list[metrics_dict]

    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        run_res = run_once_gp(dataset_name=dataset_name, seed=seed, run_id=run_idx, split=split, mode=mode, use_weights=use_weights)

        for method, uq_dict in run_res.items():
            all_results.setdefault(method, []).append(uq_dict)

    # Aggregate: per method, per metric → mean & std
    print("\n===== [GP] Aggregated results over", n_runs, "runs =====")
    for method, uq_list in all_results.items():
        print(f"\n### Method: {method}")
        if not uq_list:
            print("  (no results)")
            continue

        metric_names = uq_list[0].keys()

        for m in metric_names:
            vals = [d[m] for d in uq_list if d[m] is not None]
            if not vals:
                print(f"  {m}: no valid values")
                continue
            vals = np.asarray(vals, dtype=float)
            mean = float(vals.mean())
            std = float(vals.std(ddof=0))
            print(f"  {m}: mean={mean:.5g}, std={std:.5g}")

    # Reuse the same helper as NN, different filename prefix
    save_summary_to_csv(
        all_results,
        n_runs,
        "./cdata_{}/GP_{}_{}_c.csv".format(mode, split, dataset_name),
    )


# if __name__ == "__main__":
#     # main_svgp_ensemble_all()
#     # main_nngp_exact()
#     # main_nngp_exact_ensemble_all()
#     # main_nngp_svgp_exact_ensemble_all()
#     # main_nn()
#     tasks, train_dc, valid_dc, test_dc, transformers = load_dataset(
#         dataset_name="tox21",
#         split="random"
#     )

    # train_nn_baseline(train_dc, valid_dc, test_dc,
    #                   run_id=0, use_weights=False, mode="classification")
    # train_nn_baseline(train_dc, valid_dc, test_dc,
    #                   run_id=0, use_weights=True, mode="classification")
    # train_nn_mc_dropout(train_dc, valid_dc, test_dc,
    #                   run_id=0, use_weights=False, mode="classification")
    # train_nn_mc_dropout(train_dc, valid_dc, test_dc,
    #                   run_id=0, use_weights=True, mode="classification")
    # train_evd_baseline(train_dc, valid_dc, test_dc,
    #                   run_id=0, use_weights=False, mode="classification")
    # train_evd_baseline(train_dc, valid_dc, test_dc,
    #                   run_id=0, use_weights=True, mode="classification")
    # train_nn_deep_ensemble(train_dc, valid_dc, test_dc,
    #                   run_id=0, use_weights=False, mode="classification")
    # train_nn_deep_ensemble(train_dc, valid_dc, test_dc,
    #                   run_id=0, use_weights=True, mode="classification")
    
    # main_svgp_ensemble_all(train_dc=train_dc, valid_dc=valid_dc, test_dc=test_dc,
    #                 run_id=0, use_weights=False, mode="classification")
    # main_svgp_ensemble_all(train_dc=train_dc, valid_dc=valid_dc, test_dc=test_dc,
    #                 run_id=0, use_weights=True, mode="classification")

#
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="delaney",
                        choices=["qm7", "qm8", "delaney", "lipo", "tox21"])
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="random",
                        choices=["random", "scaffold"])
    parser.add_argument("--mode", type=str, default="regression",
                        choices=["regression", "classification"])
    args = parser.parse_args()

    main_gp_all(dataset_name=args.dataset,
                n_runs=args.n_runs,
                split=args.split,
                base_seed=args.base_seed,
                mode = args.mode,
                use_weights = (args.mode == "classification"))

    main_nn(dataset_name=args.dataset,
            n_runs=args.n_runs,
            split=args.split,
            base_seed=args.base_seed,
            mode = args.mode,
            use_weights = (args.mode == "classification"))
