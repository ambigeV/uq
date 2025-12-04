import numpy as np
import torch
import torch.nn as nn
import deepchem as dc
from scipy.stats import norm
from data_utils import evaluate_uq_metrics_from_interval
from deepchem.models.losses import Loss, _make_pytorch_shapes_consistent


# ============================================================
# 1. Base Regressors
# ============================================================

class MyTorchRegressor(nn.Module):
    """Standard feed-forward NN for regression."""
    def __init__(self, n_features: int, n_tasks: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_tasks),
        )

    def forward(self, x):
        return self.net(x)


class HeteroscedasticL2Loss(Loss):
    """
    Kendall & Gal heteroscedastic regression loss:

      L = 1/N * sum_i [ 0.5 * exp(-s_i) * (y_i - y_hat_i)^2 + 0.5 * s_i ]

    where s_i = log sigma_i^2.
    """

    def _create_pytorch_loss(self):
        import torch

        def loss(output, labels):
            # DeepChem helper: handles (N,), (N,1), multi-task shapes, etc.
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            # output: (B, 2*T)  where T = n_tasks
            # labels: (B, T)

            D = labels.shape[-1]
            y_hat   = output[..., :D]   # (B, T)
            log_var = output[..., D:]   # (B, T)

            precision = torch.exp(-log_var)     # 1 / sigma^2
            diff2     = (labels - y_hat) ** 2   # (B, T)

            # elementwise Kendall & Gal loss
            loss_elem = 0.5 * precision * diff2 + 0.5 * log_var  # (B, T)

            # mean over tasks → shape (B,) (per-sample)
            loss_per_sample = torch.mean(loss_elem, dim=-1)
            return loss_per_sample

        return loss


class MyTorchRegressorMC(nn.Module):
    """
    MC-Dropout + heteroscedastic head.

    Forward returns:
      mean   : (B, T)   -- prediction
      var    : (B, T)   -- variance = exp(log_var)
      packed : (B, 2T)  -- [mean, log_var] for the loss
    """
    def __init__(self, n_features: int, n_tasks: int = 1):
        super().__init__()
        self.n_tasks = n_tasks

        self.feature_net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

        # Output both mean and log_var in one go: 2 * n_tasks units
        self.out_head = nn.Linear(64, 2 * n_tasks)

    def forward(self, x):
        h   = self.feature_net(x)          # (B, 128)
        raw = self.out_head(h)             # (B, 2T)

        # Split into mean and log_var
        T       = self.n_tasks
        mean    = raw[..., :T]             # (B, T)
        log_var = raw[..., T:]             # (B, T)
        var     = torch.exp(log_var)       # (B, T)

        packed  = raw                      # (B, 2T), [mean, log_var]

        # Order must match output_types in TorchModel
        return mean, var, packed

# class MyTorchRegressorMC(nn.Module):
#     """NN with dropout active during inference (MC Dropout)."""
#     def __init__(self, n_features: int, n_tasks: int = 1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_features, 256),
#             nn.ReLU(),
#             nn.Dropout(p=0.1),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(p=0.1),
#             nn.Linear(128, n_tasks),
#         )
#
#     def forward(self, x):
#         return self.net(x)


# ============================================================
# 2. Helper Evaluation
# ============================================================

def mse_from_mean_prediction(mean, dataset):
    """
    Robust MSE calculation that handles:
    1. Multi-column outputs (e.g., [Prediction, Variance]) by slicing col 0.
    2. Shape mismatches (e.g., (N,1) vs (N,)) by flattening both.
    """
    # 1. Always flatten the True Labels to (N,)
    y_true = dataset.y.reshape(-1)

    # 2. Handle the Prediction 'mean'
    # Check if 'mean' is 2D and has more than 1 column (e.g., [Pred, Var])
    if mean.ndim > 1 and mean.shape[1] > 1:
        # Take only the first column (The Prediction)
        mean = mean[:, 0]

    # 3. Flatten prediction to (N,) to guarantee 1-to-1 subtraction
    mean = mean.reshape(-1)

    # 4. Calculate MSE
    return ((mean - y_true) ** 2).mean()


# ============================================================
# 3. Deep Ensemble Wrapper
# ============================================================

class DeepEnsembleRegressor:
    """
    Wrapper over M independently trained DeepChem TorchModels.
    Provides: predict() and predict_interval().
    """
    def __init__(self, models):
        self.models = models

    def predict(self, dataset):
        preds = []
        for m in self.models:
            out = m.predict(dataset)[:, 0]    
            preds.append(out)
        return np.stack(preds, axis=0)        

    def predict_interval(self, dataset, alpha=0.05):
        Y = self.predict(dataset)             
        mean = Y.mean(axis=0)
        std  = Y.std(axis=0) + 1e-8

        z = norm.ppf(1 - alpha/2)
        lower = mean - z * std
        upper = mean + z * std

        return mean, lower, upper


# ============================================================
# 4. MC Dropout Wrapper
# ============================================================

# class MCDropoutRegressor:
#     """
#     Wrapper for MC-Dropout inference.
#     Calls model.model.train() to activate dropout at inference.
#     """
#     def __init__(self, dc_model, n_samples=30):
#         self.model = dc_model
#         self.n_samples = n_samples
#
#     def predict_samples(self, dataset):
#         preds = []
#
#         self.model.model.train()
#
#         for _ in range(self.n_samples):
#             p = self.model.predict(dataset)[:, 0]  # (N,)
#             preds.append(p)
#
#         return np.stack(preds, axis=0)        # (S, N)
#
#     def predict_interval(self, dataset, alpha=0.05):
#         Y = self.predict_samples(dataset)     # (S, N)
#         mean = Y.mean(axis=0)
#         std  = Y.std(axis=0) + 1e-8
#
#         z = norm.ppf(1 - alpha/2)
#         lower = mean - z * std
#         upper = mean + z * std
#
#         return mean, lower, upper


# ============================================================
# 5. TRAINING FUNCTIONS
# ============================================================

# ------------------------------
# Baseline NN (no UQ)
# ------------------------------
def train_nn_baseline(train_dc, valid_dc, test_dc):
    n_tasks = train_dc.y.shape[1]
    n_features = train_dc.X.shape[1]

    model = MyTorchRegressor(n_features, n_tasks)
    loss = dc.models.losses.L2Loss()

    dc_model = dc.models.TorchModel(
        model=model,
        loss=loss,
        output_types=['prediction'],
        batch_size=64,
        learning_rate=1e-3,
        mode='regression',
    )

    dc_model.fit(train_dc, nb_epoch=30)

    metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
    valid_score = dc_model.evaluate(valid_dc, [metric])
    test_score  = dc_model.evaluate(test_dc,  [metric])

    print("[NN Baseline] Validation MSE:", valid_score[metric.name])
    print("[NN Baseline] Test MSE:",        test_score[metric.name])

    return {"MSE": test_score[metric.name]}


# ------------------------------
# Deep Ensemble
# ------------------------------
def train_nn_deep_ensemble(train_dc, valid_dc, test_dc, M=5):
    n_tasks = train_dc.y.shape[1]
    n_features = train_dc.X.shape[1]

    models = []

    for i in range(M):
        model = MyTorchRegressor(n_features, n_tasks)
        loss = dc.models.losses.L2Loss()

        dc_model = dc.models.TorchModel(
            model=model,
            loss=loss,
            output_types=['prediction'],
            batch_size=64,
            learning_rate=1e-3,
            mode='regression',
        )
        dc_model.fit(train_dc, nb_epoch=50)
        models.append(dc_model)

    ensemble = DeepEnsembleRegressor(models)

    # Evaluate using ensemble mean
    mean_valid, _, _ = ensemble.predict_interval(valid_dc)
    mean_test,  lower_test, upper_test = ensemble.predict_interval(test_dc)
    test_error = mse_from_mean_prediction(mean_test,  test_dc)

    print("[Deep Ensemble] Validation MSE:", mse_from_mean_prediction(mean_valid, valid_dc))
    print("[Deep Ensemble] Test MSE:",        mse_from_mean_prediction(mean_test,  test_dc))

    uq_metrics = evaluate_uq_metrics_from_interval(
        y_true=test_dc.y,
        mean=mean_test,
        lower=lower_test,
        upper=upper_test,
        alpha=0.05,
        test_error=test_error,
    )

    print("UQ (Deep Ensemble):", uq_metrics)

    return uq_metrics


# ------------------------------
# MC-Dropout
# ------------------------------

class MCDropoutRegressor:
    """
    Wrapper for MC-Dropout inference.
    Calls model.model.train() to activate dropout at inference.
    """
    def __init__(self, dc_model, n_samples=1):
        self.model = dc_model
        self.n_samples = n_samples

    def predict_samples(self, dataset):
        preds = []

        # Turn on dropout in the underlying torch model
        self.model.model.train()

        for _ in range(self.n_samples):
            p = self.model.predict(dataset)[:,0] # (N,)  (single-task)
            preds.append(p)

        return np.stack(preds, axis=0)        # (S, N)

    def predict_interval(self, dataset, alpha=0.05):
        Y = self.predict_samples(dataset)     # (S, N)
        mean = Y.mean(axis=0)
        std  = Y.std(axis=0) + 1e-8

        z = norm.ppf(1 - alpha/2)
        lower = mean - z * std
        upper = mean + z * std

        return mean, lower, upper


class MCDropoutRegressorRefined:
    def __init__(self, dc_model, n_samples=100):
        self.dc_model = dc_model
        self.n_samples = n_samples

    def predict_uncertainty(self, dataset):
        # ==================================================================
        # PART 1: The Mean (Target MSE: ~0.67)
        # We use DeepChem's standard predict(), which runs in EVAL mode.
        # ==================================================================

        # 1. Get Deterministic Prediction (Dropout OFF)
        # This returns shape (N, 2) -> [Prediction, Variance]
        raw_preds = self.dc_model.predict(dataset)

        # 2. SLICE IT! (Crucial fix for the 1.80 -> 0.67 drop)
        # We discard the variance column for the MSE calculation.
        mean_pred = raw_preds[:, 0]

        # Ensure shape is (N, 1) for consistent metric calculation
        # if mean_pred.ndim == 1:
        #     mean_pred = mean_pred.reshape(-1, 1)

        # ==================================================================
        # PART 2: The Variance (Target: Valid Coverage)
        # We manually run the loop in TRAIN mode to get uncertainty.
        # ==================================================================
        torch_model = self.dc_model.model
        torch_model.train()  # Force Dropout ON

        # Convert data to Tensor
        X_b = torch.from_numpy(dataset.X).float().to(self.dc_model.device)

        sampled_means = []
        sampled_vars = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                outputs = torch_model(X_b)

                # Handle output shape (N, 2) or list of outputs
                if isinstance(outputs, (list, tuple)):
                    mu = outputs[0]
                    v = outputs[1]
                else:
                    mu = outputs[:, 0]  # Prediction column
                    v = outputs[:, 1]  # Variance column

                sampled_means.append(mu.cpu().numpy())
                sampled_vars.append(v.cpu().numpy())

        # Stack results
        sampled_means = np.stack(sampled_means, axis=0)  # (samples, N)
        sampled_vars = np.stack(sampled_vars, axis=0)  # (samples, N)

        # Calculate Uncertainty Components
        # 1. Epistemic: Variance of the means (Model Uncertainty)
        epistemic_var = np.var(sampled_means, axis=0)

        # 2. Aleatoric: Average of the variances (Data Uncertainty)
        # Note: If your model outputs LogVar, use np.exp(aleatoric_var) here
        aleatoric_var = np.mean(sampled_vars, axis=0)

        # 3. Total Standard Deviation
        total_std = np.sqrt(epistemic_var + aleatoric_var)

        # Ensure shape matches mean_pred (N, 1)
        if total_std.ndim == 1:
            total_std = total_std.reshape(-1, 1)

        # Return the DETERMINISTIC Mean (for low MSE) and STOCHASTIC Std
        return mean_pred, total_std


def train_nn_mc_dropout(train_dc, valid_dc, test_dc, n_samples=100, alpha=0.05):
    """
    Train heteroscedastic MC-dropout NN and:
      - compute MSE with deterministic predictions (eval mode, no dropout)
      - compute UQ using DeepChem's predict_uncertainty (MC + aleatoric)
    """
    n_tasks    = train_dc.y.shape[1]
    n_features = train_dc.X.shape[1]

    # ----- Build model & DeepChem wrapper -----
    model = MyTorchRegressorMC(n_features, n_tasks)    # your MC + log-var net
    loss  = HeteroscedasticL2Loss()                   # Kendall & Gal-style loss

    dc_model = dc.models.TorchModel(
        model=model,
        loss=loss,
        output_types=['prediction', 'variance', 'loss'],
        batch_size=64,
        learning_rate=1e-3,
        mode='regression'
    )

    # ----- Train -----
    dc_model.fit(train_dc, nb_epoch=100)

    # Instantiate the FIXED wrapper
    mc_model = MCDropoutRegressorRefined(dc_model, n_samples=100)

    # Run Prediction
    mean_test, std_test = mc_model.predict_uncertainty(test_dc)

    # --- CRITICAL FIX: Ensure shapes match for MSE ---
    # Reshape mean_test to (N, 1) if necessary to match test_dc.y
    if mean_test.ndim == 1:
        mean_test = mean_test.reshape(-1, 1)

    # Calculate MSE (Should now be ~0.67)
    test_mse = mse_from_mean_prediction(mean_test, test_dc)

    alpha = 0.05
    z = norm.ppf(1 - alpha / 2.0)

    # Calculate UQ Metrics (Should now have Valid Coverage and Non-Zero Std)
    lower = mean_test - z * std_test
    upper = mean_test + z * std_test

    # Note: Ensure std_test is broadcastable. If shape mismatch, reshape it.
    if std_test.ndim == 1:
        std_test = std_test.reshape(-1, 1)

    uq_metrics = evaluate_uq_metrics_from_interval(
        y_true=test_dc.y,
        mean=mean_test,
        lower=lower,
        upper=upper,
        alpha=alpha,
        test_error=test_mse
    )

    print(f"[NN MC-DROPOUT] Test MSE: {test_mse:.6f}")
    print(f"[NN MC-DROPOUT] UQ Metrics: {uq_metrics}")

    return uq_metrics


class MyTorchRegressor(nn.Module):
    def __init__(self, n_features: int, n_tasks: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_tasks),
        )

    def forward(self, x):
        return self.net(x)


def train_nn_baseline(train_dc, valid_dc, test_dc):
    n_tasks = train_dc.y.shape[1]
    n_features = train_dc.X.shape[1]

    model = MyTorchRegressor(n_features, n_tasks)
    loss = dc.models.losses.L2Loss()

    dc_model = dc.models.TorchModel(
        model=model,
        loss=loss,
        output_types=['prediction'],
        batch_size=64,
        learning_rate=1e-3,
        mode='regression',
    )

    dc_model.fit(train_dc, nb_epoch=30)

    metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
    valid_scores = dc_model.evaluate(valid_dc, [metric])
    test_scores = dc_model.evaluate(test_dc, [metric])

    print("[NN Baseline] Validation MSE:", valid_scores[metric.name])
    print("[NN Baseline] Test MSE:",      test_scores[metric.name])
    mse_test = test_scores[metric.name]

    return {
        "alpha": None,
        "empirical_coverage": None,
        "avg_pred_std": None,
        "nll": None,
        "ce": None,
        "spearman_err_unc": None,
        "MSE": float(mse_test),
    }
