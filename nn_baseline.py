import numpy as np
import torch
from deepchem.models import TorchModel  # Assuming this base class
import torch.nn as nn
import torch.nn.functional as F
import deepchem as dc
from scipy.stats import norm
from data_utils import evaluate_uq_metrics_from_interval, calculate_cutoff_error_data
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


class EvidentialRegressionLoss(Loss):
    """
    Deep Evidential Regression Loss (NIG distribution):
    L = L_NLL + lambda * L_R

    where L_NLL is the NIG Negative Log-Likelihood (marginal likelihood)
    and L_R is the evidential regularizer.
    """

    def __init__(self, reg_coeff=0.01, reg_coeff_u=0, **kwargs):
        self.reg_coeff_r = reg_coeff
        self.reg_coeff_u = reg_coeff_u
        # super(EvidentialRegressionLoss, self).__init__(**kwargs)

    def nig_reg_U_tensor(self, y, gamma, v, alpha, beta):

        # The term nu(alpha - 1) is in the numerator
        numerator_factor = v * (alpha - 1)
        
        # The term beta(nu + 1) is in the denominator
        denominator_factor = beta * (v + 1)
        
        # Calculate the inverse of Total Uncertainty
        inv_total_unc = numerator_factor / denominator_factor
        
        # L_U = (y - gamma)^2 * (Inverse of Total Uncertainty)
        squared_error = (y - gamma) ** 2
        
        loss_u = squared_error * inv_total_unc
        
        return loss_u

    def nig_nll_tensor(self, y, gamma, v, alpha, beta):
        twoBlambda = 2 * beta * (1 + v)

        nll = 0.5 * torch.log(np.pi / v) \
              - alpha * torch.log(twoBlambda) \
              + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda) \
              + torch.lgamma(alpha) \
              - torch.lgamma(alpha + 0.5)

        return nll  # (B, T) tensor

    # Helper function to compute the Evidential Regularizer (Loss R)
    # Based on Equation 9 from the paper [cite: 147]
    def nig_reg_tensor(self, y, gamma, v, alpha):
        error = torch.abs(y - gamma)
        evi = 2 * v + alpha  # Total Evidence (Phi)
        reg = error * evi
        return reg  # (B, T) tensor

    def _create_pytorch_loss(self):

        def loss(output, labels):
            # DeepChem helper: handles shapes
            # NOTE: We assume the model outputs 4 parameters per task (T)
            # output: (B, 4*T)
            # labels: (B, T)
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            D = labels.shape[-1]

            # 1. Chunk output into the four evidential parameters (B, T)
            gamma, v, alpha, beta = output.chunk(4, dim=-1)

            # 2. Compute Loss NLL: Negative Log-Likelihood
            loss_nll_elem = self.nig_nll_tensor(labels, gamma, v, alpha, beta)

            # 3. Compute Loss R: Evidential Regularizer
            loss_reg_elem = self.nig_reg_tensor(labels, gamma, v, alpha)

            loss_u = self.nig_reg_U_tensor(labels, gamma, v, alpha, beta)

            # 4. Total Loss L = L_NLL + lambda * L_R
            loss_elem = loss_nll_elem + self.reg_coeff_r * loss_reg_elem + self.reg_coeff_u * loss_u  # (B, T)

            # mean over tasks → shape (B,) (per-sample)
            loss_per_sample = torch.mean(loss_elem, dim=-1)
            return loss_per_sample

        return loss


class DenseNormalGamma(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DenseNormalGamma, self).__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.dense = nn.Linear(self.in_dim, 4 * self.out_dim)

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        # Define the dimensions for the hidden layers
        HIDDEN_DIM_1 = 128
        HIDDEN_DIM_2 = 64

        self.dense = nn.Sequential(
            # --- Layer 1: in_dim -> HIDDEN_DIM_1 (e.g., 128) ---
            nn.Linear(self.in_dim, HIDDEN_DIM_1),
            # Add BatchNorm *before* the activation for the first hidden layer
            nn.BatchNorm1d(HIDDEN_DIM_1), 
            nn.ReLU(),
            
            # --- Layer 2: HIDDEN_DIM_1 -> HIDDEN_DIM_2 (e.g., 64) ---
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            # Add BatchNorm *before* the activation for the second hidden layer
            nn.BatchNorm1d(HIDDEN_DIM_2), 
            nn.ReLU(),
            
            # --- Output Layer: HIDDEN_DIM_2 -> 4 * out_dim ---
            # No BatchNorm needed here, as the output is directly passed for 
            # parameter splitting and transformation (softplus)
            nn.Linear(HIDDEN_DIM_2, 4 * self.out_dim),
        )

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        eps = 1e-6
        MAX_ALPHA = 100.0
        output = self.dense(x)
        mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)

        v = self.evidence(logv)
        v = torch.clamp(v, min=eps)
        alpha = self.evidence(logalpha) + 1
        alpha = torch.clamp(alpha, min=eps + 1, max=MAX_ALPHA)
        beta = self.evidence(logbeta)
        beta = torch.clamp(beta, min=eps)

        aleatoric = beta / (alpha - 1)
        epistemic = beta / (v * (alpha - 1))

        return mu, torch.cat([mu, v, alpha, beta], dim=-1), aleatoric, epistemic


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
# def train_nn_baseline(train_dc, valid_dc, test_dc):
#     n_tasks = train_dc.y.shape[1]
#     n_features = train_dc.X.shape[1]

#     model = MyTorchRegressor(n_features, n_tasks)
#     loss = dc.models.losses.L2Loss()

#     dc_model = dc.models.TorchModel(
#         model=model,
#         loss=loss,
#         output_types=['prediction'],
#         batch_size=64,
#         learning_rate=1e-3,
#         mode='regression',
#     )

#     dc_model.fit(train_dc, nb_epoch=30)

#     metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
#     valid_score = dc_model.evaluate(valid_dc, [metric])
#     test_score  = dc_model.evaluate(test_dc,  [metric])

#     print("[NN Baseline] Validation MSE:", valid_score[metric.name])
#     print("[NN Baseline] Test MSE:",        test_score[metric.name])

#     return {"MSE": test_score[metric.name]}


# ------------------------------
# Deep Ensemble
# ------------------------------
def train_nn_deep_ensemble(train_dc, valid_dc, test_dc, M=5, run_id=0):
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
    cutoff_error_df = calculate_cutoff_error_data(mean_test, upper_test-lower_test, test_dc.y)
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

    return uq_metrics, cutoff_error_df


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


def train_nn_mc_dropout(train_dc, valid_dc, test_dc, n_samples=100, alpha=0.05, run_id=0):
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

    cutoff_error_df = calculate_cutoff_error_data(mean_test, std_test, test_dc.y)
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

    return uq_metrics, cutoff_error_df


class GradientClippingCallback:
    """A DeepChem callback to perform gradient clipping after backpropagation."""

    def __init__(self, max_norm: float = 5.0):
        """
        Initializes the callback with the maximum gradient norm allowed.

        Args:
            max_norm: The maximum norm value for gradient clipping.
        """
        self.max_norm = max_norm

    # The __call__ method is executed after every optimization step.
    def __call__(self, model_wrapper: TorchModel, step: int):
        """
        Clips the gradients of the model's parameters.

        Args:
            model_wrapper: The DeepChem TorchModel instance.
            step: The current training step number.
        """
        # Ensure the model is available (it's stored in model_wrapper.model)
        if model_wrapper.model is not None:
            # Clip the gradients of all model parameters
            torch.nn.utils.clip_grad_norm_(
                model_wrapper.model.parameters(),
                self.max_norm
            )
        # Note: No need for model_wrapper.model.train() or model_wrapper.model.eval()
        # as clipping happens during training.


# Example of how to use it when instantiating the model:
# clip_callback = GradientClippingCallback(max_norm=5.0)
# dc_model.fit(train_dc, nb_epoch=100, callbacks=[clip_callback])


def train_evd_baseline(train_dc, valid_dc, test_dc, reg_coeff=1, alpha=0.05, run_id=0):
    """
    Train Deep Evidential Regression (DER) NN and:
      - compute MSE with the analytical mean prediction (gamma)
      - compute UQ using the analytical total variance (aleatoric + epistemic)

    Args:
        train_dc, valid_dc, test_dc: DeepChem datasets.
        reg_coeff (float): The regularization coefficient lambda (λ) for the loss.
        alpha (float): The significance level for confidence interval calculation.
    """
    n_tasks = train_dc.y.shape[1]
    n_features = train_dc.X.shape[1]

    # --- 1. Build Model & DeepChem wrapper (Using the structure from the second code block) ---
    # The model outputs 4 parameters (gamma, v, alpha, beta) for each task.
    # We assume 'DenseNormalGamma' is the model structure (mu, v, alpha, beta)
    model = DenseNormalGamma(n_features, n_tasks)

    # The custom evidential loss func(y_true, evidential_output)
    loss = EvidentialRegressionLoss(coeff=reg_coeff)

    gradientClip = GradientClippingCallback()

    dc_model = dc.models.TorchModel(
        model=model,
        loss=loss,
        # The output types match the return structure of DenseNormalGamma:
        output_types=['prediction', 'loss', 'var1', 'var2'],
        batch_size=128,
        learning_rate=1e-4,
        # wandb=True,  # Set to True
        # model_dir='deep-evidential-regression-run-{}'.format(run_id),
        log_frequency=40,
        mode='regression'
    )

    # --- 2. Train ---
    print(f"Training Deep Evidential Regression with lambda (reg_coeff) = {reg_coeff}")
    dc_model.fit(train_dc, nb_epoch=300, callbacks=[gradientClip])
    device = next(dc_model.model.parameters()).device

    # Convert numpy data to PyTorch tensors (DeepChem .X are typically numpy arrays)
    valid_X_tensor = torch.from_numpy(valid_dc.X).float().to(device)
    test_X_tensor = torch.from_numpy(test_dc.X).float().to(device)

    # Get predictions for the validation set
    with torch.no_grad():
        mu_valid, params_valid, aleatoric_valid, epistemic_valid = dc_model.model(valid_X_tensor)

    # Get predictions for the test set
    with torch.no_grad():
        mu_test, params_test, aleatoric_test, epistemic_test = dc_model.model(test_X_tensor)


    # The total predictive variance Var[y] is the sum of aleatoric and epistemic variance.
    # Var[y] = E[sigma^2] + Var[mu]
    total_var_test = aleatoric_test.cpu().numpy() + epistemic_test.cpu().numpy()
    std_test = np.sqrt(total_var_test)

    # --- 4. Calculate MSE (using the deterministic mean prediction, gamma) ---
    # Ensure prediction shape matches for MSE calculation
    mu_test = mu_test.cpu().numpy()
    if mu_test.ndim == 1:
        mu_test = mu_test.reshape(-1, 1)

    cutoff_error_df = calculate_cutoff_error_data(mu_test, total_var_test, test_dc.y)

    test_mse = mse_from_mean_prediction(mu_test, test_dc)

    # --- 5. Calculate UQ Metrics ---
    z = norm.ppf(1 - alpha / 2.0)

    # Confidence interval: mean ± z * standard_deviation
    lower = mu_test - z * std_test
    upper = mu_test + z * std_test

    # Ensure std_test is broadcastable.
    if std_test.ndim == 1:
        std_test = std_test.reshape(-1, 1)

    uq_metrics = evaluate_uq_metrics_from_interval(
        y_true=test_dc.y,
        mean=mu_test,
        lower=lower,
        upper=upper,
        alpha=alpha,
        test_error=test_mse
    )

    print(f"\n[EVIDENTIAL REGRESSION] Test MSE: {test_mse:.6f}")
    print(f"[EVIDENTIAL REGRESSION] UQ Metrics: {uq_metrics}")

    return uq_metrics, cutoff_error_df


def train_nn_baseline(train_dc, valid_dc, test_dc, run_id=0, use_weights=False, mode="regression"):
    n_tasks = train_dc.y.shape[1]
    n_features = train_dc.X.shape[1]

    model = MyTorchRegressor(n_features, n_tasks)
    if mode == "regression":
        loss = dc.models.losses.L2Loss()

        dc_model = dc.models.TorchModel(
            model=model,
            loss=loss,
            output_types=['prediction'],
            batch_size=64,
            learning_rate=1e-3,
            mode='regression',
        )
    else:
        loss = dc.models.losses.BinaryCrossEntropy()

        dc_model = dc.models.TorchModel(
            model=model,
            loss=loss,
            output_types=['prediction'],
            batch_size=64,
            learning_rate=1e-3,
            mode='regression',
        )

    if mode == "regression":
        dc_model.fit(train_dc, nb_epoch=80)

        metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
        valid_scores = dc_model.evaluate(valid_dc, [metric], use_sample_weights=use_weights)
        test_scores = dc_model.evaluate(test_dc, [metric], use_sample_weights=use_weights)

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
    else:
        dc_model.fit(train_dc, nb_epoch=80)

        metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        valid_scores = dc_model.evaluate(valid_dc, [metric], use_sample_weights=use_weights)
        test_scores = dc_model.evaluate(test_dc, [metric], use_sample_weights=use_weights)

        print("[NN Baseline] Validation AUC:", valid_scores[metric.name])
        print("[NN Baseline] Test AUC:", test_scores[metric.name])
        mse_test = test_scores[metric.name]

        return {
            "AUC": float(mse_test),
            "NLL": None,
            "Brier": None,
            "ECE": None,
            "Avg_Entropy": None,
            "spearman_err_unc": None,
        }
