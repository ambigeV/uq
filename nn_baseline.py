import numpy as np
import torch
from deepchem.models import TorchModel  # Assuming this base class
import torch.nn as nn
import torch.nn.functional as F
import deepchem as dc
from scipy.stats import norm
import math
from typing import Optional
from data_utils import evaluate_uq_metrics_from_interval, evaluate_uq_metrics_classification, \
    calculate_cutoff_error_data, calculate_cutoff_classification_data, roc_auc_score, auc_from_probs
from deepchem.models.losses import Loss, _make_pytorch_shapes_consistent
from model_utils import save_neural_network_model
# Import graph encoder components from nn.py
from nn import (
    BatchMolGraph,
    MolGraph,
    graphdata_to_batchmolgraph,
    BondMessagePassing,
    AtomMessagePassing,
    MeanAggregation,
    SumAggregation,
    DMPNNEncoder,
    create_dmpnn_encoder,
    UnifiedTorchModel
)


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


class MyTorchClassifier(nn.Module):
    """Standard feed-forward NN for binary classification (sigmoid output)."""
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
        logits = self.net(x)          # (B, n_tasks)
        probs = torch.sigmoid(logits) # (B, n_tasks) in [0,1]
        return probs


class HeteroscedasticClassificationLoss(Loss):
    """
    Heteroscedastic Loss for Multi-Task Binary Classification.
    
    L = Sum_over_tasks [ - log( (1/T) * sum_samples ( p_sample ) ) ]
    
    where:
      - p_sample = Sigmoid(logit_sample)      if y=1
      - p_sample = 1 - Sigmoid(logit_sample)  if y=0
    """

    def __init__(
        self,
        n_samples: int = 20,
        clamp_log_var=(-10.0, 10.0),
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_samples = int(n_samples)
        self.clamp_log_var = clamp_log_var
        self.reduction = reduction
        self.eps = float(eps)

    def _create_pytorch_loss(self):
        import torch.nn.functional as F

        def loss(output, labels):
            # output: (B, 2*T) -> [means | log_vars]
            # labels: (B, T)
            output, labels = dc.models.losses._make_pytorch_shapes_consistent(output, labels)

            n_tasks = output.shape[-1] // 2
            
            # Split Mean and Variance
            mu = output[..., :n_tasks]       # (B, n_tasks)
            log_var = output[..., n_tasks:]  # (B, n_tasks)

            if self.clamp_log_var is not None:
                lo, hi = self.clamp_log_var
                log_var = log_var.clamp(min=lo, max=hi)

            std = torch.exp(0.5 * log_var)   # (B, n_tasks)

            # --- Vectorized MC Sampling ---
            # Sample T times for every task in the batch
            # Shape: (N_samples, B, n_tasks)
            T = self.n_samples
            eps = torch.randn((T,) + mu.shape, device=mu.device, dtype=mu.dtype)
            
            # Sampled Logits
            logits_s = mu.unsqueeze(0) + std.unsqueeze(0) * eps  # (T, B, n_tasks)

            # --- Calculate Log-Likelihood (Binary) ---
            # We want log( 1/T * sum( Prob(y|x) ) )
            
            # Expand labels to match samples: (T, B, n_tasks)
            target_expanded = labels.unsqueeze(0).expand(T, -1, -1)

            # Numerical Stability Trick:
            # We use BCEWithLogitsLoss to get log_prob per sample, 
            # then logsumexp to average them.
            # BCE gives -log(p), so we take negative BCE.
            
            # This computes log(sigmoid(x)) if y=1, and log(1-sigmoid(x)) if y=0
            # reduction='none' returns (T, B, n_tasks)
            log_prob_per_sample = -F.binary_cross_entropy_with_logits(
                logits_s, target_expanded, reduction='none'
            )

            # Log-Sum-Exp trick to average probabilities in log-space:
            # log(1/T * sum(exp(log_p))) = logsumexp(log_p) - log(T)
            log_expected_prob = torch.logsumexp(log_prob_per_sample, dim=0) - math.log(T) # (B, n_tasks)

            # Loss is negative log likelihood
            loss_per_task = -log_expected_prob # (B, n_tasks)

            # --- Final Reduction ---
            # We return (B, n_tasks) so DeepChem can apply weights (B, n_tasks) element-wise.
            return loss_per_task

        return loss


class HeteroscedasticL2Loss(Loss):
    def _create_pytorch_loss(self):
        import torch

        def loss(output, labels):
            # output: (B, 2*T) -> [means | log_vars]
            # labels: (B, T)
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            
            D = labels.shape[-1]
            y_hat   = output[..., :D]   # (B, T)
            log_var = output[..., D:]   # (B, T)

            precision = torch.exp(-log_var)
            diff2     = (labels - y_hat) ** 2

            # (B, T)
            loss_elem = 0.5 * precision * diff2 + 0.5 * log_var 

            # CRITICAL FIX: Do NOT average here. 
            # Return (B, T) so it matches the shape of the weights (B, T).
            return loss_elem
        return loss


class EvidentialClassificationLoss(Loss):
    """
    Deep Evidential Classification Loss wrapper for DeepChem.
    
    Supports two operational modes based on input shapes:
    1. Single-Task Multi-Class: Uses Dirichlet distribution (competing classes).
    2. Multi-Task Binary: Uses Beta distribution (independent tasks).
    """
    def __init__(self, mode='mse', num_classes=2, annealing_step=10, epoch_tracker=None):
        self.mode = mode
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        # Use mutable list to track epoch across calls if not provided
        self.epoch_tracker = epoch_tracker if epoch_tracker is not None else [0]
        super(EvidentialClassificationLoss, self).__init__()

    def _create_pytorch_loss(self):
        
        # --- Helpers ---
        def get_annealing_coef(epoch, annealing_step):
            return torch.min(
                torch.tensor(1.0, dtype=torch.float32),
                torch.tensor(epoch / annealing_step, dtype=torch.float32),
            )

        # [Logic A] Dirichlet KL (Single-Task, Competing Classes)
        def kl_divergence_dirichlet(alpha, num_classes):
            ones = torch.ones([1, num_classes], dtype=torch.float32, device=alpha.device)
            sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
            first_term = (
                torch.lgamma(sum_alpha)
                - torch.lgamma(alpha).sum(dim=1, keepdim=True)
                + torch.lgamma(ones).sum(dim=1, keepdim=True)
                - torch.lgamma(ones.sum(dim=1, keepdim=True))
            )
            second_term = (
                (alpha - ones)
                .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
                .sum(dim=1, keepdim=True)
            )
            return first_term + second_term

        # [Logic B] Beta KL (Multi-Task, Independent Tasks)
        def kl_divergence_beta(alpha, beta):
            sum_params = alpha + beta
            term1 = torch.lgamma(sum_params) - torch.lgamma(alpha) - torch.lgamma(beta)
            term2 = (alpha - 1) * (torch.digamma(alpha) - torch.digamma(sum_params))
            term3 = (beta - 1) * (torch.digamma(beta) - torch.digamma(sum_params))
            return term1 + term2 + term3

        def loss(output, labels):
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            
            # --- Auto-Detection of Mode ---
            n_label_cols = labels.shape[-1]
            
            # === MODE 1: MULTI-TASK (Beta Distribution) ===
            if n_label_cols > 1:
                n_tasks = n_label_cols
                # Reshape: (Batch, Tasks, 2) -> alpha (Fail), beta (Success)
                params = output.view(-1, n_tasks, 2)
                
                alpha_param = params[..., 0] 
                beta_param  = params[..., 1]
                
                S = alpha_param + beta_param
                p = beta_param / S 
                
                # --- Base Loss ---
                if self.mode == 'mse':
                    sq_err = (labels - p) ** 2
                    var_term = (p * (1 - p)) / (S + 1)
                    base_loss = sq_err + var_term
                elif self.mode == 'log':
                    base_loss = -(labels * (torch.log(beta_param) - torch.log(S)) + 
                                  (1 - labels) * (torch.log(alpha_param) - torch.log(S)))
                else:
                    # Digamma for Binary
                    base_loss = -(labels * (torch.digamma(beta_param) - torch.digamma(S)) + 
                                  (1 - labels) * (torch.digamma(alpha_param) - torch.digamma(S)))
            
                
                # Formula: labels * (Target if y=1) + (1-labels) * (Target if y=0)
                alpha_reg = labels * alpha_param + (1 - labels) * 1
                beta_reg  = labels * 1 + (1 - labels) * beta_param
                
                kl = kl_divergence_beta(alpha_reg, beta_reg)
                
                current_epoch = self.epoch_tracker[0]
                annealing = get_annealing_coef(current_epoch, self.annealing_step)
                
                total_loss = base_loss + annealing * kl

                return total_loss

            # === MODE 2: SINGLE-TASK (Dirichlet Distribution) ===
            else:
                alpha = output 
                
                # One-Hot Encode
                if labels.shape[-1] == 1: 
                    y_onehot = F.one_hot(labels.long().squeeze(-1), num_classes=self.num_classes).float()
                else:
                    y_onehot = labels.float()

                S = torch.sum(alpha, dim=1, keepdim=True)
                
                if self.mode == 'mse':
                    pred_prob = alpha / S
                    loss_err = torch.sum((y_onehot - pred_prob) ** 2, dim=1, keepdim=True)
                    loss_var = torch.sum(
                        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
                    )
                    base_loss = loss_err + loss_var

                elif self.mode == 'log':
                    base_loss = torch.sum(y_onehot * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)

                elif self.mode == 'digamma':
                    base_loss = torch.sum(y_onehot * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
                
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")

                kl_alpha = (alpha - 1) * (1 - y_onehot) + 1
                current_epoch = self.epoch_tracker[0]
                annealing = get_annealing_coef(current_epoch, self.annealing_step)
                
                kl = annealing * kl_divergence_dirichlet(kl_alpha, self.num_classes)
                
                return torch.mean(base_loss + kl)

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
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            
            # 1. Chunk output (B, 4*T) -> Four (B, T) tensors
            gamma, v, alpha, beta = output.chunk(4, dim=-1)

            # 2. Compute Element-wise Losses (B, T)
            loss_nll = self.nig_nll_tensor(labels, gamma, v, alpha, beta)
            loss_reg = self.nig_reg_tensor(labels, gamma, v, alpha)
            loss_u   = self.nig_reg_U_tensor(labels, gamma, v, alpha, beta)

            # 3. Combine
            loss_elem = loss_nll + self.reg_coeff_r * loss_reg + self.reg_coeff_u * loss_u

            # CRITICAL FIX: Return (B, T), do not mean() here.
            return loss_elem

        return loss

    # def _create_pytorch_loss(self):

    #     def loss(output, labels):
    #         # DeepChem helper: handles shapes
    #         # NOTE: We assume the model outputs 4 parameters per task (T)
    #         # output: (B, 4*T)
    #         # labels: (B, T)
    #         output, labels = _make_pytorch_shapes_consistent(output, labels)
    #         D = labels.shape[-1]

    #         # 1. Chunk output into the four evidential parameters (B, T)
    #         gamma, v, alpha, beta = output.chunk(4, dim=-1)

    #         # 2. Compute Loss NLL: Negative Log-Likelihood
    #         loss_nll_elem = self.nig_nll_tensor(labels, gamma, v, alpha, beta)

    #         # 3. Compute Loss R: Evidential Regularizer
    #         loss_reg_elem = self.nig_reg_tensor(labels, gamma, v, alpha)

    #         loss_u = self.nig_reg_U_tensor(labels, gamma, v, alpha, beta)

    #         # 4. Total Loss L = L_NLL + lambda * L_R
    #         loss_elem = loss_nll_elem + self.reg_coeff_r * loss_reg_elem + self.reg_coeff_u * loss_u  # (B, T)

    #         # mean over tasks → shape (B,) (per-sample)
    #         loss_per_sample = torch.mean(loss_elem, dim=-1)
    #         return loss_per_sample

    #     return loss


class DenseDirichlet(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        Args:
            in_dim: Input feature dimension.
            out_dim: Number of classes (usually 2 for binary classification tasks).
        """
        super(DenseDirichlet, self).__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        # Define the dimensions for the hidden layers (Matching your Regression Style)
        HIDDEN_DIM_1 = 128
        HIDDEN_DIM_2 = 64

        self.dense = nn.Sequential(
            # --- Layer 1 ---
            nn.Linear(self.in_dim, HIDDEN_DIM_1),
            nn.BatchNorm1d(HIDDEN_DIM_1), 
            nn.ReLU(),
            
            # --- Layer 2 ---
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.BatchNorm1d(HIDDEN_DIM_2), 
            nn.ReLU(),
            
            # --- Output Layer ---
            # Output dimension is just 'out_dim' (number of classes), not 4x
            nn.Linear(HIDDEN_DIM_2, self.out_dim),
        )

    def forward(self, x):
        logits = self.dense(x)
        
        # Evidence is typically exp(logits) or softplus(logits)
        evidence = torch.exp(logits) 
        alpha = evidence + 1

        # --- CRITICAL FIX START ---
        
        # 1. Reshape to separate tasks from classes
        # Current shape: (Batch, 2 * n_tasks)
        # New shape:     (Batch, n_tasks, 2)
        # where dim 2 is [alpha_class0, alpha_class1]
        n_tasks = self.out_dim // 2  # Assuming out_dim is total outputs (2*T)
        alpha_reshaped = alpha.view(-1, n_tasks, 2)
        
        # 2. Calculate S per task (Sum over the LAST dimension only)
        # S shape: (Batch, n_tasks, 1)
        S = torch.sum(alpha_reshaped, dim=2, keepdim=True)
        
        # 3. Calculate K (Number of classes per task)
        # For binary tasks, K = 2
        K = 2 

        # 4. Prediction (Expected Probability): alpha_k / S_k
        # We calculate this for the "positive class" (index 1) usually, 
        # or return both. Let's return full shape (Batch, n_tasks, 2) to be safe.
        prob = alpha_reshaped / S

        # 5. Epistemic Uncertainty (Vacuity): K / S
        # Shape: (Batch, n_tasks, 1)
        epistemic = K / S

        # 6. Aleatoric Uncertainty: Entropy of the expected probability
        # Sum over the 2 classes (last dim)
        prob_safe = prob + 1e-8
        aleatoric = -torch.sum(prob * torch.log(prob_safe), dim=2, keepdim=True)

        # --- CRITICAL FIX END ---

        # Reshape back to flat vectors if your training loop expects flat tensors
        # or keep them structured. DeepChem usually expects (Batch, N) outputs.
        # Let's flatten everything back to (Batch, N) to match typical API expectations.
        
        prob = prob.view(x.shape[0], -1)       # (Batch, 2*T)
        alpha = alpha                          # (Batch, 2*T) - Already flat
        aleatoric = aleatoric.view(x.shape[0], -1) # (Batch, T)
        epistemic = epistemic.view(x.shape[0], -1) # (Batch, T)

        return prob, alpha, aleatoric, epistemic


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
# 2. Unified Model Architecture (ChemProp-style)
# ============================================================

class UnifiedModel(nn.Module):
    """
    Unified model architecture following ChemProp style with component-wise building.
    Supports baseline, MC-dropout, and evidential models for both regression and classification.
    """
    
    def __init__(self, classification: bool = False, model_type: str = "baseline", n_tasks: int = 1, **kwargs):
        """
        Args:
            classification: Whether this is a classification model
            model_type: "baseline", "mc_dropout", or "evidential"
            n_tasks: Number of output tasks
            **kwargs: Additional configuration (dropout_rate, etc.)
        """
        super(UnifiedModel, self).__init__()
        self.classification = classification
        self.model_type = model_type
        self.n_tasks = n_tasks
        self.encoder = None
        self.ffn = None
        self.config = kwargs
        self.config['n_tasks'] = n_tasks  # Store n_tasks in config for forward method
        
        # For evidential models
        if model_type == "evidential":
            self.confidence = True
            self.conf_type = "evidence"
        else:
            self.confidence = False
            self.conf_type = None
    
    def create_encoder(self, n_features: int, encoder_type: str = "identity", **kwargs):
        """
        Creates the encoder (identity for vector input, DMPNN for graph input).
        Following ChemProp's create_encoder pattern.
        
        Args:
            n_features: Input feature dimension (for identity) or atom feature dim (for graph)
            encoder_type: "identity" (default) or "dmpnn"
            **kwargs: Additional encoder config
                - For dmpnn: d_v, d_e, d_h, depth, aggregation, batch_norm, etc.
        """
        self.encoder_type = encoder_type
        
        if encoder_type == "identity":
            # Current behavior: pass-through for vector features
            self.encoder_dim = n_features
            self.encoder = nn.Identity()
        
        elif encoder_type == "dmpnn":
            # DMPNN encoder for graph input
            d_v = kwargs.get('d_v', n_features)  # Atom features (default: n_features)
            d_e = kwargs.get('d_e', 14)          # Bond features (default from chemprop)
            d_h = kwargs.get('encoder_hidden_dim', 300)  # Hidden/output dimension
            depth = kwargs.get('encoder_depth', 3)
            aggregation = kwargs.get('aggregation', 'mean')
            batch_norm = kwargs.get('encoder_batch_norm', False)
            mp_type = kwargs.get('message_passing_type', 'bond')
            dropout = kwargs.get('encoder_dropout', 0.0)
            activation = kwargs.get('encoder_activation', 'relu')
            bias = kwargs.get('encoder_bias', True)
            
            # Create DMPNN encoder
            self.encoder = create_dmpnn_encoder(
                d_v=d_v,
                d_e=d_e,
                d_h=d_h,
                depth=depth,
                aggregation=aggregation,
                batch_norm=batch_norm,
                message_passing_type=mp_type,
                dropout=dropout,
                activation=activation,
                bias=bias,
            )
            self.encoder_dim = d_h  # Output dimension of graph encoder
        
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. "
                            f"Supported: 'identity', 'dmpnn'")
    
    def create_ffn(self, n_features: int, n_tasks: int, **kwargs):
        """
        Creates the feed-forward network following ChemProp's create_ffn pattern.
        
        Args:
            n_features: Input feature dimension (DEPRECATED - use self.encoder_dim)
            n_tasks: Number of output tasks
            **kwargs: Additional FFN config (dropout_rate, etc.)
        """
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        # FIX: Use encoder_dim instead of n_features to match encoder output
        first_linear_dim = self.encoder_dim
        
        # Determine output size based on model type
        output_size = n_tasks
        if self.model_type == "mc_dropout":
            # Output mean and log_var: 2 * n_tasks
            output_size = 2 * n_tasks
        elif self.model_type == "evidential":
            if self.classification:
                # Dirichlet: 2 * n_tasks (alpha for each class per task)
                output_size = 2 * n_tasks
            else:
                # Normal-Inverse-Gamma: 4 * n_tasks (gamma, v, alpha, beta)
                output_size = 4 * n_tasks
        
        # Build FFN layers
        if self.model_type == "baseline":
            # Standard baseline: n_features -> 256 -> 128 -> n_tasks
            ffn_layers = [
                nn.Linear(first_linear_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_size)
            ]
        elif self.model_type == "mc_dropout":
            # MC-Dropout: feature net + output head
            self.feature_net = nn.Sequential(
                nn.Linear(first_linear_dim, 128),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            )
            self.out_head = nn.Linear(64, output_size)
            self.n_tasks = n_tasks
            return  # Early return for MC-dropout
        elif self.model_type == "evidential":
            # Evidential: n_features -> 128 -> 64 -> output_size
            ffn_layers = [
                nn.Linear(first_linear_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, output_size)
            ]
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        self.ffn = nn.Sequential(*ffn_layers)
    
    def forward(self, x, V_d: Optional[torch.Tensor] = None, X_d: Optional[torch.Tensor] = None):
        """
        Forward pass through encoder and FFN.
        Handles both vector input (identity encoder) and graph input (DMPNN encoder).
        
        Args:
            x: Input data
                - For identity encoder: (batch_size, n_features) tensor
                - For DMPNN encoder: BatchMolGraph object or list of GraphData objects
            V_d: Optional vertex descriptors (for graph encoder)
            X_d: Optional additional graph-level descriptors (for future pre-trained features)
        
        Returns:
            Model output (varies by model_type)
        """
        # Encode: handles both vector and graph input
        if self.encoder_type == "dmpnn":
            # Graph encoder expects BatchMolGraph
            # DeepChem may pass list of GraphData objects, so convert if needed
            if not isinstance(x, BatchMolGraph):
                # Try to convert from GraphData list
                try:
                    from nn import graphdata_to_batchmolgraph
                    if isinstance(x, (list, tuple)) and len(x) > 0:
                        # Check if it's a list of GraphData objects
                        if hasattr(x[0], 'node_features') or hasattr(x[0], 'num_node_features'):
                            x = graphdata_to_batchmolgraph(x)
                        else:
                            raise TypeError(f"Expected BatchMolGraph or list of GraphData for encoder_type='dmpnn', got {type(x)}")
                    else:
                        raise TypeError(f"Expected BatchMolGraph or list of GraphData for encoder_type='dmpnn', got {type(x)}")
                except Exception as e:
                    raise TypeError(f"Could not convert input to BatchMolGraph for encoder_type='dmpnn'. Got {type(x)}. Error: {e}")
            encoded = self.encoder(x, V_d, X_d)
        else:
            # Identity encoder expects vector input (B, n_features)
            encoded = self.encoder(x)
        
        # MC-Dropout has special structure
        if self.model_type == "mc_dropout":
            h = self.feature_net(encoded)
            raw = self.out_head(h)
            
            # Split into mean and log_var
            T = self.n_tasks
            mean = raw[..., :T]
            log_var = raw[..., T:]
            var = torch.exp(log_var)
            packed = raw

            # For classification, return packed (B, 2*T) directly for loss function
            # For regression, return tuple for output_types=['prediction', 'variance', 'loss']
            if self.classification:
                return packed
            else:
                return mean, var, packed
        
        # Standard forward through FFN
        output = self.ffn(encoded)
        
        # Evidential models need post-processing
        if self.model_type == "evidential":
            if self.classification:
                # Dirichlet: convert to alpha parameters
                evidence = torch.exp(output)
                alpha = evidence + 1
                # n_tasks is stored in self.n_tasks
                alpha_reshaped = alpha.view(-1, self.n_tasks, 2)
                S = torch.sum(alpha_reshaped, dim=2, keepdim=True)
                prob = alpha_reshaped / S
                epistemic = 2.0 / S
                prob_safe = prob + 1e-8
                aleatoric = -torch.sum(prob * torch.log(prob_safe), dim=2, keepdim=True)
                
                prob = prob.view(x.shape[0], -1)
                alpha = alpha
                aleatoric = aleatoric.view(x.shape[0], -1)
                epistemic = epistemic.view(x.shape[0], -1)
                
                return prob, alpha, aleatoric, epistemic
            else:
                # Normal-Inverse-Gamma: split and process
                eps = 1e-6
                MAX_ALPHA = 100.0
                mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
                
                v = F.softplus(logv)
                v = torch.clamp(v, min=eps)
                alpha = F.softplus(logalpha) + 1
                alpha = torch.clamp(alpha, min=eps + 1, max=MAX_ALPHA)
                beta = F.softplus(logbeta)
                beta = torch.clamp(beta, min=eps)
                
                aleatoric = beta / (alpha - 1)
                epistemic = beta / (v * (alpha - 1))
                
                return mu, torch.cat([mu, v, alpha, beta], dim=-1), aleatoric, epistemic
        
        # Baseline: return probs for classification (sigmoid applied) to match original MyTorchClassifier
        # SigmoidCrossEntropy expects probs as input
        if self.classification and self.model_type == "baseline":
            logits = output
            probs = torch.sigmoid(logits)
            return probs
        
        return output


def build_model(
    model_type: str,
    n_features: int,
    n_tasks: int,
    mode: str = "regression",
    encoder_type: str = "identity",
    **kwargs
) -> nn.Module:
    """
    Builds a unified model following ChemProp's build_model pattern.
    
    Args:
        model_type: "baseline", "mc_dropout", or "evidential"
        n_features: Input feature dimension (for identity) or atom feature dim (for graph)
        n_tasks: Number of output tasks
        mode: "regression" or "classification"
        encoder_type: "identity" (default) or "dmpnn"
        **kwargs: Additional config (dropout_rate, encoder_hidden_dim, etc.)
    
    Returns:
        A UnifiedModel instance with encoder and FFN initialized
    """
    classification = (mode == "classification")
    
    # Create model
    model = UnifiedModel(classification=classification, model_type=model_type, n_tasks=n_tasks, **kwargs)
    
    # Build components (ChemProp style)
    model.create_encoder(n_features, encoder_type=encoder_type, **kwargs)
    model.create_ffn(n_features, n_tasks, **kwargs)  # n_features kept for backward compat, but FFN uses encoder_dim
    
    # NOTE: Do NOT override PyTorch's default initialization here.
    # The original MC-Dropout head relied on default init, and reinitializing
    # with a different scheme (e.g., Xavier) degraded performance, especially
    # for QM7. If you want custom init, add it carefully and retune hyperparameters.
    
    return model


# ============================================================
# 3. Helper Evaluation
# ============================================================

import numpy as np

def mse_from_mean_prediction(mean, dataset, use_weights=False):
    """
    Calculates MSE.
    - If Single-Task: Returns a scalar (float).
    - If Multi-Task: Returns a list of scalars (one per task).
    """
    # 1. Standardize Shapes to (N, n_tasks)
    # This prevents the "hidden 1D array" issues
    y_true = dataset.y
    if y_true.ndim == 1: 
        y_true = y_true.reshape(-1, 1)
    
    if mean.ndim == 1: 
        mean = mean.reshape(-1, 1)

    n_tasks = y_true.shape[1]

    # 2. Prepare Weights (N, n_tasks)
    if use_weights and hasattr(dataset, 'w') and dataset.w is not None:
        w = dataset.w
        if w.ndim == 1: 
            w = w.reshape(-1, 1)
        
        # Safety: Check shape match
        if w.shape != y_true.shape:
            print(f"Warning: Weights shape {w.shape} != Y shape {y_true.shape}. Using unweighted.")
            w = np.ones_like(y_true)
    else:
        w = np.ones_like(y_true)

    # 3. Calculate MSE Per Task
    mse_list = []
    for t in range(n_tasks):
        y_t = y_true[:, t]
        pred_t = mean[:, t]
        w_t = w[:, t]

        sq_err = (y_t - pred_t) ** 2
        
        # Weighted Mean for this specific task
        w_sum = np.sum(w_t)
        if w_sum > 0:
            task_mse = np.average(sq_err, weights=w_t)
        else:
            task_mse = np.mean(sq_err) # Fallback if weights are all zero
            
        mse_list.append(float(task_mse))

    # 4. Return Logic
    if n_tasks == 1:
        return mse_list[0] # Return Scalar
    else:
        return mse_list    # Return List [MSE_task1, MSE_task2, ...]


# ============================================================
# 3. Deep Ensemble Wrapper
# ============================================================

class DeepEnsembleRegressor:
    """
    Wrapper over M independently trained DeepChem TorchModels.
    Compatible with both Single-Task and Multi-Task Regression.
    """
    def __init__(self, models):
        self.models = models

    def predict(self, dataset):
        preds = []
        for m in self.models:
            # [FIX] Do NOT slice [:, 0]. Keep all tasks.
            # Shape per model: (N_Samples, N_Tasks)
            out = m.predict(dataset)
            
            # DeepChem sometimes returns (N_Samples,) for single-task.
            # We enforce 2D shape (N_Samples, N_Tasks) to be safe.
            if out.ndim == 1:
                out = out.reshape(-1, 1)
                
            preds.append(out)
            
        # Stack along new axis 0 (Models)
        # Final Shape: (N_Models, N_Samples, N_Tasks)
        return np.stack(preds, axis=0)        

    def predict_interval(self, dataset, alpha=0.05):
        # Y Shape: (N_Models, N_Samples, N_Tasks)
        Y = self.predict(dataset)             
        
        # Mean/Std over the 'Models' dimension (axis 0)
        # Result Shape: (N_Samples, N_Tasks)
        mean = Y.mean(axis=0)
        std  = Y.std(axis=0) + 1e-8

        # Z-score for confidence interval
        z = norm.ppf(1 - alpha/2)
        
        # Broadcasting handles the shapes automatically
        lower = mean - z * std
        upper = mean + z * std

        return mean, lower, upper


class DeepEnsembleClassifier:
    """
    Ensemble wrapper for Multitask Binary Classification.
    Assumes base models output RAW LOGITS of shape (N_Samples, N_Tasks).
    """
    def __init__(self, models):
        self.models = models

    def predict_raw(self, dataset):
        """
        Returns stacked raw LOGITS from base models.
        Shape: (M_Models, N_Samples, N_Tasks)
        """
        preds = []
        for m in self.models:
            # Assumes m.predict returns (N, T) logits directly
            out = m.predict(dataset)
            preds.append(out)
        return np.stack(preds, axis=0)

    def predict_proba(self, dataset):
        """
        Applies Sigmoid to logits to get Probabilities.
        Returns:
          p_mean:    (N, T) -> Mean probability (Ensemble Prediction)
          p_members: (M, N, T) -> Probabilities per model
        """
        logits = self.predict_raw(dataset) # (M, N, T)
        
        # Stability clipping to prevent overflow in exp
        logits = np.clip(logits, -50, 50)
        
        # Apply Sigmoid (Logic is correct: Logits -> Probs)
        p_members = 1.0 / (1.0 + np.exp(-logits))
        
        # Average the PROBABILITIES, not the logits
        p_mean = p_members.mean(axis=0) # (N, T)
        return p_mean, p_members

    def predict_uncertainty(self, dataset, eps=1e-10):
        """
        Decomposes uncertainty for Multitask Binary Classification.
        All outputs preserve shape (N_Samples, N_Tasks).
        """
        p_mean, p_members = self.predict_proba(dataset) # (N, T), (M, N, T)

        # 1. Total Uncertainty (Entropy of the Mean Probability)
        # H[p_mean]
        # Shape: (N, T)
        H_total = -(p_mean * np.log(p_mean + eps) + (1 - p_mean) * np.log(1 - p_mean + eps))

        # 2. Aleatoric Uncertainty (Average Entropy of Members)
        # E[H(p_member)]
        # Shape: (M, N, T) -> mean -> (N, T)
        H_member = -(p_members * np.log(p_members + eps) + (1 - p_members) * np.log(1 - p_members + eps))
        H_aleatoric = H_member.mean(axis=0)

        # 3. Epistemic Uncertainty (Mutual Information)
        # 3. Epistemic Uncertainty (Mutual Information)
        # MI = Total - Aleatoric
        # Shape: (N, T)
        MI = H_total - H_aleatoric
        
        # Clip small negative values due to float precision
        MI = np.maximum(MI, 0.0)

        return p_mean, H_total, H_aleatoric, MI

# ------------------------------
# Helper function to extract n_features from dataset
# ------------------------------
def get_n_features(dataset, encoder_type: str = "identity"):
    """
    Extract n_features from dataset, handling both vector and graph data.
    
    Args:
        dataset: DeepChem dataset
        encoder_type: "identity" (vector) or "dmpnn" (graph)
    
    Returns:
        n_features: Feature dimension
    """
    if encoder_type == "dmpnn":
        # For graph data, X contains GraphData objects
        # Extract atom feature dimension from first graph
        if hasattr(dataset.X, '__len__') and len(dataset.X) > 0:
            first_graph = dataset.X[0]
            if hasattr(first_graph, 'num_node_features'):
                return first_graph.num_node_features
            elif hasattr(first_graph, 'node_features'):
                return first_graph.node_features.shape[1]
            else:
                raise ValueError("Cannot extract node_features from GraphData object")
        else:
            raise ValueError("Dataset X is empty or invalid for graph mode")
    else:
        # For vector data, X is a numpy array
        if hasattr(dataset.X, 'shape'):
            if len(dataset.X.shape) >= 2:
                return dataset.X.shape[1]
            else:
                return dataset.X.shape[0] if len(dataset.X.shape) == 1 else 1
        else:
            raise ValueError(f"Cannot extract features from dataset.X (type: {type(dataset.X)})")


# ------------------------------
# Deep Ensemble
# ------------------------------
def train_nn_deep_ensemble(train_dc, valid_dc, test_dc, M=5, run_id=0, use_weights=False, mode="regression", save_model=False, save_path="./saved_models", encoder_type="identity"):
    n_tasks = train_dc.y.shape[1]
    n_features = get_n_features(train_dc, encoder_type=encoder_type)

    models = []

    for i in range(M):
        # Ensure each model gets different initialization by using a different seed
        # This is critical for ensemble diversity
        ensemble_seed = run_id * 1000 + i  # Unique seed per model in ensemble
        import random
        random.seed(ensemble_seed)
        np.random.seed(ensemble_seed)
        torch.manual_seed(ensemble_seed)
        
        model = build_model("baseline", n_features, n_tasks, mode=mode, encoder_type=encoder_type)
        
        if mode == "regression":
            loss = dc.models.losses.L2Loss()

            dc_model = UnifiedTorchModel(
                model=model,
                loss=loss,
                output_types=['prediction'],
                batch_size=64,
                learning_rate=1e-3,
                mode=mode,
                encoder_type=encoder_type,
            )
        
        else:
            loss = dc.models.losses.SigmoidCrossEntropy()
            dc_model = UnifiedTorchModel(
                model=model,
                loss=loss,
                output_types=['prediction'],
                batch_size=64,
                learning_rate=1e-3,
                mode=mode,
                encoder_type=encoder_type,
            )

        dc_model.fit(train_dc, nb_epoch=50)
        models.append(dc_model)
    
    # Save ensemble if requested (saves all members + metadata)
    if save_model:
        from model_utils import save_neural_network_ensemble
        save_neural_network_ensemble(
            models,
            save_path,
            model_name=f"nn_deep_ensemble_{mode}_run_{run_id}",
            mode=mode,
            create_dir=True
        )

    if mode == "regression":
        ensemble = DeepEnsembleRegressor(models)

        # Evaluate using ensemble mean
        mean_valid, _, _ = ensemble.predict_interval(valid_dc)
        mean_test,  lower_test, upper_test = ensemble.predict_interval(test_dc)
        cutoff_error_df = calculate_cutoff_error_data(mean_test, upper_test-lower_test, test_dc.y, test_dc.w, use_weights=use_weights)
        test_error = mse_from_mean_prediction(mean_test, test_dc, use_weights=use_weights)

        print("[Deep Ensemble] Validation MSE:", mse_from_mean_prediction(mean_valid, valid_dc, use_weights=use_weights))
        print("[Deep Ensemble] Test MSE:",        mse_from_mean_prediction(mean_test,  test_dc, use_weights=use_weights))

        uq_metrics = evaluate_uq_metrics_from_interval(
            y_true=test_dc.y,
            mean=mean_test,
            lower=lower_test,
            upper=upper_test,
            weights=test_dc.w,
            use_weights=use_weights,
            alpha=0.05,
            test_error=test_error,
        )

        print("UQ (Deep Ensemble):", uq_metrics)
    else:
        ensemble = DeepEnsembleClassifier(models)
        mean_probs, H_total, H_exp, MI = ensemble.predict_uncertainty(test_dc)

        # Binary AUC path (keeps your previous behavior)
        n_tasks = test_dc.y.shape[1]
        if n_tasks == 1 and mean_probs.shape[1] == 2:
            probs_positive = mean_probs[:, 1].reshape(-1, 1) # Force (N, 1)
            # entropy = entropy.reshape(-1, 1)
        else:
            probs_positive = mean_probs # Already (N, T)

        if use_weights and test_dc.w is not None:
            weights = np.asarray(test_dc.w).reshape(-1)
        else:
            weights = None

        test_auc = auc_from_probs(test_dc.y, probs_positive, test_dc.w, use_weights=use_weights)

        # choose uncertainty score: total entropy (or MI)
        uncertainty = H_total

        uq_metrics = evaluate_uq_metrics_classification(
            y_true=test_dc.y,
            probs=probs_positive,
            auc=test_auc,
            uncertainty=uncertainty,
            weights=test_dc.w,
            use_weights=use_weights,
            n_bins=20
        )

        cutoff_error_df = calculate_cutoff_classification_data(
            probs_positive,
            test_dc.y,
            weights=test_dc.w,
            use_weights=use_weights
        )

        print(f"[Deep Ensemble Class] Test AUC: {test_auc}")
        print(f"[Deep Ensemble Class] UQ Metrics: {uq_metrics}")

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

        # 2. 
        # We discard the variance column for the MSE calculation.
        mean_pred = raw_preds

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


class MyTorchClassifierHeteroscedastic(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, dropout_rate: float = 0.2):
        super(MyTorchClassifierHeteroscedastic, self).__init__()
        self.n_classes = int(n_classes)

        self.dense = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            # Must be 2 * n_classes:
            # first n_classes: mean logits (mu)
            # second n_classes: log variance (log_var)
            nn.Linear(64, self.n_classes * 2)
        )

    def forward(self, x):
        return self.dense(x)


def _enable_dropout_only(model: nn.Module):
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


class MCDropoutClassifierWrapper:
    def __init__(self, dc_model, n_samples=50, clamp_log_var=(-10.0, 10.0)):
        self.dc_model = dc_model
        self.n_samples = int(n_samples)
        self.clamp_log_var = clamp_log_var

    @torch.no_grad()
    def predict_uncertainty(self, dataset):
        X = dataset.X
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        model = self.dc_model.model
        device = next(model.parameters()).device
        X_tensor = torch.from_numpy(X).float().to(device)

        _enable_dropout_only(model)

        # --- 1. DETECT MODE (Single vs Multi-Task) ---
        # We peek at the output shape to decide between Softmax and Sigmoid
        with torch.no_grad():
            dummy_out = model(X_tensor[:2])
            raw_dim = dummy_out.shape[-1]
            # Assuming model outputs [means, log_vars], so we divide by 2
            C = raw_dim // 2 
            
        n_dataset_tasks = dataset.y.shape[1]
        
        # LOGIC:
        # If we have 1 task in dataset, but model outputs 2 dims (Class 0, Class 1),
        # use SOFTMAX (standard binary classification).
        # If we have N tasks in dataset, and model outputs N dims, 
        # use SIGMOID (independent binary tasks).
        if n_dataset_tasks == 1 and C == 2:
            activation_fn = "softmax"
        else:
            activation_fn = "sigmoid"

        # --- 2. MC DROPOUT LOOP ---
        probs_list = []
        for _ in range(self.n_samples):
            out = model(X_tensor)  # (N, 2C)
            mu = out[..., :C]
            log_var = out[..., C:]

            if self.clamp_log_var is not None:
                lo, hi = self.clamp_log_var
                log_var = log_var.clamp(lo, hi)

            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            logits = mu + std * eps
            
            # [FIX 1] Apply correct activation
            if activation_fn == "softmax":
                probs = F.softmax(logits, dim=-1)  # (N, 2) -> Sums to 1
            else:
                probs = torch.sigmoid(logits)      # (N, T) -> Independent
            
            probs_list.append(probs.cpu().numpy())

        mc_probs = np.stack(probs_list, axis=0)   # (S, N, C)
        mean_prob = mc_probs.mean(axis=0)         # (N, C)
        
        e = 1e-10

        # --- 3. ENTROPY CALCULATION ---
        # [FIX 2] Calculate entropy based on the mode
        if activation_fn == "softmax":
            # Categorical Entropy: -sum(p log p)
            # Returns (N,) -> reshaped to (N, 1)
            entropy = -np.sum(mean_prob * np.log(mean_prob + e), axis=1)
            entropy = entropy.reshape(-1, 1)
        else:
            # Binary Entropy Per Task: -[p log p + (1-p) log (1-p)]
            # Returns (N, T) -> One uncertainty value per task
            entropy = -(mean_prob * np.log(mean_prob + e) + 
                       (1 - mean_prob) * np.log(1 - mean_prob + e))

        return mean_prob, entropy


def train_nn_mc_dropout(train_dc, valid_dc, test_dc, n_samples=100, alpha=0.05, run_id=0, use_weights=False, mode="regression", save_model=False, save_path="./saved_models", encoder_type="identity"):
    """
    Train heteroscedastic MC-dropout NN and:
      - compute MSE with deterministic predictions (eval mode, no dropout)
      - compute UQ using DeepChem's predict_uncertainty (MC + aleatoric)
    """
    n_tasks    = train_dc.y.shape[1]
    n_features = get_n_features(train_dc, encoder_type=encoder_type)

    if mode == "regression":
        # ----- Build model & DeepChem wrapper -----
        model = build_model("mc_dropout", n_features, n_tasks, mode=mode, dropout_rate=0.1, encoder_type=encoder_type)
        loss  = HeteroscedasticL2Loss()                   # Kendall & Gal-style loss

        if encoder_type == "dmpnn":
            dc_model = UnifiedTorchModel(
                model=model,
                loss=loss,
                output_types=['prediction', 'variance', 'loss'],
                batch_size=64,
                learning_rate=1e-3,
                mode='regression',
                encoder_type=encoder_type
            )
        else:
            dc_model = dc.models.TorchModel(
                model=model,
                loss=loss,
                output_types=['prediction', 'variance', 'loss'],
                batch_size=64,
                learning_rate=1e-3,
                mode='regression'
            )
    else:
        y = np.asarray(train_dc.y)

        if y.ndim == 2 and y.shape[1] > 1 and set(np.unique(y)).issubset({0, 1}):
            n_classes = y.shape[1]
        else:
            n_classes = 2

        # Model outputs: (B, 2*K) = [mu_logits | log_var]
        model = build_model("mc_dropout", n_features, n_tasks, mode="classification", dropout_rate=0.2, encoder_type=encoder_type)
        loss = HeteroscedasticClassificationLoss(n_samples=20)  # MC samples for training integration
        
        if encoder_type == "dmpnn":
            dc_model = UnifiedTorchModel(
                model=model,
                loss=loss,
                output_types=['prediction'],
                batch_size=64,
                learning_rate=1e-3,
                mode='classification',
                encoder_type=encoder_type
            )
        else:
            dc_model = dc.models.TorchModel(
                model=model,
                loss=loss,
                output_types=['prediction'],
                batch_size=64,
                learning_rate=1e-3,
                mode='classification'
            )

    # ----- Train -----
    dc_model.fit(train_dc, nb_epoch=100)
    
    # Save model if requested
    if save_model:
        save_neural_network_model(
            dc_model, 
            save_path, 
            model_name=f"nn_mc_dropout_{mode}_run_{run_id}",
            create_dir=True
        )

    if mode == "regression":
        # Instantiate the FIXED wrapper
        mc_model = MCDropoutRegressorRefined(dc_model, n_samples=100)
        # Run Prediction
        mean_test, std_test = mc_model.predict_uncertainty(test_dc)

        # --- CRITICAL FIX: Ensure shapes match for MSE ---
        # Reshape mean_test to (N, 1) if necessary to match test_dc.y
        if mean_test.ndim == 1:
            mean_test = mean_test.reshape(-1, 1)

        cutoff_error_df = calculate_cutoff_error_data(mean_test, std_test, test_dc.y, test_dc.w, use_weights=use_weights)
        # Calculate MSE (Should now be ~0.67)
        test_mse = mse_from_mean_prediction(mean_test, test_dc, use_weights=use_weights)

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
            test_error=test_mse,
            weights=test_dc.w,
            use_weights=use_weights
        )

        print(f"[NN MC-DROPOUT] Test MSE: {test_mse}")
        print(f"[NN MC-DROPOUT] UQ Metrics: {uq_metrics}")

    else:
        mc_wrapper = MCDropoutClassifierWrapper(dc_model, n_samples=n_samples)
        mean_probs, entropy = mc_wrapper.predict_uncertainty(test_dc)

        # Binary AUC path (keeps your previous behavior)
        n_tasks = test_dc.y.shape[1]
        if n_tasks == 1 and mean_probs.shape[1] == 2:
            probs_positive = mean_probs[:, 1].reshape(-1, 1) # Force (N, 1)
            entropy = entropy.reshape(-1, 1)
        else:
            probs_positive = mean_probs # Already (N, T)

        # 4. Calculate AUC (Scalar or List) using helper
        test_auc = auc_from_probs(test_dc.y, probs_positive, test_dc.w, use_weights=use_weights)

        # 5. Calculate UQ Metrics (Internal loop handles lists)
        uq_metrics = evaluate_uq_metrics_classification(
            y_true=test_dc.y,
            probs=probs_positive,
            auc=test_auc,        # Pass the pre-calculated AUC
            uncertainty=entropy, # Pass the entropy (N, T)
            weights=test_dc.w,
            use_weights=use_weights,
            n_bins=20
        )

        # 6. Calculate Cutoff Data (Internal loop handles DataFrame concatenation)
        cutoff_error_df = calculate_cutoff_classification_data(
            probs_positive,
            test_dc.y,
            weights=test_dc.w,
            use_weights=use_weights
        )

        # 7. Print consistent output
        print(f"[NN MC-DROPOUT] Test AUC: {test_auc}") 
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


def train_evd_baseline(train_dc, valid_dc, test_dc, reg_coeff=1, alpha=0.05, run_id=0, use_weights=False, mode="regression", save_model=False, save_path="./saved_models", encoder_type="identity"):
    """
    Train Deep Evidential Regression (DER) NN and:
      - compute MSE with the analytical mean prediction (gamma)
      - compute UQ using the analytical total variance (aleatoric + epistemic)

    Args:
        train_dc, valid_dc, test_dc: DeepChem datasets.
        reg_coeff (float): The regularization coefficient lambda (λ) for the loss.
        alpha (float): The significance level for confidence interval calculation.
        encoder_type: "identity" (vector) or "dmpnn" (graph)
    """
    n_tasks = train_dc.y.shape[1]
    n_features = get_n_features(train_dc, encoder_type=encoder_type)

    if mode == "regression":
        # --- 1. Build Model & DeepChem wrapper (Using the structure from the second code block) ---
        # The model outputs 4 parameters (gamma, v, alpha, beta) for each task.
        model = build_model("evidential", n_features, n_tasks, mode="regression", encoder_type=encoder_type)

        # The custom evidential loss func(y_true, evidential_output)
        loss = EvidentialRegressionLoss(reg_coeff=reg_coeff)
    else:
        model = build_model("evidential", n_features, n_tasks, mode="classification", encoder_type=encoder_type)

        loss = EvidentialClassificationLoss()

    gradientClip = GradientClippingCallback()

    if mode == "regression":
        dc_model = UnifiedTorchModel(
            model=model,
            loss=loss,
            # The output types match the return structure of DenseNormalGamma:
            output_types=['prediction', 'loss', 'var1', 'var2'],
            batch_size=128,
            learning_rate=1e-4,
            # wandb=True,  # Set to True
            # model_dir='deep-evidential-regression-run-{}'.format(run_id),
            log_frequency=40,
            mode='regression',
            encoder_type=encoder_type
        )

        # --- 2. Train ---
        print(f"Training Deep Evidential Regression with lambda (reg_coeff) = {reg_coeff}")
        dc_model.fit(train_dc, nb_epoch=300, callbacks=[gradientClip])
        
        # Save model if requested
        if save_model:
            save_neural_network_model(
                dc_model, 
                save_path, 
                model_name=f"nn_evd_{mode}_run_{run_id}",
                create_dir=True
            )
        device = next(dc_model.model.parameters()).device

        # Convert data to appropriate format
        # For graph data, we need to convert GraphData to BatchMolGraph
        # For vector data, convert numpy arrays to tensors
        if encoder_type == "dmpnn":
            from nn import graphdata_to_batchmolgraph
            valid_bmg = graphdata_to_batchmolgraph(valid_dc.X)
            test_bmg = graphdata_to_batchmolgraph(test_dc.X)
            valid_bmg.to(device)
            test_bmg.to(device)
            valid_X_tensor = valid_bmg
            test_X_tensor = test_bmg
        else:
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

        cutoff_error_df = calculate_cutoff_error_data(mu_test, total_var_test, test_dc.y, test_dc.w, use_weights=use_weights)

        test_mse = mse_from_mean_prediction(mu_test, test_dc, use_weights=use_weights)

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
            test_error=test_mse,
            weights=test_dc.w,
            use_weights=use_weights
        )

        print(f"\n[EVIDENTIAL REGRESSION] Test MSE: {test_mse}")
        print(f"[EVIDENTIAL REGRESSION] UQ Metrics: {uq_metrics}")

        return uq_metrics, cutoff_error_df
    else:
        dc_model = UnifiedTorchModel(
            model=model,
            loss=loss,
            # The output types match the return structure of DenseNormalGamma:
            output_types=['prediction', 'loss', 'var1', 'var2'],
            batch_size=128,
            learning_rate=1e-4,
            # wandb=True,  # Set to True
            # model_dir='deep-evidential-regression-run-{}'.format(run_id),
            log_frequency=40,
            mode='classification',
            encoder_type=encoder_type
        )

        # --- 2. Train ---
        print(f"Training Deep Evidential Classification")
        dc_model.fit(train_dc, nb_epoch=300, callbacks=[gradientClip])
        
        # Save model if requested
        if save_model:
            save_neural_network_model(
                dc_model, 
                save_path, 
                model_name=f"nn_evd_{mode}_run_{run_id}",
                create_dir=True
            )
        device = next(dc_model.model.parameters()).device

        # Convert data to appropriate format
        # For graph data, we need to convert GraphData to BatchMolGraph
        # For vector data, convert numpy arrays to tensors
        if encoder_type == "dmpnn":
            from nn import graphdata_to_batchmolgraph
            valid_bmg = graphdata_to_batchmolgraph(valid_dc.X)
            test_bmg = graphdata_to_batchmolgraph(test_dc.X)
            valid_bmg.to(device)
            test_bmg.to(device)
            valid_X_tensor = valid_bmg
            test_X_tensor = test_bmg
        else:
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

        if isinstance(mu_test, torch.Tensor):
            mu_test = mu_test.cpu().numpy()
            
        # Check if we have multiple columns (binary classification usually has >= 2)
        if mu_test.ndim > 1 and mu_test.shape[1] > 1:
            # Slice: Start at index 1 (Class 1), take every 2nd column
            mu_test = mu_test[:, 1::2]
        
        # Ensure it is at least 2D for consistency: (N_Samples, N_Tasks)
        # Even if it's single task, we want (N, 1), not (N,)
        if mu_test.ndim == 1:
            mu_test = mu_test.reshape(-1, 1)

        cutoff_error_df = calculate_cutoff_classification_data(mu_test, test_dc.y, test_dc.w, use_weights=use_weights)

        test_auc = auc_from_probs(test_dc.y, mu_test, test_dc.w, use_weights=use_weights)

        # test_mse = mse_from_mean_prediction(mu_test, test_dc)

        uq_metrics = evaluate_uq_metrics_classification(
            y_true=test_dc.y,
            probs=mu_test,
            auc=test_auc,
            uncertainty=total_var_test,
            weights=test_dc.w,
            use_weights=use_weights,
            n_bins=20
        )

        print(f"\n[EVIDENTIAL CLASSIFIACTION] Test MSE: {test_auc}")
        print(f"[EVIDENTIAL CLASSIFIACTION] UQ Metrics: {uq_metrics}")

        return uq_metrics, cutoff_error_df
    

def train_nn_baseline(train_dc, valid_dc, test_dc, run_id=0, use_weights=False, mode="regression", save_model=False, save_path="./saved_models", encoder_type="identity"):
    n_tasks = train_dc.y.shape[1]
    n_features = get_n_features(train_dc, encoder_type=encoder_type)

    model = build_model("baseline", n_features, n_tasks, mode=mode, encoder_type=encoder_type)
    if mode == "regression":
        loss = dc.models.losses.L2Loss()

        dc_model = UnifiedTorchModel(
            model=model,
            loss=loss,
            output_types=['prediction'],
            batch_size=64,
            learning_rate=1e-3,
            mode='regression',
            encoder_type=encoder_type,
        )
    else:
        loss = dc.models.losses.SigmoidCrossEntropy()

        dc_model = UnifiedTorchModel(
            model=model,
            loss=loss,
            output_types=['prediction'],
            batch_size=64,
            learning_rate=1e-3,
            mode='classification',
            encoder_type=encoder_type,
        )

    if mode == "regression":
        dc_model.fit(train_dc, nb_epoch=80)
        
        # Save model if requested
        if save_model:
            save_neural_network_model(
                dc_model, 
                save_path, 
                model_name=f"nn_baseline_{mode}_run_{run_id}",
                create_dir=True
            )

        metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
        metric_name = metric.name

        valid_scores = dc_model.evaluate(valid_dc, [metric], use_sample_weights=use_weights, per_task_metrics=True)
        test_scores = dc_model.evaluate(test_dc, [metric], use_sample_weights=use_weights, per_task_metrics=True)

        # 4. Unpack the tuples safely
        # *_agg:      Dict with scalar weighted average (mean across tasks)
        # *_detailed: Dict with detailed scores (List if multitask, float if singletask)
        valid_agg, valid_detailed = valid_scores
        test_agg,  test_detailed  = test_scores

        # 5. Extract the Main Aggregate Score (Safe for logging)
        val_score_avg = valid_agg[metric_name]
        test_score_avg = test_agg[metric_name]

        # 6. Console Logging
        print(f"[NN Baseline] Validation {metric_name} (Avg): {val_score_avg:.4f}")
        print(f"[NN Baseline] Test {metric_name} (Avg):       {test_score_avg:.4f}")

        # 7. Per-Task Logging (Handle List vs Scalar types automatically)
        task_scores_test = test_detailed[metric_name]

        if isinstance(task_scores_test, list):
            # Multi-task: Print list rounded for readability
            formatted_scores = [round(x, 4) for x in task_scores_test]
            print(f"[NN Baseline] Test {metric_name} (Per Task): {formatted_scores}")
        else:
            # Single-task: It is just a float
            print(f"[NN Baseline] Test {metric_name} (Task 0):   {task_scores_test:.4f}")

        return {
            "alpha": None,
            "empirical_coverage": None,
            "avg_pred_std": None,
            "nll": None,
            "ce": None,
            "spearman_err_unc": None,
            "MSE": task_scores_test,
        }
    else:
        # 1. Train the model
        dc_model.fit(train_dc, nb_epoch=80)
        
        # Save model if requested
        if save_model:
            save_neural_network_model(
                dc_model, 
                save_path, 
                model_name=f"nn_baseline_{mode}_run_{run_id}",
                create_dir=True
            )

        # 2. Define the metric (Using ROC-AUC as in your snippet)
        metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        metric_name = metric.name

        # 3. Evaluate with per_task_metrics=True
        # This returns a tuple: (aggregate_score_dict, per_task_score_dict)
        valid_results = dc_model.evaluate(valid_dc, [metric], use_sample_weights=use_weights, per_task_metrics=True)
        test_results = dc_model.evaluate(test_dc, [metric], use_sample_weights=use_weights, per_task_metrics=True)

        # 4. Unpack the tuples safely
        valid_agg, valid_detailed = valid_results
        test_agg,  test_detailed  = test_results

        # 5. Extract the Main Aggregate Score (Safe for logging)
        val_score_avg = valid_agg[metric_name]
        test_score_avg = test_agg[metric_name]

        # 6. Console Logging
        print(f"[NN Baseline] Validation {metric_name} (Avg): {val_score_avg:.4f}")
        print(f"[NN Baseline] Test {metric_name} (Avg):       {test_score_avg:.4f}")

        # 7. Per-Task Logging (Handle List vs Scalar types automatically)
        task_scores_test = test_detailed[metric_name]

        if isinstance(task_scores_test, list):
            # Multi-task: Print list rounded for readability
            formatted_scores = [round(x, 4) for x in task_scores_test]
            print(f"[NN Baseline] Test {metric_name} (Per Task): {formatted_scores}")
        else:
            # Single-task: It is just a float
            print(f"[NN Baseline] Test {metric_name} (Task 0):   {task_scores_test:.4f}")

        # 8. Return Dictionary
        return {
            # INVARIANT: Always returns a single float (mean over tasks).
            # This keeps your main results table clean and crash-free.
            # NEW: Stores the detailed scores (list or float) for deeper analysis.
            "AUC": task_scores_test,
            # Placeholders for other metrics
            "NLL": None,
            "Brier": None,
            "ECE": None,
            "Avg_Entropy": None,
            "Spearman_Err_Unc": None,
        }


# ============================================================
# Evaluation Functions (for reloaded models)
# ============================================================

def evaluate_nn_baseline(dc_model, test_dc, use_weights=False, mode="regression"):
    """
    Evaluate a trained neural network baseline model on test data.
    This function replicates the evaluation logic from train_nn_baseline()
    but works with a pre-trained model.
    
    Args:
        dc_model: Trained DeepChem TorchModel instance
        test_dc: Test dataset
        use_weights: Whether to use sample weights
        mode: "regression" or "classification"
        
    Returns:
        dict: UQ metrics dictionary (same format as train_nn_baseline)
    """
    if mode == "regression":
        metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
        metric_name = metric.name
        test_scores = dc_model.evaluate(test_dc, [metric], use_sample_weights=use_weights, per_task_metrics=True)
        test_agg, test_detailed = test_scores
        task_scores_test = test_detailed[metric_name]
        
        return {
            "alpha": None,
            "empirical_coverage": None,
            "avg_pred_std": None,
            "nll": None,
            "ce": None,
            "spearman_err_unc": None,
            "MSE": task_scores_test,
        }
    else:
        metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        metric_name = metric.name
        test_results = dc_model.evaluate(test_dc, [metric], use_sample_weights=use_weights, per_task_metrics=True)
        test_agg, test_detailed = test_results
        task_scores_test = test_detailed[metric_name]
        
        return {
            "AUC": task_scores_test,
            "NLL": None,
            "Brier": None,
            "ECE": None,
            "Avg_Entropy": None,
            "Spearman_Err_Unc": None,
        }


def evaluate_nn_evd(dc_model, test_dc, use_weights=False, mode="regression"):
    """
    Evaluate a trained evidential neural network model on test data.
    
    Args:
        dc_model: Trained DeepChem TorchModel with evidential model
        test_dc: Test dataset
        use_weights: Whether to use sample weights
        mode: "regression" or "classification"
        
    Returns:
        tuple: (uq_metrics, cutoff_error_df)
    """
    device = next(dc_model.model.parameters()).device
    test_X_tensor = torch.from_numpy(test_dc.X).float().to(device)
    
    with torch.no_grad():
        mu_test, params_test, aleatoric_test, epistemic_test = dc_model.model(test_X_tensor)
    
    if mode == "regression":
        total_var_test = aleatoric_test.cpu().numpy() + epistemic_test.cpu().numpy()
        std_test = np.sqrt(total_var_test)
        mu_test = mu_test.cpu().numpy()
        if mu_test.ndim == 1:
            mu_test = mu_test.reshape(-1, 1)
        
        cutoff_error_df = calculate_cutoff_error_data(mu_test, total_var_test, test_dc.y, test_dc.w, use_weights=use_weights)
        test_mse = mse_from_mean_prediction(mu_test, test_dc, use_weights=use_weights)
        
        alpha = 0.05
        z = norm.ppf(1 - alpha / 2.0)
        lower = mu_test - z * std_test
        upper = mu_test + z * std_test
        
        if std_test.ndim == 1:
            std_test = std_test.reshape(-1, 1)
        
        uq_metrics = evaluate_uq_metrics_from_interval(
            y_true=test_dc.y,
            mean=mu_test,
            lower=lower,
            upper=upper,
            alpha=alpha,
            test_error=test_mse,
            weights=test_dc.w,
            use_weights=use_weights
        )
    else:
        if isinstance(mu_test, torch.Tensor):
            mu_test = mu_test.cpu().numpy()
        
        if mu_test.ndim > 1 and mu_test.shape[1] > 1:
            mu_test = mu_test[:, 1::2]
        
        if mu_test.ndim == 1:
            mu_test = mu_test.reshape(-1, 1)
        
        total_var_test = aleatoric_test.cpu().numpy() + epistemic_test.cpu().numpy()
        cutoff_error_df = calculate_cutoff_classification_data(mu_test, test_dc.y, test_dc.w, use_weights=use_weights)
        test_auc = auc_from_probs(test_dc.y, mu_test, test_dc.w, use_weights=use_weights)
        
        uq_metrics = evaluate_uq_metrics_classification(
            y_true=test_dc.y,
            probs=mu_test,
            auc=test_auc,
            uncertainty=total_var_test,
            weights=test_dc.w,
            use_weights=use_weights,
            n_bins=20
        )
    
    return uq_metrics, cutoff_error_df


def evaluate_nn_mc_dropout(dc_model, test_dc, n_samples=100, use_weights=False, mode="regression"):
    """
    Evaluate a trained MC-Dropout neural network model on test data.
    
    Args:
        dc_model: Trained DeepChem TorchModel with MC-Dropout
        test_dc: Test dataset
        n_samples: Number of MC samples for uncertainty estimation
        use_weights: Whether to use sample weights
        mode: "regression" or "classification"
        
    Returns:
        tuple: (uq_metrics, cutoff_error_df)
    """
    if mode == "regression":
        mc_model = MCDropoutRegressorRefined(dc_model, n_samples=n_samples)
        mean_test, std_test = mc_model.predict_uncertainty(test_dc)
        
        if mean_test.ndim == 1:
            mean_test = mean_test.reshape(-1, 1)
        
        cutoff_error_df = calculate_cutoff_error_data(mean_test, std_test, test_dc.y, test_dc.w, use_weights=use_weights)
        test_mse = mse_from_mean_prediction(mean_test, test_dc, use_weights=use_weights)
        
        alpha = 0.05
        z = norm.ppf(1 - alpha / 2.0)
        lower = mean_test - z * std_test
        upper = mean_test + z * std_test
        
        if std_test.ndim == 1:
            std_test = std_test.reshape(-1, 1)
        
        uq_metrics = evaluate_uq_metrics_from_interval(
            y_true=test_dc.y,
            mean=mean_test,
            lower=lower,
            upper=upper,
            alpha=alpha,
            test_error=test_mse,
            weights=test_dc.w,
            use_weights=use_weights
        )
    else:
        mc_wrapper = MCDropoutClassifierWrapper(dc_model, n_samples=n_samples)
        mean_probs, entropy = mc_wrapper.predict_uncertainty(test_dc)
        
        n_tasks = test_dc.y.shape[1]
        if n_tasks == 1 and mean_probs.shape[1] == 2:
            probs_positive = mean_probs[:, 1].reshape(-1, 1)
            entropy = entropy.reshape(-1, 1)
        else:
            probs_positive = mean_probs
        
        test_auc = auc_from_probs(test_dc.y, probs_positive, test_dc.w, use_weights=use_weights)
        
        uq_metrics = evaluate_uq_metrics_classification(
            y_true=test_dc.y,
            probs=probs_positive,
            auc=test_auc,
            uncertainty=entropy,
            weights=test_dc.w,
            use_weights=use_weights,
            n_bins=20
        )
        
        cutoff_error_df = calculate_cutoff_classification_data(
            probs_positive,
            test_dc.y,
            weights=test_dc.w,
            use_weights=use_weights
        )
    
    return uq_metrics, cutoff_error_df


def evaluate_nn_deep_ensemble(models, test_dc, use_weights=False, mode="regression"):
    """
    Evaluate a deep ensemble on test data.
    
    Args:
        models: List of trained DeepChem TorchModel instances
        test_dc: Test dataset
        use_weights: Whether to use sample weights
        mode: "regression" or "classification"
        
    Returns:
        tuple: (uq_metrics, cutoff_error_df)
    """
    if mode == "regression":
        ensemble = DeepEnsembleRegressor(models)
        mean_test, lower_test, upper_test = ensemble.predict_interval(test_dc)
        cutoff_error_df = calculate_cutoff_error_data(mean_test, upper_test-lower_test, test_dc.y, test_dc.w, use_weights=use_weights)
        test_error = mse_from_mean_prediction(mean_test, test_dc, use_weights=use_weights)
        
        uq_metrics = evaluate_uq_metrics_from_interval(
            y_true=test_dc.y,
            mean=mean_test,
            lower=lower_test,
            upper=upper_test,
            weights=test_dc.w,
            use_weights=use_weights,
            alpha=0.05,
            test_error=test_error,
        )
    else:
        ensemble = DeepEnsembleClassifier(models)
        mean_probs, H_total, H_exp, MI = ensemble.predict_uncertainty(test_dc)
        
        n_tasks = test_dc.y.shape[1]
        if n_tasks == 1 and mean_probs.shape[1] == 2:
            probs_positive = mean_probs[:, 1].reshape(-1, 1)
            entropy = H_total.reshape(-1, 1)
        else:
            probs_positive = mean_probs
        
        test_auc = auc_from_probs(test_dc.y, probs_positive, test_dc.w, use_weights=use_weights)
        uncertainty = H_total
        
        uq_metrics = evaluate_uq_metrics_classification(
            y_true=test_dc.y,
            probs=probs_positive,
            auc=test_auc,
            uncertainty=uncertainty,
            weights=test_dc.w,
            use_weights=use_weights,
            n_bins=20
        )
        
        cutoff_error_df = calculate_cutoff_classification_data(
            probs_positive,
            test_dc.y,
            weights=test_dc.w,
            use_weights=use_weights
        )
    
    return uq_metrics, cutoff_error_df
