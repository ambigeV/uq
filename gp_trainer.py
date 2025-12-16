# gp_trainer.py
import torch
import torch.nn as nn
import gpytorch
import deepchem as dc
import numpy as np
from gpytorch.optim import NGD
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


class WeightedVariationalELBO(gpytorch.mlls.VariationalELBO):
    """
    A wrapper around VariationalELBO that supports per-sample weighting.
    Formula: Σ (weight_i * log_prob_i) - KL_Divergence
    """

    def forward(self, variational_dist_f, target, weights=None, **kwargs):
        # 1. Calculate the KL Divergence term (Regularization)
        # This is standard GPyTorch logic to handle batching properly
        kl_divergence = self.model.variational_strategy.kl_divergence().div(
            self.num_data / target.size(0)
        )

        # 2. Calculate Expected Log Probability (Likelihood) per point
        # Shape: (batch_size,)
        log_likelihood = self.likelihood.expected_log_prob(target, variational_dist_f, **kwargs)

        # 3. Apply Weights (Injecting data importance)
        if weights is not None:
            # Ensure shape match: (N,) * (N,)
            if weights.shape != log_likelihood.shape:
                weights = weights.view_as(log_likelihood)

            log_likelihood = log_likelihood * weights

        # 4. Sum and Combine
        return log_likelihood.sum() - kl_divergence.sum()


class GPTrainer:
    def __init__(
        self,
        model,
        train_dataset: dc.data.NumpyDataset,
        lr: float = 0.1,
        num_iters: int = 100,
        device: str = "cpu",
        log_interval: int = 10,
        nn_lr: float = 1e-3,
        ngd_lr: float = 0.02,
        warmup_iters: int = 5,
        clip_grad: float = 1.0,
        add_prior_term: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.num_iters = num_iters
        self.log_interval = log_interval

        self.likelihood = self.model.likelihood
        self.gp = self.model.gp_model
        self.feat_log_sample = 4096  # default for feature/predictive logging

        self.train_x, self.train_y = self._dc_to_torch(train_dataset)

        if hasattr(self.model, "train_y_norm"):
            self.train_y = self.model.train_y_norm

        # After self.gp is set
        if hasattr(self.gp, "variational_strategy"):
            # SVGP path
            self.is_svgp = True
            self.mll = gpytorch.mlls.VariationalELBO(
                self.likelihood, self.gp, num_data=self.train_x.size(0)
            )
            # --------- [MODIFIED] Optimizers: NGD(q) + Adam(other params) ----------
            self.opt_ngd = NGD(
                self.gp.variational_parameters(),
                num_data=self.train_x.size(0),
                lr=ngd_lr,
            )
            # Build Adam param groups excluding variational params
            q_param_ids = {id(p) for p in self.gp.variational_parameters()}
            other_params = [p for p in self.model.parameters() if id(p) not in q_param_ids]
            # Try to detect the feature extractor inside deep kernel to give it nn_lr
            feat_params = []
            try:
                # ScaleKernel(base_kernel=DeepFeatureKernel(...))
                feat_params = list(self.gp.covar_module.base_kernel.feature_extractor.parameters())
            except Exception:
                pass
            feat_ids = {id(p) for p in feat_params}

            groups = []
            if feat_params:
                groups.append({
                    "params": [p for p in other_params if id(p) in feat_ids],
                    "lr": nn_lr,
                    "weight_decay": 1e-4,
                })
                groups.append({
                    "params": [p for p in other_params if id(p) not in feat_ids],
                    "lr": lr,
                })
            else:
                groups.append({"params": other_params, "lr": lr})

            self.opt_adam = torch.optim.Adam(groups)

            self.warmup_iters = int(max(0, warmup_iters))
            self.clip_grad = clip_grad
            self.add_prior_term = add_prior_term

        else:
            # Exact GP path
            self.is_svgp = False  # [MODIFIED]
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _summ(self, tensor, name, k: int = 5) -> str:
        """Pretty summary for scalar or vector tensors."""
        if tensor is None:
            return f"{name}: N/A"
        t = tensor.detach().float().cpu()
        if t.ndim == 0 or t.numel() == 1:
            return f"{name}: {t.item():.3e}"
        flat = t.view(-1)
        head = " ".join(f"{v:.3e}" for v in flat[:k].tolist())
        return f"{name}[0:{min(k, flat.numel())}]: {head} | mean={flat.mean().item():.3e}"

    def _dc_to_torch(self, dataset: dc.data.NumpyDataset):
        X = torch.from_numpy(dataset.X).float().to(self.device)
        y = torch.from_numpy(dataset.y).float().to(self.device)

        # single-task case: (N, 1) -> (N,)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(-1)
        return X, y

    # --------------------- [MODIFIED] helpers for logging ---------------------
    def _has_feature_extractor(self) -> bool:
        try:
            _ = self.gp.covar_module.base_kernel.feature_extractor
            return True
        except Exception:
            return False

    def _get_bn_module(self):
        """Return the (first) BatchNorm1d in the feature extractor if present."""
        try:
            feat = self.gp.covar_module.base_kernel.feature_extractor
        except Exception:
            return None
        # Prefer attribute 'post_bn' if exists
        bn = getattr(feat, "post_bn", None)
        if isinstance(bn, nn.BatchNorm1d):
            return bn
        # Otherwise search modules
        for m in feat.modules():
            if isinstance(m, nn.BatchNorm1d):
                return m
        return None

    @torch.no_grad()
    def _feature_stats_str(self, sample_k: int = 4096, k_head: int = 5) -> str:
        """
        Summarize φ(X) distribution and BN running stats (if any).
        Prints compact stats to avoid flooding logs.
        """
        if not self._has_feature_extractor():
            return "feat: N/A"

        feat = self.gp.covar_module.base_kernel.feature_extractor
        N = self.train_x.size(0)
        if sample_k < N:
            idx = torch.randperm(N, device=self.train_x.device)[:sample_k]
            xb = self.train_x[idx]
        else:
            xb = self.train_x

        z = feat(xb).detach()  # (n, Dz)
        # per-dim stats
        z_mu = z.mean(dim=0)
        z_sd = z.std(dim=0, unbiased=False)
        # summarize
        mu_abs_mean = z_mu.abs().mean().item()
        sd_mean = z_sd.mean().item()
        sd_min = z_sd.min().item()
        sd_max = z_sd.max().item()
        # head preview
        head_mu = " ".join(f"{v:.2e}" for v in z_mu[:k_head].cpu().tolist())
        head_sd = " ".join(f"{v:.2e}" for v in z_sd[:k_head].cpu().tolist())

        # BN running stats if available
        bn = self._get_bn_module()
        if bn is not None and bn.track_running_stats:
            rm = bn.running_mean.detach().cpu()
            rv = bn.running_var.detach().cpu()
            bn_str = (f" | BNμ(mean)={rm.mean():.2e}, BNσ(mean)={(rv.clamp_min(1e-12).sqrt().mean()):.2e}")
        else:
            bn_str = ""

        return (f"feat| μ_abs(avg)={mu_abs_mean:.2e}, σ(avg/min/max)={sd_mean:.2e}/{sd_min:.2e}/{sd_max:.2e} "
                f"| μ[:{k_head}]={head_mu} | σ[:{k_head}]={head_sd}{bn_str}")

    @torch.no_grad()
    def _predictive_stats_str(self, output=None, sample_k: int = 4096) -> str:
        """
        Summarize predictive distribution on a subset of train_x (mean/std on ORIGINAL y-scale).
        Handles the ExactGP-in-train-mode restriction by reusing `output` or temporarily
        switching to eval() for subset predictions.
        """
        N = self.train_x.size(0)

        # choose subset indices (used if we need to subselect later)
        if sample_k < N:
            idx = torch.randperm(N, device=self.train_x.device)[:sample_k]
            xb = self.train_x.index_select(0, idx)
        else:
            idx = None  # [FIX] remember if we subselected
            xb = self.train_x

        # ---------- [FIX] Safe predictive path selection ----------
        if (not getattr(self, "is_svgp", False)) and self.gp.training:
            # ExactGP *in training mode* cannot take arbitrary inputs.
            if (output is not None) and (output.mean.shape[0] == N):
                # Reuse full-train output; optionally subselect if we sampled
                mean_n = output.mean
                std_n = output.stddev
                if idx is not None:
                    mean_n = mean_n.index_select(0, idx)
                    std_n = std_n.index_select(0, idx)
            else:
                # Temporarily switch to eval() to do a subset forward safely
                gp_was_train = self.gp.training
                lik_was_train = self.likelihood.training
                self.gp.eval()
                self.likelihood.eval()
                with gpytorch.settings.fast_pred_var():
                    pred = self.likelihood(self.gp(xb))
                    mean_n = pred.mean
                    std_n = pred.stddev
                # restore original modes
                self.gp.train(gp_was_train)
                self.likelihood.train(lik_was_train)
        else:
            # SVGP (OK in train) OR any model already in eval
            with gpytorch.settings.fast_pred_var():
                pred = self.likelihood(self.gp(xb))
                mean_n = pred.mean
                std_n = pred.stddev
        # ---------------------------------------------------------

        # de-normalize if model has y stats
        y_mean = getattr(self.model, "y_mean", None)
        y_std = getattr(self.model, "y_std", None)
        if isinstance(y_mean, torch.Tensor) and isinstance(y_std, torch.Tensor):
            mean = mean_n * y_std + y_mean
            std = std_n * y_std
        else:
            mean, std = mean_n, std_n

        mean = mean.detach().float().cpu().view(-1)
        std = std.detach().float().cpu().view(-1)

        mu_mean, mu_min, mu_max = mean.mean().item(), mean.min().item(), mean.max().item()
        sd_mean, sd_min, sd_max = std.mean().item(), std.min().item(), std.max().item()
        return (f"pred| μ(mean/min/max)={mu_mean:.3e}/{mu_min:.3e}/{mu_max:.3e} "
                f"| σ(mean/min/max)={sd_mean:.3e}/{sd_min:.3e}/{sd_max:.3e}")

    def train(self):
        self.model.train()
        self.likelihood.train()

        if self.is_svgp and getattr(self, "warmup_iters", 0) > 0:
            try:
                for p in self.gp.covar_module.base_kernel.feature_extractor.parameters():
                    p.requires_grad_(False)
            except Exception:
                pass

        for i in range(1, self.num_iters + 1):
            if self.is_svgp:
                self.opt_ngd.zero_grad()
                self.opt_adam.zero_grad()

                # --------- [MODIFIED] add a bit more jitter for un-whitened SVGP ----------
                with gpytorch.settings.cholesky_jitter(1e-4):
                    output = self.gp(self.train_x)

                loss = -self.mll(output, self.train_y)

                # --------- [MODIFIED] optional MAP term if you registered priors ----------
                if getattr(self, "add_prior_term", False):
                    prior_term = 0.0
                    if hasattr(self.gp, "prior_log_prob"):
                        prior_term += self.gp.prior_log_prob
                    if hasattr(self.likelihood, "prior_log_prob"):
                        prior_term += self.likelihood.prior_log_prob
                    loss = loss - prior_term

                loss.backward()

                # --------- [MODIFIED] clip gradients to stabilize deep-kernel training ----------
                if getattr(self, "clip_grad", None):
                    if self.clip_grad and self.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

                self.opt_ngd.step()
                self.opt_adam.step()

                # --------- [MODIFIED] unfreeze feature extractor after warmup ----------
                if i == getattr(self, "warmup_iters", 0) and self.warmup_iters > 0:
                    try:
                        for p in self.gp.covar_module.base_kernel.feature_extractor.parameters():
                            p.requires_grad_(True)
                    except Exception:
                        pass
            else:
                self.optimizer.zero_grad()
                output = self.gp(self.train_x)
                loss = -self.mll(output, self.train_y)
                loss.backward()
                self.optimizer.step()

            # if i % self.log_interval == 0 or i == 1 or i == self.num_iters:
            #     print(f"[Iter {i:03d}] Loss: {loss.item():.4f}")
            if i % self.log_interval == 0 or i == 1 or i == self.num_iters:
                # Safely grab hypers across kernels/likelihoods
                try:
                    ls = self.gp.covar_module.base_kernel.lengthscale
                except Exception:
                    ls = None
                try:
                    os = self.gp.covar_module.outputscale
                except Exception:
                    os = None
                # GaussianLikelihood -> scalar; FixedNoise -> per-point vector
                noise = getattr(self.likelihood, "noise", None)

                ls_str = self._summ(ls, "lengthscale")
                os_str = self._summ(os, "outputscale")
                if noise is None:
                    noise_str = "noise: N/A"
                else:
                    n = noise.detach().float().cpu()
                    if n.ndim == 0 or n.numel() == 1:
                        noise_str = f"noise: {n.item():.3e}"
                    else:
                        noise_str = f"noise(mean/min/max): {n.mean().item():.3e}/{n.min().item():.3e}/{n.max().item():.3e}"

                # print(f"[Iter {i:03d}] Loss: {loss.item():.4f} | {ls_str} | {os_str} | {noise_str}")

                # [MODIFIED] feature & BN stats (if we have a feature extractor)
                feat_str = self._feature_stats_str(sample_k=self.feat_log_sample, k_head=5)

                # [MODIFIED] predictive distribution summary (original y-scale if available)
                pred_str = self._predictive_stats_str(output=output, sample_k=self.feat_log_sample)

                print(
                    f"[Iter {i:03d}] Loss: {loss.item():.4f} | {ls_str} | {os_str} | {noise_str} | "
                    f"{feat_str} | {pred_str}"  # [MODIFIED]
                )

    def evaluate_mse(self, dataset: dc.data.NumpyDataset, use_weights=False) -> float:
        self.model.eval()
        self.likelihood.eval()

        X, y_true = self._dc_to_torch(dataset)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if use_weights:
                w_np = dataset.w
                if w_np.ndim == 2:
                    w_np = w_np[:, 0]

                w_torch = torch.from_numpy(w_np).float().to(self.device)

                # Clamp and convert to Noise Variance
                w_torch = torch.clamp(w_torch, min=1e-5)

                preds = self.model(X, w=w_torch).squeeze(-1)  # (N,)
            else:
                preds = self.model(X).squeeze(-1)

            mean_preds = preds.cpu().numpy()

        squared_errors = (y_true - mean_preds) ** 2

        if use_weights and dataset.w is not None:
            # Weighted MSE
            weights = dataset.w.flatten()
            mse = np.average(squared_errors, weights=weights)
        else:
            # Standard MSE
            # print(squared_errors.__class__)
            mse = np.mean(squared_errors.cpu().numpy())

        return mse

    def predict_mean(self, dataset: dc.data.NumpyDataset, use_weights=False) -> np.ndarray:
        self.model.eval()
        self.likelihood.eval()

        X, _ = self._dc_to_torch(dataset)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean = self.model(X).squeeze(-1)
        return mean.cpu().numpy()

    def predict_interval(self, dataset: dc.data.NumpyDataset, alpha: float = 0.05, use_weights=False):
        self.model.eval()
        self.likelihood.eval()

        X, _ = self._dc_to_torch(dataset)

        w_torch = None
        is_fixed_noise = isinstance(self.likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood)

        if use_weights and is_fixed_noise:
            if dataset.w is not None:
                w_np = dataset.w
                if w_np.ndim == 2:
                    w_np = w_np[:, 0]

                w_torch = torch.from_numpy(w_np).float().to(self.device)

                # Clamp and convert to Noise Variance
                w_torch = torch.clamp(w_torch, min=1e-5)
            else:
                print("Warning: use_weights=True but dataset.w is None. Ignoring weights.")

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean, lower, upper = self.model.predict_interval(X, alpha=alpha, w=w_torch)

        return (
            mean.cpu().numpy(),
            lower.cpu().numpy(),
            upper.cpu().numpy(),
        )


from gpytorch.optim import NGD


class GPClassificationTrainer:
    def __init__(
            self,
            model,
            train_dataset: dc.data.NumpyDataset,
            lr: float = 0.01,
            num_iters: int = 100,
            device: str = "cpu",
            log_interval: int = 10,
            nn_lr: float = 1e-3,
            ngd_lr: float = 0.1,  # Specific to Variational
            warmup_iters: int = 5,
            clip_grad: float = 1.0,  # Optional clipping
            use_weights: bool = False
    ):
        self.model = model.to(device)
        self.device = device
        self.num_iters = num_iters
        self.log_interval = log_interval
        self.warmup_iters = warmup_iters
        self.clip_grad = clip_grad
        self.use_weights = use_weights

        self.gp = self.model.gp_model
        self.likelihood = self.model.likelihood

        # Load Raw Data
        self.train_x, self.train_y = self._dc_to_torch(train_dataset)

        self.train_w = None
        if self.use_weights and train_dataset.w is not None:
            w_np = train_dataset.w.flatten()

            # CRITICAL: Normalize weights to avoid breaking the KL balance
            # If weights sum to 1e6, the likelihood overwhelms the KL prior => Overfitting.
            # We scale them so the mean is 1.0.
            w_mean = w_np.mean() + 1e-8
            w_norm = w_np / w_mean

            self.train_w = torch.from_numpy(w_norm).float().to(self.device)

        if self.use_weights:
            self.mll = WeightedVariationalELBO(
                self.likelihood, self.gp, num_data=self.train_x.size(0)
            )
        else:
            # Variational ELBO (Classification Loss)
            self.mll = gpytorch.mlls.VariationalELBO(
                self.likelihood, self.gp, num_data=self.train_x.size(0)
            )

        # --- Optimizers (NGD for Variational Params, Adam for Kernel/NN) ---
        self.opt_ngd = NGD(
            self.gp.variational_parameters(),
            num_data=self.train_x.size(0),
            lr=ngd_lr,
        )

        # Identify params for Adam
        q_param_ids = {id(p) for p in self.gp.variational_parameters()}
        other_params = [p for p in self.model.parameters() if id(p) not in q_param_ids]

        # Attempt to find Deep Kernel Feature Extractor for lower lr
        feat_params = []
        try:
            # Adjust path if using DeepFeatureKernel
            if hasattr(self.gp.covar_module, "base_kernel"):
                if hasattr(self.gp.covar_module.base_kernel, "feature_extractor"):
                    feat_params = list(self.gp.covar_module.base_kernel.feature_extractor.parameters())
        except Exception:
            pass
        feat_ids = {id(p) for p in feat_params}

        groups = []
        if feat_params:
            groups.append({"params": [p for p in other_params if id(p) in feat_ids], "lr": nn_lr})
            groups.append({"params": [p for p in other_params if id(p) not in feat_ids], "lr": lr})
        else:
            groups.append({"params": other_params, "lr": lr})

        self.opt_adam = torch.optim.Adam(groups)

    def _dc_to_torch(self, dataset):
        X = torch.from_numpy(dataset.X).float().to(self.device)
        y = torch.from_numpy(dataset.y).float().to(self.device)
        if y.ndim == 2:
            y = y.squeeze(-1)
        return X, y

    def train(self):
        self.model.train()
        self.likelihood.train()

        # Warmup: freeze feature extractor if present
        if self.warmup_iters > 0:
            self._set_feature_extractor_grad(False)

        for i in range(1, self.num_iters + 1):
            self.opt_ngd.zero_grad()
            self.opt_adam.zero_grad()

            output = self.gp(self.train_x)

            # --- INJECT WEIGHTS HERE ---
            if self.use_weights:
                loss = -self.mll(output, self.train_y, weights=self.train_w)
            else:
                loss = -self.mll(output, self.train_y)
            loss.backward()

            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.opt_ngd.step()
            self.opt_adam.step()

            # Unfreeze after warmup
            if i == self.warmup_iters and self.warmup_iters > 0:
                self._set_feature_extractor_grad(True)

            if i % self.log_interval == 0 or i == self.num_iters:
                self._log_step(i, loss.item())

    def _set_feature_extractor_grad(self, requires_grad: bool):
        try:
            # Basic attempt to find params, adjust based on exact kernel structure
            if hasattr(self.gp.covar_module, "base_kernel"):
                for p in self.gp.covar_module.base_kernel.feature_extractor.parameters():
                    p.requires_grad_(requires_grad)
        except:
            pass

    def _log_step(self, i, loss_val):
        try:
            ls = self.gp.covar_module.base_kernel.lengthscale.mean().item()
            os = self.gp.covar_module.outputscale.item()
        except:
            ls, os = 0.0, 0.0
        print(f"[Iter {i:03d}] Loss: {loss_val:.3f} | LS: {ls:.3f} | OS: {os:.3f}")

    def evaluate(self, dataset: dc.data.NumpyDataset, use_weights: bool = True):
        """
        Evaluate and return AUC and Accuracy.
        Supports weighted metrics if dataset.w is present and use_weights=True.
        """
        self.model.eval()
        self.likelihood.eval()

        X, y_true_t = self._dc_to_torch(dataset)

        with torch.no_grad():
            # Model forward accepts raw X
            probs_t = self.model(X)  # (N, 1)

        y_true = dataset.y.flatten()
        probs = probs_t.cpu().numpy().flatten()

        # Handle Weights
        weights = None
        if use_weights and dataset.w is not None:
            weights = dataset.w.flatten()
            # Basic safety: ensure no negative weights
            weights = np.clip(weights, a_min=0.0, a_max=None)

        # 1. AUC (Weighted)
        try:
            auc = roc_auc_score(y_true, probs, sample_weight=weights)
        except ValueError:
            # Can happen if only one class is present in y_true
            auc = 0.5

            # 2. ROC Curve Data (Weighted)
        # Returns: fpr, tpr, thresholds
        fpr, tpr, thresholds = roc_curve(y_true, probs, sample_weight=weights)

        # 3. Accuracy (Weighted)
        preds_cls = (probs > 0.5).astype(int)
        acc = accuracy_score(y_true, preds_cls, sample_weight=weights)

        print(f"Test Set | AUC: {auc:.4f} | Accuracy: {acc:.4f} | Weighted: {weights is not None}")

        return {
            "auc": auc,
            "acc": acc,
            "roc_data": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds},
            "probs": probs,
            "y_true": y_true
        }


class EnsembleGPTrainer:
    """
    Train multiple GP models (ExactGP or SVGP) potentially on different datasets.
    Aggregates predictions as an equal-weight mixture of Gaussians using moment matching.
    """
    def __init__(self, items, device: str = "cpu"):
        """
        items: list of dicts, each like
          {
            "model": <GPyTorchRegressor>,
            "train_dataset": <dc.data.NumpyDataset>,
            # optional per-model overrides:
            "lr": 0.1,
            "num_iters": 100,
            "log_interval": 10,
          }
        """
        self.device = device
        self.trainers = []
        for k, it in enumerate(items):
            trainer = GPTrainer(
                model=it["model"],
                train_dataset=it["train_dataset"],
                lr=it.get("lr", 0.1),
                num_iters=it.get("num_iters", 100),
                device=device,
                log_interval=it.get("log_interval", 10),
                nn_lr=it.get("nn_lr", 1e-3),
                ngd_lr=it.get("ngd_lr", 0.02),
                warmup_iters=it.get("warmup_iters", 0),
                clip_grad=it.get("clip_grad", 1.0),
            )
            self.trainers.append(trainer)

    def train(self):
        for i, t in enumerate(self.trainers, 1):
            print(f"[Ensemble] Training model {i}/{len(self.trainers)}")
            t.train()

    def _dc_to_torch(self, dataset: dc.data.NumpyDataset):
        X = torch.from_numpy(dataset.X).float().to(self.device)
        y = torch.from_numpy(dataset.y).float().to(self.device)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(-1)
        return X, y

    # ----- store / normalize weights ----------------------------------------
    def set_weights(self, w):
        w = np.asarray(w, dtype=float)
        w = np.maximum(w, 0)
        w = w / (w.sum() + 1e-12)
        self.weights_ = w

    # ----- collect per-model mean/var on ORIGINAL y-scale --------------------
    def _collect_means_vars(self, dataset):
        X, _ = self._dc_to_torch(dataset)
        Ms, Vs = [], []
        for t in self.trainers:
            m, v = self._component_mean_var(t.model, X)
            Ms.append(m.detach().cpu().numpy().reshape(-1))
            Vs.append(v.detach().cpu().numpy().reshape(-1))
        M = np.stack(Ms, axis=0)  # (K, N)
        V = np.clip(np.stack(Vs, axis=0), 1e-12, None)  # (K, N)
        y = dataset.y.reshape(-1) if getattr(dataset, "y", None) is not None else None
        return M, V, y

    # ----- moment-matched Gaussian for arbitrary weights ---------------------
    @staticmethod
    def _mixture_moment_match(M, V, w):
        w = np.asarray(w, dtype=float)
        mu = (w[:, None] * M).sum(axis=0)
        second = (w[:, None] * (V + M ** 2)).sum(axis=0)
        var = np.maximum(second - mu ** 2, 1e-12)
        return mu, var

    # ===== Strategy 1: Precision-only =======================================
    @staticmethod
    def _precision_weights(V, tau=1.0):
        prec = (1.0 / np.clip(V, 1e-12, None)).mean(axis=1)  # (K,)
        w = prec ** float(tau)
        return w / (w.sum() + 1e-12)

    @staticmethod
    def _precision_mse_hybrid(M, V, y, tau=1.0, lam=1.0):
        prec = (1.0 / np.clip(V, 1e-12, None)).mean(axis=1)  # (K,)
        mse = ((M - y[None, :]) ** 2).mean(axis=1)  # (K,)
        score = prec / (1.0 + float(lam) * mse)
        score = np.maximum(score, 1e-12)
        return score / score.sum()

    # ===== Strategy 2: Log-likelihood stacking (proper) ======================
    @staticmethod
    def _fit_nll_weights(M, V, y, l2=1e-3, dirichlet_alpha=1.0,
                         max_iters=200, tol=1e-7):

        M = torch.as_tensor(M, dtype=torch.double)
        V = torch.as_tensor(np.clip(V, 1e-12, None), dtype=torch.double)
        y = torch.as_tensor(y, dtype=torch.double)

        K, N = M.shape
        logits = torch.nn.Parameter(torch.zeros(K, dtype=torch.double))
        opt = torch.optim.LBFGS([logits], lr=1.0, max_iter=60,
                                line_search_fn='strong_wolfe')
        eps = 1e-12

        def closure():
            opt.zero_grad()
            w = torch.softmax(logits, dim=0)  # (K,)
            log_norm = -0.5 * (torch.log(2 * torch.pi * (V + eps))
                               + (y[None, :] - M) ** 2 / (V + eps))  # (K,N)
            log_mix = torch.logsumexp(torch.log(w + eps)[:, None] + log_norm, dim=0)
            nll = -log_mix.mean()
            reg = 0.5 * l2 * (logits ** 2).sum() / max(N, 1)
            if dirichlet_alpha != 1.0:
                reg = reg - (dirichlet_alpha - 1.0) * torch.log(
                    torch.clamp(torch.softmax(logits, 0), min=eps)
                ).sum() / max(N, 1)
            loss = nll + reg
            loss.backward()
            return loss

        prev = float('inf')
        for _ in range(max_iters):
            loss = opt.step(closure)
            if abs(prev - loss.item()) < tol: break
            prev = loss.item()

        with torch.no_grad():
            w = torch.softmax(logits, dim=0).cpu().numpy()
        return w

    # ===== Strategy 3: Flexible validation-based =============================
    # (a) Mean-only MSE stacking
    @staticmethod
    def _fit_mse_weights(M, y, l2=1e-3):

        M = torch.as_tensor(M, dtype=torch.double)
        y = torch.as_tensor(y, dtype=torch.double)
        K, N = M.shape

        # center to avoid explicit intercept
        M_c = M - M.mean(dim=1, keepdim=True)
        y_mu = y.mean()

        logits = torch.nn.Parameter(torch.zeros(K, dtype=torch.double))
        opt = torch.optim.LBFGS([logits], lr=1.0, max_iter=60,
                                line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            w = torch.softmax(logits, dim=0)
            yhat = (w[:, None] * M_c).sum(dim=0) + y_mu
            loss = ((yhat - y) ** 2).mean() + 0.5 * l2 * (logits ** 2).sum() / max(N, 1)
            loss.backward()
            return loss

        prev = float('inf')
        for _ in range(200):
            loss = opt.step(closure)
            if abs(prev - loss.item()) < 1e-7: break
            prev = loss.item()
        with torch.no_grad():
            w = torch.softmax(logits, dim=0).cpu().numpy()
        return w

    # (b) CRPS stacking (probabilistic) using moment-matched Gaussian
    @staticmethod
    def _fit_crps_weights(M, V, y, l2=1e-3):
        import math
        M = torch.as_tensor(M, dtype=torch.double)
        V = torch.as_tensor(np.clip(V, 1e-12, None), dtype=torch.double)
        y = torch.as_tensor(y, dtype=torch.double)
        K, N = M.shape

        logits = torch.nn.Parameter(torch.zeros(K, dtype=torch.double))
        opt = torch.optim.LBFGS([logits], lr=1.0, max_iter=60,
                                line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            w = torch.softmax(logits, dim=0)  # (K,)
            mu = (w[:, None] * M).sum(dim=0)  # (N,)
            second = (w[:, None] * (V + M ** 2)).sum(dim=0)
            var = torch.clamp(second - mu ** 2, min=1e-12)
            std = torch.sqrt(var)
            z = (y - mu) / std

            Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
            crps = std * (z * (2 * Phi - 1.0) + 2 * phi - 1.0 / math.sqrt(math.pi))
            loss = crps.mean() + 0.5 * l2 * (logits ** 2).sum() / max(N, 1)
            loss.backward()
            return loss

        prev = float('inf')
        for _ in range(200):
            loss = opt.step(closure)
            if abs(prev - loss.item()) < 1e-7: break
            prev = loss.item()
        with torch.no_grad():
            w = torch.softmax(logits, dim=0).cpu().numpy()
        return w

    # Public API: learn weights on a calibration/validation set
    def calibrate_weights(self, calib_dc, method="nll", **kwargs):
        """
        method ∈ {"precision", "precision_hybrid", "nll", "mse", "crps"}
        - precision:           label-free; w ∝ mean_i (1/var_{k,i})^tau
        - precision_hybrid:    uses y to downweight overconfident-but-wrong models
        - nll:                 proper probabilistic stacking (mixture log-likelihood)
        - mse:                 mean-only stacking (minimize MSE)
        - crps:                probabilistic, minimize Gaussian CRPS of moment-matched mixture
        """
        M, V, y = self._collect_means_vars(calib_dc)

        if method == "precision":
            w = self._precision_weights(V, tau=kwargs.get("tau", 1.0))
        elif method == "precision_hybrid":
            if y is None: raise ValueError("precision_hybrid requires labels (y).")
            w = self._precision_mse_hybrid(M, V, y,
                                           tau=kwargs.get("tau", 1.0),
                                           lam=kwargs.get("lam", 1.0))
        elif method == "nll":
            if y is None: raise ValueError("nll stacking requires labels (y).")
            w = self._fit_nll_weights(M, V, y,
                                      l2=kwargs.get("l2", 1e-3),
                                      dirichlet_alpha=kwargs.get("dirichlet_alpha", 1.0),
                                      max_iters=kwargs.get("max_iters", 200),
                                      tol=kwargs.get("tol", 1e-7))
        elif method == "mse":
            if y is None: raise ValueError("mse stacking requires labels (y).")
            w = self._fit_mse_weights(M, y, l2=kwargs.get("l2", 1e-3))
        elif method == "crps":
            if y is None: raise ValueError("crps stacking requires labels (y).")
            w = self._fit_crps_weights(M, V, y, l2=kwargs.get("l2", 1e-3))
        else:
            raise ValueError(f"Unknown method: {method}")

        self.set_weights(w)
        return self.weights_

    # Weighted predictions / metrics (fallback to uniform if no weights set)
    def _get_weights_or_uniform(self):
        if getattr(self, "weights_", None) is None:
            K = len(self.trainers)
            return np.ones(K) / K
        return self.weights_

    def predict_mixture_moments_w(self, dataset, w=None):
        M, V, _ = self._collect_means_vars(dataset)
        w = self._get_weights_or_uniform() if w is None else np.asarray(w, dtype=float)
        return self._mixture_moment_match(M, V, w)

    def predict_interval_w(self, dataset, alpha=0.05, w=None):
        mu, var = self.predict_mixture_moments_w(dataset, w=w)
        std = np.sqrt(var + 1e-12)
        if alpha == 0.05:
            z = 1.959963984540054  # high-precision 97.5% quantile
        else:
            import math
            # inverse CDF via erfinv to avoid SciPy
            from math import sqrt
            from numpy import erfinv
            z = sqrt(2) * float(erfinv(1 - 2 * alpha))
        lower, upper = mu - z * std, mu + z * std
        return mu, lower, upper

    def evaluate_mse_w(self, dataset, w=None):
        mu, _ = self.predict_mixture_moments_w(dataset, w=w)
        y = dataset.y.reshape(-1)
        return float(np.mean((mu - y) ** 2))

    @staticmethod
    def _get_y_stats(model, device):
        # y_std/y_mean may be registered buffers in your wrapper; fall back to (1,0)
        y_std = getattr(model, "y_std", torch.tensor(1.0, device=device))
        y_mean = getattr(model, "y_mean", torch.tensor(0.0, device=device))
        # Ensure shapes are broadcastable
        if y_std.ndim == 0: y_std = y_std.view(1)
        if y_mean.ndim == 0: y_mean = y_mean.view(1)
        return y_mean, y_std

    def _component_mean_var(self, model, X: torch.Tensor):
        """
        Returns mean and variance in ORIGINAL y units for a single model.
        Works for ExactGP and SVGP. Relies on model.gp_model normalizing X internally.
        """
        model.eval()
        model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            dist = model.likelihood(model.gp_model(X))  # dist is in normalized y-space if model normalizes y
            mean_n = dist.mean
            var_n  = dist.variance

        # De-normalize using model's y stats (if any)
        y_mean, y_std = self._get_y_stats(model, X.device)
        mean = mean_n * y_std + y_mean
        var  = var_n  * (y_std ** 2)
        return mean, var  # tensors shape (N,)

    # ---------- evaluation / prediction ----------
    def evaluate_mse(self, dataset: dc.data.NumpyDataset) -> float:
        """
        Compute MSE between mixture mean and true y (original units).
        """
        X, y_true = self._dc_to_torch(dataset)

        means = []
        for t in self.trainers:
            m, _ = self._component_mean_var(t.model, X)
            means.append(m)

        mean_mix = torch.stack(means, dim=0).mean(dim=0)  # (N,)
        mse = torch.mean((mean_mix - y_true) ** 2).item()
        return mse

    def predict_mean(self, dataset: dc.data.NumpyDataset) -> np.ndarray:
        """
        Mixture mean (equal weight), original units.
        """
        X, _ = self._dc_to_torch(dataset)
        means = []
        for t in self.trainers:
            m, _ = self._component_mean_var(t.model, X)
            means.append(m)
        mean_mix = torch.stack(means, dim=0).mean(dim=0)  # (N,)
        return mean_mix.detach().cpu().numpy()

    def predict_mixture_moments(self, dataset: dc.data.NumpyDataset):
        """
        Return (mean_mix, var_mix) for the equal-weight Gaussian mixture,
        using moment matching. Original units.
        """
        X, _ = self._dc_to_torch(dataset)

        comp_means = []
        comp_vars  = []
        for t in self.trainers:
            m, v = self._component_mean_var(t.model, X)
            comp_means.append(m)
            comp_vars.append(v)

        M = len(comp_means)
        means = torch.stack(comp_means, dim=0)  # (M, N)
        vars_ = torch.stack(comp_vars,  dim=0)  # (M, N)

        mean_mix = means.mean(dim=0)                           # (N,)
        second_moment = (vars_ + means**2).mean(dim=0)         # E[σ^2 + μ^2]
        var_mix = second_moment - mean_mix**2                  # Var law

        return (mean_mix.detach().cpu().numpy(),
                var_mix.clamp_min(0.0).detach().cpu().numpy())

    def predict_interval(self, dataset: dc.data.NumpyDataset, alpha: float = 0.05, method: str = "moment"):
        """
        (mean, lower, upper) for the mixture.
        method="moment": Gaussian approx using matched mean/var (fast).
        """
        mean_mix, var_mix = self.predict_mixture_moments(dataset)
        std_mix = np.sqrt(var_mix + 1e-12)
        if alpha == 0.05:
            z = 1.96
        else:
            import scipy.stats as st
            z = st.norm.ppf(1 - alpha/2.0)
        lower = mean_mix - z * std_mix
        upper = mean_mix + z * std_mix
        return mean_mix, lower, upper
