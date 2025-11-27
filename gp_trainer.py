# gp_trainer.py
import torch
import gpytorch
import deepchem as dc
import numpy as np


class GPTrainer:
    def __init__(
        self,
        model,
        train_dataset: dc.data.NumpyDataset,
        lr: float = 0.1,
        num_iters: int = 100,
        device: str = "cpu",
        log_interval: int = 10,
    ):
        self.model = model.to(device)
        self.device = device
        self.num_iters = num_iters
        self.log_interval = log_interval

        self.likelihood = self.model.likelihood
        self.gp = self.model.gp_model

        self.train_x, self.train_y = self._dc_to_torch(train_dataset)

        if hasattr(self.model, "train_y_norm"):
            self.train_y = self.model.train_y_norm

        # After self.gp is set
        if hasattr(self.gp, "variational_strategy"):
            # SVGP path
            self.mll = gpytorch.mlls.VariationalELBO(
                self.likelihood, self.gp, num_data=self.train_x.size(0)
            )
        else:
            # Exact GP path
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

    def train(self):
        self.model.train()
        self.likelihood.train()

        for i in range(1, self.num_iters + 1):
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

                print(f"[Iter {i:03d}] Loss: {loss.item():.4f} | {ls_str} | {os_str} | {noise_str}")

    def evaluate_mse(self, dataset: dc.data.NumpyDataset) -> float:
        self.model.eval()
        self.likelihood.eval()

        X, y_true = self._dc_to_torch(dataset)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean = self.model(X).squeeze(-1)  # (N,)

        mse = torch.mean((mean - y_true) ** 2).item()
        return mse

    def predict_mean(self, dataset: dc.data.NumpyDataset) -> np.ndarray:
        self.model.eval()
        self.likelihood.eval()

        X, _ = self._dc_to_torch(dataset)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean = self.model(X).squeeze(-1)
        return mean.cpu().numpy()

    def predict_interval(self, dataset: dc.data.NumpyDataset, alpha: float = 0.05):
        self.model.eval()
        self.likelihood.eval()

        X, _ = self._dc_to_torch(dataset)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean, lower, upper = self.model.predict_interval(X, alpha=alpha)

        return (
            mean.cpu().numpy(),
            lower.cpu().numpy(),
            upper.cpu().numpy(),
        )


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
