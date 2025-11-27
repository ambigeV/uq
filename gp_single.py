# gp_single.py
import torch
import torch.nn as nn
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Exact GP that normalizes X internally (optionally).
    We store x_mean/x_std in the model so that *even if*
    the trainer calls self.gp(train_x_raw), we normalize inside forward().
    """
    def __init__(self, train_x, train_y, likelihood,
                 x_mean: torch.Tensor, x_std: torch.Tensor, normalize_x: bool):
        super().__init__(train_x, train_y, likelihood)

        self.normalize_x = normalize_x
        # Keep stats in the model so trainer doesn't need to know.
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)

        input_dim = train_x.shape[-1]
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        )

    def _norm_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_x:
            return (x - self.x_mean) / self.x_std
        return x

    def forward(self, x):
        x_n = self._norm_x(x)
        mean_x = self.mean_module(x_n)
        covar_x = self.covar_module(x_n)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVGPModel(gpytorch.models.ApproximateGP):
    """
    Stochastic Variational GP with the same X-normalization convention as ExactGPModel.
    Inducing points are initialized via k-means, then snapped to the nearest
    training data points (k-medoids-style initialization).
    """
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        normalize_x: bool,
        num_inducing: int = 128,
        kmeans_iters: int = 15,
    ):
        self.normalize_x = normalize_x

        # Normalize train_x for clustering
        if self.normalize_x:
            X_n = (train_x - x_mean) / x_std
        else:
            X_n = train_x

        # If N is small, just use all points as inducing points
        if X_n.shape[0] <= num_inducing:
            inducing_points = X_n.clone()
        else:
            inducing_points = self._kmeans_medoid_init(
                X_n,
                k=num_inducing,
                num_iters=kmeans_iters,
            )

        # Build variational distribution + strategy
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[0]
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy)

        # Store normalization stats as buffers
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)

        input_dim = train_x.shape[-1]
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        )

    @staticmethod
    def _kmeans_medoid_init(X: torch.Tensor, k: int, num_iters: int = 15) -> torch.Tensor:
        """
        K-means on X to get cluster centers, then choose for each cluster
        the nearest training point (medoid) as inducing point.

        X: (N, D) normalized training inputs.
        Returns: (k, D) inducing points chosen from rows of X.
        """
        N, D = X.shape
        device = X.device

        # 1) Standard k-means to get continuous centers
        # Initialize centers by random subset
        perm = torch.randperm(N, device=device)
        centers = X[perm[:k]].clone()  # (k, D)

        for _ in range(num_iters):
            # Distances: (N, k)
            dist2 = torch.cdist(X, centers, p=2.0) ** 2
            labels = dist2.argmin(dim=1)  # (N,)

            # Update centers as mean of assigned points
            for j in range(k):
                mask = (labels == j)
                if mask.any():
                    centers[j] = X[mask].mean(dim=0)

        dist2_ck = torch.cdist(centers, X, p=2.0) ** 2  # (k, N)
        nearest_idx = dist2_ck.argmin(dim=1)  # (k,)
        inducing_points = X[nearest_idx]      # (k, D)

        return inducing_points

    def _norm_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_x:
            return (x - self.x_mean) / self.x_std
        return x

    def forward(self, x):
        x_n = self._norm_x(x)
        mean_x = self.mean_module(x_n)
        covar_x = self.covar_module(x_n)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPyTorchRegressor(nn.Module):
    """
    Single-task GP with *internal X z-norm* (auto-skips for binary ECFP).
    We DO NOT normalize y here to keep compatibility with the unchanged trainer.

    Public API:
      forward(x) -> mean in ORIGINAL y units, shape (N, 1)
      predict_interval(x, alpha) -> (mean, lower, upper) in ORIGINAL y units
    """
    def __init__(self,
                 train_x,
                 train_y,
                 normalize_x: str = "auto",
                 eps: float = 1e-8,
                 gp_model_cls=None,
                 likelihood=None,
                 **gp_model_kwargs,
                 ):
        super().__init__()

        # Store raw train tensors as buffers
        self.register_buffer("train_x_raw", train_x)
        self.register_buffer("train_y_raw", train_y)

        # Decide X normalization
        if normalize_x == "auto":
            # If all features are in {0,1}, assume ECFP -> skip norm
            is_binary = torch.all((train_x == 0) | (train_x == 1)).item()
            do_norm_x = not is_binary
        else:
            do_norm_x = bool(normalize_x)
        self.normalize_x = do_norm_x
        self.eps = eps

        # Fit X stats from raw train (per-feature)
        if self.normalize_x:
            x_mean = train_x.mean(dim=0, keepdim=True)
            x_std  = train_x.std(dim=0, keepdim=True) + eps  # std requires dim for keepdim
        else:
            x_mean = torch.zeros(1, train_x.shape[-1], device=train_x.device, dtype=train_x.dtype)
            x_std  = torch.ones(1,  train_x.shape[-1], device=train_x.device, dtype=train_x.dtype)

        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)

        train_y_1d = train_y.squeeze(-1) if train_y.ndim == 2 else train_y
        y_mean = train_y_1d.mean()
        y_std = train_y_1d.std() + eps
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)

        # Normalized targets for training / MLL
        train_y_norm = (train_y_1d - self.y_mean) / self.y_std
        self.register_buffer("train_y_norm", train_y_norm)
        train_x_norm = (train_x - self.x_mean) / self.x_std if self.normalize_x else train_x
        self.register_buffer("train_x_norm", train_x_norm)

        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood = likelihood
        # self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if gp_model_cls is None:
            gp_model_cls = ExactGPModel  # your current homoscedastic model

        self.gp_model = gp_model_cls(
            train_x=train_x,
            train_y=self.train_y_norm,
            likelihood=self.likelihood,
            x_mean=self.x_mean,
            x_std=self.x_std,
            normalize_x=self.normalize_x,
            **gp_model_kwargs,  # extra hooks if heterosced model needs them
        )

    def forward(self, x):
        """Return mean prediction in ORIGINAL y units, shape (N, 1)."""
        self.gp_model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            # gp_model.forward() will normalize x internally
            dist = self.likelihood(self.gp_model(x))
            mean_n = dist.mean
            mean = mean_n * self.y_std + self.y_mean
            return mean.unsqueeze(-1)

    def predict_interval(self, x, alpha: float = 0.05):
        """Return (mean, lower, upper) in ORIGINAL y units, each (N,)."""
        self.gp_model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            dist = self.likelihood(self.gp_model(x))
            mean_n = dist.mean
            std_n  = dist.stddev

            if alpha == 0.05:
                z = 1.96
            else:
                z = torch.distributions.Normal(0, 1).icdf(
                    torch.tensor(1 - alpha / 2, device=x.device, dtype=x.dtype)
                )

            lower_n = mean_n - z * std_n
            upper_n = mean_n + z * std_n

            # ----- EDIT: de-normalize all outputs -----
            mean = mean_n * self.y_std + self.y_mean
            lower = lower_n * self.y_std + self.y_mean
            upper = upper_n * self.y_std + self.y_mean
            return mean, lower, upper
