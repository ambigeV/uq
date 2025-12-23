# gp_single.py
from gp_multi import MultitaskExactGPModel
import torch
import torch.nn as nn
import gpytorch
from gpytorch.variational import (
    CholeskyVariationalDistribution, VariationalStrategy
)


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
        kernel: str = "matern52",  # <- NEW: "rbf" | "matern32" | "matern52"
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
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        # )
        kernel = kernel.lower()
        if kernel == "rbf":
            base = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        elif kernel == "matern32":
            base = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=input_dim)
        elif kernel == "matern52":
            base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
        else:
            raise ValueError(f"Unknown kernel '{kernel}'. Use 'rbf' | 'matern32' | 'matern52'.")

        self.covar_module = gpytorch.kernels.ScaleKernel(base)

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


class FeatureNet(nn.Module):
    """
    R^Din -> R^Dfeat via MLP + final BatchNorm1d (affine=True).
    BatchNorm stabilizes feature scale; we therefore DO NOT z-normalize X anywhere.
    """
    def __init__(self, in_dim, feat_dim=128, hidden=(256, 128), dropout=0.0):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, feat_dim)]
        self.backbone = nn.Sequential(*layers)
        self.post_bn  = nn.BatchNorm1d(feat_dim, affine=True)   # [MODIFIED] LayerNorm -> BatchNorm1d

    def forward(self, x):
        z = self.backbone(x)
        return self.post_bn(z)                                  # [MODIFIED] apply BN here


class DeepFeatureKernel(gpytorch.kernels.Kernel):
    """
    k_deep(x1, x2) = k_base( φ(x1), φ(x2) ).
    NOTE: No X normalization here—BatchNorm in FeatureNet handles scale.
    """
    is_stationary = False

    def __init__(self, feature_extractor: nn.Module, base_kernel: gpytorch.kernels.Kernel,
                 x_mean=None, x_std=None, normalize_x: bool = False):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.base_kernel = base_kernel
        # [MODIFIED] accept x_mean/x_std/normalize_x for compatibility but ignore them

    def _to_feat(self, x):
        # [MODIFIED] identity "normalization" — BN lives inside FeatureNet
        return self.feature_extractor(x)

    def forward(self, x1, x2, diag=False, **params):
        z1 = self._to_feat(x1)
        z2 = self._to_feat(x2)
        return self.base_kernel(z1, z2, diag=diag, **params)


class NNGPExactGPModel(gpytorch.models.ExactGP):
    """
    Exact GP on neural features; no X z-norm. Wrapper may pass x_mean/x_std—ignored.
    """
    def __init__(self, train_x, train_y, likelihood,
                 x_mean=None, x_std=None, normalize_x: bool = False,   # [MODIFIED] kept but unused
                 feature_extractor: nn.Module = None,
                 kernel: str = "matern52"):
        super().__init__(train_x, train_y, likelihood)

        # infer feature dimension once on raw X (BatchNorm handles scaling)
        with torch.no_grad():
            feat_dim = feature_extractor(train_x[:min(2048, train_x.size(0))]).shape[-1]

        k = kernel.lower()
        if   k == "rbf":      base = gpytorch.kernels.RBFKernel(ard_num_dims=feat_dim)
        elif k == "matern32": base = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=feat_dim)
        elif k == "matern52": base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=feat_dim)
        else: raise ValueError("kernel must be 'rbf' | 'matern32' | 'matern52'")

        self.mean_module = gpytorch.means.ConstantMean()
        deep = DeepFeatureKernel(feature_extractor, base,
                                 x_mean=None, x_std=None, normalize_x=False)
        self.covar_module = gpytorch.kernels.ScaleKernel(deep)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class NNSVGPLearnedInducing(gpytorch.models.ApproximateGP):
    """
    Inducing *inputs* initialized from a random subset of raw TRAIN X, then optimized.
    No X z-norm (BatchNorm in FeatureNet already stabilizes features).
    """
    def __init__(self, train_x, train_y, likelihood,
                 x_mean=None, x_std=None, normalize_x: bool = False,   # [MODIFIED] kept but unused
                 feature_extractor: nn.Module = None,
                 num_inducing: int = 800,
                 kernel: str = "matern52",
                 inducing_idx: torch.Tensor = None):

        N, device = train_x.size(0), train_x.device
        if inducing_idx is None:
            inducing_idx = torch.randperm(N, device=device)[:num_inducing]
        inducing_inputs = train_x[inducing_idx].contiguous()    # raw-X init

        with torch.no_grad():
            feat_dim = feature_extractor(train_x[:min(2048, N)]).shape[-1]

        k = kernel.lower()
        if   k == "rbf":      base = gpytorch.kernels.RBFKernel(ard_num_dims=feat_dim)
        elif k == "matern32": base = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=feat_dim)
        elif k == "matern52": base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=feat_dim)
        else: raise ValueError("kernel must be 'rbf' | 'matern32' | 'matern52'")

        q  = CholeskyVariationalDistribution(inducing_inputs.size(0))
        vs = VariationalStrategy(self, inducing_points=inducing_inputs,
                                 variational_distribution=q,
                                 learn_inducing_locations=True)   # learned inputs
        # vs = WhitenedVariationalStrategy(vs)
        super().__init__(vs)

        self.mean_module = gpytorch.means.ConstantMean()
        deep = DeepFeatureKernel(feature_extractor, base,
                                 x_mean=None, x_std=None, normalize_x=False)  # [MODIFIED] no norm
        self.covar_module = gpytorch.kernels.ScaleKernel(deep)
        self.likelihood = likelihood

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class SVGPClassificationModel(gpytorch.models.ApproximateGP):
    """
    Stochastic Variational GP for Classification.
    Handles internal X-normalization and Inducing Point initialization.
    """

    def __init__(
            self,
            train_x,
            train_y,  # Not strictly used by ApproximateGP, but kept for API consistency
            likelihood,
            x_mean: torch.Tensor,
            x_std: torch.Tensor,
            normalize_x: bool,
            num_inducing: int = 512,
            kmeans_iters: int = 15,
            kernel: str = "matern52",
    ):
        self.normalize_x = normalize_x

        # 1. Normalize train_x locally just for initialization (K-Means)
        if self.normalize_x:
            X_n = (train_x - x_mean) / x_std
        else:
            X_n = train_x

        # 2. Initialize Inducing Points (Centroids of the NORMALIZED space)
        if X_n.shape[0] <= num_inducing:
            inducing_points = X_n.clone()
        else:
            inducing_points = self._kmeans_medoid_init(
                X_n,
                k=num_inducing,
                num_iters=kmeans_iters,
            )

        # 3. Variational Distribution & Strategy
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

        # 4. Store normalization stats
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)

        # 5. Kernel Setup
        input_dim = train_x.shape[-1]
        self.mean_module = gpytorch.means.ConstantMean()

        kernel = kernel.lower()
        if kernel == "rbf":
            base = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        elif kernel == "matern32":
            base = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=input_dim)
        elif kernel == "matern52":
            base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
        else:
            raise ValueError(f"Unknown kernel '{kernel}'.")

        self.covar_module = gpytorch.kernels.ScaleKernel(base)

    @staticmethod
    def _kmeans_medoid_init(X: torch.Tensor, k: int, num_iters: int = 15) -> torch.Tensor:
        """K-means initialization snapping to nearest data points (Medoids)."""
        N, D = X.shape
        device = X.device

        # Init randomly
        perm = torch.randperm(N, device=device)
        centers = X[perm[:k]].clone()

        for _ in range(num_iters):
            # Distance (N, k)
            dist2 = torch.cdist(X, centers, p=2.0) ** 2
            labels = dist2.argmin(dim=1)

            # Update centers
            for j in range(k):
                mask = (labels == j)
                if mask.any():
                    centers[j] = X[mask].mean(dim=0)

        # Snap to nearest actual point (Medoid)
        dist2_ck = torch.cdist(centers, X, p=2.0) ** 2
        nearest_idx = dist2_ck.argmin(dim=1)
        return X[nearest_idx]

    def _norm_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_x:
            return (x - self.x_mean) / self.x_std
        return x

    def forward(self, x):
        # 1. Normalize Raw X
        x_n = self._norm_x(x)
        # 2. Compute Latent Distribution
        mean_x = self.mean_module(x_n)
        covar_x = self.covar_module(x_n)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPyTorchClassifier(nn.Module):
    """
    Wrapper for Binary Classification.
    Computes global stats, initializes SVGPClassificationModel, and handles probability output.
    """

    def __init__(self,
                 train_x,
                 train_y,
                 normalize_x: str = "auto",
                 eps: float = 1e-8,
                 gp_model_cls=SVGPClassificationModel,  # Default to the new class
                 likelihood=None,
                 **gp_model_kwargs,
                 ):
        super().__init__()

        # Buffers for raw data
        self.register_buffer("train_x_raw", train_x)
        self.register_buffer("train_y_raw", train_y)

        # Decide X normalization
        if normalize_x == "auto":
            # Assume ECFP (binary) -> skip norm
            is_binary = torch.all((train_x == 0) | (train_x == 1)).item()
            do_norm_x = not is_binary
        else:
            do_norm_x = bool(normalize_x)
        self.normalize_x = do_norm_x

        # Calculate Stats (on raw data)
        if self.normalize_x:
            x_mean = train_x.mean(dim=0, keepdim=True)
            x_std = train_x.std(dim=0, keepdim=True) + eps
        else:
            x_mean = torch.zeros(1, train_x.shape[-1], device=train_x.device)
            x_std = torch.ones(1, train_x.shape[-1], device=train_x.device)

        # Don't register buffers here for mean/std, they are passed to the inner GP

        # Setup Likelihood (Bernoulli)
        if likelihood is None:
            likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        self.likelihood = likelihood

        # Initialize Inner GP (It will store x_mean/x_std and use them in forward)
        self.gp_model = gp_model_cls(
            train_x=train_x,  # Pass raw X
            train_y=train_y,
            likelihood=self.likelihood,
            x_mean=x_mean,  # Pass stats
            x_std=x_std,
            normalize_x=self.normalize_x,
            **gp_model_kwargs
        )

    def forward(self, x):
        """
        Returns Probabilities P(y=1|x), shape (N, 1).
        Accepts RAW x.
        """
        # gp_model.forward() handles normalization internally
        latent_dist = self.gp_model(x)
        pred_dist = self.likelihood(latent_dist)
        probs = pred_dist.mean  # Bernoulli mean is P(y=1)

        return probs.unsqueeze(-1)


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
                 train_w=None,
                 use_weights: bool = False,
                 w_min: float = 1e-5,
                 noise_cap: float = 1e4,
                 normalize_noise_by_median: bool = False,
                 normalize_x: str = "auto",
                 eps: float = 1e-8,
                 gp_model_cls=None,
                 likelihood=None,
                 base_noise_norm: float = 1e-2,
                 ratio_clip=(0.05, 20.0),
                 **gp_model_kwargs,
                 ):
        super().__init__()

        # Store raw train tensors as buffers
        self.register_buffer("train_x_raw", train_x)
        self.register_buffer("train_y_raw", train_y)

        # multi-task head
        self.num_tasks = 1
        if train_y.ndim == 2 and train_y.shape[1] > 1:
            self.num_tasks = train_y.shape[1]

        # Decide X normalization
        if normalize_x == "auto":
            # If all features are in {0,1}, assume ECFP -> skip norm
            is_binary = torch.all((train_x == 0) | (train_x == 1)).item()
            do_norm_x = not is_binary
        else:
            do_norm_x = bool(normalize_x)
        self.normalize_x = do_norm_x
        self.eps = eps
        self.use_weights = use_weights

        # Fit X stats from raw train (per-feature)
        if self.normalize_x:
            x_mean = train_x.mean(dim=0, keepdim=True)
            x_std  = train_x.std(dim=0, keepdim=True) + eps  # std requires dim for keepdim
        else:
            x_mean = torch.zeros(1, train_x.shape[-1], device=train_x.device, dtype=train_x.dtype)
            x_std  = torch.ones(1,  train_x.shape[-1], device=train_x.device, dtype=train_x.dtype)

        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)

        # 2. Setup Y stats (Handles K tasks)
        if self.num_tasks > 1:
            # Multitask: Statistics are (1, K)
            y_mean = train_y.mean(dim=0, keepdim=True)
            y_std = train_y.std(dim=0, keepdim=True) + eps
            train_y_norm = (train_y - y_mean) / y_std
        else:
            # Single-task: Flatten to (N,)
            train_y_1d = train_y.squeeze(-1) if train_y.ndim == 2 else train_y
            y_mean = train_y_1d.mean()
            y_std = train_y_1d.std() + eps
            train_y_norm = (train_y_1d - y_mean) / y_std

        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)
        self.register_buffer("train_y_norm", train_y_norm)
        self.register_buffer("train_w", train_w)

        # keep these hyperparams for weight->noise
        self.use_weights = bool(use_weights)
        self.w_min = float(w_min)
        self.noise_cap = float(noise_cap)
        self.normalize_noise_by_median = bool(normalize_noise_by_median)
        self.base_noise_norm = float(base_noise_norm)
        self.ratio_clip = ratio_clip

        fixed_noise_norm = None
        # 3. Instantiate Likelihood & Model
        if likelihood is None:
            if self.num_tasks > 1:
                # Multitask Likelihood (Learns a noise rank, usually diagonal + low rank)
                likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_tasks)
            else:
                # Single-task Likelihood
                if self.use_weights and (train_w is not None):
                    fixed_noise_norm = self._w_to_noise_norm(train_w)
                
                if fixed_noise_norm is not None:
                    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                        noise=fixed_noise_norm, learn_additional_noise=True
                    )
                else:
                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood = likelihood

        # Select Model Class
        if gp_model_cls is None:
            if self.num_tasks > 1:
                gp_model_cls = MultitaskExactGPModel  # <--- Selects Multitask Model
                gp_model_kwargs["num_tasks"] = self.num_tasks
            else:
                gp_model_cls = ExactGPModel

        self.gp_model = gp_model_cls(
            train_x=train_x,
            train_y=self.train_y_norm,
            likelihood=self.likelihood,
            x_mean=self.x_mean,
            x_std=self.x_std,
            normalize_x=self.normalize_x,
            **gp_model_kwargs,  # extra hooks if heterosced model needs them
        )

    def _w_to_noise_norm(self, w: torch.Tensor) -> torch.Tensor:
        """
        w is precision-like (bigger => more reliable).
        We convert to *relative* noise, then scale by base_noise_norm (absolute).
        """
        w = w.view(-1).to(self.train_y_norm.device).float()
        w = torch.clamp(w, min=self.w_min)

        rel = 1.0 / w  # dimensionless relative variance proxy

        if self.normalize_noise_by_median:
            rel = rel / rel.median().clamp_min(1e-12)  # => 1 if all w equal

        lo, hi = self.ratio_clip
        rel = torch.clamp(rel, min=lo, max=hi)

        noise_norm = rel * self.base_noise_norm  # <-- key fix
        return torch.clamp(noise_norm, max=self.noise_cap)

    def forward(self, x, w=None):
        """Return mean prediction in ORIGINAL y units, shape (N, 1)."""
        self.gp_model.eval()
        self.likelihood.eval()

        test_noise = None
        is_fixed_noise = isinstance(self.likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood)

        if self.use_weights and is_fixed_noise and self.num_tasks == 1 and w is not None:
            test_noise = self._w_to_noise_norm(w)

        with torch.no_grad():
            # gp_model.forward() will normalize x internally
            f_dist = self.gp_model(x)

            if test_noise is not None:
                dist = self.likelihood(f_dist, noise=test_noise)
            else:
                dist = self.likelihood(f_dist)

            mean_n = dist.mean
            mean = mean_n * self.y_std + self.y_mean
            if self.num_tasks == 1:
                return mean.unsqueeze(-1)
            return mean

    def predict_interval(self, x, alpha: float = 0.05, w=None):
        """Return (mean, lower, upper) in ORIGINAL y units, each (N,)."""
        self.gp_model.eval()
        self.likelihood.eval()
        is_fixed_noise = isinstance(self.likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood)
        test_noise_norm = None
        if self.use_weights and is_fixed_noise and (w is not None) and self.num_tasks == 1:
            test_noise_norm = self._w_to_noise_norm(w)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_dist = self.gp_model(x)

            if test_noise_norm is not None:
                y_dist = self.likelihood(f_dist, noise=test_noise_norm)
            else:
                y_dist = self.likelihood(f_dist)

            mean_n = y_dist.mean
            std_n = y_dist.stddev

            if alpha == 0.05:
                z = 1.96
            else:
                z = torch.distributions.Normal(0, 1).icdf(
                    torch.tensor(1 - alpha / 2, device=x.device, dtype=x.dtype)
                )

            lower_n = mean_n - z * std_n
            upper_n = mean_n + z * std_n

            mean = mean_n * self.y_std + self.y_mean
            lower = lower_n * self.y_std + self.y_mean
            upper = upper_n * self.y_std + self.y_mean
            return mean, lower, upper
