# gp_multi.py
import torch
import torch.nn as nn
import gpytorch


class MultitaskExactGPModel(gpytorch.models.ExactGP):
    """
    You will need to define a multitask kernel here, e.g.,
    using gpytorch.kernels.MultitaskKernel or LMC.
    This is just a placeholder skeleton.
    """
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        input_dim = train_x.shape[-1]
        num_tasks = train_y.shape[-1]

        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            base_kernel,
            num_tasks=num_tasks,
            rank=1,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class GPyTorchMultitaskRegressor(nn.Module):
    """
    - forward(x) -> (N, T)
    - predict_interval(x, alpha) -> (mean, lower, upper), each (N, T)
    """
    def __init__(self, train_x, train_y):
        super().__init__()
        self.register_buffer("train_x", train_x)
        self.register_buffer("train_y", train_y)

        self.num_tasks = train_y.shape[-1]
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks
        )
        self.gp_model = MultitaskExactGPModel(self.train_x, self.train_y, self.likelihood)

    def forward(self, x):
        self.gp_model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            dist = self.likelihood(self.gp_model(x))
            mean = dist.mean  # (N, T)
            return mean

    def predict_interval(self, x, alpha: float = 0.05):
        self.gp_model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            dist = self.likelihood(self.gp_model(x))
            mean = dist.mean
            std = dist.stddev

            if alpha == 0.05:
                z = 1.96
            else:
                z = torch.distributions.Normal(0, 1).icdf(
                    torch.tensor(1 - alpha / 2, device=x.device)
                )

            lower = mean - z * std
            upper = mean + z * std
            return mean, lower, upper
