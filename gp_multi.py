# gp_multi.py
import torch
import torch.nn as nn
import gpytorch


class MultitaskExactGPModel(gpytorch.models.ExactGP):
    """
    Multitask GP (LCM) that normalizes X internally, supporting K tasks.
    Matches the API of your single-task ExactGPModel.
    """
    def __init__(self, train_x, train_y, likelihood, num_tasks,
                 x_mean: torch.Tensor, x_std: torch.Tensor, normalize_x: bool):
        super().__init__(train_x, train_y, likelihood)

        self.normalize_x = normalize_x
        self.num_tasks = num_tasks
        
        # Register normalization buffers (same as single-task)
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)

        input_dim = train_x.shape[-1]
        
        # 1. Multitask Mean: Wraps a ConstantMean (or others)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), 
            num_tasks=num_tasks
        )
        
        # 2. Multitask Kernel (LCM): Wraps your base kernel
        # rank=1 is standard for LCM (Linear Model of Coregionalization)
        base_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            base_kernel, num_tasks=num_tasks, rank=1
        )

    def _norm_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_x:
            return (x - self.x_mean) / self.x_std
        return x

    def forward(self, x):
        x_n = self._norm_x(x)
        
        mean_x = self.mean_module(x_n)
        covar_x = self.covar_module(x_n)
        
        # Crucial: Must return MultitaskMultivariateNormal for MLL to work
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class MultitaskBernoulliLikelihood(gpytorch.likelihoods.Likelihood):
    def forward(self, function_samples, **kwargs):
        # 1. Probit Link (Normal CDF)
        output_probs = torch.distributions.Normal(0, 1).cdf(function_samples)
        
        # 2. Independent Wrapper (Treats the last dim 'K' as the event dim)
        return torch.distributions.Independent(torch.distributions.Bernoulli(probs=output_probs), 1)
