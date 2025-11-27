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

        # if hasattr(self.model, "train_x_norm") and hasattr(self.model, "train_y_norm"):
        #     self.train_x = self.model.train_x_norm
        #     self.train_y = self.model.train_y_norm
        # keep raw X (model normalizes internally)
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

            if i % self.log_interval == 0 or i == 1 or i == self.num_iters:
                print(f"[Iter {i:03d}] Loss: {loss.item():.4f}")

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
