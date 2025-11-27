import numpy as np
import torch
import torch.nn as nn
import deepchem as dc
from scipy.stats import norm
from data_utils import evaluate_uq_metrics_from_interval


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


class MyTorchRegressorMC(nn.Module):
    """NN with dropout active during inference (MC Dropout)."""
    def __init__(self, n_features: int, n_tasks: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, n_tasks),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 2. Helper Evaluation
# ============================================================

def mse_from_mean_prediction(mean, dataset):
    y_true = dataset.y.reshape(-1)
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

class MCDropoutRegressor:
    """
    Wrapper for MC-Dropout inference.
    Calls model.model.train() to activate dropout at inference.
    """
    def __init__(self, dc_model, n_samples=30):
        self.model = dc_model
        self.n_samples = n_samples

    def predict_samples(self, dataset):
        preds = []

        self.model.model.train()

        for _ in range(self.n_samples):
            p = self.model.predict(dataset)[:, 0]  # (N,)
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

    return dc_model


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

    print("[Deep Ensemble] Validation MSE:", mse_from_mean_prediction(mean_valid, valid_dc))
    print("[Deep Ensemble] Test MSE:",        mse_from_mean_prediction(mean_test,  test_dc))

    uq_metrics = evaluate_uq_metrics_from_interval(
        y_true=test_dc.y,
        mean=mean_test,
        lower=lower_test,
        upper=upper_test,
        alpha=0.05,
    )

    print("UQ (Deep Ensemble):", uq_metrics)

    return ensemble


# ------------------------------
# MC-Dropout
# ------------------------------
def train_nn_mc_dropout(train_dc, valid_dc, test_dc, n_samples=100):
    n_tasks = train_dc.y.shape[1]
    n_features = train_dc.X.shape[1]

    model = MyTorchRegressorMC(n_features, n_tasks)
    loss = dc.models.losses.L2Loss()

    dc_model = dc.models.TorchModel(
        model=model,
        loss=loss,
        output_types=['prediction'],
        batch_size=64,
        learning_rate=1e-3,
        mode='regression',
    )

    dc_model.fit(train_dc, nb_epoch=100)

    mc_model = MCDropoutRegressor(dc_model, n_samples=n_samples)

    # Evaluate using MC mean
    mean_valid, _, _ = mc_model.predict_interval(valid_dc)
    mean_test,  lower_test, upper_test = mc_model.predict_interval(test_dc)

    print("[MC-Dropout] Validation MSE:", mse_from_mean_prediction(mean_valid, valid_dc))
    print("[MC-Dropout] Test MSE:",        mse_from_mean_prediction(mean_test,  test_dc))

    uq_metrics = evaluate_uq_metrics_from_interval(
        y_true=test_dc.y,
        mean=mean_test,
        lower=lower_test,
        upper=upper_test,
        alpha=0.05,
    )

    print("UQ (MC):", uq_metrics)

    return mc_model



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
