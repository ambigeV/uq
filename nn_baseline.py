# nn_baseline.py
import deepchem as dc
import torch
import torch.nn as nn


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
