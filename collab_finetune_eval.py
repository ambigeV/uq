#!/usr/bin/env python3
"""
collab_finetune_eval.py
========================
Self-contained fine-tune and evaluation script for evidential / MC-dropout
binary classification models originally trained with bii.py.

No project-internal imports — only standard scientific Python.

Requirements:
    pip install torch numpy pandas deepchem scikit-learn

Checkpoint compatibility:
    .pt files saved by bii.py (torch.save) with keys:
        model_type, backend, n_features, n_tasks, ecfp_size, radius,
        encoder_type, model_state_dict

--- Mode: finetune ---
    Load an existing checkpoint, continue training on new labelled data,
    then save an updated checkpoint.

    python collab_finetune_eval.py \\
        --mode finetune \\
        --checkpoint saved_models/my_model.pt \\
        --train_csv new_train.csv \\
        --val_csv   new_val.csv \\
        --smiles_column SMILES --label_column Outcome \\
        --epochs 30 --lr 5e-5 \\
        --output_checkpoint saved_models/my_model_ft.pt

--- Mode: train ---
    Train a fresh model from scratch on provided data, then save.

    python collab_finetune_eval.py \\
        --mode train \\
        --model_type evidential \\
        --train_csv new_train.csv --val_csv new_val.csv \\
        --smiles_column SMILES --label_column Outcome \\
        --ecfp_size 1024 --radius 2 \\
        --epochs 50 \\
        --output_checkpoint saved_models/new_model.pt

--- Mode: eval ---
    Load checkpoint, run inference, write predictions CSV with columns:
        prob_class1_positive, epistemic_uncertainty, aleatoric_uncertainty

    python collab_finetune_eval.py \\
        --mode eval \\
        --checkpoint saved_models/my_model.pt \\
        --test_csv  test_set.csv \\
        --smiles_column SMILES \\
        --label_column Outcome \\
        --output_csv predictions.csv

Output predictions CSV columns:
    prob_class1_positive  — P(positive class)
    epistemic_uncertainty — model/knowledge uncertainty
    aleatoric_uncertainty — data/irreducible uncertainty
    (y_true, pred_label   — appended when --label_column is provided)
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import deepchem as dc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.feat import GraphData
from torch.utils.data import DataLoader, Dataset as TorchDataset

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DMPNN graph components (inlined from nn.py)
# Only needed when encoder_type="dmpnn". Safe to ignore for ECFP models.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MolGraph:
    V: np.ndarray          # (num_atoms, d_v) atom features
    E: np.ndarray          # (num_bonds, d_e) bond features
    edge_index: np.ndarray # (2, num_bonds)
    rev_edge_index: np.ndarray  # (num_bonds,) reverse-edge mapping


class BatchMolGraph:
    def __init__(self, V, E, edge_index, rev_edge_index, batch):
        self.V = V
        self.E = E
        self.edge_index = edge_index
        self.rev_edge_index = rev_edge_index
        self.batch = batch

    @classmethod
    def from_molgraphs(cls, mgs: List[MolGraph]) -> "BatchMolGraph":
        Vs, Es, eis, reis, bis = [], [], [], [], []
        n_nodes = n_edges = 0
        for i, mg in enumerate(mgs):
            Vs.append(mg.V)
            Es.append(mg.E)
            eis.append(mg.edge_index + n_nodes)
            reis.append(mg.rev_edge_index + n_edges)
            bis.append([i] * len(mg.V))
            n_nodes += mg.V.shape[0]
            n_edges += mg.edge_index.shape[1]
        return cls(
            V=torch.from_numpy(np.concatenate(Vs)).float(),
            E=torch.from_numpy(np.concatenate(Es)).float(),
            edge_index=torch.from_numpy(np.hstack(eis)).long(),
            rev_edge_index=torch.from_numpy(np.concatenate(reis)).long(),
            batch=torch.tensor(np.concatenate(bis)).long(),
        )

    def to(self, device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)
        return self


def graphdata_to_batchmolgraph(graph_data_list: List[GraphData]) -> BatchMolGraph:
    mgs = []
    for gd in graph_data_list:
        ei = gd.edge_index
        n_e = ei.shape[1]
        if n_e == 0:
            rev = np.array([], dtype=np.int64)
        else:
            src, dst = ei[0], ei[1]
            sm = src[:, None] == dst[None, :]
            dm = dst[:, None] == src[None, :]
            mask = sm & dm
            has_rev = np.any(mask, axis=1)
            rev = np.where(has_rev, np.argmax(mask, axis=1), np.arange(n_e))
        ef = gd.edge_features if gd.edge_features is not None else np.zeros((n_e, 14), dtype=np.float32)
        mgs.append(MolGraph(V=gd.node_features, E=ef, edge_index=ei, rev_edge_index=rev))
    return BatchMolGraph.from_molgraphs(mgs)


class BondMessagePassing(nn.Module):
    def __init__(self, d_v=133, d_e=14, d_h=300, depth=3, dropout=0.0, bias=True):
        super().__init__()
        self.d_h = d_h
        self.depth = depth
        self.drop = nn.Dropout(dropout)
        self.W_i = nn.Linear(d_v + d_e, d_h, bias=bias)
        self.W_h = nn.Linear(d_h, d_h, bias=bias)
        self.W_o = nn.Linear(d_v + d_h, d_h, bias=bias)

    def forward(self, bmg: BatchMolGraph, V_d=None):
        V, E, ei, rei = bmg.V, bmg.E, bmg.edge_index, bmg.rev_edge_index
        n_nodes, n_edges = V.shape[0], E.shape[0]
        src = ei[0]
        H0 = F.relu(self.W_i(torch.cat([V[src], E], dim=1)))
        H0 = self.drop(H0)
        H = H0
        for _ in range(1, self.depth):
            dst = ei[1]
            agg = torch.zeros(n_nodes, self.d_h, device=H.device, dtype=H.dtype)
            agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.d_h), H)
            M = agg[src]
            idx = torch.arange(n_edges, device=rei.device, dtype=rei.dtype)
            M = M - H[rei] * (rei != idx).unsqueeze(1).float()
            H = F.relu(H0 + self.W_h(M))
            H = self.drop(H)
        M_v = torch.zeros(n_nodes, self.d_h, device=H.device, dtype=H.dtype)
        M_v.scatter_add_(0, src.unsqueeze(1).expand(-1, self.d_h), H)
        Hv = F.relu(self.W_o(torch.cat([V, M_v], dim=1)))
        return self.drop(Hv)


class MeanAggregation(nn.Module):
    def forward(self, Hv, batch):
        B = int(batch.max().item()) + 1
        d = Hv.shape[1]
        out = torch.zeros(B, d, device=Hv.device, dtype=Hv.dtype)
        out.scatter_add_(0, batch.unsqueeze(1).expand(-1, d), Hv)
        cnt = torch.zeros(B, device=batch.device, dtype=torch.long)
        cnt.scatter_add_(0, batch, torch.ones_like(batch))
        return out / cnt.unsqueeze(1).float().clamp(min=1)


class DMPNNEncoder(nn.Module):
    def __init__(self, d_v=133, d_e=14, d_h=300, depth=3, dropout=0.0, batch_norm=False):
        super().__init__()
        self.mp = BondMessagePassing(d_v, d_e, d_h, depth, dropout)
        self.agg = MeanAggregation()
        self.bn = nn.BatchNorm1d(d_h) if batch_norm else nn.Identity()
        self.output_dim = d_h

    def forward(self, bmg: BatchMolGraph, V_d=None, X_d=None):
        Hv = self.mp(bmg, V_d)
        H = self.agg(Hv, bmg.batch)
        return self.bn(H)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Model architecture (inlined from nn_baseline.py)
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedModel(nn.Module):
    """
    Unified architecture for evidential and mc_dropout binary classification.

    encoder_type = "identity" : ECFP vector → identity pass-through
    encoder_type = "dmpnn"    : molecular graph → DMPNNEncoder

    model_type = "evidential"  : 4-tuple output (prob, alpha, aleatoric, epistemic)
    model_type = "mc_dropout"  : single tensor (B, 2*n_tasks) = [mu | log_var]
    """

    def __init__(self, model_type: str, n_tasks: int, classification: bool = True, **kwargs):
        super().__init__()
        self.model_type = model_type
        self.n_tasks = n_tasks
        self.classification = classification
        self.encoder_type: str = "identity"
        self.encoder: Optional[nn.Module] = None
        self.encoder_dim: int = 0
        self.ffn: Optional[nn.Module] = None
        self.feature_net: Optional[nn.Module] = None
        self.out_head: Optional[nn.Module] = None

    def create_encoder(self, n_features: int, encoder_type: str = "identity", **kwargs):
        self.encoder_type = encoder_type
        if encoder_type == "identity":
            self.encoder = nn.Identity()
            self.encoder_dim = n_features
        elif encoder_type == "dmpnn":
            d_h = kwargs.get("encoder_hidden_dim", 300)
            self.encoder = DMPNNEncoder(
                d_v=n_features,
                d_e=kwargs.get("d_e", 14),
                d_h=d_h,
                depth=kwargs.get("encoder_depth", 3),
                dropout=kwargs.get("encoder_dropout", 0.0),
                batch_norm=kwargs.get("encoder_batch_norm", False),
            )
            self.encoder_dim = d_h
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def create_ffn(self, n_features: int, n_tasks: int, **kwargs):
        dropout_rate = kwargs.get("dropout_rate", 0.2)
        first_dim = self.encoder_dim

        if self.model_type == "mc_dropout":
            self.feature_net = nn.Sequential(
                nn.Linear(first_dim, 128), nn.ReLU(), nn.Dropout(p=dropout_rate),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p=dropout_rate),
            )
            self.out_head = nn.Linear(64, 2 * n_tasks)
            return

        if self.model_type == "evidential":
            output_size = 2 * n_tasks  # Dirichlet: alpha for each of 2 classes per task
            self.ffn = nn.Sequential(
                nn.Linear(first_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Linear(64, output_size),
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _encode(self, x):
        if self.encoder_type == "dmpnn":
            if not isinstance(x, BatchMolGraph):
                if isinstance(x, (list, tuple)):
                    x = graphdata_to_batchmolgraph(list(x))
                else:
                    raise TypeError(f"Expected BatchMolGraph for dmpnn encoder, got {type(x)}")
            return self.encoder(x)
        return self.encoder(x)

    def forward(self, x):
        encoded = self._encode(x)

        if self.model_type == "mc_dropout":
            h = self.feature_net(encoded)
            raw = self.out_head(h)           # (B, 2*n_tasks)
            return raw                        # single tensor; used differently at train vs infer

        # evidential
        logits = self.ffn(encoded)            # (B, 2*n_tasks)
        evidence = torch.exp(logits)
        alpha = evidence + 1.0
        B = logits.shape[0]
        alpha_r = alpha.view(B, self.n_tasks, 2)
        S = alpha_r.sum(dim=2, keepdim=True)
        prob = alpha_r / S                    # (B, n_tasks, 2)
        epistemic = 2.0 / S                   # (B, n_tasks, 1)
        aleatoric = -(prob * torch.log(prob + 1e-8)).sum(dim=2, keepdim=True)

        return (
            prob.view(B, 2 * self.n_tasks),   # output[0] — positive prob at col [1]
            alpha,                             # output[1] — for loss
            aleatoric.view(B, self.n_tasks),   # output[2]
            epistemic.view(B, self.n_tasks),   # output[3]
        )


def build_model(
    model_type: str,
    n_features: int,
    n_tasks: int,
    encoder_type: str = "identity",
    dropout_rate: float = 0.2,
) -> UnifiedModel:
    m = UnifiedModel(model_type=model_type, n_tasks=n_tasks, classification=True)
    m.create_encoder(n_features, encoder_type=encoder_type)
    m.create_ffn(n_features, n_tasks, dropout_rate=dropout_rate)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Loss functions (standalone PyTorch, no DeepChem dependency)
# ─────────────────────────────────────────────────────────────────────────────

def evidential_clf_loss(
    alpha: torch.Tensor,   # (B, 2) dirichlet params for single-task binary
    labels: torch.Tensor,  # (B, 1) in {0, 1}
    epoch: int = 0,
    annealing_step: int = 10,
) -> torch.Tensor:
    """Deep evidential classification loss (Dirichlet / single-task mode)."""
    S = alpha.sum(dim=1, keepdim=True)                          # (B, 1)
    y_oh = F.one_hot(labels.long().squeeze(-1), num_classes=2).float()  # (B, 2)
    pred_prob = alpha / S
    loss_mse = ((y_oh - pred_prob) ** 2).sum(dim=1)
    loss_var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=1)
    base = loss_mse + loss_var

    ann = min(1.0, epoch / max(annealing_step, 1))
    kl_alpha = labels * alpha[:, :1] + (1.0 - labels) * 1.0
    kl_beta  = labels * 1.0          + (1.0 - labels) * alpha[:, 1:]
    sum_p = kl_alpha + kl_beta
    kl = (
        torch.lgamma(sum_p) - torch.lgamma(kl_alpha) - torch.lgamma(kl_beta)
        + (kl_alpha - 1) * (torch.digamma(kl_alpha) - torch.digamma(sum_p))
        + (kl_beta  - 1) * (torch.digamma(kl_beta)  - torch.digamma(sum_p))
    ).squeeze(-1)
    return base + ann * kl


def mc_dropout_clf_loss(
    output: torch.Tensor,   # (B, 2): [mu | log_var] for single-task binary
    labels: torch.Tensor,   # (B, 1)
    n_samples: int = 20,
    clamp: Tuple[float, float] = (-10.0, 10.0),
) -> torch.Tensor:
    """Heteroscedastic classification loss via MC sampling."""
    n_tasks = output.shape[-1] // 2
    mu      = output[..., :n_tasks]   # (B, n_tasks)
    log_var = output[..., n_tasks:].clamp(*clamp)
    std     = torch.exp(0.5 * log_var)

    T = n_samples
    eps = torch.randn((T,) + mu.shape, device=mu.device, dtype=mu.dtype)
    logits_s = mu.unsqueeze(0) + std.unsqueeze(0) * eps          # (T, B, n_tasks)
    tgt = labels.unsqueeze(0).expand(T, -1, -1)                  # (T, B, n_tasks)
    log_p = -F.binary_cross_entropy_with_logits(logits_s, tgt, reduction="none")
    log_exp = torch.logsumexp(log_p, dim=0) - math.log(T)
    return -log_exp                                                # (B, n_tasks)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Data loading and featurization
# ─────────────────────────────────────────────────────────────────────────────

def _safe_load_file(path: str) -> pd.DataFrame:
    if path.endswith(".pkl"):
        try:
            obj = pd.read_pickle(path)
        except Exception:
            with open(path, "rb") as f:
                obj = pickle.load(f, encoding="latin1")
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"Expected DataFrame in {path}")
        return obj
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}. Use .csv or .pkl")


def featurize_smiles(
    smiles: List[str],
    encoder_type: str = "identity",
    ecfp_size: int = 1024,
    radius: int = 2,
) -> Tuple[Any, List[int]]:
    """
    Returns (features, valid_indices).

    For identity: features is np.ndarray float32 (N_valid, ecfp_size).
    For dmpnn:    features is List[GraphData] of length N_valid.
    """
    if encoder_type == "dmpnn":
        featurizer = dc.feat.DMPNNFeaturizer()
    else:
        featurizer = dc.feat.CircularFingerprint(size=ecfp_size, radius=radius)

    raw = featurizer.featurize(smiles)
    valid = [i for i, f in enumerate(raw) if f is not None and (
        f.size > 0 if isinstance(f, np.ndarray) else True
    )]
    if not valid:
        raise ValueError("No valid molecules after featurization.")

    if encoder_type == "dmpnn":
        return [raw[i] for i in valid], valid
    return np.stack([raw[i] for i in valid]).astype(np.float32), valid


class _ECFPDataset(TorchDataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float() if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return (self.X[idx],)


class _DMPNNDataset(TorchDataset):
    def __init__(self, graphs: List[GraphData], y: Optional[np.ndarray] = None):
        self.graphs = graphs
        self.y = y

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        if self.y is not None:
            return g, float(self.y[idx])
        return (g,)


def _dmpnn_collate(batch):
    if len(batch[0]) == 2:
        graphs, ys = zip(*batch)
        bmg = graphdata_to_batchmolgraph(list(graphs))
        return bmg, torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
    graphs = [b[0] for b in batch]
    return (graphdata_to_batchmolgraph(list(graphs)),)


def make_loader(
    X: Any,
    y: Optional[np.ndarray],
    encoder_type: str,
    batch_size: int = 128,
    shuffle: bool = True,
) -> DataLoader:
    if encoder_type == "dmpnn":
        ds = _DMPNNDataset(X, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=_dmpnn_collate)
    ds = _ECFPDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def load_split(
    path: str,
    smiles_col: str,
    label_col: Optional[str],
    encoder_type: str,
    ecfp_size: int,
    radius: int,
) -> Tuple[Any, Optional[np.ndarray], List[str]]:
    """Returns (X_features, y_or_None, smiles_list)."""
    df = _safe_load_file(path)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in {path}")
    smiles = df[smiles_col].astype(str).tolist()
    X, valid = featurize_smiles(smiles, encoder_type, ecfp_size, radius)
    clean_smiles = [smiles[i] for i in valid]
    y = None
    if label_col and label_col in df.columns:
        y = df[label_col].values[valid].reshape(-1, 1).astype(np.float32)
    return X, y, clean_smiles


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Training / fine-tuning loop
# ─────────────────────────────────────────────────────────────────────────────

def _enable_dropout_only(model: nn.Module) -> None:
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def _val_brier(model: UnifiedModel, loader: DataLoader, device: torch.device) -> float:
    """Quick Brier score on validation loader for early stopping."""
    model.eval()
    sq_errs, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 1:
                continue
            x_b, y_b = batch[0], batch[1].to(device)
            if isinstance(x_b, BatchMolGraph):
                x_b = x_b.to(device)
            else:
                x_b = x_b.to(device)
            out = model(x_b)
            if model.model_type == "evidential":
                prob = out[0]
                p_pos = prob[:, 1::2].clamp(0, 1)
            else:
                raw = out
                mu = raw[:, :1]
                p_pos = torch.sigmoid(mu)
            y_true = y_b[:, :1]
            sq_errs += float(((p_pos - y_true) ** 2).sum())
            n += len(y_b)
    return sq_errs / max(n, 1)


def train_model(
    model: UnifiedModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    grad_clip: float = 5.0,
    mc_train_samples: int = 20,
    verbose: bool = True,
) -> UnifiedModel:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, patience_counter = float("inf"), None, 0
    patience = max(10, epochs // 5)

    for epoch in range(epochs):
        model.train()
        total_loss, n_batch = 0.0, 0
        for batch in train_loader:
            if len(batch) < 2:
                continue
            x_b, y_b = batch[0], batch[1].to(device)
            if isinstance(x_b, BatchMolGraph):
                x_b = x_b.to(device)
            else:
                x_b = x_b.to(device)

            optimizer.zero_grad()
            out = model(x_b)
            if model.model_type == "evidential":
                alpha = out[1]   # (B, 2)
                loss = evidential_clf_loss(alpha, y_b, epoch=epoch).mean()
            else:
                loss = mc_dropout_clf_loss(out, y_b, n_samples=mc_train_samples).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        avg_loss = total_loss / max(n_batch, 1)
        if val_loader is not None:
            val_brier = _val_brier(model, val_loader, device)
            if verbose:
                print(f"  Epoch {epoch + 1:3d}/{epochs} | train_loss={avg_loss:.4f} | val_brier={val_brier:.4f}")
            if val_brier < best_val:
                best_val = val_brier
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stop at epoch {epoch + 1} (patience={patience})")
                    break
        elif verbose:
            print(f"  Epoch {epoch + 1:3d}/{epochs} | train_loss={avg_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"  Restored best checkpoint (val_brier={best_val:.4f})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Inference with uncertainty decomposition
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_with_uncertainty(
    model: UnifiedModel,
    X: Any,
    encoder_type: str,
    device: torch.device,
    batch_size: int = 512,
    mc_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (prob_positive, epistemic, aleatoric), each shape (N,).

    Evidential:
        prob_positive = alpha_pos / (alpha_pos + alpha_neg)
        epistemic     = 2 / S  (vacuity)
        aleatoric     = entropy of expected probability

    MC-Dropout:
        prob_positive = mean sigmoid(sampled_logit)
        epistemic     = mutual information (total H - mean member H)
        aleatoric     = mean entropy across MC samples
    """
    model.eval()
    model = model.to(device)

    if encoder_type == "dmpnn":
        loader: DataLoader = DataLoader(
            _DMPNNDataset(X),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_dmpnn_collate,
        )
    else:
        loader = DataLoader(_ECFPDataset(X), batch_size=batch_size, shuffle=False)

    all_prob, all_epi, all_ale = [], [], []

    for batch in loader:
        x_b = batch[0]
        if isinstance(x_b, BatchMolGraph):
            x_b = x_b.to(device)
        else:
            x_b = x_b.to(device)
        N = x_b.V.shape[0] if isinstance(x_b, BatchMolGraph) else x_b.shape[0]

        if model.model_type == "evidential":
            out = model(x_b)
            prob_b = out[0].cpu().numpy()    # (N, 2)
            ale_b  = out[2].cpu().numpy()    # (N, 1)
            epi_b  = out[3].cpu().numpy()    # (N, 1)
            all_prob.append(prob_b[:, 1])    # P(positive)
            all_epi.append(epi_b.reshape(-1))
            all_ale.append(ale_b.reshape(-1))

        else:  # mc_dropout
            _enable_dropout_only(model)
            probs_mc = []
            for _ in range(mc_samples):
                raw = model(x_b)             # (N, 2)
                c = raw.shape[-1] // 2
                mu = raw[..., :c]
                lv = raw[..., c:].clamp(-10, 10)
                std = torch.exp(0.5 * lv)
                logits = mu + std * torch.randn_like(std)
                p = torch.sigmoid(logits) if c == 1 else torch.softmax(logits, dim=-1)
                probs_mc.append(p.cpu().numpy())
            mc = np.stack(probs_mc, axis=0)   # (S, N, c)
            mean_p = mc.mean(axis=0)           # (N, c)
            eps = 1e-10
            if c == 1:
                p_pos = mean_p[:, 0]
                p_neg = 1.0 - p_pos
                H_tot = -(p_pos * np.log(p_pos + eps) + p_neg * np.log(p_neg + eps))
                mc_p = mc[:, :, 0]
                H_mem = -(mc_p * np.log(mc_p + eps) + (1 - mc_p) * np.log(1 - mc_p + eps))
                H_ale = H_mem.mean(axis=0)
            else:
                p_pos = mean_p[:, 1]
                H_tot = -(mean_p * np.log(mean_p + eps)).sum(axis=1)
                H_mem = -(mc * np.log(mc + eps)).sum(axis=2)
                H_ale = H_mem.mean(axis=0)
            mi = np.maximum(H_tot - H_ale, 0.0)
            all_prob.append(p_pos)
            all_epi.append(mi)
            all_ale.append(H_ale)

    return (
        np.concatenate(all_prob),
        np.concatenate(all_epi),
        np.concatenate(all_ale),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Checkpoint I/O (compatible with bii.py format)
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: str) -> Tuple[UnifiedModel, Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta: Dict[str, Any] = {
        "model_type":        str(ckpt.get("model_type", "evidential")),
        "n_features":        int(ckpt["n_features"]),
        "n_tasks":           int(ckpt.get("n_tasks", 1)),
        "ecfp_size":         int(ckpt.get("ecfp_size", 1024)),
        "radius":            int(ckpt.get("radius", 2)),
        "encoder_type":      str(ckpt.get("encoder_type", "identity")),
        "mc_dropout_rate":   float(ckpt.get("mc_dropout_rate", 0.2)),
        "mc_dropout_samples": int(ckpt.get("mc_dropout_samples", 100)),
        "smiles_column":     str(ckpt.get("smiles_column", "SMILES")),
        "label_column":      str(ckpt.get("label_column", "Outcome")),
    }
    if meta["model_type"] not in {"evidential", "mc_dropout"}:
        raise ValueError(
            f"This script only supports 'evidential' and 'mc_dropout'. "
            f"Got '{meta['model_type']}'. For TabPFN use the original bii.py."
        )
    model = build_model(
        model_type=meta["model_type"],
        n_features=meta["n_features"],
        n_tasks=meta["n_tasks"],
        encoder_type=meta["encoder_type"],
        dropout_rate=meta["mc_dropout_rate"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {path}")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    return model, meta


def save_checkpoint(
    model: UnifiedModel,
    path: str,
    meta: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ckpt = dict(meta)
    ckpt["backend"] = "torch"
    ckpt["model_state_dict"] = model.state_dict()
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y = y_true.reshape(-1).astype(int)
    p = np.clip(prob.reshape(-1).astype(float), 1e-8, 1 - 1e-8)
    pred = (p >= threshold).astype(int)

    # AUC (Mann-Whitney)
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    if n_pos > 0 and n_neg > 0:
        order = np.argsort(p)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(p) + 1)
        auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    else:
        auc = float("nan")

    acc   = float((pred == y).mean())
    brier = float(np.mean((p - y) ** 2))
    nll   = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    tp    = float(np.sum((pred == 1) & (y == 1)))
    fp    = float(np.sum((pred == 1) & (y == 0)))
    fn    = float(np.sum((pred == 0) & (y == 1)))
    tn    = float(np.sum((pred == 0) & (y == 0)))
    prec  = tp / max(tp + fp, 1)
    rec   = tp / max(tp + fn, 1)
    f1    = 2 * prec * rec / max(prec + rec, 1e-12)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc   = (tp * tn - fp * fn) / denom if denom > 0 else float("nan")
    return dict(AUC=auc, F1=f1, MCC=mcc, Accuracy=acc, NLL=nll, Brier=brier,
                TP=tp, FP=fp, FN=fn, TN=tn)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune / evaluate evidential or MC-dropout binary classifiers."
    )
    p.add_argument("--mode", choices=["finetune", "train", "eval"], required=True)

    # Data
    p.add_argument("--train_csv",  type=str, default="")
    p.add_argument("--val_csv",    type=str, default="")
    p.add_argument("--test_csv",   type=str, default="")
    p.add_argument("--smiles_column", type=str, default="SMILES")
    p.add_argument("--label_column",  type=str, default="Outcome")

    # Checkpoint
    p.add_argument("--checkpoint",        type=str, default="",
                   help="Path to .pt checkpoint (required for finetune/eval).")
    p.add_argument("--output_checkpoint", type=str, default="",
                   help="Where to save the updated checkpoint after finetune/train.")
    p.add_argument("--output_csv",        type=str, default="predictions.csv",
                   help="Predictions output CSV (eval mode).")

    # Training hyperparameters
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--grad_clip",   type=float, default=5.0)
    p.add_argument("--mc_train_samples", type=int, default=20)
    p.add_argument("--mc_infer_samples", type=int, default=100)

    # Model (only used when --mode train, i.e. training from scratch)
    p.add_argument("--model_type",   choices=["evidential", "mc_dropout"], default="evidential")
    p.add_argument("--encoder_type", choices=["identity", "dmpnn"],        default="identity")
    p.add_argument("--ecfp_size",    type=int,   default=1024)
    p.add_argument("--radius",       type=int,   default=2)
    p.add_argument("--dropout_rate", type=float, default=0.2)

    return p.parse_args()


def main():
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── FINETUNE ────────────────────────────────────────────────────────────
    if args.mode == "finetune":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for mode=finetune")
        if not args.train_csv:
            raise ValueError("--train_csv is required for mode=finetune")

        model, meta = load_checkpoint(args.checkpoint)
        enc  = meta["encoder_type"]
        ecsz = meta["ecfp_size"]
        rad  = meta["radius"]
        sc   = meta.get("smiles_column", args.smiles_column)
        lc   = meta.get("label_column",  args.label_column)

        X_tr, y_tr, _ = load_split(args.train_csv, sc, lc, enc, ecsz, rad)
        if y_tr is None:
            raise ValueError("Training data must have labels (--label_column).")

        val_loader = None
        if args.val_csv:
            X_val, y_val, _ = load_split(args.val_csv, sc, lc, enc, ecsz, rad)
            val_loader = make_loader(X_val, y_val, enc, args.batch_size, shuffle=False)

        train_loader = make_loader(X_tr, y_tr, enc, args.batch_size, shuffle=True)

        print(f"\nFine-tuning {meta['model_type']} model for {args.epochs} epochs …")
        model = train_model(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, grad_clip=args.grad_clip,
            mc_train_samples=args.mc_train_samples,
        )

        # Optionally evaluate on val
        if val_loader and y_val is not None:
            pp, ep, al = predict_with_uncertainty(
                model, X_val, enc, device, args.batch_size, args.mc_infer_samples
            )
            metrics = compute_metrics(y_val, pp)
            print("\nValidation metrics after finetune:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        out_ckpt = args.output_checkpoint or args.checkpoint.replace(".pt", "_finetuned.pt")
        save_checkpoint(model, out_ckpt, meta)

    # ── TRAIN FROM SCRATCH ──────────────────────────────────────────────────
    elif args.mode == "train":
        if not args.train_csv:
            raise ValueError("--train_csv is required for mode=train")

        enc  = args.encoder_type
        ecsz = args.ecfp_size
        rad  = args.radius
        sc   = args.smiles_column
        lc   = args.label_column

        X_tr, y_tr, _ = load_split(args.train_csv, sc, lc, enc, ecsz, rad)
        if y_tr is None:
            raise ValueError("Training data must have labels (--label_column).")

        n_features = (
            X_tr[0].node_features.shape[1] if enc == "dmpnn" else X_tr.shape[1]
        )
        n_tasks = y_tr.shape[1]

        model = build_model(args.model_type, n_features, n_tasks, enc, args.dropout_rate)
        print(f"\nBuilt {args.model_type} model | n_features={n_features} | n_tasks={n_tasks}")

        val_loader = None
        if args.val_csv:
            X_val, y_val, _ = load_split(args.val_csv, sc, lc, enc, ecsz, rad)
            val_loader = make_loader(X_val, y_val, enc, args.batch_size, shuffle=False)

        train_loader = make_loader(X_tr, y_tr, enc, args.batch_size, shuffle=True)
        print(f"Training from scratch for {args.epochs} epochs …")
        model = train_model(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, grad_clip=args.grad_clip,
            mc_train_samples=args.mc_train_samples,
        )

        meta: Dict[str, Any] = {
            "model_type": args.model_type,
            "n_features": n_features,
            "n_tasks": n_tasks,
            "ecfp_size": ecsz,
            "radius": rad,
            "encoder_type": enc,
            "mc_dropout_rate": args.dropout_rate,
            "mc_dropout_samples": args.mc_infer_samples,
            "smiles_column": sc,
            "label_column": lc,
        }
        out_ckpt = args.output_checkpoint or "trained_model.pt"
        save_checkpoint(model, out_ckpt, meta)

    # ── EVAL ────────────────────────────────────────────────────────────────
    elif args.mode == "eval":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for mode=eval")
        if not args.test_csv:
            raise ValueError("--test_csv is required for mode=eval")

        model, meta = load_checkpoint(args.checkpoint)
        enc  = meta["encoder_type"]
        ecsz = meta["ecfp_size"]
        rad  = meta["radius"]
        sc   = meta.get("smiles_column", args.smiles_column)
        lc   = meta.get("label_column",  args.label_column) if args.label_column else None

        X_te, y_te, smiles = load_split(args.test_csv, sc, lc, enc, ecsz, rad)
        print(f"\nRunning inference on {len(smiles)} molecules …")
        pp, ep, al = predict_with_uncertainty(
            model, X_te, enc, device, args.batch_size, args.mc_infer_samples
        )

        out_df = pd.DataFrame({
            "SMILES": smiles,
            "prob_class1_positive":  pp,
            "epistemic_uncertainty": ep,
            "aleatoric_uncertainty": al,
        })
        if y_te is not None:
            out_df["y_true"]     = y_te.reshape(-1)
            out_df["pred_label"] = (pp >= 0.5).astype(int)
            metrics = compute_metrics(y_te, pp)
            print("\nTest metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved predictions: {args.output_csv}")


if __name__ == "__main__":
    main()
