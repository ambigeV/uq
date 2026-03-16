import argparse
import json
from pathlib import Path

import deepchem as dc
import numpy as np
import pandas as pd
import torch

from nn import UnifiedTorchModel
from nn_baseline import build_model, get_n_features


def _safe_read_pickle(path: Path) -> pd.DataFrame:
    obj = pd.read_pickle(path)
    if not isinstance(obj, pd.DataFrame):
        raise ValueError(f"Expected DataFrame in {path}, got {type(obj)}")
    return obj


def _build_ecfp_features_from_smiles(
    smiles: np.ndarray,
    ecfp_size: int,
    ecfp_radius: int,
) -> np.ndarray:
    featurizer = dc.feat.CircularFingerprint(size=int(ecfp_size), radius=int(ecfp_radius))
    raw = featurizer.featurize([str(s) for s in smiles.tolist()])
    feats = np.zeros((len(raw), int(ecfp_size)), dtype=np.float32)
    for i, f in enumerate(raw):
        if f is None:
            continue
        arr = np.asarray(f, dtype=np.float32).reshape(-1)
        if arr.size == int(ecfp_size):
            feats[i] = arr
    return feats


def _build_graph_features_from_smiles(smiles: np.ndarray) -> np.ndarray:
    featurizer = dc.feat.DMPNNFeaturizer()
    raw = featurizer.featurize([str(s) for s in smiles.tolist()])
    clean = [f for f in raw if f is not None]
    if not clean:
        raise ValueError("No valid graph features produced by DMPNN featurizer.")
    return np.array(clean, dtype=object)


def _build_split_features(
    *,
    label_dir: Path,
    split_name: str,
    encoder_type: str,
    smiles_column: str,
    ecfp_size: int,
    ecfp_radius: int,
) -> np.ndarray:
    pkl_path = label_dir / f"{split_name}.pkl"
    df = _safe_read_pickle(pkl_path)
    if smiles_column not in df.columns:
        raise ValueError(f"Missing '{smiles_column}' in {pkl_path}")
    smiles = df[smiles_column].astype(str).to_numpy()
    if encoder_type == "dmpnn":
        return _build_graph_features_from_smiles(smiles)
    return _build_ecfp_features_from_smiles(smiles, ecfp_size=ecfp_size, ecfp_radius=ecfp_radius)


def _to_dc_dataset(x: np.ndarray, y: np.ndarray, encoder_type: str):
    ids = np.array([str(i) for i in range(len(x))])
    w = np.ones_like(y, dtype=float)
    if encoder_type == "dmpnn":
        x_obj = np.array(list(x), dtype=object)
        return dc.data.NumpyDataset(X=x_obj, y=y, w=w, ids=ids)
    return dc.data.DiskDataset.from_numpy(X=x.astype(np.float32), y=y, w=w, ids=ids)


def _extract_positive_probs(raw_probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(raw_probs)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    if probs.ndim > 1 and probs.shape[1] > 1:
        probs = probs[:, 1::2]
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    return probs.astype(np.float64)


def _train_domain_classifier_and_save(
    *,
    label_dir: Path,
    val_split: str,
    test_split: str,
    out_path: Path,
    encoder_type: str,
    label_column: str,
    smiles_column: str,
    ecfp_size: int,
    ecfp_radius: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> dict:
    _ = label_column
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    x_val = _build_split_features(
        label_dir=label_dir,
        split_name=val_split,
        encoder_type=encoder_type,
        smiles_column=smiles_column,
        ecfp_size=ecfp_size,
        ecfp_radius=ecfp_radius,
    )
    x_test = _build_split_features(
        label_dir=label_dir,
        split_name=test_split,
        encoder_type=encoder_type,
        smiles_column=smiles_column,
        ecfp_size=ecfp_size,
        ecfp_radius=ecfp_radius,
    )
    y_val = np.zeros((len(x_val), 1), dtype=np.float32)
    y_test = np.ones((len(x_test), 1), dtype=np.float32)

    x = np.concatenate([x_val, x_test], axis=0)
    if encoder_type != "dmpnn":
        x = x.astype(np.float32)
    y = np.concatenate([y_val, y_test], axis=0).astype(np.float32)
    perm = np.random.permutation(len(x))
    x = x[perm]
    y = y[perm]

    if encoder_type == "dmpnn":
        ds = dc.data.NumpyDataset(X=np.array(list(x), dtype=object), y=y, w=np.ones_like(y))
        n_features = int(get_n_features(ds, encoder_type="dmpnn"))
    else:
        n_features = int(x.shape[1])

    domain_dc = _to_dc_dataset(x=x, y=y, encoder_type=encoder_type)

    model = build_model(
        model_type="baseline",
        n_features=n_features,
        n_tasks=1,
        mode="classification",
        encoder_type=encoder_type,
    )
    loss = dc.models.losses.SigmoidCrossEntropy()
    dc_model = UnifiedTorchModel(
        model=model,
        loss=loss,
        output_types=["prediction", "loss"],  # prediction=probs, loss=logits
        batch_size=int(batch_size),
        learning_rate=float(lr),
        mode="classification",
        encoder_type=encoder_type,
    )
    dc_model.fit(domain_dc, nb_epoch=int(epochs))

    raw_probs = dc_model.predict(domain_dc)
    probs = _extract_positive_probs(raw_probs).reshape(-1)
    train_acc = float(((probs >= 0.5).astype(np.float32) == y.reshape(-1)).mean())

    metadata = {
        "model_family": "nn_baseline",
        "encoder_type": encoder_type,
        "label_column": label_column,
        "smiles_column": smiles_column,
        "ecfp_size": int(ecfp_size),
        "ecfp_radius": int(ecfp_radius),
        "n_features": int(n_features),
        "n_tasks": 1,
        "val_split": val_split,
        "test_split": test_split,
        "train_accuracy": train_acc,
        "seed": int(seed),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": dc_model.model.state_dict(), "metadata": metadata}, out_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/store domain classifier for weighted conformal CP."
    )
    parser.add_argument("--label_dir", type=str, default="cytotoxicity_data")
    parser.add_argument("--val_split", type=str, default="HEK293_test_BM")
    parser.add_argument("--test_split", type=str, default="tox21_all")
    parser.add_argument(
        "--out_model",
        type=str,
        default="saved_models/weighted_cp_domain_classifier.pt",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="dmpnn",
        choices=["identity", "dmpnn"],
    )
    parser.add_argument("--label_column", type=str, default="Outcome")
    parser.add_argument("--smiles_column", type=str, default="SMILES")
    parser.add_argument("--ecfp_size", type=int, default=1024)
    parser.add_argument("--ecfp_radius", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    label_dir = Path(args.label_dir)
    if not label_dir.is_absolute():
        label_dir = repo_root / label_dir
    out_model = Path(args.out_model)
    if not out_model.is_absolute():
        out_model = repo_root / out_model

    metadata = _train_domain_classifier_and_save(
        label_dir=label_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        out_path=out_model,
        encoder_type=args.encoder_type,
        label_column=args.label_column,
        smiles_column=args.smiles_column,
        ecfp_size=args.ecfp_size,
        ecfp_radius=args.ecfp_radius,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"Saved domain classifier: {out_model}")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
