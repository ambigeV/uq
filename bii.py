import argparse
import hashlib
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import deepchem as dc
import numpy as np
import pandas as pd
import torch


def _ensure_deepchem_torchmodel_alias() -> None:
    """
    Some DeepChem versions don't expose dc.models.TorchModel directly.
    model_utils/nn_baseline may still reference it in annotations.
    """
    if hasattr(dc.models, "TorchModel"):
        return

    torch_model_cls = None
    try:
        from deepchem.models.torch_models.torch_model import TorchModel as _TorchModel
        torch_model_cls = _TorchModel
    except Exception:
        try:
            from deepchem.models.torch_models import TorchModel as _TorchModel
            torch_model_cls = _TorchModel
        except Exception:
            torch_model_cls = None

    if torch_model_cls is not None:
        dc.models.TorchModel = torch_model_cls


_ensure_deepchem_torchmodel_alias()

from nn_baseline import (  # noqa: E402
    EvidentialClassificationLoss,
    GradientClippingCallback,
    UnifiedTorchModel,
    build_model,
    get_n_features,
)


def _safe_read_pickle(file_path: str):
    """
    Read pickle with a robust fallback path.
    """
    try:
        return pd.read_pickle(file_path)
    except Exception:
        # Fallback using native pickle loading.
        with open(file_path, "rb") as f:
            return pickle.load(f, encoding="latin1")


def _is_valid_feature(feat) -> bool:
    if feat is None:
        return False
    if isinstance(feat, np.ndarray):
        return feat.size > 0
    if hasattr(feat, "size"):
        try:
            return int(feat.size) > 0
        except Exception:
            return True
    return True


def _finalize_feature_array(clean_feats: List, encoder_type: str) -> np.ndarray:
    if encoder_type == "dmpnn":
        return np.array(clean_feats, dtype=object)
    return np.stack(clean_feats).astype(np.float32)


def _load_split_from_pkl(
    file_path: str,
    featurizer: dc.feat.Featurizer,
    encoder_type: str = "identity",
    smiles_column: str = "SMILES",
    label_column: str = "Outcome",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Split file not found: {file_path}")

    df = _safe_read_pickle(file_path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected DataFrame in {file_path}, got {type(df)}")
    if smiles_column not in df.columns:
        raise ValueError(f"Missing '{smiles_column}' in: {file_path}")
    if label_column not in df.columns:
        raise ValueError(f"Missing '{label_column}' in: {file_path}")

    smiles = df[smiles_column].astype(str).tolist()
    features = featurizer.featurize(smiles)

    valid_inds = [i for i, f in enumerate(features) if _is_valid_feature(f)]
    if len(valid_inds) == 0:
        raise ValueError(f"No valid molecules after featurization in: {file_path}")

    x = _finalize_feature_array([features[i] for i in valid_inds], encoder_type=encoder_type)
    y = df[label_column].values[valid_inds].reshape(-1, 1).astype(np.float32)
    ids = np.array(smiles)[valid_inds]
    return x, y, ids


def load_custom_ecfp_tox_pkl(
    data_dir: str = "cytotoxicity_data",
    ecfp_size: int = 1024,
    radius: int = 2,
    encoder_type: str = "identity",
    smiles_column: str = "SMILES",
    label_column: str = "Outcome",
    save_dir: str = "save_ecfp_pkl",
    reload: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    files = {
        "train": "HEK293_train_BM.pkl",
        "val": "HEK293_test_BM.pkl",
        "test": "tox21_all.pkl",
    }

    # Cache featurized arrays so repeated runs can skip featurization.
    cache_tag = (
        f"{encoder_type}_ecfp{ecfp_size}_r{radius}_{smiles_column}_{label_column}"
    )
    cache_root = os.path.join(save_dir, cache_tag)
    os.makedirs(cache_root, exist_ok=True)

    def _cache_file(split_name: str) -> str:
        return os.path.join(cache_root, f"{split_name}.npz")

    if reload:
        cached = True
        for split in files.keys():
            if not os.path.exists(_cache_file(split)):
                cached = False
                break
        if cached:
            out: Dict[str, Dict[str, np.ndarray]] = {}
            for split in files.keys():
                with np.load(_cache_file(split), allow_pickle=True) as data:
                    x_cached = data["X"]
                    if encoder_type != "dmpnn":
                        x_cached = x_cached.astype(np.float32)
                    out[split] = {
                        "X": x_cached,
                        "y": data["y"].astype(np.float32),
                        "ids": data["ids"],
                    }
                print(f"Loaded cached {split}: {len(out[split]['X'])} samples")
            return out

    if encoder_type == "dmpnn":
        featurizer = dc.feat.DMPNNFeaturizer()
    else:
        featurizer = dc.feat.CircularFingerprint(size=ecfp_size, radius=radius)
    out: Dict[str, Dict[str, np.ndarray]] = {}

    for split, name in files.items():
        path = os.path.join(data_dir, name)
        x, y, ids = _load_split_from_pkl(
            file_path=path,
            featurizer=featurizer,
            encoder_type=encoder_type,
            smiles_column=smiles_column,
            label_column=label_column,
        )
        out[split] = {"X": x, "y": y, "ids": ids}
        print(f"Loaded {split}: {len(x)} samples from {name}")
        if reload:
            np.savez_compressed(_cache_file(split), X=x, y=y, ids=ids)
            print(f"Saved cache for {split}: {_cache_file(split)}")

    return out


def _to_dc_dataset(
    x: np.ndarray,
    y: np.ndarray,
    ids: Optional[np.ndarray] = None,
    encoder_type: str = "identity",
):
    if ids is None:
        ids = np.array([str(i) for i in range(len(x))])
    w = np.ones_like(y, dtype=float)
    if encoder_type == "dmpnn":
        # GraphData objects are best handled through in-memory datasets.
        x_obj = np.array(list(x), dtype=object)
        return dc.data.NumpyDataset(X=x_obj, y=y, w=w, ids=ids)
    return dc.data.DiskDataset.from_numpy(X=x, y=y, w=w, ids=ids)


def _extract_positive_probs(raw_probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(raw_probs)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    if probs.ndim > 1 and probs.shape[1] > 1:
        probs = probs[:, 1::2]
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    return probs


def evaluate_classification_dc(
    dc_model: UnifiedTorchModel,
    x: np.ndarray,
    y: np.ndarray,
    encoder_type: str = "identity",
) -> Dict[str, float]:
    ds = _to_dc_dataset(x, y, encoder_type=encoder_type)
    probs = _extract_positive_probs(dc_model.predict(ds))
    y_true = y.reshape(-1, 1).astype(np.float32)
    pred = (probs >= 0.5).astype(np.float32)
    acc = float((pred == y_true).mean())
    brier = float(np.mean((probs - y_true) ** 2))
    return {"acc": acc, "brier": brier}


def train_evidential_binary(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    n_features: int,
    encoder_type: str = "identity",
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 128,
    grad_clip: float = 5.0,
) -> UnifiedTorchModel:
    n_tasks = train_y.shape[1]

    model = build_model(
        model_type="evidential",
        n_features=n_features,
        n_tasks=n_tasks,
        mode="classification",
        encoder_type=encoder_type,
    )
    loss = EvidentialClassificationLoss()
    clip_callback = GradientClippingCallback(max_norm=grad_clip)

    dc_model = UnifiedTorchModel(
        model=model,
        loss=loss,
        output_types=["prediction", "loss", "var1", "var2"],
        batch_size=batch_size,
        learning_rate=lr,
        log_frequency=40,
        mode="classification",
        encoder_type=encoder_type,
    )

    train_ds = _to_dc_dataset(train_x, train_y, encoder_type=encoder_type)
    print(f"Training nn_baseline evidential classifier for {epochs} epochs...")
    dc_model.fit(train_ds, nb_epoch=epochs, callbacks=[clip_callback])

    val_metrics = evaluate_classification_dc(dc_model, val_x, val_y, encoder_type=encoder_type)
    print(
        f"Post-train val metrics | acc={val_metrics['acc']:.4f} "
        f"| brier={val_metrics['brier']:.4f}"
    )
    return dc_model


def save_checkpoint(
    dc_model: UnifiedTorchModel,
    checkpoint_path: str,
    n_features: int,
    ecfp_size: int,
    radius: int = 2,
    encoder_type: str = "identity",
    smiles_column: str = "SMILES",
    label_column: str = "Outcome",
    batch_size: int = 128,
    learning_rate: float = 1e-4,
) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    ckpt = {
        "model_state_dict": dc_model.model.state_dict(),
        "model_type": "nn_baseline_evidential_classification",
        "n_features": int(n_features),
        "n_tasks": 1,
        "ecfp_size": int(ecfp_size),
        "radius": int(radius),
        "encoder_type": encoder_type,
        "smiles_column": smiles_column,
        "label_column": label_column,
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
    }
    torch.save(ckpt, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def inspect_checkpoint_hparams(checkpoint_path: str) -> Dict[str, object]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    meta = {
        "model_type": ckpt.get("model_type"),
        "n_features": ckpt.get("n_features"),
        "n_tasks": ckpt.get("n_tasks"),
        "ecfp_size": ckpt.get("ecfp_size"),
        "radius": ckpt.get("radius"),
        "encoder_type": ckpt.get("encoder_type", "identity"),
        "smiles_column": ckpt.get("smiles_column"),
        "label_column": ckpt.get("label_column"),
        "batch_size": ckpt.get("batch_size"),
        "learning_rate": ckpt.get("learning_rate"),
    }
    print(f"Checkpoint metadata from: {checkpoint_path}")
    for k, v in meta.items():
        print(f"  - {k}: {v}")
    return meta


def load_checkpoint(checkpoint_path: str) -> Tuple[torch.nn.Module, Dict[str, object]]:
    meta = inspect_checkpoint_hparams(checkpoint_path)
    n_features = meta.get("n_features")
    n_tasks = meta.get("n_tasks", 1)
    if n_features is None:
        raise ValueError("Checkpoint missing n_features.")

    encoder_type = str(meta.get("encoder_type", "identity"))
    model = build_model(
        model_type="evidential",
        n_features=int(n_features),
        n_tasks=int(n_tasks),
        mode="classification",
        encoder_type=encoder_type,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, meta


def _featurize_smiles_for_inference(
    smiles: List[str], ecfp_size: int, radius: int, encoder_type: str = "identity"
) -> Tuple[np.ndarray, List[str], List[int]]:
    if encoder_type == "dmpnn":
        featurizer = dc.feat.DMPNNFeaturizer()
    else:
        featurizer = dc.feat.CircularFingerprint(size=ecfp_size, radius=radius)
    features = featurizer.featurize(smiles)
    valid_inds = [i for i, f in enumerate(features) if _is_valid_feature(f)]
    if len(valid_inds) == 0:
        raise ValueError("No valid SMILES for inference.")
    x = _finalize_feature_array([features[i] for i in valid_inds], encoder_type=encoder_type)
    clean_smiles = [smiles[i] for i in valid_inds]
    return x, clean_smiles, valid_inds


@torch.no_grad()
def _infer_from_features(
    model: torch.nn.Module,
    x: np.ndarray,
    smiles_out: List[str],
    input_indices: List[int],
    encoder_type: str = "identity",
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if encoder_type == "dmpnn":
        from nn import graphdata_to_batchmolgraph

        batch_molgraph = graphdata_to_batchmolgraph(list(x))
        batch_molgraph.to(device)
        output = model(batch_molgraph)
    else:
        x_t = torch.from_numpy(x).float().to(device)
    output = model(x_t)
    if not isinstance(output, tuple) or len(output) < 4:
        raise RuntimeError("Unexpected model output format for evidential classification.")

    probs_raw = output[0].detach().cpu().numpy()
    aleatoric_raw = output[2].detach().cpu().numpy()
    epistemic_raw = output[3].detach().cpu().numpy()

    probs = np.asarray(probs_raw)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    if probs.shape[1] < 2:
        raise RuntimeError(
            f"Expected at least 2 class probability columns, got shape {probs.shape}"
        )

    # nn_baseline evidential classification flattens per-task class probs as [class0, class1].
    p_negative = probs[:, 0]
    p_positive = probs[:, 1]

    aleatoric = np.asarray(aleatoric_raw).reshape(-1)
    epistemic = np.asarray(epistemic_raw).reshape(-1)

    return pd.DataFrame(
        {
            "prob_class1_positive": p_positive,
            "prob_class2_negative": p_negative,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
        }
    )


@torch.no_grad()
def infer_from_smiles(
    model: torch.nn.Module,
    smiles: List[str],
    ecfp_size: int,
    radius: int,
    encoder_type: str = "identity",
) -> pd.DataFrame:
    x, clean_smiles, valid_inds = _featurize_smiles_for_inference(
        smiles=smiles, ecfp_size=ecfp_size, radius=radius, encoder_type=encoder_type
    )
    return _infer_from_features(
        model=model,
        x=x,
        smiles_out=clean_smiles,
        input_indices=valid_inds,
        encoder_type=encoder_type,
    )


def infer_from_file(
    model: torch.nn.Module,
    input_path: str,
    smiles_column: str,
    ecfp_size: int,
    radius: int,
    encoder_type: str = "identity",
    cache_dir: str = "save_ecfp_pkl",
    reload: bool = True,
) -> pd.DataFrame:
    cache_root = os.path.join(cache_dir, "inference")
    os.makedirs(cache_root, exist_ok=True)

    cache_key_raw = (
        f"{os.path.abspath(input_path)}|{ecfp_size}|{radius}|{smiles_column}|{encoder_type}"
    )
    cache_key = hashlib.sha1(cache_key_raw.encode("utf-8")).hexdigest()[:16]
    cache_file = os.path.join(cache_root, f"infer_{cache_key}.npz")

    if reload and os.path.exists(cache_file):
        with np.load(cache_file, allow_pickle=True) as cached:
            x = cached["X"]
            if encoder_type != "dmpnn":
                x = x.astype(np.float32)
            clean_smiles = cached["SMILES"].tolist()
            valid_inds = cached["input_index"].tolist()
        pred_df = _infer_from_features(
            model=model,
            x=x,
            smiles_out=clean_smiles,
            input_indices=valid_inds,
            encoder_type=encoder_type,
        )
        print(f"Loaded cached inference features: {cache_file}")
        return pred_df

    if input_path.endswith(".pkl"):
        df = _safe_read_pickle(input_path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected DataFrame in {input_path}, got {type(df)}")
    elif input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        raise ValueError("inference_input must end with .pkl or .csv")

    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in inference input.")
    smiles = df[smiles_column].astype(str).tolist()

    # Save featurization cache for repeated inference runs.
    x, clean_smiles, valid_inds = _featurize_smiles_for_inference(
        smiles=smiles,
        ecfp_size=ecfp_size,
        radius=radius,
        encoder_type=encoder_type,
    )
    if reload:
        np.savez_compressed(
            cache_file,
            X=x,
            SMILES=np.array(clean_smiles, dtype=object),
            input_index=np.array(valid_inds, dtype=int),
        )
        print(f"Saved inference feature cache: {cache_file}")

    return _infer_from_features(
        model=model,
        x=x,
        smiles_out=clean_smiles,
        input_indices=valid_inds,
        encoder_type=encoder_type,
    )


def build_default_inference_output_path(
    inference_input: Path,
    method_name: str,
    run_name: str,
    repo_root: Path,
) -> Path:
    dataset_name = inference_input.stem
    output_dir = repo_root / "inference_outputs" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{method_name}_{run_name}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PKL + (ECFP or DMPNN graph) + Deep Evidential Binary model script."
    )
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--data_dir", type=str, default="cytotoxicity_data")
    parser.add_argument(
        "--featurized_save_dir",
        type=str,
        default="save_ecfp_pkl",
        help="Directory to cache featurized arrays (save_dir behavior)",
    )
    parser.add_argument(
        "--reload",
        dest="reload",
        action="store_true",
        help="Load featurized arrays from cache if available",
    )
    parser.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help="Disable cache loading and re-featurize from source PKLs",
    )
    parser.set_defaults(reload=True)
    parser.add_argument("--ecfp_size", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument(
        "--use_graph",
        action="store_true",
        help="Use DMPNN graph featurization; otherwise identity/ECFP mode is used",
    )
    parser.add_argument("--smiles_column", type=str, default="SMILES")
    parser.add_argument("--label_column", type=str, default="Outcome")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--model_name", type=str, default="bii_evidential_binary")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--inference_input", type=str, default=None)
    parser.add_argument(
        "--inference_all_splits",
        action="store_true",
        help="Run inference on all default PKL splits (train/val/test)",
    )
    parser.add_argument("--inference_output", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir

    # Enforce one-to-one mapping from use_graph -> encoder_type.
    args.encoder_type = "dmpnn" if args.use_graph else "identity"

    if args.mode == "train":
        splits = load_custom_ecfp_tox_pkl(
            data_dir=str(data_dir),
            ecfp_size=args.ecfp_size,
            radius=args.radius,
            encoder_type=args.encoder_type,
            smiles_column=args.smiles_column,
            label_column=args.label_column,
            save_dir=args.featurized_save_dir,
            reload=args.reload,
        )
        train_x, train_y = splits["train"]["X"], splits["train"]["y"]
        val_x, val_y = splits["val"]["X"], splits["val"]["y"]
        test_x, test_y = splits["test"]["X"], splits["test"]["y"]
        train_ds_for_shape = _to_dc_dataset(train_x, train_y, encoder_type=args.encoder_type)
        n_features = get_n_features(train_ds_for_shape, encoder_type=args.encoder_type)

        dc_model = train_evidential_binary(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            n_features=n_features,
            encoder_type=args.encoder_type,
            epochs=args.epochs,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            grad_clip=args.grad_clip,
        )

        test_metrics = evaluate_classification_dc(
            dc_model, test_x, test_y, encoder_type=args.encoder_type
        )
        print(
            f"Test metrics | acc={test_metrics['acc']:.4f} "
            f"| brier={test_metrics['brier']:.4f}"
        )

        save_dir = Path(args.save_dir)
        if not save_dir.is_absolute():
            save_dir = repo_root / save_dir
        ckpt_path = save_dir / f"{args.model_name}.pt"
        save_checkpoint(
            dc_model=dc_model,
            checkpoint_path=str(ckpt_path),
            n_features=n_features,
            ecfp_size=args.ecfp_size,
            radius=args.radius,
            encoder_type=args.encoder_type,
            smiles_column=args.smiles_column,
            label_column=args.label_column,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        return

    # Inference mode
    if args.checkpoint_path is None:
        raise ValueError("--checkpoint_path is required for mode=infer")
    if args.inference_input is None and not args.inference_all_splits:
        raise ValueError("--inference_input is required for mode=infer")

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo_root / checkpoint_path
    model, meta = load_checkpoint(str(checkpoint_path))
    run_name = checkpoint_path.stem
    ecfp_size = int(meta.get("ecfp_size", args.ecfp_size))
    radius = int(meta.get("radius", args.radius))
    encoder_type = str(meta.get("encoder_type", args.encoder_type))
    smiles_column = str(meta.get("smiles_column", args.smiles_column))
    if args.inference_all_splits:
        split_files = {
            "train": data_dir / "HEK293_train_BM.pkl",
            "val": data_dir / "HEK293_test_BM.pkl",
            "test": data_dir / "tox21_all.pkl",
        }
        for split_name, inference_input in split_files.items():
            pred_df = infer_from_file(
                model=model,
                input_path=str(inference_input),
                smiles_column=smiles_column,
                ecfp_size=ecfp_size,
                radius=radius,
                encoder_type=encoder_type,
                cache_dir=args.featurized_save_dir,
                reload=args.reload,
            )
            print(f"\n[{split_name}]")
            print(pred_df.head(10))

            default_output_path = build_default_inference_output_path(
                inference_input=inference_input,
                method_name=encoder_type,
                run_name=run_name,
                repo_root=repo_root,
            )
            pred_df.to_csv(default_output_path, index=False)
            print(f"Saved inference output to: {default_output_path}")
    else:
        inference_input = Path(args.inference_input)
        if not inference_input.is_absolute():
            inference_input = repo_root / inference_input

        pred_df = infer_from_file(
            model=model,
            input_path=str(inference_input),
            smiles_column=smiles_column,
            ecfp_size=ecfp_size,
            radius=radius,
            encoder_type=encoder_type,
            cache_dir=args.featurized_save_dir,
            reload=args.reload,
        )
        print(pred_df.head(10))

        default_output_path = build_default_inference_output_path(
            inference_input=inference_input,
            method_name=encoder_type,
            run_name=run_name,
            repo_root=repo_root,
        )
        pred_df.to_csv(default_output_path, index=False)
        print(f"Saved inference output to: {default_output_path}")

        if args.inference_output is not None:
            out_path = Path(args.inference_output)
            if not out_path.is_absolute():
                out_path = repo_root / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pred_df.to_csv(out_path, index=False)
            print(f"Saved additional inference output to: {out_path}")


if __name__ == "__main__":
    main()
