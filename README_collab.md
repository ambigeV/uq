# Collaborator Scripts

Two **self-contained** scripts (no repo-internal imports):

- `collab_finetune_eval.py` — train / finetune / eval a binary UQ classifier → checkpoints + prediction CSVs.
- `collab_ensemble_stack.py` — combine prediction CSVs into an ensemble (+ optional conformal rejection).

Pipeline: `collab_finetune_eval.py` (eval) → prediction CSVs → `collab_ensemble_stack.py`.

## Folders

| folder | contains |
|---|---|
| `sample_data/` | `train.csv`, `val.csv`, `test.csv` — columns `SMILES,Outcome` |
| `saved_models/` | `.pt` checkpoints |
| `inference_outputs/<split>/` | prediction CSVs named `<method>_run_<id>.csv` |
| `ensemble_results/` | stacker output CSVs |

Prediction CSV columns: `prob_class1_positive, epistemic_uncertainty, aleatoric_uncertainty`.

## Install

```bash
# collab_finetune_eval.py
pip install torch numpy pandas deepchem scikit-learn
# collab_ensemble_stack.py
pip install numpy pandas scikit-learn
```

---

## 1. `collab_finetune_eval.py`

**Input:** data CSV(s) with `SMILES` (+ `Outcome` for train/eval-metrics); `.pt` checkpoint for finetune/eval.
**Output:** a checkpoint (train/finetune) **or** a predictions CSV (eval).

| option | meaning |
|---|---|
| `--mode` | `train` / `finetune` / `eval` |
| `--train_csv` / `--val_csv` / `--test_csv` | data files |
| `--checkpoint` | input `.pt` (required for finetune/eval) |
| `--output_checkpoint` | output `.pt` (train/finetune) |
| `--output_csv` | predictions path (eval) |
| `--model_type` | `evidential` / `mc_dropout` (train only) |
| `--encoder_type` | `identity` (ECFP) / `dmpnn` (train only) |
| `--epochs` / `--lr` / `--batch_size` | training hyperparams |
| `--mc_infer_samples` | MC-dropout passes at inference |

> In finetune/eval, encoder/ecfp/radius are read from the checkpoint, so you don't pass them.

```bash
python collab_finetune_eval.py --mode eval \
  --checkpoint saved_models/sample.pt \
  --test_csv sample_data/test.csv \
  --output_csv preds/test.csv
```

---

## 2. `collab_ensemble_stack.py`

**Input:** prediction CSVs (convention dir **or** explicit paths) + label files.
**Output (in `--output_dir`):** `ensemble_metrics_per_run.csv`, `ensemble_metrics_aggregate.csv`, `ensemble_predictions_all_runs.csv`.

| option | meaning |
|---|---|
| `--pred_root` | root of `<split>/<method>_run_<id>.csv` (convention mode) |
| `--val_pred_files` / `--test_pred_files` | explicit CSV paths (`;`=runs, `,`=methods); bypasses `--pred_root` |
| `--label_dir` | dir with `<split>.csv`/`.pkl` labels |
| `--val_split` / `--test_split` | split names (also the label filename stems) |
| `--methods` | base-model name(s) = filename prefix before `_run_<id>` |
| `--run_ids` | run ids to aggregate |
| `--strategy` | `iso_inverse` / `iso_inverse_reject` / `both` |
| `--alpha` | conformal level (rejection only); `alpha=0` rejects everything |
| `--uncertainty_source` | `total` / `epistemic` |
| `--no_test_labels` | skip test metrics |

> Output columns: `iso_inv_*` (no rejection) and/or `iso_inv_reject_*` (with rejection).

---

## End-to-end: stacking `saved_models/bii_mc_ucb_dmpnn_balanced_run_{}.pt`

```bash
# 1) Eval each run on val + test (encoder/model_type read from each checkpoint)
for r in 0 1 2 3 4; do
  python collab_finetune_eval.py --mode eval \
    --checkpoint saved_models/bii_mc_ucb_dmpnn_balanced_run_${r}.pt \
    --test_csv sample_data/val.csv \
    --output_csv inference_outputs/val/bii_mc_ucb_dmpnn_balanced_run_${r}.csv
  python collab_finetune_eval.py --mode eval \
    --checkpoint saved_models/bii_mc_ucb_dmpnn_balanced_run_${r}.pt \
    --test_csv sample_data/test.csv \
    --output_csv inference_outputs/test/bii_mc_ucb_dmpnn_balanced_run_${r}.csv
done

# 2) Stack the 5 runs
python collab_ensemble_stack.py \
  --pred_root inference_outputs \
  --label_dir sample_data \
  --val_split val --test_split test \
  --methods bii_mc_ucb_dmpnn_balanced \
  --run_ids 0,1,2,3,4 \
  --strategy both --alpha 0.10 \
  --output_dir ensemble_results
```

`--methods` is the filename prefix before `_run_<id>.csv`; labels are read from
`sample_data/val.csv` and `sample_data/test.csv` (stem = split name).
