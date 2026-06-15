# Holly — Inference + 2-Model Ensemble Runbook

End-to-end CLI for: (1) running inference from **already-finetuned** checkpoints in
`saved_models/`, then (2) stacking two base models in `bii_super_learner.py` to evaluate
prediction quality and uncertainty quantification.

No training is performed. Models are loaded and inferred only.

---

## 0. Assumptions

- Finetuned checkpoints already exist in `saved_models/`:
  - `bii_mc_dmpnn_balanced_run_{id}_ft.pt`  — dmpnn encoder, **mc_dropout**
  - `bii_new_dmpnn_balanced_run_{id}_ft.pt` — dmpnn encoder, **evidential**
  - for `id` in `0,1,2,3,4`
- `encoder_type` and `model_type` are read from each checkpoint's saved meta,
  so they do **not** need to be passed at inference.
- Label pkls live in `cytotoxicity_data/` (e.g. `HEK293_test_BM.pkl`, `tox21_all.pkl`).

### Naming contract (important)

`bii.py` writes:
```
inference_outputs/<pkl_stem>/<encoder_type>_<checkpoint_stem>.csv
```
`bii_super_learner.py` reads `inference_outputs/<split>/<method>_run_<id>.csv`, i.e. the
filename **must end in `_run_<id>.csv`**.

The checkpoint name `bii_mc_dmpnn_balanced_run_0_ft` produces
`dmpnn_bii_mc_dmpnn_balanced_run_0_ft.csv`, whose `_ft` comes **after** the run id — this
will NOT be found by the stacker. Two options:

- **Recommended:** rename checkpoints to `bii_mc_dmpnn_balanced_ft_run_{id}.pt` so the CSV
  becomes `dmpnn_bii_mc_dmpnn_balanced_ft_run_0.csv` (method = `dmpnn_bii_mc_dmpnn_balanced_ft`).
- **Alternative:** patch the filename pattern in `bii_super_learner.py:319` to allow a suffix.

The commands below assume the recommended `_ft_run_{id}` ordering. If you keep
`_run_{id}_ft`, the inference still works — only the stacker step needs the patch.

---

## 1. Inference (val + test only)

We do **not** infer the train split: the inverse-uncertainty ensemble, all ensemble weights,
and isotonic calibration are fit on `--val_split` and applied to test only. Train predictions
are used solely for the optional `--export_base_per_molecule_csv_dir --train_split ...` dump.

```bash
# Family 1: mc_dropout
for id in 0 1 2 3 4; do
  python bii.py --mode infer \
    --checkpoint_path "saved_models/bii_mc_dmpnn_balanced_ft_run_${id}.pt" \
    --inference_all_splits \
    --train_file "" \
    --val_file HEK293_test_BM.pkl \
    --test_file tox21_all.pkl
done

# Family 2: evidential
for id in 0 1 2 3 4; do
  python bii.py --mode infer \
    --checkpoint_path "saved_models/bii_new_dmpnn_balanced_ft_run_${id}.pt" \
    --inference_all_splits \
    --train_file "" \
    --val_file HEK293_test_BM.pkl \
    --test_file tox21_all.pkl
done
```

Notes:
- `--inference_all_splits` is driven by `--train_file/--val_file/--test_file`. Passing
  `--train_file ""` skips the train split.
- Output split-directory name = pkl stem, so it must match the `--val_split` /
  `--test_pred_split` you pass to the stacker below.

Produces, per run:
```
inference_outputs/HEK293_test_BM/dmpnn_bii_mc_dmpnn_balanced_ft_run_{id}.csv
inference_outputs/tox21_all/dmpnn_bii_mc_dmpnn_balanced_ft_run_{id}.csv
inference_outputs/HEK293_test_BM/dmpnn_bii_new_dmpnn_balanced_ft_run_{id}.csv
inference_outputs/tox21_all/dmpnn_bii_new_dmpnn_balanced_ft_run_{id}.csv
```

---

## 2. Stacking + UQ evaluation (2 base models)

Activity-cliff analysis is force-disabled in `bii_super_learner.py`, so no `ac_label` column
is required.

```bash
TS=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="inference_outputs/stacking/plain_cp_2base_${TS}"

python bii_super_learner.py \
  --pred_root inference_outputs \
  --label_dir cytotoxicity_data \
  --val_split HEK293_test_BM \
  --test_split tox21_all \
  --test_pred_split tox21_all \
  --label_column Outcome \
  --methods "dmpnn_bii_mc_dmpnn_balanced_ft,dmpnn_bii_new_dmpnn_balanced_ft" \
  --run_ids "0,1,2,3,4" \
  --summary_csv "${OUT_DIR}/stacking_summary_all_metrics.csv" \
  --conformal_summary_csv "${OUT_DIR}/stacking_summary_conformal_accepted.csv" \
  --uncertainty_dist_dir "${OUT_DIR}/artifacts" \
  --uncertainty_tau 2.0 \
  --no_per_molecule_csv
```

The stacker auto-computes every ensemble scheme (uniform, uncertainty_inverse_isotonic,
softmax_brier_model_score, brier_opt, logloss_opt) and evaluates both prediction metrics and
UQ (ECE/reliability, conformal coverage, uncertainty distributions) for each.

Optional knobs:
- `--no_test_labels` — if test ground-truth is unavailable; CP on test still runs from
  val-calibrated scores.
- `--conformal_method csa` (or `csa_lc`) — make 2-model disagreement explicit in conformal sets.
- `--uncertainty_weight_source epistemic` and `--disagreement_lambda <x>` — tune the fused UQ score.

---

## Pre-flight checklist

1. Both checkpoint families exist in `saved_models/` for runs 0–4.
2. After inference, the four CSV families exist under `inference_outputs/HEK293_test_BM/`
   and `inference_outputs/tox21_all/` and end in `_run_<id>.csv`.
3. `cytotoxicity_data/{HEK293_test_BM,tox21_all}.pkl` have the `Outcome` column and row counts
   matching their prediction CSVs.
