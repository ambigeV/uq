# Weighted CP CLI Runbook

This runbook provides the full CLI workflow for:

1. training the domain classifier,
2. running `bii_super_learner.py` for **all data + plain CP**,
3. running `bii_super_learner.py` for **all data + weighted CP**.

The five model names are fixed as:

- `dmpnn_bii_mc_dmpnn_balanced`
- `dmpnn_bii_mc_unb_dmpnn_balanced`
- `dmpnn_bii_new_dmpnn_balanced`
- `dmpnn_bii_new_unb_dmpnn_balanced`
- `ensemble`

---

## 1) Train Domain Classifier (for weighted CP)

```bash
python train_weighted_cp_domain_classifier.py \
  --label_dir cytotoxicity_data \
  --val_split HEK293_test_BM \
  --test_split tox21_all \
  --encoder_type identity \
  --label_column Outcome \
  --smiles_column SMILES \
  --epochs 30 \
  --batch_size 256 \
  --lr 3e-3 \
  --seed 0 \
  --out_model saved_models/weighted_cp_domain_classifier.pt
```

---

## 2) Super Learner: All Data + Plain CP (unweighted conformal)

This run records:

- all-data metrics (`scenario=all`)
- plain conformal metrics (`scenario=conformal`, without domain-ratio weighting)

```bash
TS=$(date +"%Y%m%d_%H%M%S")
OUT_DIR_PLAIN="inference_outputs/stacking/plain_cp_${TS}"

python bii_super_learner.py \
  --pred_root inference_outputs \
  --label_dir cytotoxicity_data \
  --val_split HEK293_test_BM \
  --test_split tox21_all \
  --label_column Outcome \
  --methods "dmpnn_bii_mc_dmpnn_balanced,dmpnn_bii_mc_unb_dmpnn_balanced,dmpnn_bii_new_dmpnn_balanced,dmpnn_bii_new_unb_dmpnn_balanced,ensemble" \
  --run_ids "0,1,2,3,4" \
  --summary_csv "${OUT_DIR_PLAIN}/stacking_summary_all_metrics.csv" \
  --conformal_summary_csv "${OUT_DIR_PLAIN}/stacking_summary_conformal_accepted.csv" \
  --uncertainty_dist_dir "${OUT_DIR_PLAIN}/artifacts"
```

---

## 3) Super Learner: All Data + Weighted CP

This run records:

- all-data metrics (`scenario=all`)
- weighted conformal metrics (`scenario=conformal`, with domain-ratio weights)

```bash
TS=$(date +"%Y%m%d_%H%M%S")
OUT_DIR_WEIGHTED="inference_outputs/stacking/weighted_cp_${TS}"

python bii_super_learner.py \
  --pred_root inference_outputs \
  --label_dir cytotoxicity_data \
  --val_split HEK293_test_BM \
  --test_split tox21_all \
  --label_column Outcome \
  --methods "dmpnn_bii_mc_dmpnn_balanced,dmpnn_bii_mc_unb_dmpnn_balanced,dmpnn_bii_new_dmpnn_balanced,dmpnn_bii_new_unb_dmpnn_balanced,ensemble" \
  --run_ids "0,1,2,3,4" \
  --weighted_cp_domain_model saved_models/weighted_cp_domain_classifier.pt \
  --weighted_cp_prob_clip 1e-2 \
  --weighted_cp_ratio_offset 1e-3 \
  --weighted_cp_weight_clip_max 1000 \
  --summary_csv "${OUT_DIR_WEIGHTED}/stacking_summary_all_metrics.csv" \
  --conformal_summary_csv "${OUT_DIR_WEIGHTED}/stacking_summary_conformal_accepted.csv" \
  --uncertainty_dist_dir "${OUT_DIR_WEIGHTED}/artifacts"
```

---

## Result Files To Compare

- Plain CP run:
  - `${OUT_DIR_PLAIN}/stacking_summary_all_metrics.csv`
  - `${OUT_DIR_PLAIN}/stacking_summary_conformal_accepted.csv`
- Weighted CP run:
  - `${OUT_DIR_WEIGHTED}/stacking_summary_all_metrics.csv`
  - `${OUT_DIR_WEIGHTED}/stacking_summary_conformal_accepted.csv`

