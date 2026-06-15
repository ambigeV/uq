# OOD Uncertainty Envelope — CLI Runbook

A post-hoc analysis tool that fits a CSA-style geometric envelope in
K-dimensional epistemic uncertainty space (one axis per base model) to
characterise which test molecules are in-distribution vs out-of-distribution,
and how OOD status correlates with AC / non-AC / non-AC (no MMP match) labels.

No new model is trained. Everything runs on top of existing inference outputs.

---

## New CLI flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--ood_envelope` | bool flag | off | Enable OOD envelope fitting and export. |
| `--ood_normalisation` | str | `none` | Per-model uncertainty normalisation fitted on val. Choices: `none`, `standardise`, `rank`, `percentile`. |
| `--ood_percentile_q` | float | `95.0` | Percentile used when `--ood_normalisation percentile`. |
| `--ood_envelope_source` | str | `all` | Val samples used to fit envelope. `all` = entire val set; `non_ac_only` = only val samples with `ac_label="non-AC"`. |
| `--ood_alpha` | float | `None` | OOD significance level. If omitted, falls back to `--conformal_alpha`. |

Reused existing flags (no changes needed):

| Flag | Role in OOD |
|---|---|
| `--csa_n_directions` | Number of random projection directions M (default 128). |
| `--csa_split_ratio` | Fraction of val for stage-1 envelope shape fitting (default 0.5). |
| `--csa_seed` | RNG seed for direction sampling and split. |
| `--conformal_alpha` | Fallback OOD threshold level if `--ood_alpha` is not set. |

---

## New output files

All files land in the same directory as `--summary_csv`.
`{norm}` = value of `--ood_normalisation`, `{src}` = value of `--ood_envelope_source`.

| File | Description |
|---|---|
| `ood_uncertainty_scatter_{norm}_{src}.png` | 2D (K=2) or 3D (K=3) scatter of test molecules in epistemic uncertainty space. Colour = AC category. Filled circles = in-distribution, hollow X = OOD. |
| `ood_uncertainty_scatter_{norm}_{src}.csv` | Per-molecule: `molecule_idx`, `T_score`, `ood` (bool), `ac_label`. |
| `ood_ac_label_grid_{norm}_{src}.png` | Heatmap: rows = In-dist / OOD, cols = AC label categories. Column-normalised (within each category, % in-dist vs OOD). |
| `ood_ac_label_grid_{norm}_{src}_row_norm.png` | Same heatmap, row-normalised (within in-dist/OOD, % per category). |
| `ood_ac_label_grid_{norm}_{src}.csv` | Raw counts and percentages backing both heatmaps. |

Notes:
- Scatter plot is only generated for K=2 or K=3 base models.
- AC label grid is only generated when `--test_split` points to a pkl with an `ac_label` column (e.g. `tox21_all_ACA`).
- `T_score` is a continuous OOD score; higher = further outside the val envelope.

---

## Normalisation options

| `--ood_normalisation` | What it does |
|---|---|
| `none` | Raw epistemic uncertainties as-is. Envelope shape can be dominated by whichever model has the largest absolute scale. |
| `standardise` | Subtract val mean, divide by val std per model. Equal axis contribution, assumes roughly Gaussian uncertainty distributions. |
| `rank` | Map each model's val uncertainties to [0,1] rank positions; apply the same val-fitted mapping to test via interpolation. Robust to outliers. |
| `percentile` | Divide by val `--ood_percentile_q` percentile per model. Interpretable: "fraction of max val uncertainty". |

Normalisation is always **fitted on val only** and applied to both val and test — no leakage.

---

## Envelope source options

| `--ood_envelope_source` | Val samples used |
|---|---|
| `all` | Entire val set. Envelope represents the full distribution of in-distribution chemistry. |
| `non_ac_only` | Only val samples where `ac_label="non-AC"`. Envelope represents clean, non-cliff in-distribution chemistry. AC and non-MMP-match test samples can then be assessed against a stricter baseline. Requires val pkl to have an `ac_label` column. |

---

## Example CLIs

### Minimal (no normalisation, all val, default alpha)

```bash
TS=$(date +"%Y%m%d_%H%M%S")
OUT="inference_outputs/stacking/ood_analysis_${TS}"

python bii_super_learner.py \
  --pred_root inference_outputs \
  --label_dir cytotoxicity_data \
  --val_split HEK293_test_BM \
  --test_split tox21_all_ACA \
  --label_column Outcome \
  --methods "dmpnn_bii_mc_dmpnn_balanced,dmpnn_bii_new_dmpnn_balanced" \
  --run_ids "0,1,2,3,4" \
  --summary_csv "${OUT}/stacking_summary_all_metrics.csv" \
  --conformal_summary_csv "${OUT}/stacking_summary_conformal_accepted.csv" \
  --uncertainty_dist_dir "${OUT}/artifacts" \
  --uncertainty_tau 2.0 \
  --no_per_molecule_csv \
  --ood_envelope \
  --ood_normalisation none \
  --ood_envelope_source all
```

### Percentile normalisation, non-AC envelope, tighter alpha

```bash
TS=$(date +"%Y%m%d_%H%M%S")
OUT="inference_outputs/stacking/ood_nonac_${TS}"

python bii_super_learner.py \
  --pred_root inference_outputs \
  --label_dir cytotoxicity_data \
  --val_split HEK293_test_BM \
  --test_split tox21_all_ACA \
  --label_column Outcome \
  --methods "dmpnn_bii_mc_dmpnn_balanced,dmpnn_bii_new_dmpnn_balanced" \
  --run_ids "0,1,2,3,4" \
  --summary_csv "${OUT}/stacking_summary_all_metrics.csv" \
  --conformal_summary_csv "${OUT}/stacking_summary_conformal_accepted.csv" \
  --uncertainty_dist_dir "${OUT}/artifacts" \
  --uncertainty_tau 2.0 \
  --no_per_molecule_csv \
  --ood_envelope \
  --ood_normalisation percentile \
  --ood_percentile_q 95.0 \
  --ood_envelope_source non_ac_only \
  --ood_alpha 0.05 \
  --csa_n_directions 256
```

### 3-base-model variant (produces 3D scatter)

```bash
TS=$(date +"%Y%m%d_%H%M%S")
OUT="inference_outputs/stacking/ood_3base_${TS}"

python bii_super_learner.py \
  --pred_root inference_outputs \
  --label_dir cytotoxicity_data \
  --val_split HEK293_test_BM \
  --test_split tox21_all_ACA \
  --label_column Outcome \
  --methods "dmpnn_bii_mc_dmpnn_balanced,dmpnn_bii_new_dmpnn_balanced,ensemble" \
  --run_ids "0,1,2,3,4" \
  --summary_csv "${OUT}/stacking_summary_all_metrics.csv" \
  --conformal_summary_csv "${OUT}/stacking_summary_conformal_accepted.csv" \
  --uncertainty_dist_dir "${OUT}/artifacts" \
  --uncertainty_tau 2.0 \
  --no_per_molecule_csv \
  --ood_envelope \
  --ood_normalisation standardise \
  --ood_envelope_source all
```

---

## How it works (summary)

1. Collect per-model epistemic uncertainty vectors across all runs and average → shape `(K, N)`.
2. Optionally normalise each model's axis using val statistics only.
3. Optionally restrict val to `non-AC` samples before fitting.
4. Split val into two halves (I1, I2).
5. Stage 1 — sample M random directions in the positive orthant of the K-dimensional sphere; binary-search for the tightest envelope `beta*` that covers `>= 1-alpha` of I1.
6. Stage 2 — compute scalar T-scores on I2 and set final conformal threshold `t_hat`.
7. Score each test molecule: `T(u) > t_hat` → OOD.
8. Export scatter plot and AC-label grid.

The approach deliberately avoids isotonic rescaling or any scalarisation — the K-dimensional geometry of inter-model uncertainty is preserved throughout.
