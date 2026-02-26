# `bii.py` Commands (`--reload`)

Run from project root:

```bash
cd /Users/tingyangwei/PycharmProjects/UQ
```

Thread limits (recommended for runtime stability):

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

---

## A) Epoch=1 Debug Commands (Train + Infer)

### A1. Identity debug train (`epochs=1`)

```bash
python bii.py \
  --mode train \
  --data_dir cytotoxicity_data \
  --reload \
  --epochs 1 \
  --model_name bii_identity_debug
```

### A2. Identity debug inference (single file)

```bash
python bii.py \
  --mode infer \
  --checkpoint_path saved_models/bii_identity_debug.pt \
  --inference_input cytotoxicity_data/tox21_all.pkl \
  --reload
```

### A3. Identity debug inference (all PKL splits)

```bash
python bii.py \
  --mode infer \
  --checkpoint_path saved_models/bii_identity_debug.pt \
  --inference_all_splits \
  --reload
```

Expected output files include:

```text
inference_outputs/tox21_all/identity_bii_identity_debug.csv
inference_outputs/HEK293_train_BM/identity_bii_identity_debug.csv
inference_outputs/HEK293_test_BM/identity_bii_identity_debug.csv
```

### A4. DMPNN debug train (`epochs=1`)

```bash
python bii.py \
  --mode train \
  --use_graph \
  --data_dir cytotoxicity_data \
  --reload \
  --epochs 1 \
  --model_name bii_dmpnn_debug
```

### A5. DMPNN debug inference (single file)

```bash
python bii.py \
  --mode infer \
  --checkpoint_path saved_models/bii_dmpnn_debug.pt \
  --inference_input cytotoxicity_data/tox21_all.pkl \
  --reload
```

### A6. DMPNN debug inference (all PKL splits)

```bash
python bii.py \
  --mode infer \
  --checkpoint_path saved_models/bii_dmpnn_debug.pt \
  --inference_all_splits \
  --reload
```

Expected output files include:

```text
inference_outputs/tox21_all/dmpnn_bii_dmpnn_debug.csv
inference_outputs/HEK293_train_BM/dmpnn_bii_dmpnn_debug.csv
inference_outputs/HEK293_test_BM/dmpnn_bii_dmpnn_debug.csv
```

---

## B) Full Multi-Run Commands (5 runs) -> all `.pt` + all CSVs

### B1. Identity 5-run training

```bash
for i in 0 1 2 3 4; do
  python bii.py \
    --mode train \
    --data_dir cytotoxicity_data \
    --reload \
    --epochs 1 \
    --model_name "bii_identity_run_${i}"
done
```

### B2. Identity 5-run inference on all splits

```bash
for i in 0 1 2 3 4; do
  python bii.py \
    --mode infer \
    --checkpoint_path "saved_models/bii_identity_run_${i}.pt" \
    --inference_all_splits \
    --reload
done
```

### B3. DMPNN 5-run training

```bash
for i in 0 1 2 3 4; do
  python bii.py \
    --mode train \
    --use_graph \
    --data_dir cytotoxicity_data \
    --reload \
    --epochs 1 \
    --model_name "bii_dmpnn_run_${i}"
done
```

### B4. DMPNN 5-run inference on all splits

```bash
for i in 0 1 2 3 4; do
  python bii.py \
    --mode infer \
    --checkpoint_path "saved_models/bii_dmpnn_run_${i}.pt" \
    --inference_all_splits \
    --reload
done
```

Example generated files:

```text
saved_models/bii_identity_run_0.pt
saved_models/bii_dmpnn_run_0.pt
inference_outputs/tox21_all/identity_bii_identity_run_0.csv
inference_outputs/HEK293_train_BM/dmpnn_bii_dmpnn_run_0.csv
```

---

## C) Windows-Friendly Commands (PowerShell)

Use PowerShell in project root:

```powershell
Set-Location "C:\Users\tingyangwei\PycharmProjects\UQ"
```

Optional thread limits (session-only):

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:VECLIB_MAXIMUM_THREADS="1"
```

### C1. Epoch=1 debug (Identity)

```powershell
python bii.py --mode train --data_dir cytotoxicity_data --reload --epochs 1 --model_name bii_identity_debug
python bii.py --mode infer --checkpoint_path saved_models/bii_identity_debug.pt --inference_input cytotoxicity_data/tox21_all.pkl --reload
python bii.py --mode infer --checkpoint_path saved_models/bii_identity_debug.pt --inference_all_splits --reload
```

### C2. Epoch=1 debug (DMPNN)

```powershell
python bii.py --mode train --use_graph --data_dir cytotoxicity_data --reload --epochs 1 --model_name bii_dmpnn_debug
python bii.py --mode infer --checkpoint_path saved_models/bii_dmpnn_debug.pt --inference_input cytotoxicity_data/tox21_all.pkl --reload
python bii.py --mode infer --checkpoint_path saved_models/bii_dmpnn_debug.pt --inference_all_splits --reload
```

### C3. 5 runs (Identity) + all-split inference

```powershell
0..4 | ForEach-Object {
  python bii.py --mode train --data_dir cytotoxicity_data --reload --epochs 1 --model_name "bii_identity_run_$($_)"
}

0..4 | ForEach-Object {
  python bii.py --mode infer --checkpoint_path "saved_models/bii_identity_run_$($_).pt" --inference_all_splits --reload
}
```

### C4. 5 runs (DMPNN) + all-split inference

```powershell
0..4 | ForEach-Object {
  python bii.py --mode train --use_graph --data_dir cytotoxicity_data --reload --epochs 1 --model_name "bii_dmpnn_run_$($_)"
}

0..4 | ForEach-Object {
  python bii.py --mode infer --checkpoint_path "saved_models/bii_dmpnn_run_$($_).pt" --inference_all_splits --reload
}
```
