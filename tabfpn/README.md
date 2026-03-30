# TabPFN Client API Quick UCI Test

This folder provides a minimal setup to test **TabPFN client (API)** on small UCI-style tabular datasets using on-demand train/test splits.

## What is included

- `requirements.txt`: dependencies with `tabpfn-client`
- `uci_incontext_eval.py`: benchmark script on:
  - Iris (UCI)
  - Wine (UCI)
  - Breast Cancer Wisconsin (UCI)

The script uses API model path `v2.5_default` by default, so it targets the TabPFN v2.5 family.

## Setup

```bash
cd tabfpn
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Authentication

Option 1 (recommended): set a token in env var

```bash
export TABPFN_ACCESS_TOKEN="your_token_here"
```

Option 2: pass token at runtime

```bash
python uci_incontext_eval.py --access-token "your_token_here"
```

You can also run the interactive login/init flow via the client automatically (default behavior).

## Run

```bash
python uci_incontext_eval.py
```

Optional settings:

```bash
python uci_incontext_eval.py \
  --train-sizes 16 32 64 128 \
  --n-splits 5 \
  --model-path v2.5_default \
  --n-estimators 8
```

This prints mean accuracy and macro-F1 for each dataset and train size, which gives a quick signal of TabPFN's in-context learning behavior under low-data, on-demand splits.
