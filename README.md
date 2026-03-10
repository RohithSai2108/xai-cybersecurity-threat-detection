# XAI Cybersecurity Threat Detection (UNSW-NB15 + TON_IoT)

This repository provides a reproducible baseline for your capstone workflow using a **2-dataset strategy**:

- `UNSW-NB15`
- `TON_IoT`

It implements the three next steps you requested:

1. **Common attack mapping** across both datasets.
2. **Unified preprocessing pipeline** with label-style encoding, standard scaling, and SMOTE.
3. **Model comparison + SHAP** with within-dataset and cross-dataset reporting.

## What is included

- `xai_threat_pipeline.py`: end-to-end experiment runner.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected input data

Prepare two CSV files:

- UNSW-NB15 CSV (must contain a target label column)
- TON_IoT CSV (must contain a target label column)

You can set label column names via CLI flags.

## Run

```bash
python xai_threat_pipeline.py \
  --unsw-csv /path/to/UNSW_NB15.csv \
  --ton-csv /path/to/TON_IoT.csv \
  --unsw-label-col attack_cat \
  --ton-label-col type \
  --out-dir outputs
```

## Outputs

The script writes:

- `within_dataset_metrics.csv` (train/test split metrics on source dataset)
- `cross_dataset_metrics.csv` (train on one dataset, test on the other)
- `shap_summary_train_<dataset>.png` (SHAP bar plot for RF model)
- `common_features.json` (aligned shared feature list)

## Notes

- The script maps raw dataset labels into a common taxonomy:
  - `Normal`
  - `DoS/DDoS`
  - `Recon/Probe`
  - `R2L/BruteForce`
  - `Other`
- Cross-dataset evaluation uses **shared columns only**.
- SHAP is generated for the `RandomForest` model (best fit for TreeExplainer).
