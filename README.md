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
1. Create + activate a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\\Scripts\\activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

## Expected input data
The pipeline expects these files by default:

- `data/UNSW_NB15.csv`
- `data/TON_IoT.csv`

Each CSV must contain a label column. If your CSVs use different label column names, override them with CLI flags.

## Run

Run end-to-end from the project folder:

```bash
python xai_threat_pipeline.py --out-dir outputs
```

Optional: point to your own dataset files:

```bash
python xai_threat_pipeline.py \
  --unsw-csv /path/to/UNSW_NB15.csv \
  --ton-csv /path/to/TON_IoT.csv \
  --unsw-label-col attack_cat \
  --ton-label-col label \
  --out-dir outputs
```

## Outputs

The script writes:

- `within_dataset_metrics.csv` (train/test split metrics on source dataset)
- `cross_dataset_metrics.csv` (train on one dataset, test on the other)
- `shap_summary_train_<dataset>.png` (SHAP bar plot for RF model)
- `common_features.json` (aligned shared feature list)

## Notes
- Label handling:
  - If the label columns contain string attack categories (like `normal`, `dos`, etc.), the script maps them into a common taxonomy (`Normal`, `DoS/DDoS`, `Recon/Probe`, `R2L/BruteForce`, `Other`).
  - If the label columns are numeric/binary (like `0/1`), the script keeps them as-is so training/evaluation still works.
- Cross-dataset evaluation uses **shared columns only**.
- SHAP is generated for the `RandomForest` model (best fit for TreeExplainer).
- The script uses `matplotlib`'s non-GUI `Agg` backend, so it works reliably in background/CLI runs.
- For speed on large datasets, the pipeline caps oversampled training size and may skip the expensive models (SVM/ANN) automatically.
