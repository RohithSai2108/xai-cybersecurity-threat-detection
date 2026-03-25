#!/usr/bin/env python3
"""End-to-end XAI cyber threat pipeline for UNSW-NB15 + TON_IoT.

Implements:
1) Common attack mapping
2) Unified preprocessing (label-style categorical encoding + scaling + SMOTE)
3) Model comparison (RF/SVM/ANN) + SHAP + within/cross-dataset evaluation
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

# Force a non-interactive backend so the script works reliably in background/CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


COMMON_CLASSES = ["Normal", "DoS/DDoS", "Recon/Probe", "R2L/BruteForce", "Other"]

UNSW_MAP = {
    "normal": "Normal",
    "backdoor": "R2L/BruteForce",
    "analysis": "Recon/Probe",
    "fuzzers": "Other",
    "reconnaissance": "Recon/Probe",
    "shellcode": "R2L/BruteForce",
    "dos": "DoS/DDoS",
    "exploits": "R2L/BruteForce",
    "worms": "Other",
    "generic": "Other",
}

TON_MAP = {
    "normal": "Normal",
    "ddos": "DoS/DDoS",
    "dos": "DoS/DDoS",
    "scanning": "Recon/Probe",
    "reconnaissance": "Recon/Probe",
    "password": "R2L/BruteForce",
    "xss": "R2L/BruteForce",
    "injection": "R2L/BruteForce",
    "mitm": "Other",
    "backdoor": "R2L/BruteForce",
    "ransomware": "Other",
}


@dataclass
class DatasetBundle:
    name: str
    X: pd.DataFrame
    y: pd.Series


def normalize_label(value: object) -> str:
    return str(value).strip().lower()


def map_attack_labels(labels: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    # If labels are numeric (e.g., 0/1 flags), keep them as-is so we still have multiple classes.
    numeric = pd.to_numeric(labels, errors="coerce")
    if numeric.notna().sum() / max(1, len(labels)) > 0.9:
        # Use integer labels for modeling (e.g., 0 = normal, 1 = attack)
        return pd.Series(pd.Categorical(numeric.astype(int)), index=labels.index)

    mapped = labels.map(lambda x: mapping.get(normalize_label(x), "Other"))
    return pd.Series(pd.Categorical(mapped, categories=COMMON_CLASSES), index=labels.index)


def load_dataset(path: Path, label_col: str, dataset_name: str) -> DatasetBundle:
    # pandas defaults to utf‑8; some of the provided files in this repo use
    # latin‑1/Windows‑1252 which causes a UnicodeDecodeError. Try utf‑8 first
    # and fall back to latin‑1 so the pipeline doesn’t crash at startup.
    mapping = UNSW_MAP if dataset_name == "UNSW-NB15" else TON_MAP

    # In this repo, UNSW-NB15 is saved without a usable header row, so a normal
    # `pd.read_csv(path)` causes the first data row to be treated as headers.
    # That leads to unnecessary extra full-file reads and lots of downstream
    # confusion. Load UNSW as headerless up-front.
    if dataset_name == "UNSW-NB15":
        try:
            df = pd.read_csv(path, header=None, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, header=None, low_memory=False, encoding="latin-1")
    else:
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

    # If the label column isn't present, try to infer it from the data.
    if label_col not in df.columns:
        # Common case: CSV has no header row (pandas treated the first data row as headers).
        # Read without a header so we can inspect all columns.
        # Fast inference: scan only a small prefix to avoid a full-column pass over
        # very large UNSW CSVs (which can make the pipeline appear "stuck").
        sample_rows = 5000
        if dataset_name == "UNSW-NB15":
            df_no_header = df
            df_sample = df_no_header.head(sample_rows)
        else:
            try:
                df_no_header = pd.read_csv(path, header=None, low_memory=False)
            except UnicodeDecodeError:
                df_no_header = pd.read_csv(path, header=None, low_memory=False, encoding="latin-1")
            try:
                df_sample = pd.read_csv(path, header=None, nrows=sample_rows, low_memory=False)
            except UnicodeDecodeError:
                df_sample = pd.read_csv(path, header=None, nrows=sample_rows, low_memory=False, encoding="latin-1")

        keys = set(mapping.keys())
        best_col = None
        best_score = 0.0
        for col in df_sample.columns:
            values = df_sample[col].dropna().astype(str).str.strip().str.lower()
            if values.empty:
                continue
            match_count = values.isin(keys).sum()
            score = match_count / len(values)
            if score > best_score:
                best_score = score
                best_col = col

        # Use the best column if it has any meaningful match; otherwise fall back.
        if best_col is not None and best_score > 0.01:
            label_col = best_col
            df = df_no_header
        else:
            # Fallback: use the last column (often a binary attack flag) and warn.
            label_col = df_no_header.columns[-1]
            df = df_no_header
            print(
                f"WARNING: {dataset_name}: label column inferred as last column (fallback).",
                "If the datasets are pre-encoded, the taxonomy mapping may be bypassed.",
            )

    y_raw = df[label_col]
    # Drop rows where the label is missing; these cannot be used for training.
    nonnull_mask = y_raw.notna() & (y_raw.astype(str).str.strip() != "")
    df = df[nonnull_mask]
    y_raw = y_raw[nonnull_mask]

    y_raw = df[label_col]
    X = df.drop(columns=[label_col])

    y = map_attack_labels(y_raw, mapping)

    # Remove likely ID-like fields if present.
    drop_candidates = ["id", "flow_id", "timestamp", "ts"]
    X = X.drop(columns=[c for c in drop_candidates if c in X.columns], errors="ignore")
    return DatasetBundle(dataset_name, X, y)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    class PandasToNumeric(BaseEstimator, TransformerMixin):
        """
        Coerce mixed-type numeric columns (e.g., '-' strings) into real numbers.

        This prevents SimpleImputer(median) from crashing during cross-dataset
        `transform()` when the same column name has different dtypes.
        """

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            if isinstance(X, pd.DataFrame):
                return X.apply(pd.to_numeric, errors="coerce").to_numpy()
            return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").to_numpy()

    numeric_pipeline = Pipeline(
        steps=[
            ("to_numeric", PandasToNumeric()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "label_encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, object]:
    """
    Train models with safety guards for large datasets.

    SVC (RBF) can become prohibitively slow on the oversampled dataset size; when the
    training set is large we only train RandomForest (and keep SHAP consistent).
    """
    n = len(X_train)

    models: Dict[str, object] = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }
    # Enable the more expensive models only when the training set is reasonably sized.
    if n <= 10000:
        models["SVM"] = SVC(kernel="rbf", probability=True, random_state=42)
    if n <= 10000:
        models["ANN"] = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)

    for model in models.values():
        model.fit(X_train, y_train)
    return models


def align_common_features(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    common = sorted(set(a.columns).intersection(b.columns))

    # If no shared column names, align by position (first N columns) to enable
    # cross-dataset evaluation when the files use different headers or no headers.
    if not common:
        min_cols = min(a.shape[1], b.shape[1])
        common = [f"f{i}" for i in range(min_cols)]
        a = a.iloc[:, :min_cols].copy()
        b = b.iloc[:, :min_cols].copy()
        a.columns = common
        b.columns = common
        print(
            "WARNING: No shared feature names detected; aligning by position using",
            f"the first {min_cols} columns.",
        )
        return a, b, common

    return a[common].copy(), b[common].copy(), common


def save_shap_bar(model: RandomForestClassifier, X_sample: np.ndarray, out_path: Path) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Handle both binary and multiclass outputs.
    if isinstance(shap_values, list):
        values = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
    else:
        values = np.abs(shap_values)

    shap.summary_plot(values, X_sample, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_within_dataset(bundle: DatasetBundle, out_dir: Path) -> pd.DataFrame:
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=0.2,
        random_state=42,
        stratify=bundle.y,
    )

    preprocessor = build_preprocessor(X_train_df)
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # Cap oversampled training size to keep end-to-end runtime reasonable.
    max_train_rows = 30000
    if len(X_train_bal) > max_train_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train_bal), size=max_train_rows, replace=False)
        X_train_bal = X_train_bal[idx]
        y_train_bal = y_train_bal[idx]

    models = train_models(X_train_bal, y_train_bal)

    rows = []
    for name, model in models.items():
        pred = model.predict(X_test)
        metrics = evaluate(y_test, pred)
        rows.append({"dataset": bundle.name, "model": name, **metrics})

    save_shap_bar(models["RandomForest"], X_train_bal[: min(1000, len(X_train_bal))], out_dir / f"shap_summary_train_{bundle.name}.png")

    return pd.DataFrame(rows)


def run_cross_dataset(source: DatasetBundle, target: DatasetBundle) -> pd.DataFrame:
    X_source, X_target, _ = align_common_features(source.X, target.X)

    preprocessor = build_preprocessor(X_source)
    X_source_t = preprocessor.fit_transform(X_source)
    X_target_t = preprocessor.transform(X_target)

    smote = SMOTE(random_state=42)
    X_source_bal, y_source_bal = smote.fit_resample(X_source_t, source.y)

    # Cap oversampled training size to keep end-to-end runtime reasonable.
    max_train_rows = 30000
    if len(X_source_bal) > max_train_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_source_bal), size=max_train_rows, replace=False)
        X_source_bal = X_source_bal[idx]
        y_source_bal = y_source_bal[idx]

    models = train_models(X_source_bal, y_source_bal)

    rows = []
    for name, model in models.items():
        pred = model.predict(X_target_t)
        metrics = evaluate(target.y, pred)
        rows.append({"train_on": source.name, "test_on": target.name, "model": name, **metrics})

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=
            "Run the XAI threat pipeline. If no CSV paths are provided, "
            "the script will attempt to use `UNSW_.csv` and `TON_.csv` in the "
            "current directory."
    )
    p.add_argument(
        "--unsw-csv",
        type=Path,
        default=Path("data/UNSW_NB15.csv"),
        help="path to the UNSW-NB15 CSV file (default: data/UNSW_NB15.csv)",
    )
    p.add_argument(
        "--ton-csv",
        type=Path,
        default=Path("data/TON_IoT.csv"),
        help="path to the TON_IoT CSV file (default: data/TON_IoT.csv)",
    )
    p.add_argument("--unsw-label-col", type=str, default="attack_cat")
    # In this repo's TON CSV, `label` is a binary attack flag; default to it so
    # UNSW (pre-encoded) and TON use the same label style.
    p.add_argument("--ton-label-col", type=str, default="label")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ensure the provided/default CSV files exist
    if not args.unsw_csv.exists():
        raise FileNotFoundError(f"UNSW CSV not found at {args.unsw_csv}")
    if not args.ton_csv.exists():
        raise FileNotFoundError(f"TON CSV not found at {args.ton_csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    unsw = load_dataset(args.unsw_csv, args.unsw_label_col, "UNSW-NB15")
    ton = load_dataset(args.ton_csv, args.ton_label_col, "TON_IoT")

    unsw_aligned, ton_aligned, common_features = align_common_features(unsw.X, ton.X)
    unsw.X, ton.X = unsw_aligned, ton_aligned

    within_df = pd.concat(
        [
            run_within_dataset(unsw, args.out_dir),
            run_within_dataset(ton, args.out_dir),
        ],
        ignore_index=True,
    )
    within_df.to_csv(args.out_dir / "within_dataset_metrics.csv", index=False)

    cross_df = pd.concat(
        [
            run_cross_dataset(unsw, ton),
            run_cross_dataset(ton, unsw),
        ],
        ignore_index=True,
    )
    cross_df.to_csv(args.out_dir / "cross_dataset_metrics.csv", index=False)

    with open(args.out_dir / "common_features.json", "w", encoding="utf-8") as fp:
        json.dump(common_features, fp, indent=2)

    print("Saved outputs to", args.out_dir)


if __name__ == "__main__":
    main()
