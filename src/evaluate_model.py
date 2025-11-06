# src/evaluate_model.py
"""
Load saved model and evaluate on test set; returns metrics and confusion matrix.
This script prints metrics when run standalone.
"""

import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_patient_data.csv"
MODEL_FILE = ROOT / "models" / "readmit_model.joblib"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["patient_id", "readmitted_30d"])
    y = df["readmitted_30d"]
    return X, y


def evaluate():
    X, y = load_data()
    model = joblib.load(MODEL_FILE)

    # split same way as training script
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(
        model, "predict_proba") else None

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist()
    }
    return results


if __name__ == "__main__":
    r = evaluate()
    print("Evaluation results:")
    for k, v in r.items():
        print(f"{k}: {v}")
