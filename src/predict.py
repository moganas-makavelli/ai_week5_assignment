# src/predict.py
"""
Utility to load model and preprocessor and make predictions for a DataFrame.
Returns predicted label and probability.
"""

import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "readmit_model.joblib"
FEATURES_FILE = ROOT / "models" / "feature_columns.joblib"


def load_model():
    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURES_FILE)
    return model, features


def predict_dataframe(df: pd.DataFrame):
    model, features = load_model()
    X = df[features]
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(
        model, "predict_proba") else None
    result = df.copy()
    result["predicted_readmit"] = preds
    if proba is not None:
        result["readmit_probability"] = proba
    return result


if __name__ == "__main__":
    # simple demo when run standalone (requires models to exist)
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "sample_patient_data.csv").head(10)
    print(predict_dataframe(df))
