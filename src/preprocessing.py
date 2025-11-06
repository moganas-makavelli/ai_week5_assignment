# src/preprocessing.py
"""
Preprocessing pipeline using sklearn:
- imputes missing values
- encodes categorical variables
- scales numeric features
Saves the fitted preprocessor (joblib) so the same transforms are used in production.
"""

import pandas as pd
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_patient_data.csv"
OUT_PATH = ROOT / "models"
OUT_PATH.mkdir(parents=True, exist_ok=True)

PREPROCESSOR_FILE = OUT_PATH / "preprocessor.joblib"
FEATURES_FILE = OUT_PATH / "feature_columns.joblib"


def build_preprocessor(df: pd.DataFrame):
    # identify columns
    target_col = "readmitted_30d"
    feature_cols = [c for c in df.columns if c !=
                    target_col and c != "patient_id"]
    numeric_cols = df[feature_cols].select_dtypes(
        include=["int64", "float64"]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(
        include=["object", "category"]).columns.tolist()

    # numeric pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # categorical pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor, feature_cols, numeric_cols, categorical_cols


def fit_and_save_preprocessor():
    df = pd.read_csv(DATA_PATH)
    preprocessor, feature_cols, numeric_cols, categorical_cols = build_preprocessor(
        df)
    print("Fitting preprocessor on full dataset (demo).")
    preprocessor.fit(df[feature_cols])

    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    joblib.dump(feature_cols, FEATURES_FILE)
    print(f"Saved preprocessor to {PREPROCESSOR_FILE}")
    print(f"Saved feature column names to {FEATURES_FILE}")


if __name__ == "__main__":
    fit_and_save_preprocessor()
