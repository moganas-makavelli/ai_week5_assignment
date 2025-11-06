# src/train_model.py
"""
Train a RandomForest classifier using the preprocessor.
Performs a quick GridSearch on n_estimators and max_depth.
Saves the best model as models/readmit_model.joblib
"""

import pandas as pd
from pathlib import Path
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_patient_data.csv"
PREPROCESSOR_FILE = ROOT / "models" / "preprocessor.joblib"
MODEL_FILE = ROOT / "models" / "readmit_model.joblib"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["patient_id", "readmitted_30d"])
    y = df["readmitted_30d"]
    return X, y


def train():
    X, y = load_data()
    preprocessor = joblib.load(PREPROCESSOR_FILE)

    # split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # pipeline
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [5, 10, None]
    }

    # lightweight grid search for demo purposes
    grid = GridSearchCV(pipe, param_grid, cv=3,
                        scoring="recall", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    best = grid.best_estimator_

    # Evaluate on test set
    y_pred = best.predict(X_test)
    print("Classification report (test):")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(best, MODEL_FILE)
    print(f"Saved trained model to {MODEL_FILE}")


if __name__ == "__main__":
    train()
