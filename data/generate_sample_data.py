# data/generate_sample_data.py
"""
Generate a synthetic patient dataset for demo/testing.
Columns:
- patient_id, age, gender, num_prior_admissions, length_of_stay,
  comorbidity_score (0-5), lab_result_1, lab_result_2, discharge_disposition, readmitted_30d
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT / "sample_patient_data.csv"


def generate(n=2000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    patient_id = np.arange(1, n+1)
    age = np.random.randint(18, 95, size=n)
    gender = np.random.choice(
        ["Male", "Female", "Other"], size=n, p=[0.48, 0.48, 0.04])
    num_prior_admissions = np.random.poisson(0.8, size=n)
    length_of_stay = np.clip(np.random.exponential(
        scale=3, size=n).astype(int) + 1, 1, 60)
    comorbidity_score = np.random.choice([0, 1, 2, 3, 4, 5], size=n, p=[
                                         0.2, 0.25, 0.2, 0.15, 0.12, 0.08])
    lab_result_1 = np.round(np.random.normal(
        loc=100, scale=20, size=n), 1)  # e.g., glucose
    lab_result_2 = np.round(np.random.normal(
        loc=7.4, scale=0.5, size=n), 2)  # e.g., pH
    discharge_disposition = np.random.choice(
        ["Home", "SNF", "AMA", "Other"], size=n, p=[0.7, 0.15, 0.05, 0.1])

    # Create a synthetic risk score to derive readmission label (not realistic, just for demo)
    risk_score = (
        0.02 * (age - 18) +
        0.5 * num_prior_admissions +
        0.3 * comorbidity_score +
        0.05 * length_of_stay +
        np.where(discharge_disposition == "SNF", 1.0, 0.0) +
        np.random.normal(0, 1, size=n)
    )
    # Convert risk into probability and sample binary outcome
    prob_readmit = 1 / (1 + np.exp(- (risk_score - 3.5)))  # sigmoid shift
    readmitted_30d = (np.random.rand(n) < prob_readmit).astype(int)

    df = pd.DataFrame({
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "num_prior_admissions": num_prior_admissions,
        "length_of_stay": length_of_stay,
        "comorbidity_score": comorbidity_score,
        "lab_result_1": lab_result_1,
        "lab_result_2": lab_result_2,
        "discharge_disposition": discharge_disposition,
        "readmitted_30d": readmitted_30d
    })
    return df


if __name__ == "__main__":
    df = generate(3000)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved synthetic dataset to {OUT_CSV}")
