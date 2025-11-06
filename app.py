# app.py
"""
Streamlit app to demo the readmission prediction system.
Features:
- Upload CSV or use sample dataset
- Show data preview
- Run predictions using saved model
- Show confusion matrix, ROC AUC, and top feature importances (if available)
- Download predictions as CSV
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score
import seaborn as sns  # allowed for app visuals

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "readmit_model.joblib"
FEATURES_FILE = ROOT / "models" / "feature_columns.joblib"

st.set_page_config(page_title="Patient Readmission Predictor", layout="wide")

st.title("🏥 Patient 30-Day Readmission Risk — Demo")
st.markdown(
    "Upload a CSV with patient features (same columns as sample dataset) or use the sample dataset.")


@st.cache_data
def load_sample():
    return pd.read_csv(ROOT / "data" / "sample_patient_data.csv")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURES_FILE)
    return model, features


def predict(model, df):
    X = df[features]
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(
        model, "predict_proba") else np.zeros(len(preds))
    out = df.copy()
    out["predicted_readmit"] = preds
    out["readmit_probability"] = proba
    return out


with st.sidebar:
    st.header("Options")
    use_sample = st.checkbox("Use sample dataset", value=True)
    uploaded_file = st.file_uploader("Or upload CSV", type=["csv"])
    show_metrics = st.checkbox(
        "Show evaluation metrics (requires true label column 'readmitted_30d')", value=True)
    thresh = st.slider(
        "Probability threshold for positive prediction", 0.0, 1.0, 0.5)

# Load data
if use_sample or (uploaded_file is None):
    df = load_sample()
else:
    df = pd.read_csv(uploaded_file)

st.subheader("Data preview")
st.dataframe(df.head(10))

# Ensure model & features exist
try:
    model, features = load_model()
except Exception as e:
    st.error(
        f"Could not load model. Make sure 'models/readmit_model.joblib' and 'models/preprocessor.joblib' exist. Error: {e}")
    st.stop()

# Check required columns
missing = [c for c in features if c not in df.columns]
if missing:
    st.error(f"Uploaded data is missing required columns: {missing}")
    st.stop()

# Predict
with st.spinner("Predicting..."):
    out = predict(model, df)

st.subheader("Predictions")
st.dataframe(out[["patient_id"] + [c for c in out.columns if c not in df.columns or c in [
             'readmit_probability', 'predicted_readmit']]].head(15))

# Download predictions
csv = out.to_csv(index=False).encode('utf-8')
st.download_button(label="Download predictions CSV", data=csv,
                   file_name="predictions.csv", mime="text/csv")

# If true labels present and user wants metrics, compute them
if show_metrics and "readmitted_30d" in df.columns:
    y_true = df["readmitted_30d"].values
    y_proba = out["readmit_probability"].values
    y_pred_thresh = (y_proba >= thresh).astype(int)

    precision = precision_score(y_true, y_pred_thresh)
    recall = recall_score(y_true, y_pred_thresh)
    st.metric("Precision", f"{precision:.3f}")
    st.metric("Recall", f"{recall:.3f}")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred_thresh)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # ROC curve
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_proba):.3f}")
        ax2.plot([0, 1], [0, 1], "--")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend()
        st.pyplot(fig2)

# Feature importances (if RandomForest)
if hasattr(model.named_steps["clf"], "feature_importances_"):
    st.subheader("Top feature importances (model)")
    # reconstruct feature names after preprocessing: numeric + onehot names
    preprocessor = model.named_steps["preprocessor"]
    # numeric names:
    num_cols = preprocessor.transformers_[0][2]
    cat_pipe = preprocessor.transformers_[1][1]
    cat_cols = preprocessor.transformers_[1][2]
    # onehot feature names
    try:
        onehot: 'OneHotEncoder' = cat_pipe.named_steps["onehot"]
        onehot_names = list(onehot.get_feature_names_out(cat_cols))
    except Exception:
        onehot_names = cat_cols
    all_feature_names = list(num_cols) + onehot_names
    importances = model.named_steps["clf"].feature_importances_
    fi = pd.DataFrame({"feature": all_feature_names,
                      "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(15)
    st.table(fi.reset_index(drop=True))

st.info("This demo is for educational purposes. Do not use the model clinically without rigorous validation, regulatory approvals, and privacy safeguards.")
