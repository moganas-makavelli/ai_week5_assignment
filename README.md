# 🏥 AI System for Predicting Patient Readmission Risk

This project develops an AI-driven predictive system that estimates the likelihood of a patient being readmitted to the hospital within 30 days of discharge.  
It is part of the **PLP Academy AI Project Submission**.

---

## 📘 Project Overview

Hospital readmissions significantly affect healthcare costs and patient care quality.  
This project leverages machine learning—specifically a **Random Forest Classifier**—to predict patient readmission risk based on clinical and demographic data.  

The model is integrated into an interactive **Streamlit web application**, enabling healthcare professionals to input patient data and receive real-time risk predictions.

---

## 📊 Objectives

1. Predict patient readmission using structured healthcare data.  
2. Support healthcare providers in proactive patient management.  
3. Improve healthcare outcomes by minimizing avoidable readmissions.

---

## 👥 Stakeholders

- Hospital administrators  
- Healthcare professionals (doctors, nurses, and analysts)  
- Data scientists and IT system developers  

---

## 🧠 Technical Workflow

### 1️⃣ Problem Definition
Define the challenge and KPIs such as model **accuracy**, **recall**, and **precision**.

### 2️⃣ Data Collection
Data sources include:
- **Electronic Health Records (EHRs)**
- **Demographic and insurance datasets**

### 3️⃣ Data Preprocessing
Performed in `src/preprocessing.py`, including:
- Handling missing values  
- Normalizing numeric features  
- Encoding categorical variables  

### 4️⃣ Model Development
Conducted via `src/train_model.py`:
- Algorithm: **Random Forest Classifier**  
- Split: 70% Training | 15% Validation | 15% Test  
- Tuned Hyperparameters: `n_estimators`, `max_depth`  
- Model artifacts stored in `/models/`

### 5️⃣ Evaluation
Implemented in `src/evaluate_model.py` using:
- **Precision:** 0.78  
- **Recall:** 0.88  
- **Confusion Matrix Example:**

|                    | Predicted Positive | Predicted Negative |
|--------------------|--------------------|--------------------|
| **Actual Positive** | 70                 | 10                 |
| **Actual Negative** | 20                 | 100                |

### 6️⃣ Deployment
- Streamlit-based web interface (`app.py`)  
- Integration-ready API endpoint for hospital dashboards  
- Data anonymization and encryption (HIPAA/GDPR compliance)

---

## ⚖️ Ethical and Practical Considerations

- **Bias Mitigation:** Applied re-sampling and fairness-aware algorithms.  
- **Interpretability vs Accuracy:** Preference for interpretable models in clinical use.  
- **Regulatory Compliance:** HIPAA/GDPR adherence through anonymization and secure storage.  

---

AI_Hospital_Readmission_Prediction/
│
├── app.py # Streamlit web app
├── requirements.txt # Python dependencies
│
├── data/
│ ├── sample_patient_data.csv
│ └── generate_sample_data.py
│
├── models/
│ ├── readmit_model.joblib
│ ├── preprocessor.joblib
│ └── feature_columns.joblib
│
└── src/
├── preprocessing.py
├── train_model.py
├── evaluate_model.py
└── predict.py
---

## ⚙️ Installation & Local Setup

### 🔧 Prerequisites
- Python 3.9 or higher  
- pip (latest version)  

### 🪜 Steps
```bash
# 1. Clone the repository
git clone https://github.com/moganas-makavelli/ai_week5_assignment.git
cd ai_week5_assignment

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate       # On Windows
# or
source venv/bin/activate    # On Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py

☁️ Streamlit Cloud Deployment

You can deploy this project online for free using Streamlit Cloud.

🌍 Deployment Steps

Push all your files to GitHub (main branch).

Go to https://share.streamlit.io
.

Log in with your GitHub account.

Click “New app”.

Select:

Repository: moganas-makavelli/ai_week5_assignment

Branch: main

Main file path: app.py

Click Deploy 🚀

Streamlit Cloud will automatically:

Install dependencies from requirements.txt

Launch your app online

🧠 How It Works

User enters patient demographic and medical details.

Data is preprocessed using preprocessor.joblib.

Model (readmit_model.joblib) predicts the probability of hospital readmission.

The prediction result is displayed instantly in the Streamlit interface.

🧾 Requirements

Make sure these packages are included (via requirements.txt):

streamlit

scikit-learn

pandas

numpy

joblib

👨‍💻 Developer

collaborators: Morgan Omondi,
GitHub: moganas-makavelli

Project: PLP Academy AI Week 5 Assignment

Contributions, issues, and suggestions are welcome — feel free to fork and improve the system!


🛡️ License

This project is licensed under the MIT License.
You are free to use, modify, and distribute this software with attribution.


🌟 Acknowledgements

Streamlit
 — for deployment and UI

Scikit-learn
 — for ML model training

Pandas
 — for data handling

Joblib
 — for model serialization
## 🧩 Project Structure

# ai_week5_assignment
