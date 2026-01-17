# ============================================================================
# streamlit_app.py
# Heart Disease Classification – ML Assignment 2 (BITS Pilani – M.Tech AIML/DSE)
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================  
# PAGE CONFIG  
# ============================================================================
st.set_page_config(
    page_title="Heart Disease Prediction – ML Assignment 2",
    page_icon="❤️",
    layout="wide"
)

# ============================================================================  
# HEADER  
# ============================================================================
st.title("❤️ Heart Disease Prediction – ML Assignment 2")
st.markdown("""
This Streamlit web application demonstrates 6 machine learning models trained on  
the **Heart Disease UCI dataset**.  
You may **upload test data**, select a model, and view **evaluation metrics**.

**Models Implemented:**  
- Logistic Regression  
- Decision Tree  
- k‑Nearest Neighbors  
- Naive Bayes  
- Random Forest  
- XGBoost  
""")

# ============================================================================  
# LOAD ARTIFACTS  
# ============================================================================
MODEL_DIR = "model"

try:
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
except:
    st.error("❌ scaler.pkl or feature_names.pkl missing inside /model folder.")
    st.stop()

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "k_nearest_neighbors.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

# ============================================================================  
# SIDEBAR – DATASET INFO + SAMPLE DOWNLOAD  
# ============================================================================
st.sidebar.header("📘 Dataset Information")
st.sidebar.markdown("""
This project uses the **Heart Disease UCI dataset**  
containing **13 numerical features** + **1 optional target**.
""")

sample_path = "data/test_sample.csv"
if os.path.exists(sample_path):
    sample_df = pd.read_csv(sample_path)
    st.sidebar.download_button(
        label="📥 Download Sample Test Data (100+ rows)",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_test_data.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")

# ============================================================================  
# STEP 1 — REQUIRED CSV FORMAT + TEMPLATE DOWNLOADS  
# ============================================================================
st.header("📤 Step 1: Upload Test Dataset (CSV Only)")

st.markdown("""
Your uploaded CSV **must contain exactly the following 13 features**:

| Feature | Meaning |
|--------|---------|
| age | Age in years |
| sex | Sex (1=male, 0=female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Cholesterol |
| fbs | Fasting blood sugar >120 mg/dl |
| restecg | Resting ECG |
| thalach | Max heart rate |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of ST segment |
| ca | Number of vessels (0–4) |
| thal | Thalassemia (0–3) |

**Optional:** Add a `target` column (0/1) to compute evaluation metrics.
""")

# Template downloads
st.subheader("📥 Download CSV Templates")

template_without_target = pd.DataFrame(columns=feature_names)
st.download_button(
    label="📄 Blank Template (13 Features)",
    data=template_without_target.to_csv(index=False).encode("utf-8"),
    file_name="template_without_target.csv",
    mime="text/csv"
)

template_with_target = pd.DataFrame(columns=feature_names + ["target"])
st.download_button(
    label="📄 Template with Target Column",
    data=template_with_target.to_csv(index=False).encode("utf-8"),
    file_name="template_with_target.csv",
    mime="text/csv"
)

# Upload CSV
uploaded_file = st.file_uploader("Upload your test data CSV here:", type=["csv"])

if uploaded_file is None:
    st.info("👆 Please upload a CSV file to continue.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except:
    st.error("❌ Could not read file. Make sure it is a valid CSV.")
    st.stop()

if df.empty:
    st.error("❌ Uploaded CSV is empty.")
    st.stop()

st.success("📄 File uploaded successfully.")
with st.expander("Preview Uploaded Data"):
    st.dataframe(df.head(), use_container_width=True)

# Validate feature columns
expected = set(feature_names)
uploaded = set(df.columns)

missing = expected - uploaded
extra = uploaded - (expected | {"target"})

if missing:
    st.error(f"❌ Missing required columns: {list(missing)}")
    st.stop()

if extra:
    st.error(f"❌ Unexpected extra columns: {list(extra)}")
    st.stop()

# Extract features + optional target
if "target" in df.columns:
    X_test = df[feature_names]
    y_test = df["target"]
    has_target = True
else:
    X_test = df[feature_names]
    y_test = None
    has_target = False
    st.warning("Target column not found → metrics will NOT be computed.")

# Validate numeric data
try:
    X_test = X_test.astype(float)
except:
    st.error("❌ Non-numeric values found in feature columns.")
    st.stop()

# ============================================================================  
# STEP 2 — MODEL SELECTION  
# ============================================================================
st.header("🤖 Step 2: Select a Model")

model_choice = st.selectbox("Choose a model:", list(model_files.keys()), index=0)
model_path = f"{MODEL_DIR}/{model_files[model_choice]}"

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()

model = joblib.load(model_path)
st.success(f"✅ Loaded: {model_choice}")

# ============================================================================  
# STEP 3 — SCALING & PREDICTION  
# ============================================================================
X_scaled = scaler.transform(X_test)
pred = model.predict(X_scaled)
pred_prob = model.predict_proba(X_scaled)[:, 1]

# ============================================================================  
# STEP 4 — METRICS  
# ============================================================================
st.header("📊 Step 3: Evaluation Metrics")

if has_target:
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, pred_prob)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    mcc = matthews_corrcoef(y_test, pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.4f}")
    col1.metric("AUC", f"{auc:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col2.metric("Recall", f"{recall:.4f}")
    col3.metric("F1 Score", f"{f1:.4f}")
    col3.metric("MCC", f"{mcc:.4f}")

else:
    st.info("Upload CSV WITH a `target` column to compute metrics.")

# ============================================================================  
# STEP 5 — CONFUSION MATRIX + CLASSIFICATION REPORT  
# ============================================================================
st.header("🔢 Step 4: Confusion Matrix & Classification Report")

if has_target:
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"], ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(y_test, pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("Upload target column to view confusion matrix and classification report.")

# ============================================================================  
# STEP 6 — DOWNLOAD FINAL PREDICTIONS  
# ============================================================================
st.header("📥 Step 5: Download Predictions")

output_df = X_test.copy()
output_df["Predicted"] = pred
output_df["Confidence (%)"] = (pred_prob * 100).round(2)
output_df["Label"] = output_df["Predicted"].map({0: "No Disease", 1: "Disease"})

if has_target:
    output_df["Actual"] = y_test
    output_df["Correct"] = (pred == y_test).map({True: "Yes", False: "No"})

st.dataframe(output_df, use_container_width=True)

st.download_button(
    label="📥 Download Predictions CSV",
    data=output_df.to_csv(index=False).encode("utf-8"),
    file_name=f"predictions_{model_choice.replace(' ', '_')}.csv",
    mime="text/csv"
)

# ============================================================================  
# FOOTER  
# ============================================================================
st.markdown("---")
st.markdown(
    "<center>Developed as part of BITS Pilani – ML Assignment 2</center>",
    unsafe_allow_html=True
)
