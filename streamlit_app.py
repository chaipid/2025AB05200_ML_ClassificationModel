# ============================================================================
# streamlit_app.py
# ML Assignment 2 - Heart Disease Classification
# BITS Pilani – M.Tech AIML / DSE
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

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
This Streamlit application demonstrates 6 ML classification models trained on the 
Heart Disease UCI dataset.  
To evaluate a model, upload a **test dataset (CSV)** containing **all 13 features and a target column**.

**Models Implemented:** Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost  
""")

# ============================================================================
# SIDEBAR — DATASET INFORMATION
# ============================================================================
st.sidebar.header("📘 Dataset Information")
st.sidebar.markdown("""
This project uses the **Heart Disease UCI dataset** with:

- **13 numerical clinical features**
- **Binary target:** 0 = No Disease, 1 = Disease  
- **1025 total records**

Your uploaded CSV **must contain exactly these 14 columns**:

1. age  
2. sex  
3. cp  
4. trestbps  
5. chol  
6. fbs  
7. restecg  
8. thalach  
9. exang  
10. oldpeak  
11. slope  
12. ca  
13. thal  
14. target  (required for metrics)
""")

st.sidebar.info(f"📦 Using XGBoost version: {xgb.__version__}")

# ============================================================================
# LOAD MODELS + SCALER
# ============================================================================
MODEL_DIR = "model"

try:
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
except:
    st.error("❌ scaler.pkl or feature_names.pkl missing inside /model.")
    st.stop()

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "k_nearest_neighbors.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

# ============================================================================
# SECTION 1 — TEMPLATE & SAMPLE DATA DOWNLOADS
# ============================================================================
st.header("📁 Step 1: Download Template / Sample Data")

# Template WITH target (required)
template = pd.DataFrame(columns=feature_names + ["target"])
st.download_button(
    "📄 Download Required Template (with target)",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="template_with_target.csv",
    mime="text/csv"
)

# Sample dataset
if os.path.exists("data/test_sample.csv"):
    sample_df = pd.read_csv("data/test_sample.csv")
    st.download_button(
        "📥 Download Sample Test Dataset (100+ rows)",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_test_data.csv",
        mime="text/csv"
    )
else:
    st.warning("Sample test_sample.csv not found.")

st.markdown("---")

# ============================================================================
# SECTION 2 — UPLOAD DATA
# ============================================================================
st.header("📤 Step 2: Upload Test Dataset (CSV Required)")

uploaded = st.file_uploader("Upload CSV file:", type=["csv"])

# Auto-load sample if available
if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Test dataset uploaded successfully.")
elif os.path.exists("data/test_sample.csv"):
    df = pd.read_csv("data/test_sample.csv")
    st.info("Using default sample test data.")
else:
    st.error("Please upload a CSV to continue.")
    st.stop()

# Show preview
with st.expander("📄 Preview Uploaded Data"):
    st.dataframe(df.head(), use_container_width=True)

# ============================================================================
# VALIDATION — TARGET REQUIRED
# ============================================================================
required_cols = set(feature_names + ["target"])
uploaded_cols = set(df.columns)

missing = required_cols - uploaded_cols
extra = uploaded_cols - required_cols

if missing:
    st.error(f"❌ Missing required columns: {missing}")
    st.stop()

if extra:
    st.error(f"❌ Unexpected extra columns: {extra}")
    st.stop()

# Extract features & target
X_test = df[feature_names]
y_test = df["target"]

# Ensure numeric
try:
    X_test = X_test.astype(float)
except:
    st.error("❌ Non-numeric values detected in feature columns.")
    st.stop()

# ============================================================================
# SECTION 3 — MODEL SELECTION
# ============================================================================
st.header("🤖 Step 3: Select a Model")

model_choice = st.selectbox("Choose one:", list(model_files.keys()), index=0)
model_path = f"{MODEL_DIR}/{model_files[model_choice]}"

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

model = joblib.load(model_path)
st.success(f"Loaded model: {model_choice}")

# ============================================================================
# SECTION 4 — RESULTS
# ============================================================================
st.header("📊 Step 4: Model Results & Metrics")

# Scale + Predict
X_scaled = scaler.transform(X_test)
pred = model.predict(X_scaled)
pred_prob = model.predict_proba(X_scaled)[:, 1]

# Evaluation Metrics
st.subheader("📈 Evaluation Metrics")

acc = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, pred_prob)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
mcc = matthews_corrcoef(y_test, pred)

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{acc:.4f}")
col1.metric("AUC Score", f"{auc:.4f}")
col2.metric("Precision", f"{prec:.4f}")
col2.metric("Recall", f"{rec:.4f}")
col3.metric("F1 Score", f"{f1:.4f}")
col3.metric("MCC", f"{mcc:.4f}")

st.markdown("---")

# Confusion Matrix
st.subheader("🔢 Confusion Matrix")

cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"],
    ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.markdown("---")

# Classification Report
st.subheader("📄 Classification Report")
report = classification_report(y_test, pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

# ============================================================================
# SECTION 5 — DOWNLOAD PREDICTIONS
# ============================================================================
st.header("📥 Step 5: Download Predictions")

output = X_test.copy()
output["Actual"] = y_test
output["Predicted"] = pred
output["Confidence (%)"] = (pred_prob * 100).round(2)
output["Correct"] = (pred == y_test).map({True: "Yes", False: "No"})

st.dataframe(output.head(), use_container_width=True)

st.download_button(
    "📥 Download Predictions CSV",
    data=output.to_csv(index=False).encode("utf-8"),
    file_name=f"predictions_{model_choice.replace(' ', '_')}.csv",
    mime="text/csv"
)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    "<center>Developed for ML Assignment 2 – BITS Pilani</center>",
    unsafe_allow_html=True
)
