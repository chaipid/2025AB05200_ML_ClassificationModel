import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------
# Streamlit Page Settings
# ------------------------------------------------------------
st.set_page_config(page_title="ML Model Comparison", layout="wide")

# ------------------------------------------------------------
# Load Preprocessor
# ------------------------------------------------------------
@st.cache_resource
def load_preprocessor():
    try:
        return joblib.load("model/preprocessor.pkl")
    except:
        st.error("❌ preprocessor.pkl not found.")
        return None

preprocessor = load_preprocessor()

# ------------------------------------------------------------
# Load Saved ML Models
# ------------------------------------------------------------
def load_model(name):
    path = f"model/{name.lower().replace(' ', '_')}.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"Model file not found: {path}")
        return None

# ------------------------------------------------------------
# Load Metrics
# ------------------------------------------------------------
with open("model/metrics.json", "r") as f:
    metrics = json.load(f)

model_list = list(metrics.keys())

# ------------------------------------------------------------
# Dataset Choices
# ------------------------------------------------------------
st.title("📊 Machine Learning Classification – Model Comparison Dashboard")

dataset_choice = st.selectbox(
    "Select Dataset Source:",
    ["Telco Churn Dataset (Default)", "Upload Custom Dataset"]
)

required_cols = [
    "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod",
    "MonthlyCharges","TotalCharges"
]

# ------------------------------------------------------------
# Data Cleaning (Matches your training notebook)
# ------------------------------------------------------------
def clean_data(df):
    df = df.copy()

    categorical_cols = [
        "gender","Partner","Dependents","PhoneService","MultipleLines",
        "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies","Contract",
        "PaperlessBilling","PaymentMethod"
    ]
    numeric_cols = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]

    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    yes_no_cols = [
        "Partner","Dependents","PhoneService","MultipleLines","OnlineSecurity",
        "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
        "StreamingMovies","PaperlessBilling"
    ]
    for col in yes_no_cols:
        df[col] = df[col].replace({
            "yes": "yes", "no": "no",
            "y": "yes", "n": "no",
            "true": "yes", "false": "no"
        })

    df["InternetService"] = df["InternetService"].replace({
        "fiber optic": "fiber optic",
        "fiber": "fiber optic",
        "dsl": "dsl",
        "none": "no"
    })

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
if dataset_choice == "Telco Churn Dataset (Default)":
    df = pd.read_csv("model/test_default.csv")
    st.info("Loaded default Telco test dataset.")
    df = clean_data(df)

else:
    uploaded = st.file_uploader("Upload custom CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"❌ Missing required columns: {missing}")
            st.stop()

        df = clean_data(df)
        st.success("Custom dataset loaded and cleaned.")
    else:
        st.warning("Upload a CSV to proceed.")
        st.stop()

st.subheader("📌 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# ------------------------------------------------------------
# Model Selection
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🤖 Select a Model")

selected_model = st.selectbox("Choose model:", model_list)
model = load_model(selected_model)

if preprocessor is None or model is None:
    st.stop()

# ------------------------------------------------------------
# Preprocess & Predict
# ------------------------------------------------------------
X = preprocessor.transform(df)
pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

results = df.copy()
results["Prediction"] = pred
results["Probability"] = prob.round(4)

# ------------------------------------------------------------
# Show Metrics
# ------------------------------------------------------------
m = metrics[selected_model]

st.markdown("### 📈 Performance Metrics")
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

col1.metric("Accuracy", round(m["accuracy"],4))
col2.metric("AUC", round(m["auc"],4))
col3.metric("Precision", round(m["precision"],4))
col4.metric("Recall", round(m["recall"],4))
col5.metric("F1 Score", round(m["f1"],4))
col6.metric("MCC", round(m["mcc"],4))

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------
st.markdown("### 🔢 Confusion Matrix")

cm = np.array(m["confusion_matrix"])

fig, ax = plt.subplots(figsize=(4,3))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ------------------------------------------------------------
# Classification Report
# ------------------------------------------------------------
st.markdown("### 📄 Classification Report")
st.json(m["classification_report"])

# ------------------------------------------------------------
# Predictions Output
# ------------------------------------------------------------
st.markdown("### 🔮 Prediction Results")
st.dataframe(results.head(50), use_container_width=True)

st.download_button(
    "📥 Download Predictions",
    results.to_csv(index=False).encode("utf-8"),
    "predictions.csv",
    "text/csv"
)
