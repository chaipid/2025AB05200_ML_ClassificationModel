import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json
import os

st.set_page_config(page_title="ML Classification Comparison", layout="wide")

# ------------------------------------------------------------
# Load Preprocessor
# ------------------------------------------------------------
@st.cache_resource
def load_preprocessor():
    return joblib.load("model/preprocessor.pkl")

preprocessor = load_preprocessor()

# ------------------------------------------------------------
# Load Models + Metrics
# ------------------------------------------------------------
def load_model(name):
    return joblib.load(f"model/{name.lower().replace(' ', '_')}.pkl")

with open("model/metrics.json", "r") as f:
    metrics = json.load(f)

model_list = list(metrics.keys())

# ------------------------------------------------------------
# Cleaning Function
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
        "Partner","Dependents","PhoneService","MultipleLines",
        "OnlineSecurity","OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies","PaperlessBilling"
    ]

    for col in yes_no_cols:
        df[col] = df[col].replace({"y": "yes", "n": "no", "true": "yes", "false": "no"})

    df["InternetService"] = df["InternetService"].replace({"fiber": "fiber optic", "none": "no"})

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df

# ------------------------------------------------------------
# SECTION 1: Dataset Overview
# ------------------------------------------------------------
st.header("📁 Dataset Information")

required_cols = [
    "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod",
    "MonthlyCharges","TotalCharges"
]

# Sample Template
sample_df = pd.DataFrame(columns=required_cols)

st.download_button(
    "📥 Download Sample Dataset Template",
    sample_df.to_csv(index=False).encode("utf-8"),
    "sample_telco_template.csv",
    "text/csv"
)

uploaded = st.file_uploader("Upload CSV (matching required structure)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Missing columns: {missing}")
        st.stop()
    df = clean_data(df)
    st.success("Custom dataset loaded successfully.")
else:
    st.info("Using default test dataset (no file uploaded).")
    df = pd.read_csv("model/test_default.csv")
    df = clean_data(df)

# Dataset Quick Stats
st.subheader("📊 Quick Stats")
colA, colB, colC, colD = st.columns(4)

colA.metric("Instances", len(df))
colB.metric("Features", len(df.columns))
colC.metric("Classes", 2)
colD.metric("Type", "Classification")

# Dataset Preview
st.subheader("🔎 First Five Rows")
st.dataframe(df.head(), width="stretch")

# ------------------------------------------------------------
# SECTION 2: Model Evaluation
# ------------------------------------------------------------
st.markdown("---")
st.header("🤖 Model Evaluation")

selected_model = st.selectbox("Select a Model", model_list)
model = load_model(selected_model)

X = preprocessor.transform(df)

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
st.subheader("📈 Evaluation Metrics Table")

m = metrics[selected_model]

metric_table = pd.DataFrame({
    "Metric": ["Accuracy","AUC","Precision","Recall","F1 Score","MCC"],
    "Value": [
        m["accuracy"], m["auc"], m["precision"],
        m["recall"], m["f1"], m["mcc"]
    ]
})
metric_table["Value"] = metric_table["Value"].round(4)

st.dataframe(metric_table, width="stretch")

# ------------------------------------------------------------
# Confusion Matrix (smaller size)
# ------------------------------------------------------------
st.subheader("📉 Confusion Matrix")

cm = np.array(m["confusion_matrix"])

fig, ax = plt.subplots(figsize=(4,3))  # reduced size
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ------------------------------------------------------------
# Classification Report - Table
# ------------------------------------------------------------
st.subheader("📄 Classification Report")

report_df = pd.DataFrame(m["classification_report"]).T.round(3)
report_df.index.name = "Class"

st.dataframe(report_df, width="stretch")
