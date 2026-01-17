import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json
import os

# ------------------------------------------------------------
# Streamlit Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="Classification Model Comparison", layout="wide")


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

    # Normalize categorical
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

    # Numeric conversion
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ------------------------------------------------------------
# SECTION 1 – Dataset Information
# ------------------------------------------------------------
st.header("📁 Dataset Information")

st.write("""
This dataset contains customer subscription details, service enrollments, 
billing information, and churn labels. It supports binary classification tasks 
for predicting customer churn in the telecommunications domain.
""")


# ------------------------------------------------------------
# SECTION 2 – Feature Information
# ------------------------------------------------------------
st.header("📌 Feature Overview")

st.write("""
The attributes in this dataset fall into the following categories:
""")

st.markdown("""
- **Customer Demographics:** gender, SeniorCitizen, Partner, Dependents  
- **Account Details:** tenure, Contract, PaymentMethod, PaperlessBilling  
- **Services Subscribed:** PhoneService, InternetService, StreamingTV, TechSupport, etc.  
- **Billing Metrics:** MonthlyCharges, TotalCharges  
- **Target Variable:** Churn (Yes / No)
""")


# ------------------------------------------------------------
# Dataset Upload + Template + Sample Data
# ------------------------------------------------------------
st.subheader("📂 Upload or Download Dataset")

required_cols = [
    "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod",
    "MonthlyCharges","TotalCharges"
]

template_df = pd.DataFrame(columns=required_cols)

colT, colS = st.columns(2)

# Template Download
colT.download_button(
    "📄 Download Input Template",
    template_df.to_csv(index=False).encode("utf-8"),
    "template_dataset.csv",
    "text/csv"
)

# Sample 100 rows
default_data_path = "model/test_default.csv"
if os.path.exists(default_data_path):
    default_df = pd.read_csv(default_data_path)
    sample100 = default_df.sample(n=min(100, len(default_df)), random_state=42)

    colS.download_button(
        "📥 Download Sample Test Data (100 rows)",
        sample100.to_csv(index=False).encode("utf-8"),
        "sample_test_100.csv",
        "text/csv"
    )


uploaded = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    df = clean_data(df)
    st.success("Dataset uploaded successfully.")
else:
    st.info("Using default evaluation dataset.")
    df = pd.read_csv(default_data_path)
    df = clean_data(df)


# ------------------------------------------------------------
# SECTION 3 – Dataset Exploration
# ------------------------------------------------------------
st.header("📊 Dataset Exploration")

colA, colB, colC, colD = st.columns(4)

colA.metric("Total Records", len(df))
colB.metric("Number of Features", df.shape[1])
colC.metric("Target Classes", 2)
colD.metric("Task Type", "Binary Classification")


# ------------------------------------------------------------
# SECTION 4 – Dataset Preview (Separate Section)
# ------------------------------------------------------------
st.header("🔍 Dataset Preview")
st.dataframe(df.head(), width="stretch")


# ------------------------------------------------------------
# SECTION 5 – Implemented Models
# ------------------------------------------------------------
st.header("🧠 Models Included in Comparison")

model_info = pd.DataFrame([
    ["Logistic Regression", "Linear Model", "Interpretable, strong baseline"],
    ["Decision Tree", "Tree-Based", "Handles non-linearity, easy to visualize"],
    ["KNN", "Instance-Based", "Distance-driven, sensitive to scaling"],
    ["Naive Bayes", "Probabilistic", "Fast, assumption-based"],
    ["Random Forest", "Ensemble (Bagging)", "Reduces overfitting, robust"],
    ["XGBoost", "Ensemble (Boosting)", "High performance for tabular data"]
], columns=["Model", "Category", "Characteristics"])

st.dataframe(model_info, width="stretch")


# ------------------------------------------------------------
# SECTION 6 – Evaluation Metric Definitions
# ------------------------------------------------------------
st.header("📈 Metric Definitions")

metric_def = pd.DataFrame([
    ["Accuracy", "Proportion of correctly classified samples", "Higher is better"],
    ["AUC-ROC", "Ability to distinguish positive vs negative classes", "Higher is better"],
    ["Precision", "Correctness of predicted positive labels", "Higher is better"],
    ["Recall", "Coverage of actual positive labels", "Higher is better"],
    ["F1 Score", "Balance between precision and recall", "Higher is better"],
    ["MCC", "Correlation measure for binary outcomes", "Closer to +1 is better"]
], columns=["Metric", "Description", "Desired Direction"])

st.dataframe(metric_def, width="stretch")


# ------------------------------------------------------------
# SECTION 7 – Model Performance Evaluation
# ------------------------------------------------------------
st.header("📊 Model Performance Evaluation")

selected_model = st.selectbox("Select Model", model_list)
model = load_model(selected_model)

X = preprocessor.transform(df)
m = metrics[selected_model]


# Metrics Table
st.subheader("Evaluation Results")

result_table = pd.DataFrame({
    "Metric": ["Accuracy","AUC","Precision","Recall","F1 Score","MCC"],
    "Value": [
        m["accuracy"], m["auc"], m["precision"],
        m["recall"], m["f1"], m["mcc"]
    ]
}).round(4)

st.dataframe(result_table, width="stretch")


# Confusion Matrix
st.subheader("Confusion Matrix")

cm = np.array(m["confusion_matrix"])

fig, ax = plt.subplots(figsize=(2.2, 1.8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)


# Classification Report
st.subheader("Classification Report")

report_df = pd.DataFrame(m["classification_report"]).T.round(3)
report_df.index.name = "Class"

st.dataframe(report_df, width="stretch")
