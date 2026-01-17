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
# Radio Navigation
# ------------------------------------------------------------
section = st.radio(
    "Navigate",
    ["Dataset Overview", "Model Evaluation"],
    horizontal=True
)


# ============================================================
# SECTION 1 — DATASET OVERVIEW
# ============================================================
if section == "Dataset Overview":

    st.header("📁 Dataset Overview")

    st.write("""
    This dataset contains customer subscription, service usage, and billing characteristics, 
    used to predict the likelihood of customer churn in a telecom environment.
    """)

    # Dataset Source + Credits
    st.subheader("Dataset Description")
    st.write("""
    - **Source:** IBM Sample Data Repository  
    - **URL:** https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/  
    - **Credits:** IBM Corporation – Telco Customer Churn Dataset  
    """)

    # ---------------------------
    # Feature Overview Table
    # ---------------------------
    st.subheader("📌 Feature Overview")

    feature_info = pd.DataFrame([
        ["gender", "Customer gender", "Categorical"],
        ["SeniorCitizen", "Whether customer is a senior", "Numeric (0/1)"],
        ["Partner", "Has a partner", "Binary"],
        ["Dependents", "Has dependents", "Binary"],
        ["tenure", "Months with company", "Numeric"],
        ["PhoneService", "Phone service subscription", "Binary"],
        ["MultipleLines", "Multiple phone lines", "Categorical"],
        ["InternetService", "Internet technology", "Categorical"],
        ["OnlineSecurity", "Security add‑on", "Binary"],
        ["OnlineBackup", "Cloud backup", "Binary"],
        ["DeviceProtection", "Device protection", "Binary"],
        ["TechSupport", "Tech support service", "Binary"],
        ["StreamingTV", "TV streaming", "Binary"],
        ["StreamingMovies", "Movie streaming", "Binary"],
        ["Contract", "Contract type", "Categorical"],
        ["PaperlessBilling", "Paperless billing enabled", "Binary"],
        ["PaymentMethod", "Payment type", "Categorical"],
        ["MonthlyCharges", "Monthly fee", "Numeric"],
        ["TotalCharges", "Total lifetime charges", "Numeric"]
    ], columns=["Feature", "Description", "Type"])

    st.dataframe(feature_info, width="stretch")

    # ---------------------------
    # Dataset Upload + Template + Sample
    # ---------------------------
    st.subheader("📥 Upload or Download Dataset")

    required_cols = feature_info["Feature"].tolist()
    template_df = pd.DataFrame(columns=required_cols)

    colT, colS = st.columns(2)

    colT.download_button(
        "📄 Download Input Template",
        template_df.to_csv(index=False).encode("utf-8"),
        "template_dataset.csv",
        "text/csv"
    )

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
        df = clean_data(default_df)

    # ---------------------------
    # Dataset Exploration
    # ---------------------------
    st.subheader("📊 Dataset Exploration")

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Total Records", len(df))
    colB.metric("Number of Features", df.shape[1])
    colC.metric("Target Classes", 2)
    colD.metric("Task Type", "Binary Classification")

    # ---------------------------
    # Dataset Preview
    # ---------------------------
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(), width="stretch")


# ============================================================
# SECTION 2 — MODEL EVALUATION
# ============================================================
else:
    st.header("🤖 Model Evaluation Dashboard")

    # --------------------------
    # Implemented Models
    # --------------------------
    st.subheader("Models Included in Comparison")

    model_info = pd.DataFrame([
        ["Logistic Regression", "Linear Model", "Interpretable, robust baseline"],
        ["Decision Tree", "Tree-Based", "Handles non-linear splits"],
        ["KNN", "Instance-Based", "Distance-driven classification"],
        ["Naive Bayes", "Probabilistic", "Efficient and assumption-based"],
        ["Random Forest", "Ensemble (Bagging)", "Reduces variance, prevents overfitting"],
        ["XGBoost", "Gradient Boosting", "High-performance boosting model"]
    ], columns=["Model", "Category", "Characteristics"])

    st.dataframe(model_info, width="stretch")

    # --------------------------
    # Metric Definitions
    # --------------------------
    st.subheader("📈 Metric Guide")

    metric_def = pd.DataFrame([
        ["Accuracy", "Correct prediction ratio", "Higher is better"],
        ["AUC-ROC", "Class separability measure", "Higher is better"],
        ["Precision", "Correctness of positive predictions", "Higher is better"],
        ["Recall", "Coverage of actual positive cases", "Higher is better"],
        ["F1 Score", "Balance of precision & recall", "Higher is better"],
        ["MCC", "Balanced correlation measure", "+1 is ideal"]
    ], columns=["Metric", "Meaning", "Preferred Direction"])

    st.dataframe(metric_def, width="stretch")

    # --------------------------
    # Model Selection
    # --------------------------
    st.subheader("Model Results")

    selected_model = st.selectbox("Choose Model", model_list)
    model = load_model(selected_model)

    X = preprocessor.transform(df)
    m = metrics[selected_model]

    # Metrics Table
    result_table = pd.DataFrame({
        "Metric": ["Accuracy","AUC","Precision","Recall","F1 Score","MCC"],
        "Value": [
            m["accuracy"], m["auc"], m["precision"],
            m["recall"], m["f1"], m["mcc"]
        ]
    }).round(4)

    st.dataframe(result_table, width="stretch")

    # --------------------------
    # Confusion Matrix
    # --------------------------
    st.subheader(f"{selected_model} – Confusion Matrix")

    cm = np.array(m["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(2.2, 1.8))  # small grid size
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                annot_kws={"size": 8})
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    ax.tick_params(labelsize=7)
    st.pyplot(fig)

    # --------------------------
    # Classification Report
    # --------------------------
    st.subheader("Classification Report")

    report_df = pd.DataFrame(m["classification_report"]).T.round(3)
    report_df.index.name = "Class"
    st.dataframe(report_df, width="stretch")
