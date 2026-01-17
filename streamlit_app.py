import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# ------------------------------------------------------------
# Streamlit Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="Classification Model Comparison", layout="wide")

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = 0   # 0 = Dataset Overview, 1 = Model Evaluation


# ------------------------------------------------------------
# Load Preprocessor
# ------------------------------------------------------------
@st.cache_resource
def load_preprocessor():
    return joblib.load("model/preprocessor.pkl")

preprocessor = load_preprocessor()


# ------------------------------------------------------------
# Load Models
# ------------------------------------------------------------
def load_model(name):
    return joblib.load(f"model/{name.lower().replace(' ', '_')}.pkl")

model_list = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]


# ------------------------------------------------------------
# Cleaning Function
# ------------------------------------------------------------
def clean_data(df):
    df = df.copy()

    # Normalize categorical
    cat_cols = [
        "gender","Partner","Dependents","PhoneService","MultipleLines",
        "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies","Contract",
        "PaperlessBilling","PaymentMethod"
    ]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()

    # Normalize yes/no
    yn_cols = [
        "Partner","Dependents","PhoneService","MultipleLines",
        "OnlineSecurity","OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies","PaperlessBilling"
    ]
    for col in yn_cols:
        df[col] = df[col].replace({"y": "yes", "n": "no", "true": "yes", "false": "no"})

    # InternetService fixes
    df["InternetService"] = df["InternetService"].replace({"fiber": "fiber optic", "none": "no"})

    # Fix numeric
    numeric_cols = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Normalize Churn if present
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].astype(str).str.lower().str.strip()

    return df


# ------------------------------------------------------------
# Load Dataset BEFORE Sections (Global df)
# ------------------------------------------------------------
required_cols = [
    "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod",
    "MonthlyCharges","TotalCharges","Churn"
]

default_data_path = "model/test_default.csv"
template_df = pd.DataFrame(columns=required_cols)

uploaded = st.file_uploader("Upload CSV File", type=["csv"], key="global_upload")

if uploaded:
    df = pd.read_csv(uploaded)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()
    df = clean_data(df)
else:
    df = pd.read_csv(default_data_path)
    df = clean_data(df)


# ------------------------------------------------------------
# RADIO NAVIGATION
# ------------------------------------------------------------
radio_choice = st.radio(
    "Navigate",
    ["Dataset Overview and Selection", "Model Evaluation"],
    index=st.session_state.page,
    horizontal=True
)

if radio_choice == "Dataset Overview and Selection":
    st.session_state.page = 0
else:
    st.session_state.page = 1


# ============================================================
# SECTION 1 — DATASET OVERVIEW AND SELECTION
# ============================================================
if st.session_state.page == 0:

    st.header("📁 Dataset Overview and Selection")

    st.write("""
    This dataset represents telecom customer details used to predict churn (whether 
    the customer has left the company). The Churn column must be present to evaluate 
    model performance.
    """)

    # Dataset Description and Credits
    st.subheader("Dataset Description")
    st.write("""
    - **Source:** IBM Sample Data Repository  
    - **URL:** https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/  
    - **Credits:** IBM Corporation – Telco Customer Churn Dataset  
    """)

    # Feature Overview Table
    st.subheader("📌 Feature Overview")

    feature_info = pd.DataFrame([
        ["gender", "Customer gender", "Categorical"],
        ["SeniorCitizen", "Is senior citizen", "Numeric"],
        ["Partner", "Has partner", "Binary"],
        ["Dependents", "Has dependents", "Binary"],
        ["tenure", "Months active", "Numeric"],
        ["PhoneService", "Phone plan", "Binary"],
        ["MultipleLines", "Multiple lines", "Categorical"],
        ["InternetService", "Internet plan", "Categorical"],
        ["OnlineSecurity", "Security addon", "Binary"],
        ["OnlineBackup", "Backup addon", "Binary"],
        ["DeviceProtection", "Protection addon", "Binary"],
        ["TechSupport", "Tech support", "Binary"],
        ["StreamingTV", "TV streaming", "Binary"],
        ["StreamingMovies", "Movies streaming", "Binary"],
        ["Contract", "Contract type", "Categorical"],
        ["PaperlessBilling", "Paperless billing", "Binary"],
        ["PaymentMethod", "Payment type", "Categorical"],
        ["MonthlyCharges", "Monthly fee", "Numeric"],
        ["TotalCharges", "Lifetime fee", "Numeric"],
        ["Churn", "Target variable", "Binary (yes/no)"]
    ], columns=["Feature", "Description", "Type"])

    st.dataframe(feature_info, width="stretch")

    # Template + Sample Data
    st.subheader("📥 Download Template or Sample Dataset")

    colT, colS = st.columns(2)

    colT.download_button(
        "📄 Download Input Template (with Churn)",
        template_df.to_csv(index=False).encode("utf-8"),
        "template_dataset.csv",
        "text/csv"
    )

    default_df = pd.read_csv(default_data_path)
    sample100 = default_df.sample(n=min(100, len(default_df)), random_state=42)

    colS.download_button(
        "📥 Download Sample Test Data (100 rows)",
        sample100.to_csv(index=False).encode("utf-8"),
        "sample_test_100.csv",
        "text/csv"
    )

    # Dataset Exploration
    st.subheader("📊 Dataset Exploration")

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Total Records", len(df))
    colB.metric("Features", df.shape[1])
    colC.metric("Target Present?", "Yes" if "Churn" in df.columns else "No")
    colD.metric("Type", "Binary Classification")

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(), width="stretch")

    if st.button("Next →"):
        st.session_state.page = 1
        st.rerun()


# ============================================================
# SECTION 2 — MODEL EVALUATION (DYNAMIC)
# ============================================================
if st.session_state.page == 1:

    st.header("🤖 Model Evaluation")

    if "Churn" not in df.columns:
        st.error("❌ Dataset does not contain 'Churn'. Cannot compute evaluation metrics.")
        st.stop()

    # Metrics Info
    st.subheader("📈 Metric Definitions")

    metric_table = pd.DataFrame([
        ["Accuracy", "Correct predictions"],
        ["AUC‑ROC", "Discrimination power"],
        ["Precision", "Correctness of predicted positives"],
        ["Recall", "Correctly detected positives"],
        ["F1 Score", "Balance of Precision and Recall"],
        ["MCC", "Balanced accuracy-like measure"]
    ], columns=["Metric", "Meaning"])

    st.dataframe(metric_table, width="stretch")

    # Model Selection
    st.subheader("Model Performance Summary")

    selected_model = st.selectbox("Select Model", model_list)
    model = load_model(selected_model)

    X = preprocessor.transform(df.drop(columns=["Churn"]))
    y_true = df["Churn"].replace({"yes": 1, "no": 0}).astype(int)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Dynamic Metrics
    dyn = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cm": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, output_dict=True)
    }

    # Metrics Table
    result_table = pd.DataFrame({
        "Metric": ["Accuracy","AUC","Precision","Recall","F1 Score","MCC"],
        "Value": [
            dyn["accuracy"], dyn["auc"], dyn["precision"],
            dyn["recall"], dyn["f1"], dyn["mcc"]
        ]
    }).round(4)

    st.dataframe(result_table, width="stretch")

    # Confusion Matrix
    st.subheader(f"{selected_model} – Confusion Matrix")

    fig, ax = plt.subplots(figsize=(1.8, 1.3))
    sns.heatmap(dyn["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                cbar=False, annot_kws={"size": 7})
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("Actual", fontsize=7)
    ax.tick_params(labelsize=6)
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(dyn["report"]).T.round(3), width="stretch")

    if st.button("← Previous"):
        st.session_state.page = 0
        st.rerun()
