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
    """Load saved model file."""
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

    categorical_cols = [
        "gender","Partner","Dependents","PhoneService","MultipleLines",
        "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies","Contract",
        "PaperlessBilling","PaymentMethod"
    ]
    numeric_cols = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]

    # Normalize categorical
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()

    yes_no_cols = [
        "Partner","Dependents","PhoneService","MultipleLines",
        "OnlineSecurity","OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies","PaperlessBilling"
    ]

    for col in yes_no_cols:
        df[col] = df[col].replace({
            "y": "yes", "n": "no",
            "true": "yes", "false": "no"
        })

    df["InternetService"] = df["InternetService"].replace({
        "fiber": "fiber optic",
        "none": "no"
    })

    # Numeric conversion
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ------------------------------------------------------------
# GLOBAL DATA LOADING (before navigation)
# ------------------------------------------------------------

required_cols = [
    "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod",
    "MonthlyCharges","TotalCharges", "Churn"
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
# RADIO NAVIGATION (sync with session_state)
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
    This dataset covers telecom customer attributes used for predicting 
    churn behavior in a binary classification setting.
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
        ["SeniorCitizen", "Customer is senior citizen", "Numeric"],
        ["Partner", "Has partner", "Binary"],
        ["Dependents", "Has dependents", "Binary"],
        ["tenure", "Months active", "Numeric"],
        ["PhoneService", "Phone plan", "Binary"],
        ["MultipleLines", "Multiple phone lines", "Categorical"],
        ["InternetService", "Internet plan", "Categorical"],
        ["OnlineSecurity", "Security addon", "Binary"],
        ["OnlineBackup", "Backup addon", "Binary"],
        ["DeviceProtection", "Protection service", "Binary"],
        ["TechSupport", "Technical support", "Binary"],
        ["StreamingTV", "TV streaming", "Binary"],
        ["StreamingMovies", "Movie streaming", "Binary"],
        ["Contract", "Contract type", "Categorical"],
        ["PaperlessBilling", "Paperless billing", "Binary"],
        ["PaymentMethod", "Payment type", "Categorical"],
        ["MonthlyCharges", "Monthly fee", "Numeric"],
        ["TotalCharges", "Lifetime fee", "Numeric"],
        ["Churn", "Target variable", "Binary"]
    ], columns=["Feature", "Description", "Type"])

    st.dataframe(feature_info, width="stretch")

    # ---------------------------
    # Template + Sample Data
    # ---------------------------
    st.subheader("📥 Download Template or Sample Dataset")

    colT, colS = st.columns(2)

    # Template
    colT.download_button(
        "📄 Download Input Template",
        template_df.to_csv(index=False).encode("utf-8"),
        "template_dataset.csv",
        "text/csv"
    )

    # Sample 100 rows
    default_df = pd.read_csv(default_data_path)
    sample100 = default_df.sample(n=min(100, len(default_df)), random_state=42)

    colS.download_button(
        "📥 Download Sample Test Data (100 rows)",
        sample100.to_csv(index=False).encode("utf-8"),
        "sample_test_100.csv",
        "text/csv"
    )

    # ---------------------------
    # Dataset Exploration
    # ---------------------------
    st.subheader("📊 Dataset Exploration")

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Total Records", len(df))
    colB.metric("Features", df.shape[1])
    colC.metric("Classes", 2)
    colD.metric("Type", "Binary Classification")

    # ---------------------------
    # Dataset Preview
    # ---------------------------
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(), width="stretch")

    # NEXT BUTTON
    if st.button("Next →"):
        st.session_state.page = 1
        st.rerun()


# ============================================================
# SECTION 2 — MODEL EVALUATION (DYNAMIC METRICS)
# ============================================================
if st.session_state.page == 1:

    st.header("🤖 Model Evaluation")

    # --------------------------
    # Implemented Models Table
    # --------------------------
    st.subheader("Models Included")

    model_info = pd.DataFrame([
        ["Logistic Regression", "Linear Model", "Interpretable baseline"],
        ["Decision Tree", "Tree-Based", "Handles non-linear splits"],
        ["KNN", "Instance-Based", "Distance-driven algorithm"],
        ["Naive Bayes", "Probabilistic", "Efficient for high-dimensional data"],
        ["Random Forest", "Bagging Ensemble", "Reduces variance"],
        ["XGBoost", "Boosting Ensemble", "High performance"]
    ], columns=["Model", "Category", "Characteristics"])

    st.dataframe(model_info, width="stretch")

    # --------------------------
    # Metric Definitions Table
    # --------------------------
    st.subheader("📈 Metric Definitions")

    metric_def = pd.DataFrame([
        ["Accuracy", "Overall correctness", "Higher = Better"],
        ["AUC-ROC", "Class separation ability", "Higher = Better"],
        ["Precision", "Correctness of predicted positives", "Higher = Better"],
        ["Recall", "Coverage of actual positives", "Higher = Better"],
        ["F1 Score", "Balance of precision & recall", "Higher = Better"],
        ["MCC", "Balanced correlation", "+1 = Ideal"]
    ], columns=["Metric", "Meaning", "Preferred"])

    st.dataframe(metric_def, width="stretch")

    # --------------------------
    # MODEL SELECTION + DYNAMIC EVALUATION
    # --------------------------
    st.subheader("Model Performance Summary")

    selected_model = st.selectbox("Select Model", model_list)
    model = load_model(selected_model)

    # Prepare X, y
    y_true = df["Churn"].replace({"yes": 1, "no": 0}).astype(int)
    X = preprocessor.transform(df)

    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Dynamic Metrics
    dynamic_metrics = {
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
            dynamic_metrics["accuracy"],
            dynamic_metrics["auc"],
            dynamic_metrics["precision"],
            dynamic_metrics["recall"],
            dynamic_metrics["f1"],
            dynamic_metrics["mcc"]
        ]
    }).round(4)

    st.dataframe(result_table, width="stretch")

    # --------------------------
    # Confusion Matrix
    # --------------------------
    st.subheader(f"{selected_model} – Confusion Matrix")

    cm = dynamic_metrics["cm"]

    fig, ax = plt.subplots(figsize=(1.8, 1.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                annot_kws={"size": 7})
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("Actual", fontsize=7)
    ax.tick_params(labelsize=6)
    st.pyplot(fig)

    # --------------------------
    # Classification Report
    # --------------------------
    st.subheader("Classification Report")

    report_df = pd.DataFrame(dynamic_metrics["report"]).T.round(3)
    report_df.index.name = "Class"

    st.dataframe(report_df, width="stretch")

    # PREVIOUS BUTTON
    if st.button("← Previous"):
        st.session_state.page = 0
        st.rerun()
