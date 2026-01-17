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

# Ensure session_state page exists
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
        df[col] = df[col].replace({"y":"yes","n":"no","true":"yes","false":"no"})

    df["InternetService"] = df["InternetService"].replace(
        {"fiber":"fiber optic", "none":"no"}
    )

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ------------------------------------------------------------
# Load Dataset BEFORE Sections (Global df)
# ------------------------------------------------------------
required_cols = [
    "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod",
    "MonthlyCharges","TotalCharges"
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
# Radio Navigation (Synced with session_state.page)
# ------------------------------------------------------------
radio_choice = st.radio(
    "Navigate",
    ["Dataset Overview and Selection", "Model Evaluation"],
    index=st.session_state.page,
    horizontal=True
)

# Sync radio to session_state
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
    This dataset contains customer subscription and service usage attributes 
    used for predicting telecom customer churn (binary classification).
    """)

    # Dataset Source + Credits
    st.subheader("Dataset Description")
    st.write("""
    - **Source:** IBM Sample Data Repository  
    - **URL:** https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/  
    - **Credits:** IBM Corporation – Telco Customer Churn Dataset  
    """)

    # ---------------------------
    # Feature Overview
    # ---------------------------
    st.subheader("📌 Feature Overview")

    feature_info = pd.DataFrame([
        ["gender", "Customer gender", "Categorical"],
        ["SeniorCitizen", "Whether customer is senior", "Numeric"],
        ["Partner", "Customer has partner", "Binary"],
        ["Dependents", "Customer dependents", "Binary"],
        ["tenure", "Active months", "Numeric"],
        ["PhoneService", "Phone plan", "Binary"],
        ["MultipleLines", "Multiple phone lines", "Categorical"],
        ["InternetService", "Internet technology", "Categorical"],
        ["OnlineSecurity", "Security add-on", "Binary"],
        ["OnlineBackup", "Backup service", "Binary"],
        ["DeviceProtection", "Protection plan", "Binary"],
        ["TechSupport", "Technical support", "Binary"],
        ["StreamingTV", "TV streaming", "Binary"],
        ["StreamingMovies", "Movie streaming", "Binary"],
        ["Contract", "Contract term", "Categorical"],
        ["PaperlessBilling", "Paperless billing", "Binary"],
        ["PaymentMethod", "Billing method", "Categorical"],
        ["MonthlyCharges", "Monthly fee", "Numeric"],
        ["TotalCharges", "Lifetime fee", "Numeric"]
    ], columns=["Feature", "Description", "Type"])

    st.dataframe(feature_info, width="stretch")

    # ---------------------------
    # Template + Sample Downloads
    # ---------------------------
    st.subheader("📥 Download Template or Sample Dataset")

    colT, colS = st.columns(2)

    colT.download_button(
        "📄 Download Input Template",
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

    # NEXT BUTTON (move to section 2)
    if st.button("Next →"):
        st.session_state.page = 1
        st.rerun()


# ============================================================
# SECTION 2 — MODEL EVALUATION
# ============================================================
if st.session_state.page == 1:

    st.header("🤖 Model Evaluation")

    # --------------------------
    # Implemented Models Section
    # --------------------------
    st.subheader("Models Included")

    model_info = pd.DataFrame([
        ["Logistic Regression", "Linear Model", "Interpretable baseline"],
        ["Decision Tree", "Tree-Based", "Non-linear splits"],
        ["KNN", "Instance-Based", "Distance-driven method"],
        ["Naive Bayes", "Probabilistic", "Assumption-based"],
        ["Random Forest", "Bagging Ensemble", "Reduces overfitting"],
        ["XGBoost", "Boosting Ensemble", "High performance"]
    ], columns=["Model", "Category", "Characteristics"])

    st.dataframe(model_info, width="stretch")

    # --------------------------
    # Metric Definitions
    # --------------------------
    st.subheader("📈 Metric Definitions")

    metric_def = pd.DataFrame([
        ["Accuracy", "Correct predictions ratio", "Higher = Better"],
        ["AUC-ROC", "Class separability", "Higher = Better"],
        ["Precision", "Correct positive predictions", "Higher = Better"],
        ["Recall", "Detection of true positives", "Higher = Better"],
        ["F1 Score", "Balance of precision & recall", "Higher = Better"],
        ["MCC", "Correlation coefficient", "+1 = Ideal"]
    ], columns=["Metric", "Meaning", "Preferred"])

    st.dataframe(metric_def, width="stretch")

    # --------------------------
    # Model Selection + Auto Evaluation
    # --------------------------
    st.subheader("Model Performance Summary")

    selected_model = st.selectbox("Select Model", model_list)
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

    report_df = pd.DataFrame(m["classification_report"]).T.round(3)
    report_df.index.name = "Class"
    st.dataframe(report_df, width="stretch")

    # PREVIOUS BUTTON (back to section 1)
    if st.button("← Previous"):
        st.session_state.page = 0
        st.rerun()
