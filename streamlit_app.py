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
st.set_page_config(page_title="ML Classification Model Evaluation", layout="wide")

# Initialize session page state
if "page" not in st.session_state:
    st.session_state.page = 0   # 0 = Dataset Overview, 1 = Model Evaluation

if "nav" not in st.session_state:
    st.session_state.nav = "Dataset Overview and Selection"


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

    # Normalize strings
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.lower().str.strip()

    # Fix numeric conversions
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Normalize Churn if present
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].astype(str).str.lower().str.strip()

    return df


# ------------------------------------------------------------
# Required Columns
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

# ------------------------------------------------------------
# File Upload
# ------------------------------------------------------------
uploaded = st.file_uploader("Upload CSV File", type=["csv"], key="global_upload")

if uploaded:
    df = pd.read_csv(uploaded)
    df = clean_data(df)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.stop()

else:
    df = pd.read_csv(default_data_path)
    df = clean_data(df)


# ------------------------------------------------------------
# NAVIGATION RADIO (with sync)
# ------------------------------------------------------------
radio_choice = st.radio(
    "Navigate",
    ["Dataset Overview and Selection", "Model Evaluation"],
    index=st.session_state.page,
    horizontal=True,
    key="nav"
)

if st.session_state.nav == "Dataset Overview and Selection":
    st.session_state.page = 0
else:
    st.session_state.page = 1


# ====================================================================
# PAGE 1 — DATASET OVERVIEW
# ====================================================================
if st.session_state.page == 0:

    st.header("📁 Dataset Overview and Selection")

    st.write("""
    Upload a CSV file containing customer information.  
    The dataset **must include the 'Churn' column** for evaluation.
    """)

    # Dataset Preview
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Dataset Stats
    st.subheader("📊 Dataset Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(df))
    col2.metric("Columns", df.shape[1])
    col3.metric("Churn Present?", "Yes" if "Churn" in df.columns else "No")

    # Download Template + Sample Data
    st.subheader("📥 Download Files")
    c1, c2 = st.columns(2)

    c1.download_button(
        "📄 Download Input Template",
        template_df.to_csv(index=False).encode("utf-8"),
        "template_dataset.csv",
        "text/csv"
    )

    default_df = pd.read_csv(default_data_path)
    sample100 = default_df.sample(n=min(100, len(default_df)), random_state=42)

    c2.download_button(
        "📥 Sample Test Data (100 rows)",
        sample100.to_csv(index=False).encode("utf-8"),
        "sample_test_100.csv",
        "text/csv"
    )

    # NEXT BUTTON
    if st.button("Next →"):
        st.session_state.page = 1
        st.session_state.nav = "Model Evaluation"
        st.rerun()


# ====================================================================
# PAGE 2 — MODEL EVALUATION
# ====================================================================
if st.session_state.page == 1:

    st.header("🤖 Model Evaluation")

    if "Churn" not in df.columns:
        st.error("❌ 'Churn' column missing. Cannot evaluate model.")
        st.stop()

    # SAFETY CHECKS (Prevents 0-sample errors)
    if df.empty:
        st.error("❌ The dataset has no rows after loading/cleaning.")
        st.stop()

    df_features = df.drop(columns=["Churn"])

    if df_features.empty:
        st.error("❌ No feature columns found after dropping target.")
        st.stop()

    # Try preprocessing
    try:
        X = preprocessor.transform(df_features)
    except Exception as e:
        st.error(f"❌ Preprocessing failed: {e}")
        st.stop()

    y_true = df["Churn"].replace({"yes": 1, "no": 0}).astype(int)

    # Model Selection
    selected_model = st.selectbox("Select Model", model_list)
    model = load_model(selected_model)

    # XGBoost Fix (legacy model compatibility)
    if hasattr(model, "use_label_encoder"):
        try:
            delattr(model, "use_label_encoder")
        except:
            pass

    # Predictions
    try:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(f"❌ Model prediction failed: {e}")
        st.stop()

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

    # Display Metrics
    st.subheader("📈 Evaluation Metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy","AUC","Precision","Recall","F1 Score","MCC"],
        "Value": [
            dyn["accuracy"], dyn["auc"], dyn["precision"],
            dyn["recall"], dyn["f1"], dyn["mcc"]
        ]
    }).round(4)
    st.dataframe(metrics_df, use_container_width=True)

    # Confusion Matrix
    st.subheader("📌 Confusion Matrix")

    fig, ax = plt.subplots(figsize=(1.8, 1.3))
    sns.heatmap(dyn["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("Actual", fontsize=7)
    ax.tick_params(labelsize=6)
    st.pyplot(fig)

    # Classification Report
    st.subheader("📄 Classification Report")
    st.dataframe(pd.DataFrame(dyn["report"]).T.round(3), use_container_width=True)

    # PREVIOUS BUTTON
    if st.button("← Previous"):
        st.session_state.page = 0
        st.session_state.nav = "Dataset Overview and Selection"
        st.rerun()
