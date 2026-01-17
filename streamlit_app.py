import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json
import os

st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

# ------------------------------------------------------------
# Load saved metrics
# ------------------------------------------------------------
with open("model/metrics.json", "r") as f:
    metrics = json.load(f)

def load_model(name):
    path = f"model/{name.lower().replace(' ', '_')}.pkl"
    return joblib.load(path)

# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------
required_cols = [
    "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod",
    "MonthlyCharges","TotalCharges"
]

categorical_cols = [
    "gender","Partner","Dependents","PhoneService","MultipleLines",
    "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
    "TechSupport","StreamingTV","StreamingMovies","Contract",
    "PaperlessBilling","PaymentMethod"
]

numeric_cols = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]

# ------------------------------------------------------------
# Auto-clean categorical & numeric values
# ------------------------------------------------------------
def clean_uploaded_data(df):

    # Strip whitespace + lowercase
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Normalize yes/no style columns
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

    # Normalize InternetService
    df["InternetService"] = df["InternetService"].replace({
        "fiber optic": "fiber optic",
        "fiber": "fiber optic",
        "dsl": "dsl",
        "none": "no"
    })

    # Convert numerics safely
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df

# ------------------------------------------------------------
# Load default test dataset
# ------------------------------------------------------------
default_path = "model/test_default.csv"
default_df = pd.read_csv(default_path) if os.path.exists(default_path) else pd.DataFrame({col:[] for col in required_cols})
default_df = clean_uploaded_data(default_df)

# ------------------------------------------------------------
# PAGE HEADER
# ------------------------------------------------------------
st.markdown("## 📊 Telco Customer Churn Prediction Dashboard")
st.markdown("---")

st.write(
    "This dashboard allows you to upload a test dataset, select a model, "
    "view performance metrics, and generate churn predictions. "
)

st.info(f"Showing predictions for default test dataset ({len(default_df)} rows). "
        "Upload your own CSV to update the results.")

# ------------------------------------------------------------
# SAMPLE CSV + EXPECTED FORMAT
# ------------------------------------------------------------
st.markdown("### 📥 Download Sample CSV (Optional)")
sample_df = pd.DataFrame({col: ["sample_value"] for col in required_cols})
st.download_button(
    "Download Sample CSV Template",
    sample_df.to_csv(index=False).encode("utf-8"),
    "sample_telco_test.csv",
    "text/csv"
)

st.markdown("### 📘 Expected CSV Format:")
st.write(required_cols)
st.warning("Only CSV files with the above columns will be accepted.")

st.markdown("---")

# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        st.error(f"Uploaded CSV is missing required columns: {missing}")
        st.stop()

    st.success("Custom CSV uploaded. Predictions will now use this dataset.")
    data = clean_uploaded_data(data)
else:
    data = default_df.copy()

st.markdown("### 🔍 Preview of Dataset")
st.dataframe(data.head())

st.markdown("---")

# ------------------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------------------
st.markdown("## 🤖 Model Selection & Performance Metrics")

model_list = list(metrics.keys())

# Default selected model should be Logistic Regression
default_index = model_list.index("Logistic Regression") if "Logistic Regression" in model_list else 0

model_name = st.selectbox("Choose a model:", model_list, index=default_index)
model = load_model(model_name)
m = metrics[model_name]

# ------------------------------------------------------------
# METRICS DISPLAY
# ------------------------------------------------------------
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", round(m["accuracy"],4))
col2.metric("AUC", round(m["auc"],4))
col3.metric("Precision", round(m["precision"],4))
col4.metric("Recall", round(m["recall"],4))
col5.metric("F1 Score", round(m["f1"],4))
col6.metric("MCC", round(m["mcc"],4))

st.markdown("---")

# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------
st.markdown("### 📊 Confusion Matrix")
cm = m["confusion_matrix"]

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.markdown("---")

# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------
st.markdown("### 📄 Classification Report")
st.json(m["classification_report"])

st.markdown("---")

# ------------------------------------------------------------
# AUTOMATIC PREDICTION
# ------------------------------------------------------------
st.markdown("## 🔮 Prediction Results (Auto‑generated)")

pred_df = data.copy()
pred_df["Churn_Prediction"] = model.predict(pred_df)

st.dataframe(pred_df)

st.download_button(
    "📥 Download Predictions CSV",
    pred_df.to_csv(index=False).encode("utf-8"),
    "predictions.csv",
    "text/csv"
)
