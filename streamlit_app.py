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
st.set_page_config(page_title="ML Model Comparison", layout="wide")


# ------------------------------------------------------------
# Load Preprocessor
# ------------------------------------------------------------
@st.cache_resource
def load_preprocessor():
    return joblib.load("model/preprocessor.pkl")

preprocessor = load_preprocessor()


# ------------------------------------------------------------
# Load ML Models
# ------------------------------------------------------------
def load_model(model_name):
    file_path = f"model/{model_name.lower().replace(' ', '_')}.pkl"
    return joblib.load(file_path)


# ------------------------------------------------------------
# Load Metrics
# ------------------------------------------------------------
with open("model/metrics.json", "r") as f:
    metrics = json.load(f)

model_list = list(metrics.keys())


# ------------------------------------------------------------
# Clean Uploaded Data (same logic as training)
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

    # Clean categorical values
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

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

    # Convert numerics
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ------------------------------------------------------------
# SECTION 1: Dataset Section
# ------------------------------------------------------------
st.header("📁 Dataset")

required_cols = [
    "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod",
    "MonthlyCharges","TotalCharges"
]

# Allow user to download sample CSV
sample_df = pd.DataFrame({col: ["sample_value"] for col in required_cols})

st.download_button(
    "📥 Download Sample Dataset Template",
    sample_df.to_csv(index=False).encode("utf-8"),
    "sample_telco_input.csv",
    "text/csv",
)

st.write("Upload a dataset matching the above structure:")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ The following columns are missing: {missing}")
        st.stop()

    df = clean_data(df)
    st.success("Dataset uploaded successfully!")

else:
    st.info("No file uploaded. Using default test dataset.")
    df = pd.read_csv("model/test_default.csv")
    df = clean_data(df)

# Show dataset preview
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(), width="stretch")


# ------------------------------------------------------------
# SECTION 2: Model Selection
# ------------------------------------------------------------
st.markdown("---")
st.header("🤖 Model Evaluation")

selected_model = st.selectbox("Choose model:", model_list)
model = load_model(selected_model)

X = preprocessor.transform(df)
pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

results = df.copy()
results["Prediction"] = pred
results["Probability"] = prob.round(4)


# ------------------------------------------------------------
# SECTION 3: Metrics + Confusion Matrix Side-by-Side
# ------------------------------------------------------------
st.markdown("## 📈 Evaluation Metrics")

m = metrics[selected_model]

col_metrics, col_cm = st.columns([1, 2])

with col_metrics:
    st.markdown("### 📊 Metrics Table")

    metric_table = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
        "Value": [
            m["accuracy"],
            m["auc"],
            m["precision"],
            m["recall"],
            m["f1"],
            m["mcc"]
        ]
    })
    st.dataframe(metric_table.style.format({"Value": "{:.4f}"}), width="stretch")


with col_cm:
    st.markdown(f"### 📉 Confusion Matrix – {selected_model}")

    cm = np.array(m["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


# ------------------------------------------------------------
# SECTION 4: Classification Report Table
# ------------------------------------------------------------
st.markdown("## 📄 Classification Report")

report_df = pd.DataFrame(m["classification_report"]).T
report_df = report_df.round(3)
report_df.index.name = "Class"

st.dataframe(report_df, width="stretch")


# ------------------------------------------------------------
# SECTION 5: Predictions
# ------------------------------------------------------------
st.markdown("## 🔮 Prediction Results")
st.dataframe(results.head(50), width="stretch")

st.download_button(
    "📥 Download Predictions CSV",
    results.to_csv(index=False).encode("utf-8"),
    "predictions.csv",
    "text/csv"
)
