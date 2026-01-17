import streamlit as st
import pandas as pd
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# Load saved metrics
# ------------------------------------------------------------
METRICS_PATH = "model/metrics.json"

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

# ------------------------------------------------------------
# Streamlit App Title
# ------------------------------------------------------------
st.title("📊 Telco Customer Churn Prediction App")

st.write(
    "This application allows you to upload a test CSV file, "
    "select a trained model, view evaluation metrics, and predict churn."
)

# ------------------------------------------------------------
# Expected Columns for Telco Dataset
# ------------------------------------------------------------
required_columns = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
    "TotalCharges"
]

# ------------------------------------------------------------
# Sample CSV Download Section
# ------------------------------------------------------------
st.header("1. Upload Test Dataset (CSV)")

st.info(
    "The app is loaded. You may upload your own test CSV file, "
    "or download a sample CSV template below."
)

# Create a sample CSV template
sample_df = pd.DataFrame({col: ["sample_value"] for col in required_columns})
sample_csv = sample_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Sample Test CSV",
    data=sample_csv,
    file_name="sample_telco_test.csv",
    mime="text/csv"
)

# Show expected columns
st.subheader("Expected CSV Format")
st.write("Your uploaded CSV must contain the following columns:")
st.write(required_columns)

st.warning("Only CSV files with the required columns will be accepted.")

# ------------------------------------------------------------
# File Upload
# ------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your Test CSV", type=["csv"])

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)

    # Validate required columns
    missing = [col for col in required_columns if col not in test_df.columns]
    if missing:
        st.error(f"The uploaded file is missing required columns: {missing}")
        st.stop()

    st.success("File uploaded and validated successfully!")
    st.write("Preview of uploaded data:")
    st.dataframe(test_df.head())

else:
    test_df = None
    st.info("Please upload a CSV file to proceed.")

# ------------------------------------------------------------
# Model Selection
# ------------------------------------------------------------
st.header("2. Select a Model")

model_name = st.selectbox("Choose a model:", list(metrics.keys()))
model_path = f"model/{model_name.lower().replace(' ', '_')}.pkl"
model = joblib.load(model_path)

st.success(f"Selected Model: {model_name}")

# ------------------------------------------------------------
# Display Evaluation Metrics
# ------------------------------------------------------------
st.header("3. Model Evaluation Metrics")

selected_metrics = metrics[model_name]

st.json({
    "Accuracy": selected_metrics["accuracy"],
    "AUC": selected_metrics["auc"],
    "Precision": selected_metrics["precision"],
    "Recall": selected_metrics["recall"],
    "F1 Score": selected_metrics["f1"],
    "MCC": selected_metrics["mcc"]
})

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------
st.header("4. Confusion Matrix")

cm = selected_metrics["confusion_matrix"]

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ------------------------------------------------------------
# Classification Report
# ------------------------------------------------------------
st.header("5. Classification Report")
st.json(selected_metrics["classification_report"])

# ------------------------------------------------------------
# Prediction Section
# ------------------------------------------------------------
st.header("6. Predict Churn Using Uploaded CSV")

if test_df is not None:
    if st.button("Run Prediction"):
        predictions = model.predict(test_df)

        result_df = test_df.copy()
        result_df["Churn_Prediction"] = predictions

        st.write("Prediction Results:")
        st.dataframe(result_df)

        # Download option
        csv_data = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_data,
            file_name="prediction_results.csv",
            mime="text/csv"
        )
else:
    st.info("Upload a valid CSV file to enable prediction.")
