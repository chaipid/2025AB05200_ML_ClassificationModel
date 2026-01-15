import streamlit as st
import pandas as pd
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# Load Saved Metrics
# ------------------------------------------------------------
METRICS_PATH = "model/metrics.json"

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

# ------------------------------------------------------------
# Streamlit App Title
# ------------------------------------------------------------
st.title("Customer Churn Prediction App")

st.markdown("### Upload Test CSV → Select Model → View Metrics → Predict Churn")

# ------------------------------------------------------------
# Dataset Upload Section
# ------------------------------------------------------------
st.header("1. Upload Test Dataset (CSV)")

uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"])

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data:")
    st.dataframe(test_df.head())
else:
    test_df = None


# ------------------------------------------------------------
# Model Selection Dropdown
# ------------------------------------------------------------
st.header("2. Select a Model")

model_name = st.selectbox(
    "Choose a model:",
    list(metrics.keys())
)

st.success(f"You selected: {model_name}")

# Load selected model
model_path = f"model/{model_name.lower().replace(' ', '_')}.pkl"
model = joblib.load(model_path)


# ------------------------------------------------------------
# Display Evaluation Metrics
# ------------------------------------------------------------
st.header("3. Model Evaluation Metrics")

selected_metrics = metrics[model_name]

st.write("### Key Metrics")
st.json({
    "Accuracy": selected_metrics["accuracy"],
    "AUC": selected_metrics["auc"],
    "Precision": selected_metrics["precision"],
    "Recall": selected_metrics["recall"],
    "F1 Score": selected_metrics["f1"],
    "MCC": selected_metrics["mcc"]
})


# ------------------------------------------------------------
# Display Confusion Matrix
# ------------------------------------------------------------
st.header("4. Confusion Matrix")

cm = selected_metrics["confusion_matrix"]

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)


# ------------------------------------------------------------
# Display Classification Report
# ------------------------------------------------------------
st.header("5. Classification Report")
st.json(selected_metrics["classification_report"])


# ------------------------------------------------------------
# Prediction Section
# ------------------------------------------------------------
st.header("6. Predict Churn on Uploaded CSV")

if test_df is not None:
    if st.button("Run Prediction"):
        predictions = model.predict(test_df)

        test_df["Churn_Prediction"] = predictions

        st.write("### Prediction Output:")
        st.dataframe(test_df)

        # Option to download
        csv = test_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
else:
    st.info("Upload a test CSV to enable prediction.")
