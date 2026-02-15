import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

st.title("ML Assignment 2 - Classification Models")

# Load full dataset for test data download
full_data = pd.read_csv("adult.csv")
full_data = full_data.replace("?", np.nan)
full_data = full_data.dropna()

# Create test split (same logic as training)
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(
    full_data,
    test_size=0.2,
    random_state=42
)

st.download_button(
    label="Download Test Data",
    data=test_data.to_csv(index=False),
    file_name="test_data.csv",
    mime="text/csv"
)


st.write("Upload a CSV file with the same structure as the training dataset.")

# Model selection
model_option = st.selectbox(
    "Select Model",
    ["KNN", "Logistic Regression", "Decision Tree", 
     "Naive Bayes", "Random Forest", "XGBoost"]
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    data = data.replace("?", np.nan)
    data = data.dropna()

    data['income'] = data['income'].map({'>50K': 1, '<=50K': 0})
    data = pd.get_dummies(data, drop_first=True)

    X = data.drop("income", axis=1)
    y = data["income"]

    # Ensure same feature structure as training
    trained_columns = joblib.load("model/feature_columns.pkl")
    X = X.reindex(columns=trained_columns, fill_value=0)


    # Load selected model
    if model_option == "KNN":
        model = joblib.load("model/knn_model.pkl")
        scaler = joblib.load("model/scaler.pkl")
        X = scaler.transform(X)

    elif model_option == "Logistic Regression":
        model = joblib.load("model/logistic_model.pkl")
        scaler = joblib.load("model/logistic_scaler.pkl")
        X = scaler.transform(X)

    elif model_option == "Decision Tree":
        model = joblib.load("model/decision_tree_model.pkl")

    elif model_option == "Naive Bayes":
        model = joblib.load("model/naive_bayes_model.pkl")

    elif model_option == "Random Forest":
        model = joblib.load("model/random_forest_model.pkl")

    elif model_option == "XGBoost":
        model = joblib.load("model/xgboost_model.pkl")

    # Predictions
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = y_pred

    # Metrics
    st.subheader("Evaluation Metrics")

    st.write("Accuracy:", accuracy_score(y, y_pred))
    st.write("AUC:", roc_auc_score(y, y_prob))
    st.write("Precision:", precision_score(y, y_pred))
    st.write("Recall:", recall_score(y, y_pred))
    st.write("F1 Score:", f1_score(y, y_pred))
    st.write("MCC:", matthews_corrcoef(y, y_pred))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    st.write(cm)
