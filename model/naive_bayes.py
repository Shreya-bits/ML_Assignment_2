import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

print("Starting Naive Bayes training...")

# Load dataset
data = pd.read_csv("../adult.csv")

# Basic cleaning
data = data.replace("?", np.nan)
data = data.dropna()

# Convert target to binary
data['income'] = data['income'].map({'>50K': 1, '<=50K': 0})

# One-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Separate features and target
X = data.drop("income", axis=1)
y = data["income"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print("\nNaive Bayes Evaluation Metrics:")
print("Accuracy:", accuracy)
print("AUC:", auc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("MCC:", mcc)

# Save model
joblib.dump(model, "naive_bayes_model.pkl")

print("\nNaive Bayes model trained and saved successfully.")
