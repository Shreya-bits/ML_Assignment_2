import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

print("Starting KNN training...")

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

print("Feature size:", X.shape[1])
print("Number of instances:", X.shape[0])


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print("\nKNN Evaluation Metrics:")
print("Accuracy:", accuracy)
print("AUC:", auc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("MCC:", mcc)

# Save model and scaler
joblib.dump(knn, "../model/knn_model.pkl")
joblib.dump(scaler, "../model/scaler.pkl")

print("\nKNN model trained and saved successfully.")
