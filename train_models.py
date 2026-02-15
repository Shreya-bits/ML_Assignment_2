import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

print("Starting training pipeline...")

# Create model directory if not exists
os.makedirs("model", exist_ok=True)

# Load dataset
data = pd.read_csv("adult.csv")
data = data.replace("?", np.nan)
data = data.dropna()

# Encode target
data['income'] = data['income'].map({'>50K': 1, '<=50K': 0})

# One-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Split features/target
X = data.drop("income", axis=1)
y = data["income"]

# Save feature columns
joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = []

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return metrics


# =========================
# KNN
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

joblib.dump(knn, "model/knn_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

results.append(evaluate_model("KNN", knn, X_test_scaled, y_test))


# =========================
# Logistic Regression
# =========================
log_scaler = StandardScaler()
X_train_scaled = log_scaler.fit_transform(X_train)
X_test_scaled = log_scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

joblib.dump(log_model, "model/logistic_model.pkl")
joblib.dump(log_scaler, "model/logistic_scaler.pkl")

results.append(evaluate_model("Logistic Regression", log_model, X_test_scaled, y_test))


# =========================
# Decision Tree
# =========================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

joblib.dump(dt, "model/decision_tree_model.pkl")
results.append(evaluate_model("Decision Tree", dt, X_test, y_test))


# =========================
# Naive Bayes
# =========================
nb = GaussianNB()
nb.fit(X_train, y_train)

joblib.dump(nb, "model/naive_bayes_model.pkl")
results.append(evaluate_model("Naive Bayes", nb, X_test, y_test))


# =========================
# Random Forest
# =========================
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

joblib.dump(rf, "model/random_forest_model.pkl")
results.append(evaluate_model("Random Forest", rf, X_test, y_test))


# =========================
# XGBoost
# =========================
xgb = XGBClassifier(
    eval_metric="logloss",
    use_label_encoder=False
)
xgb.fit(X_train, y_train)

joblib.dump(xgb, "model/xgboost_model.pkl")
results.append(evaluate_model("XGBoost", xgb, X_test, y_test))


# Save comparison table
comparison_df = pd.DataFrame(results)
comparison_df.to_csv("model/model_comparison.csv", index=False)

print("\nTraining completed successfully.")
print(comparison_df)
