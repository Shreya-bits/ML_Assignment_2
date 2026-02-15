# ML Assignment 2 - Classification Models
Machine Learning Assignment 2 - BITS WILP

## Live Streamlit App
[Click here to access the deployed app](https://mlassignment2-gkpodq6anc2k9cwr3mscdp.streamlit.app/)

a. Problem Statement

The objective of this assignment is to build and compare multiple classification models for predicting whether an individual's income exceeds $50K per year based on demographic and employment-related attributes. The task involves model implementation, evaluation using multiple performance metrics, and deployment through an interactive Streamlit web application.
---

## Dataset Details

- Dataset Name: Adult Income Dataset
- Source: Kaggle (UCI Census Income Dataset)
- Original Dataset Link: https://www.kaggle.com/datasets/uciml/adult-census-income
- Original Number of Instances: 48,842
- Instances After Preprocessing: 30,162
- Original Number of Features: 14
- Final Number of Features After One-Hot Encoding: 96
- Target Variable: Income (>50K or <=50K)

---

## Preprocessing Steps

- Replaced missing values ("?") with NaN
- Removed rows containing missing values
- Converted target variable to binary (0 = <=50K, 1 = >50K)
- Applied one-hot encoding to categorical variables (drop_first=True)
- Applied feature scaling (StandardScaler) for KNN and Logistic Regression
- Train-test split performed with 80% training and 20% testing data
- Random state set to 42 for reproducibility

---

## Implemented Models

1. K-Nearest Neighbors (KNN)
2. Logistic Regression
3. Decision Tree
4. Naive Bayes
5. Random Forest
6. XGBoost

---

## Evaluation Metrics Used

- Accuracy
- Area Under Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

These metrics were chosen to provide a comprehensive evaluation considering class imbalance.

---

## Model Performance Comparison

| Model               | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
|--------------------|----------|-------|-----------|--------|----------|-------|
| KNN                | 0.8140   | 0.8307| 0.6443    | 0.5627 | 0.6007   | 0.4821|
| Logistic Regression| 0.8420   | 0.8970| 0.7190    | 0.5987 | 0.6533   | 0.5560|
| Decision Tree      | 0.8023   | 0.7432| 0.5977    | 0.6260 | 0.6115   | 0.4792|
| Naive Bayes        | 0.7824   | 0.8223| 0.6406    | 0.2840 | 0.3935   | 0.3192|
| Random Forest      | 0.8417   | 0.8964| 0.7144    | 0.6053 | 0.6554   | 0.5567|
| XGBoost            | 0.8654   | 0.9237| 0.7788    | 0.6407 | 0.7030   | 0.6220|


## Best Performing Model

XGBoost achieved the highest overall performance across:
- Accuracy
- AUC
- F1 Score
- MCC

---
## Observations on Model Performance 

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Performed well with high AUC and balanced precision-recall tradeoff. Good generalization with moderate MCC score. |
| Decision Tree | Lower AUC compared to ensemble models. Slight tendency to overfit, but recall was relatively strong. |
| K-Nearest Neighbors (KNN) | Moderate performance. Sensitive to feature scaling and high dimensionality due to one-hot encoding. |
| Naive Bayes | Lowest recall and F1 score. Independence assumption between features likely reduced effectiveness. |
| Random Forest (Ensemble) | Strong and stable performance. Improved generalization compared to single Decision Tree. |
| XGBoost (Ensemble) | Best overall model. Achieved highest Accuracy, AUC, F1 Score, and MCC on the dataset. |


