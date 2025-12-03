## 📌 Insurance Fraud Detection — Machine Learning Project

AdaBoost + SMOTE + End-to-End Data Pipeline

## 📝 Project Overview

This project builds a fraud detection system for insurance claims using a robust end-to-end Machine Learning pipeline.
It covers:
- Feature engineering
- Data cleaning
- Mixed-type preprocessing (numeric + categorical)
- Handling imbalanced data using SMOTE
- Training an AdaBoost classifier
- Hyperparameter tuning
- Evaluation with AUC, Recall, Precision, F1
- Full ML pipeline integration (ColumnTransformer → SMOTE → Model)

The final model achieves strong real-world fraud detection performance, with an excellent trade-off between catching fraud and minimizing false alarms.

## 📊 Dataset Summary

Source: Kaggle

Link: https://www.kaggle.com/datasets/antopravinjohnbosco/auto-insurance-claims-fraud-detection

- Rows: 1,000
- Target:
1 → Fraud
0 → Not Fraud
- Contains customer, policy, and incident-level information
- Includes engineered time & duration features
- High-cardinality categorical features
- Imbalanced dataset (~75% non-fraud, ~25% fraud)

## 🧹 Data Cleaning & Feature Engineering
Steps Performed:
- Removed irrelevant, noisy, or duplicate columns:
_c39, policy_number, insured_zip, fraud_reported, insured_hobbies,
incident_location, auto_model, capital-gains, capital-loss, vehicle_claim
- Converted dirty categorical entries ("?", "None", "") → "Unknown"
- Converted date columns to datetime
- Engineered new features: policy_duration_days, incident_month, incident_day_of_week
- Identified suitable numeric/categorical columns
- Handled mixed data types with a ColumnTransformer pipeline

## ⚙️ Preprocessing Pipeline

Implemented with Scikit-Learn ColumnTransformer:

- Numeric Features: 
months_as_customer
age
policy_annual_premium
total_claim_amount
injury_claim
property_claim
policy_duration_days

- Categorical Features: 
policy_state, policy_csl, insured_sex, insured_education_level,
insured_occupation, insured_relationship, incident_type,
collision_type, incident_severity, authorities_contacted,
incident_state, incident_city, property_damage,
police_report_available, auto_make, policy_deductable,
incident_hour_of_the_day, number_of_vehicles_involved,
bodily_injuries, witnesses, incident_month,
incident_day_of_week, umbrella_limit, auto_year

Transformations:

- Numeric → Median Imputer + StandardScaler

- Categorical → Constant Imputer + OneHotEncoder

## 🤖 Final Model: SMOTE + AdaBoost (Selected Model)

To handle class imbalance, SMOTE oversampling is applied inside the training pipeline, followed by an AdaBoost classifier.

- Model Architecture:
SMOTE → AdaBoost(estimator = DecisionTreeClassifier(max_depth=3, class_weight={0:1,1:2}))

- Parameters:
n_estimators = 200
learning_rate = 0.1
max_depth = 3
class_weight = {0:1, 1:2}

This setup provided the strongest performance on fraud recall and ROC-AUC.

## 🧪 Final Model Performance
📌 Confusion Matrix:

[[124  27]

 [ 13  36]]

📌 Classification Report
Metric|	Score
|---|---|
Accuracy|	0.80
Fraud Recall (class 1)|	0.73
Fraud Precision|	0.57
Fraud F1-score|	0.64
ROC-AUC|	0.823
## 🔥 Why This Model Was Selected

- Highest ROC-AUC (0.823) among all models

- Excellent fraud recall (0.73) → detects more fraudulent cases

- Balanced precision–recall trade-off

- Stable and interpretable weak learners (DecisionTree)

- Handles imbalanced data effectively with SMOTE

- This makes it well-suited for real-world fraud detection workflows.

## 📈 Visualizations Included

- Fraud vs Non-Fraud distribution

- Correlation analysis

- Categorical fraud patterns

- SMOTE oversampling demonstration

- ROC Curve

- Confusion Matrix Heatmap

## ⚙️Tech Stack

Category|	Tools
|---|---|
Language|	Python 3
Libraries|	pandas, numpy, matplotlib, seaborn
Environment|	Jupyter Notebook / VS Code
Version Control|	Git & GitHub
Dataset Hosting|	Kaggle Datasets

## 📄 License

This project is licensed under the MIT License.
