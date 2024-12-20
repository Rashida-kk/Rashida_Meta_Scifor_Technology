# -*- coding: utf-8 -*-
"""Loan Approval Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ceiw47oFfPd41W3qCr_wy5_s_H5vvSni
"""

#import laibraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("loan_approval_dataset.csv")

df.head()

print(df.shape)

print(df.info())

print(df.describe())

print(df.isnull().sum())

df = df.drop("loan_id", axis=1)
if 'loan_id' in df.columns:
    df = df.drop("loan_id", axis=1)
print(df.head())

df.columns = df.columns.str.strip()        #Removes any leading or trailing whitespace

df["Assets"] = df.residential_assets_value + df.commercial_assets_value + df.luxury_assets_value +df.bank_asset_value

df.drop(["residential_assets_value", "commercial_assets_value", "luxury_assets_value", "bank_asset_value"], axis=1, inplace=True)

def clean_data(st):
  st = st.strip()
  return st

clean_data(' Graduate')
df.education = df.education.apply(clean_data)
df["education"] = df["education"].replace(["Graduate","Not Graduate"],[1,0])

df.self_employed = df.self_employed.apply(clean_data)
df.self_employed = df.self_employed.replace(["Yes","No"],[1,0])

df.loan_status = df.loan_status.apply(clean_data)
df.loan_status = df.loan_status.replace(["Approved","Rejected"],[1,0])

# Countplot for loan_status
sns.countplot(x='loan_status', data=df)
plt.title("Loan Status Distribution")
plt.show()

#outlier detection
numerical_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets']
sns.boxplot(data=df[numerical_cols])

# Function to detect and remove outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    cleaned_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return cleaned_df

# Apply to specific columns
for col in numerical_cols:
    df = remove_outliers(df, col)

print("Outliers removed (IQR method).")

# Correlation heatmap for numerical features
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Feature-target split
input_data = df.drop("loan_status", axis=1)
target_data = df["loan_status"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.2, random_state=2)

# Standardize numerical features
from sklearn.preprocessing import LabelEncoder, StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Initialize models
lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train Logistic Regression
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

# Train Random Forest
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

# Model Evaluation
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

print("Random Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(x_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Predict using the best Random Forest model
y_pred = best_model.predict(x_test)
y_proba = best_model.predict_proba(x_test)[:, 1]

# Calculate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# Display the metrics
print("\nRandom Forest Performance After Hyperparameter Tuning:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Random Forest")
plt.legend()
plt.show()

#Predict New Loan Applications
#Make predictions on new loan applications.
# new data
new_application = pd.DataFrame({
    'no_of_dependents': [2],
    'education': [1],
    'self_employed': [0],
    'income_annum': [600000],
    'loan_amount': [200000],
    'loan_term': [15],
    'cibil_score': [720],
    'Assets': [500000]})

# Preprocess new application
new_application = scaler.transform(new_application)
prediction = best_model.predict(new_application)
print("Loan Status Prediction:", "Approved" if prediction[0] == 1 else "Rejected")

import pickle

# Instead of best_model = best_models['Random Forest'],
# use the already assigned best_model variable
with open('loan_approval_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

#Build a Web App

import streamlit as st
import joblib

# Load the model
with open('loan_approval_model.pkl', 'rb') as file:
    model = pickle.load(file)

#Use Streamlit to create an interactive loan approval application.

import streamlit as st

# Streamlit UI
st.title("Loan Prediction App")

no_of_dependents = st.slider("Choose No of Dependents", 0, 10)
education = st.selectbox("Choose Education", ["Graduate", "Non-Graduate"])
self_employed = st.selectbox("Self-Employed", ["No", "Yes"])
income_annum = st.slider("Choose Annual Income", 0, 10000000)
loan_amount = st.slider("Choose Loan Amount", 0, 10000000)
loan_term = st.number_input("Loan Term (in months)")
cibil_score = st.slider("Choose CIBIL Score", 0, 1000)
assets = st.slider("Choose Asset ", 0, 10000000)


if st.button("Predict Loan Status"):
    # Preprocess input
    input_data = [[no_of_dependents, 1 if education == "Graduate" else 0, 1 if self_employed == "Yes" else 0,
                   income_annum, loan_amount, loan_term, cibil_score, assets]]
    prediction = model.predict(input_data)
    result = "Approved" if prediction[0] == 1 else "Rejected"
    st.success(f"Loan Status: {result}")



df.head()









































