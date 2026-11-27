# Loan Approval Prediction System

A machine learning model and Streamlit web app that predicts whether a loan application will be approved or rejected, to support faster, more consistent lending decisions.

## Problem
Lenders review many applications manually, which is slow and inconsistent. This project predicts approval from applicant and loan details so reviewers can prioritise and double-check decisions.

## Data
- 4,269 loan applications (about 62% approved, 38% rejected)
- Features: number of dependents, education, self-employment, annual income, loan amount, loan term, CIBIL (credit) score, and total assets
- Preparation: removed the ID column, combined four asset columns into one `Assets` feature, encoded categories, checked for outliers (IQR method), and used an 80/20 train-test split

## Approach
1. Exploratory analysis: class balance, outliers, correlation heatmap
2. Baseline: Logistic Regression (about 81% accuracy)
3. Main model: Random Forest, tuned with grid search and 5-fold cross-validation
4. Deployment: interactive Streamlit app for real-time predictions

## Results (Random Forest, test set)
| Metric | Score |
|---|---|
| Accuracy | 97.9% |
| Precision | 98.5% |
| Recall | 98.1% |
| F1-score | 98.3% |
| ROC-AUC | 0.998 |

The tuned Random Forest clearly beat the Logistic Regression baseline.

**Key finding:** CIBIL score is by far the most important feature (about 81% of the model's feature importance), followed by loan term, loan amount, assets, and income.

## Limitations
- Because one feature dominates, the high accuracy partly reflects how strongly credit score drives approval in this dataset. I would validate on newer, real-world data before any real use.
- The dataset is a public, cleaned sample, so real lending data would be messier.
- Approval decisions can affect people, so a model like this should support human review, not replace it.

## App
![App screenshot](app_screenshot.png)

## Run it locally
```bash
pip install -r requirements.txt
streamlit run loan_approval_prediction.py
```

## Tools
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit
