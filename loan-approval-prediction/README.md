# Loan Approval Prediction

Predicts loan approval status from applicant and asset financial data, comparing a scaled Logistic Regression pipeline against tree-based models.

## Dataset
`loan_approval_dataset.csv` — applicant income, loan amount, asset values, CIBIL score, and related fields → loan_status (approved/rejected)

## Approach
1. Label-encoded the target variable
2. EDA: bar plots of key financial features against loan status, correlation heatmap to rank feature relevance
3. One-hot encoded categorical features; stratified 80/20 split
4. Built a `Pipeline` (StandardScaler + LogisticRegression) and tuned it with `GridSearchCV`
5. Trained and tuned Decision Tree and Random Forest classifiers for comparison
6. Evaluated all models on accuracy, precision, recall, F1, and ROC-AUC

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (pipeline) | 92.15% | 91.56% | 87.31% | 89.38% | 91.21% |
| Decision Tree (tuned) | 97.54% | 97.48% | 95.98% | 96.72% | 97.23% |
| **Random Forest (best, tuned)** | **98.24%** | **99.04%** | **96.28%** | **97.65%** | **97.86%** |

## Tech stack
Python, scikit-learn, pandas, NumPy, seaborn, matplotlib

## Possible extensions
- Feature importance analysis to identify which financial signals drive approval most
- Check for data leakage / correlated proxy features given the very high accuracy
- Calibrate probability outputs if this were used for real risk scoring
