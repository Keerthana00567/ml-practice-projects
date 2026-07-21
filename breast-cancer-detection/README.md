# Breast Cancer Detection (Logistic Regression)

Binary classification model to predict whether a tumor is malignant or benign, using the classic Wisconsin Breast Cancer dataset.

## Dataset
- `sklearn.datasets.load_breast_cancer` — 569 samples, 30 numeric features (cell nuclei measurements)
- No missing values; used directly from scikit-learn's built-in dataset loader

## Approach
1. Split data 80/20 (train/test)
2. Standardized features with `StandardScaler`
3. Trained a `LogisticRegression` classifier
4. Evaluated with accuracy, precision, recall, F1, and confusion matrix
5. Persisted the trained model with `joblib` and verified reload produces identical predictions

## Results

| Metric | Score |
|---|---|
| Accuracy | 97.37% |
| Precision | 97.22% |
| Recall | 98.59% |
| F1 Score | 97.90% |

## Tech stack
Python, scikit-learn, pandas, NumPy, joblib

## Possible extensions
- Compare against Random Forest / SVM / Gradient Boosting
- Add SHAP-based feature importance to explain predictions
- Wrap the trained model in a small Flask/FastAPI inference endpoint
