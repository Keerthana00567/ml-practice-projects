# ML Practice Projects

A collection of end-to-end classification workflows built while practicing core machine learning skills — EDA, preprocessing, model comparison, and hyperparameter tuning — on public Kaggle datasets.

These are learning/practice projects, distinct from my primary research work (see [oral-cancer-detection-cnn](../oral-cancer-detection-cnn) for that).

| Project | Task | Best Model | Accuracy |
|---|---|---|---|
| [Breast Cancer Detection](./breast-cancer-detection) | Binary classification | Logistic Regression | 97.4% |
| [Customer Purchase Prediction](./customer-purchase-prediction) | Binary classification | Random Forest | 91.3% |
| [Loan Approval Prediction](./loan-approval-prediction) | Binary classification | Random Forest | 98.2% |

## What these demonstrate
- Structured EDA (distribution checks, skewness, correlation heatmaps, outlier inspection)
- Feature preprocessing (scaling, one-hot encoding, label encoding)
- Training and comparing multiple algorithms (Logistic Regression, Decision Tree, Random Forest, KNN)
- Systematic hyperparameter tuning with `GridSearchCV`
- Model evaluation beyond accuracy (precision, recall, F1, ROC-AUC, confusion matrices)
- Model persistence with `joblib`

## Tech stack
Python, scikit-learn, pandas, NumPy, seaborn, matplotlib
