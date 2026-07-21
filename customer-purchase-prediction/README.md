# Customer Purchase Prediction

Predicts whether a customer will purchase a product based on age, estimated salary, and gender, using the Social Network Ads dataset. Trains and compares four classifiers rather than settling for one.

## Dataset
`Social_Network_Ads.csv` — Age, EstimatedSalary, Gender → Purchased (binary)

## Approach
1. EDA: distribution plots, skewness check, boxplots for outliers, correlation matrix, class balance check
2. One-hot encoded categorical features
3. Stratified 80/20 train/test split
4. Trained and tuned four models with `GridSearchCV`:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - K-Nearest Neighbors
5. Compared all four on accuracy, precision, recall, and F1; selected the best performer
6. Persisted the winning model with `joblib`

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 81.25% | 81.82% | 62.07% | 70.59% |
| Decision Tree | 81.25% | 75.00% | 72.41% | 73.68% |
| **Random Forest (best)** | **91.25%** | **86.67%** | **89.66%** | **88.14%** |
| KNN | 87.50% | 80.65% | 86.21% | 83.33% |

## Tech stack
Python, scikit-learn, pandas, NumPy, seaborn, matplotlib, joblib

## Possible extensions
- Try gradient boosting (XGBoost/LightGBM) for comparison
- Feature-engineer salary bands or age groups
- Serve the saved model via a simple API
