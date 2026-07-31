# Hotel Booking Cancellation Prediction

## 📌 Project Overview

This project predicts whether a hotel booking will be cancelled based on customer booking information using machine learning classification algorithms.

The project demonstrates a complete machine learning workflow, including:
- Data preprocessing
- Feature engineering
- Model training
- Hyperparameter tuning
- Model evaluation
- Model saving

---

## 🎯 Objective

To build a classification model that predicts whether a hotel booking will be cancelled before the customer's arrival.

---

## 📂 Dataset

The dataset contains hotel booking information such as:

- Lead time
- Arrival date
- Market segment
- Deposit type
- Customer type
- Previous cancellations
- Special requests
- Booking status (Target)

**Target Variable:**

- `0` → Booking Not Cancelled
- `1` → Booking Cancelled

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

## ⚙️ Machine Learning Workflow

1. Data Loading
2. Data Cleaning
3. Feature Engineering
4. Encoding Categorical Features
5. Train-Test Split
6. Feature Scaling
7. Model Training
8. Hyperparameter Tuning using GridSearchCV
9. Model Evaluation
10. Model Saving

---

## 🤖 Models Trained

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

## 📊 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

---

## 📁 Project Structure

```
hotel-booking-cancellation-prediction/
│
├── Hotel_Booking_Cancellation_Prediction.ipynb
├── booking.csv
├── hotel_booking_cancellation_model.joblib
├── requirements.txt
└── README.md
```

---

## 🚀 Future Improvements

- Add feature importance visualization
- Add confusion matrix visualization
- Deploy using Streamlit
- Improve feature engineering
- Perform cross-validation

---

## 👩‍💻 Author

**Keerthana N**
