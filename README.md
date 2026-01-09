# 🚗 Vehicle Loan Fraud & Default Risk Detection System

An **AI-powered web application** that predicts **vehicle loan default / fraud risk** using **Machine Learning combined with banking policy rules**.  
The system simulates **real-world bank/NBFC decision-making** by combining predictive analytics with rule-based overrides and explainability.

---

## 📌 Problem Statement

Vehicle loan providers face significant financial losses due to **loan defaults and fraudulent applications**.  
Pure rule-based systems are rigid, while standalone ML models lack regulatory and policy enforcement.

This project bridges that gap by:
- Predicting default risk using ML
- Enforcing **banking policies** through hard rules
- Providing **transparent reason codes** for decisions

---

## 📊 Dataset Used

- **Dataset Name:** LT Vehicle Loan Default Prediction  
- **Source:** Kaggle  

🔗 **Dataset Link:**  
https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction

### Dataset Description
The dataset contains real-world vehicle loan application data including:
- Customer demographic information
- Loan details (disbursed amount, asset cost, EMI)
- Credit bureau score
- Employment type
- Loan performance indicators

**Target Variable:**
```text
loan_default
1 = Default
0 = No Default

---

🧠 Model Training Approach
1️⃣ Data Preprocessing

Handled missing values using median (numerical) and mode (categorical)

Encoded categorical variables (Employment Type)

Removed data leakage columns

Balanced class impact using scale_pos_weight

2️⃣ Feature Engineering

Key engineered features:

Loan-to-Value (LTV)

Income-to-EMI Ratio

Employment-based risk thresholds

3️⃣ Model Used

XGBoost Classifier

Why XGBoost?

Handles non-linear patterns well

Performs strongly on tabular financial data

Robust to missing values

Initial Parameters:

XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=class_imbalance_ratio,
    eval_metric="logloss"
)

4️⃣ Hyperparameter Tuning

Used RandomizedSearchCV to optimize:

n_estimators

max_depth

learning_rate

subsample

colsample_bytree

Best tuned parameters were saved and used in the final model.

5️⃣ Model Evaluation

Metrics used:

Precision

Recall (especially for default class)

F1-score

Confusion Matrix

Focus was on recall for defaulters, as missing a risky customer is more costly than false positives.

🏦 Real-World Decision Logic (IMPORTANT)
ML Model Output

Produces default probability

Policy Override Layer

Even if ML probability is moderate, the system auto-rejects when:

Credit Score < 500

Loan-to-Value (LTV) > 90%

EMI-to-Income ratio exceeds threshold

➡ If 2 or more conditions are violated, the loan is Auto-Rejected.

This mirrors real banking systems.

🧩 Application Features

✅ Auto-calculated LTV

✅ Income-to-EMI ratio (employment-aware)

✅ ML-based risk probability

✅ Risk levels (Low / Medium / High)

✅ Policy-based Auto Reject

✅ Explainable Reason Codes

✅ Streamlit-based interactive UI

🖥️ Tech Stack

Python

XGBoost

Pandas / NumPy

Scikit-learn

Streamlit

Joblib

🚀 How to Run Locally
1️⃣ Clone the Repository
git clone https://github.com/your-username/vehicle-loan-fraud-detection.git
cd vehicle-loan-fraud-detection

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App
streamlit run app.py

🌐 Deployment

The application is deployed using Streamlit Cloud, directly connected to this GitHub repository.

📈 Future Enhancements

SHAP-based explainability plots

Admin dashboard for risk monitoring

Threshold configuration panel

PDF / CSV loan decision reports

API version for integration with core banking systems

👤 Author

Aman Jai
