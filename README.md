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
loan_default
1 = Default
0 = No Default

---

## 🧠 Model Training Approach

### 1️⃣ Data Preprocessing
- Handled missing values using **median** (numerical features) and **mode** (categorical features)
- Encoded categorical variables (Employment Type)
- Removed data leakage columns
- Balanced class impact using `scale_pos_weight` to handle class imbalance

---

### 2️⃣ Feature Engineering
Key engineered features include:
- **Loan-to-Value (LTV)**
- **Income-to-EMI Ratio**
- **Employment-based risk thresholds**

These features help capture borrower affordability and repayment capacity.

---

### 3️⃣ Model Used

**XGBoost Classifier**

**Why XGBoost?**
- Handles non-linear patterns effectively
- Performs strongly on tabular financial datasets
- Robust to missing values and noisy data

**Initial Parameters:**
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=class_imbalance_ratio,
    eval_metric="logloss"
)


### 4️⃣ Hyperparameter Tuning

Used **RandomizedSearchCV** to optimize key XGBoost hyperparameters for improved performance and generalization:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`

The best-performing parameter combination was selected based on cross-validation results and used to train the **final production model**.

---

### 5️⃣ Model Evaluation

The trained model was evaluated using the following metrics:

- **Precision**
- **Recall** (with special focus on the default class)
- **F1-score**
- **Confusion Matrix**

The optimization objective prioritized **high recall for defaulters**, as failing to identify a risky borrower is more costly than falsely flagging a safe applicant.


## 🏦 Real-World Decision Logic (IMPORTANT)

### ML Model Output
- The trained machine learning model produces a **default probability** for each loan application.

### Policy Override Layer
Even if the ML probability is moderate, the system **auto-rejects** a loan when the following high-risk conditions are observed:

- Credit Score < 500  
- Loan-to-Value (LTV) > 90%  
- EMI-to-Income ratio exceeds the allowed threshold  

➡ If **two or more conditions** are violated, the loan is **Auto-Rejected**.

This approach mirrors **real-world banking and NBFC decision systems**, where **policy and compliance rules override model predictions** to manage risk effectively.

---

## 🧩 Application Features

- ✅ Auto-calculated Loan-to-Value (LTV)
- ✅ Income-to-EMI ratio (employment-aware)
- ✅ ML-based risk probability
- ✅ Risk levels (Low / Medium / High)
- ✅ Policy-based Auto Reject
- ✅ Explainable Reason Codes
- ✅ Streamlit-based interactive UI

---

## 🖥️ Tech Stack

- Python  
- XGBoost  
- Pandas / NumPy  
- Scikit-learn  
- Streamlit  
- Joblib  


## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
git clone https://github.com/your-username/vehicle-loan-fraud-detection.git
cd vehicle-loan-fraud-detection

###2️⃣ Install Dependencies
pip install -r requirements.txt

###3️⃣ Run the App
streamlit run app.py

##🌐 Deployment

- The application is deployed using Streamlit Cloud, directly connected to this GitHub repository.
- 🔗 **Link:**
  https://vehicle-loan-fraud-detection-system.streamlit.app/

##📈 Future Enhancements

- SHAP-based explainability plots

- Admin dashboard for risk monitoring

- Threshold configuration panel

- PDF / CSV loan decision reports

- API version for integration with core banking systems

##👤 Author
Aman Jaiswal
