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
