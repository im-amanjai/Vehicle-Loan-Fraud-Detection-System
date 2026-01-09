import streamlit as st
import numpy as np
import pandas as pd
import joblib

# --------------------------------------------------
# Load model & metadata
# --------------------------------------------------
model = joblib.load("vehicle_loan_xgb.pkl")
feature_names = joblib.load("feature_names.pkl")
feature_medians = joblib.load("feature_medians.pkl")

st.set_page_config(
    page_title="Vehicle Loan Fraud Detection",
    layout="centered"
)

st.title("🚗 Vehicle Loan Fraud Detection System")
st.write("AI-based system to assess **vehicle loan default / fraud risk**")
st.divider()

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
disbursed_amount = st.number_input("Disbursed Amount (₹)", min_value=0, step=1000)
asset_cost = st.number_input("Asset Cost (₹)", min_value=0, step=1000)
monthly_income = st.number_input("Monthly Income (₹)", min_value=0, step=1000)

# --------------------------------------------------
# Auto LTV Calculation
# --------------------------------------------------
if asset_cost > 0:
    raw_ltv = (disbursed_amount / asset_cost) * 100
    ltv = round(min(raw_ltv, 100), 2)
else:
    raw_ltv = 0
    ltv = 0

st.metric("Loan to Value (LTV %)", ltv)

if raw_ltv > 90:
    st.warning(f"⚠️ Very High LTV detected ({raw_ltv:.2f}%)")

# --------------------------------------------------
# Other Inputs
# --------------------------------------------------
credit_score = st.slider("Credit Score", 300, 900, 650)
emi = st.number_input("EMI Amount (₹)", min_value=0, step=500)

employment = st.selectbox("Employment Type", ["Salaried", "Self-employed"])
employment_encoded = 1 if employment == "Self-employed" else 0

# --------------------------------------------------
# Income-to-EMI Ratio (Employment-aware)
# --------------------------------------------------
if monthly_income > 0:
    emi_ratio = round((emi / monthly_income) * 100, 2)
else:
    emi_ratio = 0

st.metric("Income-to-EMI Ratio (%)", emi_ratio)

if employment == "Self-employed":
    safe_limit = 25
    risk_limit = 40
else:
    safe_limit = 30
    risk_limit = 50

if emi_ratio > risk_limit:
    st.error("🚨 High EMI burden considering income stability")
elif emi_ratio > safe_limit:
    st.warning("⚠️ Moderate EMI burden")

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Check Loan Risk"):
    if asset_cost == 0 or monthly_income == 0:
        st.warning("⚠️ Asset cost and monthly income must be greater than zero")
    else:
        # Build full feature vector
        input_dict = feature_medians.copy()
        input_dict["disbursed_amount"] = disbursed_amount
        input_dict["asset_cost"] = asset_cost
        input_dict["ltv"] = ltv
        input_dict["PERFORM_CNS_SCORE"] = credit_score
        input_dict["PRIMARY.INSTAL.AMT"] = emi
        input_dict["Employment.Type"] = employment_encoded

        input_df = pd.DataFrame(
            [[input_dict[col] for col in feature_names]],
            columns=feature_names
        )

        # ML Probability
        prob = model.predict_proba(input_df)[0][1]

        # --------------------------------------------------
        # Reason Codes
        # --------------------------------------------------
        reasons = []

        if credit_score < 650:
            reasons.append("Low Credit Score")

        if raw_ltv > 70:
            reasons.append("High Loan-to-Value (LTV)")

        if emi_ratio > safe_limit:
            reasons.append("High EMI-to-Income burden")

        if employment == "Self-employed":
            reasons.append("Income variability due to self-employment")

        # --------------------------------------------------
        # HARD AUTO-REJECT POLICY (Bank Override)
        # --------------------------------------------------
        hard_reject_count = 0

        if credit_score < 500:
            hard_reject_count += 1

        if raw_ltv > 90:
            hard_reject_count += 1

        if emi_ratio > risk_limit:
            hard_reject_count += 1

        if hard_reject_count >= 2:
            st.error(
                "🔴 HIGH RISK – AUTO REJECT (Policy Override)\n\n"
                f"Risk Probability (ML): **{prob:.2f}**\n\n"
                "Decision: **Auto Reject**"
            )

            st.markdown("### 🔍 Reason Codes")
            for r in reasons:
                st.write(f"• {r}")

            st.stop()   # ⛔ stop further ML-based decision

        # --------------------------------------------------
        # Risk Levels (ML-Based)
        # --------------------------------------------------
        if prob < 0.3:
            risk = "Low"
            decision = "Auto Approval"
        elif prob < 0.6:
            risk = "Medium"
            decision = "Manual Review Required"
        else:
            risk = "High"
            decision = "Reject / Fraud Investigation"

        # --------------------------------------------------
        # Display Result
        # --------------------------------------------------
        if risk == "Low":
            st.success(
                f"🟢 Low Risk\n\n"
                f"Risk Probability: **{prob:.2f}**\n\n"
                f"Decision: **{decision}**"
            )

        elif risk == "Medium":
            st.warning(
                f"🟡 Medium Risk\n\n"
                f"Risk Probability: **{prob:.2f}**\n\n"
                f"Decision: **{decision}**"
            )

        else:
            st.error(
                f"🔴 High Risk\n\n"
                f"Risk Probability: **{prob:.2f}**\n\n"
                f"Decision: **{decision}**"
            )

        if reasons:
            st.markdown("### 🔍 Reason Codes")
            for r in reasons:
                st.write(f"• {r}")
