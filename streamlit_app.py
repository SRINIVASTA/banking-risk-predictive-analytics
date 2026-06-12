import streamlit as st

st.set_page_config(page_title="CareerDream Banking Analytics", page_icon="🚀", layout="centered")
st.title("🚀 Executive Customer Churn Prediction Engine")
st.write("Enter profile traits below to check operational retention risks in real-time.")

# Interactive Parameter UI Inputs
age = st.slider("Customer Age Span", 18, 80, 42)
income = st.number_input("Annual Gross Income ($)", 10000, 200000, 55000)
credit_score = st.slider("Credit Rating Score Matrix", 300, 850, 680)
recency = st.slider("Days Since Last Transaction (Recency)", 0, 180, 14)
tx_count = st.slider("Monthly Transaction Volume Frequency", 0, 100, 22)
total_spend = st.number_input("Cumulative Card Expenditure ($)", 0, 50000, 1800)
has_loan = st.selectbox("Active Debt Liability Holder", ["No Active Loan", "Has Active Bank Loan"])
has_loan_numeric = 1 if has_loan == "Has Active Bank Loan" else 0

# Predictive Calculation Heuristics Mapping logic 
churn_probability = (recency / 180) * 0.75 + (1 - (credit_score / 850)) * 0.25 - (has_loan_numeric * 0.15)
churn_probability = max(0.0, min(1.0, churn_probability))

st.markdown("---")
st.subheader("🔮 Predictive Risk Scoring Diagnostic Evaluation:")
if churn_probability > 0.55:
    st.error(f"🔴 ALERT: HIGH RISK OF PROFILE CHURN ({churn_probability*100:.1f}%)")
    st.write("⚠️ Recommended Action: Dispatch retention discounts and targeted customer support offers.")
else:
    st.success(f"🟢 STABLE PROFILE: ACTIVE USER ({churn_probability*100:.1f}%)")
    st.write("✅ Account metrics show steady user engagement and account health.")
