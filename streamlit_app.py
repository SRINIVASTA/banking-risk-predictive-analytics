import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# =====================================================================
# 1. STREAMLIT GLOBAL VIEWPORT INITIALIZATION (Strict Screen Fit)
# =====================================================================
st.set_page_config(
    page_title="Banking Analytics Platform", 
    page_icon="🚀", 
    layout="wide"  
)

# Using compact markdown typography to save vital vertical screen rows
st.markdown("<h2 style='margin:0; padding:0;'>🚀 Executive Banking Performance Analytics Platform</h2>", unsafe_allow_html=True)
st.markdown("<hr style='margin:4px 0;'/>", unsafe_allow_html=True)

# =====================================================================
# 2. RELATIONAL DATA PROCESSING PIPELINE
# =====================================================================
@st.cache_data
def verify_and_load_relational_database():
    """
    Attempts to read data from the designated repository workspace structure.
    If CSV files are missing, it triggers an in-memory fallback synthesizer.
    """
    cust_path = 'data/customer_profiles.csv'
    tx_path = 'data/transaction_history.csv'
    loan_path = 'data/loan_records.csv'
    
    try:
        df_cust = pd.read_csv(cust_path)
        df_tx = pd.read_csv(tx_path)
        df_loans = pd.read_csv(loan_path)
        df_tx['Timestamp'] = pd.to_datetime(df_tx['Timestamp'])
        return df_cust, df_tx, df_loans
    except FileNotFoundError:
        np.random.seed(42)
        NUM_CUSTOMERS, NUM_TRANSACTIONS, NUM_LOANS = 200, 1500, 80

        customer_ids = [f"CUST_{i:04d}" for i in range(1, NUM_CUSTOMERS + 1)]
        df_cust = pd.DataFrame({
            'CustomerID': customer_ids,
            'Age': np.random.randint(18, 75, size=NUM_CUSTOMERS),
            'Income': np.random.randint(20000, 160000, size=NUM_CUSTOMERS),
            'CreditScore': np.random.randint(450, 850, size=NUM_CUSTOMERS),
            'AccountTier': np.random.choice(['Silver', 'Gold', 'Platinum'], size=NUM_CUSTOMERS, p=[0.5, 0.3, 0.2])
        })

        tx_amounts = np.round(np.random.exponential(scale=80, size=NUM_TRANSACTIONS) + 2, 2)
        fraud_indices = np.random.choice(range(NUM_TRANSACTIONS), size=int(NUM_TRANSACTIONS * 0.015), replace=False)
        tx_amounts[fraud_indices] = np.round(np.random.uniform(3000, 12000, size=len(fraud_indices)), 2)
        timestamps = [datetime(2026, 1, 1) + timedelta(days=int(np.random.randint(0, 160))) for _ in range(NUM_TRANSACTIONS)]

        df_tx = pd.DataFrame({
            'TransactionID': [f"TX_{i:06d}" for i in range(1, NUM_TRANSACTIONS + 1)],
            'CustomerID': np.random.choice(customer_ids, size=NUM_TRANSACTIONS),
            'Timestamp': pd.to_datetime(timestamps),
            'Amount': tx_amounts,
            'Category': np.random.choice(['Groceries', 'Utilities', 'Travel', 'Dining Out'], size=NUM_TRANSACTIONS),
            'Channel': np.random.choice(['Mobile App', 'ATM', 'Branch'], size=NUM_TRANSACTIONS),
            'IsFlaggedFraud': [1 if i in fraud_indices else 0 for i in range(NUM_TRANSACTIONS)]
        })

        loan_cust_ids = np.random.choice(customer_ids, size=NUM_LOANS, replace=False)
        loan_status = []
        for cid in loan_cust_ids:
            score = df_cust.loc[df_cust['CustomerID'] == cid, 'CreditScore'].values
            p_dist = [0.4, 0.3, 0.3] if score < 580 else ([0.7, 0.2, 0.1] if score < 670 else [0.95, 0.04, 0.01])
            loan_status.append(np.random.choice(['Current', 'Late', 'Defaulted'], p=p_dist))

        df_loans = pd.DataFrame({
            'LoanID': [f"LN_{i:04d}" for i in range(1, NUM_LOANS + 1)],
            'CustomerID': loan_cust_ids,
            'LoanAmount': np.round(np.random.randint(5000, 60000, size=NUM_LOANS), -2),
            'InterestRate': np.round(np.random.uniform(5.0, 18.0, size=NUM_LOANS), 2),
            'TermMonths': np.random.choice([12, 24, 36, 48, 60], size=NUM_LOANS),
            'CurrentStatus': loan_status
        })
        return df_cust, df_tx, df_loans

df_cust, df_tx, df_loans = verify_and_load_relational_database()

# =====================================================================
# 3. EXECUTIVE MANAGEMENT METRIC HIGHLIGHTS
# =====================================================================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(label="Total Active Customers", value=f"{len(df_cust)}")
with kpi2:
    st.metric(label="Total Audited Transactions", value=f"{len(df_tx)}")
with kpi3:
    mean_amt = df_tx['Amount'].mean()
    std_amt = df_tx['Amount'].std()
    fraud_threshold = mean_amt + (3 * std_amt)
    anomalies_count = len(df_tx[df_tx['Amount'] > fraud_threshold])
    st.metric(label="Flagged Fraud Anomalies", value=anomalies_count, delta="Outlier Threshold >3 SD", delta_color="inverse")
with kpi4:
    total_exposure = df_loans['LoanAmount'].sum()
    st.metric(label="Active Capital Exposure", value=f"${total_exposure:,.0f}")

st.markdown("<hr style='margin:4px 0;'/>", unsafe_allow_html=True)

# =====================================================================
# 4. UNIFIED 4-PANEL LAPTOP VIEWPORT OPTIMISED VISUAL MATRIX
# =====================================================================
st.markdown("<h4 style='margin:0 0 4px 0; padding:0;'>📊 Executive Operational Workspace Visual Matrix</h4>", unsafe_allow_html=True)

sns.set_theme(style="whitegrid")
# COMPRESSED: Changed figure vertical scale down to 4.3 to pack charts tightly into the screen space
fig, axes = plt.subplots(2, 2, figsize=(11, 4.3))
plt.rcParams.update({
    'font.size': 7.5, 
    'axes.labelsize': 8.5, 
    'axes.titlesize': 9
})

# Panel A: Feature Importance Plot
feature_data = pd.DataFrame({
    'Feature': ['Recency', 'Tx_Count', 'CreditScore', 'Total_Spend', 'Income', 'Age', 'Avg_Tx_Value', 'Has_Active_Loan'],
    'Importance': [0.555690, 0.259590, 0.045125, 0.044271, 0.034610, 0.028467, 0.027646, 0.004600]
}).sort_values(by='Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_data, ax=axes[0, 0], palette='Blues_d', hue='Feature', legend=False)
axes[0, 0].set_title("💡 Churn Predictive Feature Drivers", fontweight='bold', pad=4)
axes[0, 0].set_xlabel("")
axes[0, 0].set_ylabel("")
axes[0, 0].tick_params(axis='both', labelsize=7.5)

# Panel B: Loan Default Rates Plot
df_loan_risk = pd.merge(df_loans, df_cust, on='CustomerID')
def assign_credit_tier(score):
    if score < 580: return 'Poor (<580)'
    elif score < 670: return 'Fair (580-669)'
    elif score < 740: return 'Good (670-739)'
    else: return 'Excellent (740+)'
df_loan_risk['Credit_Tier'] = df_loan_risk['CreditScore'].apply(assign_credit_tier)
tier_order = ['Poor (<580)', 'Fair (580-669)', 'Good (670-739)', 'Excellent (740+)']

loan_summary = df_loan_risk.groupby('Credit_Tier').agg(
    Total_Loans=('LoanID', 'count'),
    Defaults=('CurrentStatus', lambda x: (x == 'Defaulted').sum())
).reindex(tier_order).fillna(0).reset_index()
loan_summary['Default_Rate'] = (loan_summary['Defaults'] / loan_summary['Total_Loans']) * 100

sns.barplot(x='Credit_Tier', y='Default_Rate', data=loan_summary, ax=axes[0, 1], palette='Oranges_r', hue='Credit_Tier', legend=False)
axes[0, 1].set_title("📉 Asset Delinquency: Loan Default Rates", fontweight='bold', pad=4)
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylabel("")
axes[0, 1].tick_params(axis='both', labelsize=7.5)
for p in axes[0, 1].patches:
    axes[0, 1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.2),
                        ha='center', va='center', xytext=(0, 2), textcoords='offset points', fontsize=7.5)

# Panel C: Fraud Outlier Distribution
df_tx['Z_Score'] = (df_tx['Amount'] - mean_amt) / std_amt
df_tx['Status'] = np.where(df_tx['Z_Score'] > 3, 'Flagged Outlier (>3 SD)', 'Normal Core Process')
sns.scatterplot(x=df_tx.index, y='Amount', hue='Status', data=df_tx,
                palette={'Normal Core Process': '#cccccc', 'Flagged Outlier (>3 SD)' : '#cc0000'},
                ax=axes[1, 0], alpha=0.4, s=8, edgecolor=None)
axes[1, 0].set_title("🚨 Transaction Auditing: Fraud Anomaly Plot", fontweight='bold', pad=4)
axes[1, 0].set_xlabel("")
axes[1, 0].set_ylabel("")
axes[1, 0].get_legend().remove()
axes[1, 0].tick_params(axis='both', labelsize=7.5)

# Panel D: Revenue Contribution Per Card Tier
tier_spend = df_tx.merge(df_cust, on='CustomerID').groupby('AccountTier')['Amount'].sum().reset_index()
axes[1, 1].pie(tier_spend['Amount'], labels=tier_spend['AccountTier'], autopct='%1.1f%%',
               startangle=140, colors=['#cfd8dc', '#ffd54f', '#90caf9'], textprops={'fontsize': 7.5})
axes[1, 1].set_title("💎 Capital Contribution: Share per Card Tier", fontweight='bold', pad=4)

# Tight spacing adjustments to compress vertical margins completely
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
st.pyplot(fig, use_container_width=True)
st.markdown("<hr style='margin:4px 0;'/>", unsafe_allow_html=True)
# =====================================================================
# 5. 🔮 SIDEBAR-NESTED CUSTOMER CHURN RISK SIMULATOR ENGINE
# =====================================================================
st.sidebar.markdown("<h3 style='margin:0;'>🔮 Churn Risk Simulator</h3>", unsafe_allow_html=True)

credit_score = st.sidebar.slider("Customer Credit Score Metric", min_value=300, max_value=850, value=710, step=10)
age = st.sidebar.slider("Customer Age", min_value=18, max_value=80, value=35, step=1)
recency = st.sidebar.slider("Days Since Last Transaction (Recency)", min_value=0, max_value=180, value=14, step=1)
tx_count = st.sidebar.slider("Monthly Transaction Counts", min_value=0, max_value=100, value=38, step=1)
debt_status = st.sidebar.selectbox(
    "Active Account Debt Liability Status", 
    ["No Active Loan", "Healthy Active Loan", "Delinquent / Default"]
)

# Churn Mathematical Calculation Models
base_risk = 35.0
base_risk -= (credit_score - 300) * 0.06   
base_risk += recency * 0.42                
base_risk -= tx_count * 0.32               
if debt_status == "Delinquent / Default":
    base_risk += 28.0                      
elif debt_status == "Healthy Active Loan":
    base_risk -= 4.0                       

churn_probability = max(0.0, min(100.0, base_risk))

# Highly compressed output section to prevent screen overflow
st.markdown("<h4 style='margin:0 0 2px 0; padding:0;'>🔮 Live Churn Probability Monitor Output</h4>", unsafe_allow_html=True)

# Low-profile flat bar gauge sizing configuration
fig_sim, ax_sim = plt.subplots(figsize=(10, 0.4))

if churn_probability < 30.0:
    bar_color = '#2ecc71'  
    status_text = f"🟢 HEALTHY ACCOUNT FRAME: ACTIVE PROFILE ({churn_probability:.1f}%) — Standard background monitoring metrics hold safe."
    alert_func = st.success
elif churn_probability < 70.0:
    bar_color = '#f1c40f'  
    status_text = f"🟡 ELEVATED RISK FRAME: WATCHLIST PROFILE ({churn_probability:.1f}%) — Initiate customer retention loops."
    alert_func = st.warning
else:
    bar_color = '#e74c3c'  
    status_text = f"🔴 HIGH CHURN RISK: CRITICAL PROFILE ({churn_probability:.1f}%) — Trigger immediate outreach protocols."
    alert_func = st.error

ax_sim.barh(["Risk"], [churn_probability], color=bar_color, height=0.5, edgecolor='none')
ax_sim.set_xlim(0, 100)
ax_sim.xaxis.set_visible(False) # Hiding axes ticks saves important vertical pixel heights
ax_sim.yaxis.set_visible(False)

# Overlay value text string explicitly near the bar tracking edge
ax_sim.text(churn_probability + 1.2, 0, f"{churn_probability:.1f}%", va='center', ha='left', fontweight='bold', color='#333333', fontsize=10)

sns.despine(left=True, bottom=True)
plt.tight_layout(pad=0)

# Display charts and short feedback banner cleanly without adding vertical empty spaces
st.pyplot(fig_sim, use_container_width=True)
alert_func(status_text)
