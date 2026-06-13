import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# =====================================================================
# 1. STREAMLIT GLOBAL VIEWPORT INITIALIZATION (Auto-Fit Tuning)
# =====================================================================
st.set_page_config(
    page_title="CareerDream Banking Analytics Platform", 
    page_icon="🚀", 
    layout="wide"  # Uses fluid responsive widths to wrap neatly on laptop displays
)

st.title("🚀 CareerDream.in — Executive Banking Performance Analytics Platform")
st.markdown("---")

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

st.markdown("---")

# =====================================================================
# 4. UNIFIED 4-PANEL LAPTOP OPTIMISED VISUAL MATRIX
# =====================================================================
st.subheader("📊 High-Resolution Executive Workspace Visual Matrix")

sns.set_theme(style="whitegrid")
# FIXED: Re-scaled to a highly compact 11x6.5 layout size to avoid screen spillover errors
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
plt.rcParams.update({
    'font.size': 8, 
    'axes.labelsize': 9, 
    'axes.titlesize': 10
})

# Panel A: Feature Importance Plot
feature_data = pd.DataFrame({
    'Feature': ['Recency', 'Tx_Count', 'CreditScore', 'Total_Spend', 'Income', 'Age', 'Avg_Tx_Value', 'Has_Active_Loan'],
    'Importance': [0.555690, 0.259590, 0.045125, 0.044271, 0.034610, 0.028467, 0.027646, 0.004600]
}).sort_values(by='Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_data, ax=axes[0, 0], palette='Blues_d', hue='Feature', legend=False)
axes[0, 0].set_title("💡 Churn Predictive Feature Drivers", fontweight='bold')
axes[0, 0].set_xlabel("Predictive Weight Value")
axes[0, 0].set_ylabel("")
axes[0, 0].tick_params(axis='both', labelsize=8)

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
axes[0, 1].set_title("📉 Asset Delinquency: Loan Default Rates", fontweight='bold')
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylabel("Default Rate (%)")
axes[0, 1].tick_params(axis='both', labelsize=8)
for p in axes[0, 1].patches:
    axes[0, 1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.3),
                        ha='center', va='center', xytext=(0, 3), textcoords='offset points', fontsize=8)

# Panel C: Fraud Outlier Distribution
df_tx['Z_Score'] = (df_tx['Amount'] - mean_amt) / std_amt
df_tx['Status'] = np.where(df_tx['Z_Score'] > 3, 'Flagged Outlier (>3 SD)', 'Normal Core Process')
sns.scatterplot(x=df_tx.index, y='Amount', hue='Status', data=df_tx,
                palette={'Normal Core Process': '#cccccc', 'Flagged Outlier (>3 SD)' : '#cc0000'},
                ax=axes[1, 0], alpha=0.5, s=10, edgecolor=None)
axes[1, 0].set_title("🚨 Transaction Auditing: Fraud Anomaly Plot", fontweight='bold')
axes[1, 0].set_xlabel("Sequential Transaction Reference ID")
axes[1, 0].set_ylabel("Volume Magnitude ($)")
axes[1, 0].legend(loc='upper right', fontsize=7)
axes[1, 0].tick_params(axis='both', labelsize=8)

# Panel D: Revenue Contribution Per Card Tier
tier_spend = df_tx.merge(df_cust, on='CustomerID').groupby('AccountTier')['Amount'].sum().reset_index()
axes[1, 1].pie(tier_spend['Amount'], labels=tier_spend['AccountTier'], autopct='%1.1f%%',
               startangle=140, colors=['#cfd8dc', '#ffd54f', '#90caf9'], textprops={'fontsize': 8})
axes[1, 1].set_title("💎 Capital Contribution: Share per Card Tier", fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Forces strict matching scaling onto responsive CSS grid blocks
st.pyplot(fig, use_container_width=True)
st.markdown("---")
# =====================================================================
# 5. 🔮 SCREEN-SAFE REAL-TIME CUSTOMER CHURN RISK SIMULATOR ENGINE
# =====================================================================
st.subheader("🔮 Interactive Real-Time Customer Churn Risk Simulator")
st.markdown("Adjust parameters dynamically. The prediction layout scales cleanly within laptop viewport grids.")

# FIXED: Nested row layout arrays to cluster sliders cleanly without taking side-by-side screen space
row_input1, row_input2, row_input3 = st.columns(3)

with row_input1:
    credit_score = st.slider("Customer Credit Score Metric", min_value=300, max_value=850, value=710, step=10)
    age = st.slider("Customer Age", min_value=18, max_value=80, value=35, step=1)

with row_input2:
    recency = st.slider("Days Since Last Transaction (Recency)", min_value=0, max_value=180, value=14, step=1)
    tx_count = st.slider("Monthly Transaction Counts", min_value=0, max_value=100, value=38, step=1)

with row_input3:
    debt_status = st.selectbox(
        "Active Account Debt Liability Status", 
        ["No Active Loan", "Healthy Active Loan", "Delinquent / Default"]
    )

# Model Formulation Calculation Code
base_risk = 35.0
base_risk -= (credit_score - 300) * 0.06   
base_risk += recency * 0.42                
base_risk -= tx_count * 0.32               
if debt_status == "Delinquent / Default":
    base_risk += 28.0                      
elif debt_status == "Healthy Active Loan":
    base_risk -= 4.0                       

churn_probability = max(0.0, min(100.0, base_risk))

# FIXED: Replaced split columns with vertical block stacking to keep metrics nicely within screen bounds
st.markdown("### Calculated Live Risk Profile Matrix Output:")

# Compile highly responsive horizontal gauge tracker asset bar
fig_sim, ax_sim = plt.subplots(figsize=(10, 1.2))

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

ax_sim.barh(["Churn Risk"], [churn_probability], color=bar_color, height=0.4, edgecolor='none')
ax_sim.set_xlim(0, 100)
ax_sim.set_xlabel("Probability Rate Metric (%)", fontsize=8)
ax_sim.tick_params(axis='both', which='major', labelsize=8)

# Overlay direct statistical percentage labels inside the visual timeline track boundaries
ax_sim.text(churn_probability + 1.5, 0, f"{churn_probability:.1f}%", va='center', ha='left', fontweight='bold', color='#333333', fontsize=10)

sns.despine(left=True, bottom=False)
plt.tight_layout()

# Render output canvas elements sequentially inside responsive container parameters
st.pyplot(fig_sim, use_container_width=True)
alert_func(status_text)
