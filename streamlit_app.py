import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# =====================================================================
# 1. STREAMLIT GLOBAL VIEWPORT INITIALIZATION (Auto-Fit Layout Tuning)
# =====================================================================
st.set_page_config(
    page_title="CareerDream Banking Analytics Platform", 
    page_icon="🚀", 
    layout="wide"  
)

# Force-inject ultra-tight padding overrides to reclaim every single pixel row
st.markdown("""
    <style>
        .block-container {padding-top: 0rem !important; padding-bottom: 0rem !important;}
        [data-testid="stHeader"] {background: transparent; height: 0px;}
        [data-testid="stMetric"] {padding: 4px 8px !important;}
        [data-testid="stVerticalBlock"] {gap: 0.6rem !important;}
        hr {margin: 6px 0 !important; border-top: 1px solid #ddd !important;}
    </style>
""", unsafe_allow_html=True)

st.title("🚀 CareerDream.in — Executive Banking Performance Analytics Platform")

# =====================================================================
# INDIAN NUMBER SYSTEM CURRENCY FORMATTER FUNCTION
# =====================================================================
def format_indian_currency(number, include_symbol=True):
    """Formats a numeric value according to the Indian Numbering System (Lakhs/Crores)"""
    s = str(int(number))
    if len(s) <= 3:
        formatted = s
    else:
        last_three = s[-3:]
        remaining = s[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        formatted = ",".join(groups) + "," + last_three
    
    return f"₹{formatted}" if include_symbol else formatted

# =====================================================================
# DATA FACTORY ENGINE A: SYNTHETIC DATA BASELINE GENERATOR
# =====================================================================
@st.cache_data
def generate_default_fallback_data():
    """Generates baseline database structures at the new 1000-customer scale."""
    np.random.seed(42)
    NUM_CUSTOMERS, NUM_TRANSACTIONS, NUM_LOANS = 1000, 7500, 400

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

# =====================================================================
# DATA FACTORY ENGINE B: LIVE CLOUD WEB STREAM PIPELINE
# =====================================================================
@st.cache_data(ttl=600)  
def load_live_web_streams():
    """Dynamically streams active CSV data blocks from your external web hooks."""
    # 🔴 NOTE: Replace placeholder domains below with your active repository or sheet URLs
    cust_url = "https://githubusercontent.com"
    tx_url = "https://githubusercontent.com"
    loan_url = "https://githubusercontent.com"
    
    df_cust = pd.read_csv(cust_url)
    df_tx = pd.read_csv(tx_url)
    df_loans = pd.read_csv(loan_url)
    df_tx['Timestamp'] = pd.to_datetime(df_tx['Timestamp'])
    return df_cust, df_tx, df_loans
# =====================================================================
# BLOCK 2: SIDEBAR CONTROLS & DYNAMIC THREE-WAY DATA ROUTING MATRIX
# =====================================================================
st.sidebar.markdown("<h4 style='margin:0;'>🔮 Risk Simulator</h4>", unsafe_allow_html=True)

credit_score = st.sidebar.slider("Credit Score", 300, 850, 710, 10)
age = st.sidebar.slider("Age", 18, 80, 35, 1)
recency = st.sidebar.slider("Recency (Days)", 0, 180, 14, 1)
tx_count = st.sidebar.slider("Monthly Tx", 0, 100, 38, 1)
debt_status = st.sidebar.selectbox("Debt Status", ["No Active Loan", "Healthy Active Loan", "Delinquent / Default"])

st.sidebar.markdown("---")
st.sidebar.markdown("<h5 style='margin:0; color:#003366;'>📂 Operational Data Engine Sourcing</h5>", unsafe_allow_html=True)

# THE THREE-WAY SWITCHER: Router execution selection
data_source = st.sidebar.radio(
    "Select data engine input framework:",
    ["Synthetic Baseline Generator", "Live Cloud Web Streams", "Local Batch CSV Upload"]
)

# Variable allocation definitions
df_cust, df_tx, df_loans = None, None, None

if data_source == "Synthetic Baseline Generator":
    df_cust, df_tx, df_loans = generate_default_fallback_data()
    st.sidebar.success("🤖 Core: Synthetic baseline engines active.")

elif data_source == "Live Cloud Web Streams":
    try:
        df_cust, df_tx, df_loans = load_live_web_streams()
        st.sidebar.success("⚡ Cloud: Web data streams synchronized!")
    except Exception as e:
        st.sidebar.error(f"⚠️ Web Stream Unreachable. Using fallback engine data. Error: {e}")
        df_cust, df_tx, df_loans = generate_default_fallback_data()

elif data_source == "Local Batch CSV Upload":
    st.sidebar.markdown("##### Upload Unified CSV Ledger Sheets:")
    uploaded_cust = st.sidebar.file_uploader("Upload customer profiles CSV", type=["csv"], key="cust_up")
    uploaded_tx = st.sidebar.file_uploader("Upload transaction history CSV", type=["csv"], key="tx_up")
    uploaded_loans = st.sidebar.file_uploader("Upload loan records CSV", type=["csv"], key="loan_up")
    
    if uploaded_cust and uploaded_tx and uploaded_loans:
        try:
            df_cust = pd.read_csv(uploaded_cust)
            df_tx = pd.read_csv(uploaded_tx)
            df_loans = pd.read_csv(uploaded_loans)
            df_tx['Timestamp'] = pd.to_datetime(df_tx['Timestamp'])
            st.sidebar.success("📤 Success: Batch profiles active!")
        except Exception as e:
            st.sidebar.error("❌ Invalid CSV layout formatting structure.")
            df_cust, df_tx, df_loans = generate_default_fallback_data()
    else:
        st.sidebar.info("📥 Waiting for all 3 relational database CSV files to be dropped...")
        df_cust, df_tx, df_loans = generate_default_fallback_data()

# AUTOMATED TIER PROMOTION ALERTS
st.sidebar.markdown("---")
st.sidebar.markdown("<h5 style='margin:0; color:#003366;'>💎 Automated Business Promotion</h5>", unsafe_allow_html=True)
is_qualified = (credit_score >= 650) and (debt_status == "No Active Loan") and (tx_count >= 15)

if is_qualified:
    if credit_score >= 740:
        recommended_promotion, credit_increase_limit, badge_func = "🚀 **Gold ➔ Platinum Premium Migration**", "₹2,00,000 Limit Increase", st.sidebar.success
    elif credit_score >= 670:
        recommended_promotion, credit_increase_limit, badge_func = "🌟 **Silver ➔ Gold Growth Promotion**", "₹1,00,000 Limit Increase", st.sidebar.info
    else:
        recommended_promotion, credit_increase_limit, badge_func = "📈 **Silver Retention Baseline Promotion**", "₹50,000 Limit Increase", st.sidebar.info
    badge_func(f"{recommended_promotion}\n\n**Approved:** {credit_increase_limit}")
else:
    if debt_status == "Delinquent / Default" or credit_score < 580:
        st.sidebar.error("🛑 **Promotion Blocked:** High default risk or active delinquency detected.")
    else:
        st.sidebar.warning("⚠️ **Not Eligible:** Requires Monthly Tx ≥ 15 and Credit Score ≥ 650 to qualify.")

# =====================================================================
# RENDER GLOBAL METRICS CARD ROW
# =====================================================================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1: st.metric(label="Active Customers", value=format_indian_currency(len(df_cust), include_symbol=False))
with kpi2: st.metric(label="Audited Transactions", value=format_indian_currency(len(df_tx), include_symbol=False))
with kpi3:
    mean_amt = df_tx['Amount'].mean()
    std_amt = df_tx['Amount'].std()
    fraud_threshold = mean_amt + (3 * std_amt)
    anomalies_count = len(df_tx[df_tx['Amount'] > fraud_threshold])
    st.metric(label="Fraud Anomalies", value=format_indian_currency(anomalies_count, include_symbol=False), delta="Outliers >3 SD", delta_color="inverse")
with kpi4:
    total_exposure = df_loans['LoanAmount'].sum()
    st.metric(label="Capital Exposure", value=format_indian_currency(total_exposure))

st.markdown("<hr/>", unsafe_allow_html=True)

# =====================================================================
# RENDER VISUAL PLOTTING INTERACTION ROW CANVAS
# =====================================================================
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9.5})
col_left, col_right = st.columns(2)

with col_left:
    fig_a, ax_a = plt.subplots(figsize=(6, 1.9))
    feature_data = pd.DataFrame({
        'Feature': ['Recency', 'Tx_Count', 'CreditScore', 'Total_Spend', 'Income', 'Age', 'Avg_Tx_Value', 'Has_Active_Loan'],
        'Importance': [0.555690, 0.259590, 0.045125, 0.044271, 0.034610, 0.028467, 0.027646, 0.004600]
    }).sort_values(by='Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=feature_data, ax=ax_a, palette='Blues_d', hue='Feature', legend=False)
    ax_a.set_title("💡 Churn Predictive Feature Drivers", fontweight='bold', pad=4)
    ax_a.set_xlabel(""); ax_a.set_ylabel("")
    plt.tight_layout(pad=0.1)
    st.pyplot(fig_a, use_container_width=True)
    plt.close(fig_a)

    fig_c, ax_c = plt.subplots(figsize=(6, 1.9))
    df_tx['Z_Score'] = (df_tx['Amount'] - mean_amt) / std_amt
    df_tx['Status'] = np.where(df_tx['Z_Score'] > 3, 'Outlier', 'Normal')
    sns.scatterplot(x=df_tx.index, y='Amount', hue='Status', data=df_tx, palette={'Normal': '#cccccc', 'Outlier' : '#cc0000'}, ax=ax_c, alpha=0.4, s=6, edgecolor=None)
    ax_c.set_title("🚨 Transaction Auditing: Fraud Anomaly Plot", fontweight='bold', pad=4)
    ax_c.set_xlabel(""); ax_c.set_ylabel(""); ax_c.get_legend().remove()
    plt.tight_layout(pad=0.1)
    st.pyplot(fig_c, use_container_width=True)
    plt.close(fig_c)

with col_right:
    fig_b, ax_b = plt.subplots(figsize=(6, 1.9))
    df_loan_risk = pd.merge(df_loans, df_cust, on='CustomerID')
    def score_to_bucket(s): 
        return 'Poor (<580)' if s < 580 else ('Fair (580-66)' if s < 670 else ('Good' if s < 740 else 'Excellent'))
    df_loan_risk['Credit_Tier'] = df_loan_risk['CreditScore'].apply(score_to_bucket)
    tier_order = ['Poor (<580)', 'Fair (580-66)', 'Good', 'Excellent']
    loan_summary = df_loan_risk.groupby('Credit_Tier').agg(Total_Loans=('LoanID', 'count'), Defaults=('CurrentStatus', lambda x: (x == 'Defaulted').sum())).reindex(tier_order).fillna(0).reset_index()
    loan_summary['Default_Rate'] = (loan_summary['Defaults'] / loan_summary['Total_Loans']) * 100
    sns.barplot(x='Credit_Tier', y='Default_Rate', data=loan_summary, ax=ax_b, palette='Oranges_r', hue='Credit_Tier', legend=False)
    ax_b.set_title("📉 Asset Delinquency: Loan Default Rates", fontweight='bold', pad=4)
    ax_b.set_xlabel(""); ax_b.set_ylabel("")
    for p in ax_b.patches:
        ax_b.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.3), ha='center', va='center', xytext=(0, 2), textcoords='offset points', fontsize=7.5)
    plt.tight_layout(pad=0.1)
    st.pyplot(fig_b, use_container_width=True)
    plt.close(fig_b)

    fig_d, ax_d = plt.subplots(figsize=(6, 1.9))
    tier_spend = df_tx.merge(df_cust, on='CustomerID').groupby('AccountTier')['Amount'].sum().reset_index()
    ax_d.pie(tier_spend['Amount'], labels=tier_spend['AccountTier'], autopct='%1.1f%%', startangle=140, colors=['#cfd8dc', '#ffd54f', '#90caf9'], textprops={'fontsize': 7.5})
    ax_d.set_title("💎 Capital Contribution: Share per Card Tier", fontweight='bold', pad=4)
    plt.tight_layout(pad=0.1)
    st.pyplot(fig_d, use_container_width=True)
    plt.close(fig_d)

st.markdown("<hr/>", unsafe_allow_html=True)
# =====================================================================
# BLOCK 3: LIVE RISK TRACKER MONITOR & BATCH LEAD EXPORTER GENERATOR
# =====================================================================
# Re-calculate predictive risk probability score layers reactively
base_risk = 35.0
base_risk -= (credit_score - 300) * 0.06   
base_risk += recency * 0.42                
base_risk -= tx_count * 0.32               
if debt_status == "Delinquent / Default": base_risk += 28.0                      
elif debt_status == "Healthy Active Loan": base_risk -= 4.0                       
churn_probability = max(0.0, min(100.0, base_risk))

st.markdown("<h4 style='color:#003366; margin:0;'>🔮 Live Churn Probability Tracker Monitor Output</h4>", unsafe_allow_html=True)

# 1. Compile flat low-profile horizontal bar gauge
fig_sim, ax_sim = plt.subplots(figsize=(10, 0.22))

if churn_probability < 30.0:
    bar_color, status_text, alert_func = '#2ecc71', f"🟢 HEALTHY PROFILE ({churn_probability:.1f}%) — Standard background monitoring applies.", st.success
elif churn_probability < 70.0:
    bar_color, status_text, alert_func = '#f1c40f', f"🟡 WATCHLIST PROFILE ({churn_probability:.1f}%) — Initiate customer retention loops.", st.warning
else:
    bar_color, status_text, alert_func = '#e74c3c', f"🔴 CRITICAL PROFILE ({churn_probability:.1f}%) — Trigger immediate outreach protocols.", st.error

ax_sim.barh(["Risk"], [churn_probability], color=bar_color, height=0.6, edgecolor='none')
ax_sim.set_xlim(0, 100)
ax_sim.xaxis.set_visible(False)
ax_sim.yaxis.set_visible(False)
ax_sim.text(churn_probability + 1.0, 0, f"{churn_probability:.1f}%", va='center', ha='left', fontweight='bold', color='#333333', fontsize=9)

sns.despine(left=True, bottom=True, right=True, top=True)
plt.tight_layout(pad=0)

# Render tracking visual elements into responsive viewport layout grids
st.pyplot(fig_sim, use_container_width=True)
plt.close(fig_sim)
alert_func(status_text)

# 2. 📞 HIGH-VALUE TARGET CALLING GENERATOR PIPELINE
st.markdown("---")
st.markdown("<h4 style='color:#003366;'>📞 Premium Up-Sell Cohort Batch Target Calling List Generator</h4>", unsafe_allow_html=True)
st.markdown("Generates actionable lead records matching target criteria from the active database tables.")

# Process behavioral parameter matrices from streaming files
tx_summary = df_tx.groupby('CustomerID').agg(
    Monthly_Transactions=('TransactionID', 'count'), 
    Total_Spend_Volume=('Amount', 'sum')
).reset_index()

df_leads = pd.merge(df_cust, tx_summary, on='CustomerID', how='inner')
active_borrowers = df_loans['CustomerID'].unique()

# Extract records based on relaxed density parameters to keep data populated (Prevents 0 records bug)
df_high_value_targets = df_leads[
    (df_leads['CreditScore'] >= 650) & 
    (df_leads['Monthly_Transactions'] >= 15) & 
    (~df_leads['CustomerID'].isin(active_borrowers))
].copy()

# Append Indian localized CRM processing variables using string accessor (Fixed Attribute Crash)
np.random.seed(42)
df_high_value_targets['Priority_Score'] = np.random.randint(85, 100, size=len(df_high_value_targets))
df_high_value_targets['Corporate_Email'] = df_high_value_targets['CustomerID'].str.lower() + "@careerdreambank.in"
df_high_value_targets['Campaign_Status'] = "Ready to Call"

df_final_calling_sheet = df_high_value_targets[[
    'CustomerID', 'Age', 'CreditScore', 'AccountTier', 
    'Monthly_Transactions', 'Corporate_Email', 'Priority_Score', 'Campaign_Status'
]].sort_values(by='Priority_Score', ascending=False)

# Compile download actions row entry configurations
call_col1, call_col2 = st.columns(2)
with call_col1:
    formatted_lead_count = format_indian_currency(len(df_final_calling_sheet), include_symbol=False)
    st.markdown(f"🎯 **System Found:** `{formatted_lead_count}` Customer records matching the premium high-engagement, zero-debt profile criteria.")
with call_col2:
    csv_bytes = df_final_calling_sheet.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Lead List (CSV)", 
        data=csv_bytes, 
        file_name=f"banking_upsell_call_list_{datetime.now().strftime('%Y%m%d')}.csv", 
        mime="text/csv", 
        use_container_width=True
    )

# Render responsive interactive dataframe window sheet
st.dataframe(df_final_calling_sheet, use_container_width=True, height=160)
