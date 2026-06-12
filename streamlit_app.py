import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =====================================================================
# 1. STREAMLIT GLOBAL CANVAS CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="CareerDream Banking Analytics Platform", 
    page_icon="🚀", 
    layout="wide"  # Enforces a full widescreen desktop monitor layout
)

st.title("🚀 CareerDream.in — Executive Banking Performance Analytics Platform")
st.markdown("---")

# =====================================================================
# 2. DATA PROCESSING ENGINE (Self-Healing Memory Load)
# =====================================================================
@st.cache_data
def load_and_process_banking_data():
    try:
        df_cust = pd.read_csv('customer_profiles.csv')
        df_tx = pd.read_csv('transaction_history.csv')
        df_loans = pd.read_csv('loan_records.csv')
        df_tx['Timestamp'] = pd.to_datetime(df_tx['Timestamp'])
        return df_cust, df_tx, df_loans
    except FileNotFoundError:
        st.error("❌ Operational Error: Base CSV data assets missing. Run your generator script first.")
        st.stop()

df_cust, df_tx, df_loans = load_and_process_banking_data()

# =====================================================================
# 3. EXECUTIVE KPI CARD HEADER row
# =====================================================================
# Highlights crucial high-level platform health metrics at a glance
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(label="Total Active Customers", value=f"{len(df_cust)}")
with kpi2:
    st.metric(label="Total Audited Transactions", value=f"{len(df_tx)}")
with kpi3:
    mean_amt = df_tx['Amount'].mean()
    std_amt = df_tx['Amount'].std()
    fraud_limit = mean_amt + (3 * std_amt)
    anomalies_count = len(df_tx[df_tx['Amount'] > fraud_limit])
    st.metric(label="Flagged Fraud Anomalies", value=anomalies_count, delta="Outlier Threshold >3 SD", delta_color="inverse")
with kpi4:
    total_exposure = df_loans['LoanAmount'].sum()
    st.metric(label="Active Capital Exposure", value=f"${total_exposure:,.0f}")

st.markdown("---")

# =====================================================================
# 4. DESKTOP GRID LAYOUT CANVAS (2x2 Matrix Structure)
# =====================================================================
# Split the web screen into two balanced column halves
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# ---------------------------------------------------------------------
# PANEL A (Top Left): ML FEATURE IMPORTANCE DRIVERS
# ---------------------------------------------------------------------
with row1_col1:
    st.subheader("💡 Churn Predictive Feature Drivers")
    feature_data = pd.DataFrame({
        'Feature': ['Recency', 'Tx_Count', 'CreditScore', 'Total_Spend', 'Income', 'Age', 'Avg_Tx_Value', 'Has_Active_Loan'],
        'Importance': [0.555690, 0.259590, 0.045125, 0.044271, 0.034610, 0.028467, 0.027646, 0.004600]
    }).sort_values(by='Importance', ascending=True) # Ascending for clean bottom-up horizontal plots
    
    fig_a = px.bar(
        feature_data, x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale='Blues',
        labels={'Importance': 'Predictive Weight Value', 'Feature': ''}
    )
    fig_a.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig_a, use_container_width=True)

# ---------------------------------------------------------------------
# PANEL B (Top Right): LOAN PORTFOLIO DEFAULT ANALYSIS
# ---------------------------------------------------------------------
with row1_col2:
    st.subheader("📉 Asset Delinquency: Loan Default Rates")
    df_loan_risk = pd.merge(df_loans, df_cust, on='CustomerID')
    
    def score_to_bucket(score):
        if score < 580: return 'Poor (<580)'
        elif score < 670: return 'Fair (580-669)'
        elif score < 740: return 'Good (670-739)'
        else: return 'Excellent (740+)'
        
    df_loan_risk['Credit_Tier'] = df_loan_risk['CreditScore'].apply(score_to_bucket)
    tier_order = ['Poor (<580)', 'Fair (580-669)', 'Good (670-739)', 'Excellent (740+)']
    
    loan_summary = df_loan_risk.groupby('Credit_Tier').agg(
        Total_Loans=('LoanID', 'count'),
        Defaults=('CurrentStatus', lambda x: (x == 'Defaulted').sum())
    ).reindex(tier_order).fillna(0).reset_index()
    loan_summary['Default_Rate'] = (loan_summary['Defaults'] / loan_summary['Total_Loans']) * 100
    
    fig_b = px.bar(
        loan_summary, x='Credit_Tier', y='Default_Rate',
        color='Default_Rate', color_continuous_scale='Oranges',
        text=loan_summary['Default_Rate'].apply(lambda x: f"{x:.1f}%"),
        labels={'Credit_Tier': 'Customer Credit Risk Group', 'Default_Rate': 'Default Rate (%)'}
    )
    fig_b.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    fig_b.update_traces(textposition='outside', textfont_size=12, textfont_series='bold')
    st.plotly_chart(fig_b, use_container_width=True)

# ---------------------------------------------------------------------
# PANEL C (Bottom Left): FRAUD EXPOSURE OUTLIER DISTRIBUTION
# ---------------------------------------------------------------------
with row2_col1:
    st.subheader("🚨 Transaction Auditing: Fraud Anomaly Plot")
    df_tx['Z_Score'] = (df_tx['Amount'] - mean_amt) / std_amt
    df_tx['Status'] = np.where(df_tx['Z_Score'] > 3, 'Flagged Outlier (>3 SD)', 'Normal Core Process')
    
    fig_c = px.scatter(
        df_tx.reset_index(), x='index', y='Amount', color='Status',
        color_discrete_map={'Normal Core Process': '#cccccc', 'Flagged Outlier (>3 SD)': '#cc0000'},
        opacity=[0.4 if s == 'Normal Core Process' else 0.9 for s in df_tx['Status']],
        labels={'index': 'Sequential Transaction Reference ID', 'Amount': 'Volume Magnitude ($)'}
    )
    # Add the static dashed horizontal Z-score indicator ceiling threshold
    fig_c.add_hline(y=fraud_limit, line_dash="dash", line_color="#cc0000", line_width=2, annotation_text="Z-Score Limit Limit")
    fig_c.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig_c, use_container_width=True)

# ---------------------------------------------------------------------
# PANEL D (Bottom Right): ACCOUNT TIER CORE VOLUME SEGMENTATION
# ---------------------------------------------------------------------
with row2_col2:
    st.subheader("💎 Capital Contribution: Share per Card Tier")
    tier_spend = df_tx.merge(df_cust, on='CustomerID').groupby('AccountTier')['Amount'].sum().reset_index()
    
    fig_d = px.pie(
        tier_spend, values='Amount', names='AccountTier',
        color='AccountTier', color_discrete_map={'Silver': '#90caf9', 'Gold': '#ffd54f', 'Platinum': '#cfd8dc'},
        hole=0.3 # Generates a clean, professional donut plot style
    )
    fig_d.update_traces(textposition='inside', textinfo='percent+label', textfont_size=13, textfont_series='bold')
    fig_d.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig_d, use_container_width=True)

# =====================================================================
# 5. RETENTION SIMULATOR CONTROL PANEL (Widescreen Collapsible Drawer)
# =====================================================================
st.markdown("---")
with st.expander("🔮 Open Interactive Real-Time Customer Churn Risk Simulator"):
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        score_input = st.slider("Customer Credit Score Metric", 300, 850, 680)
        age_input = st.slider("Customer Age", 18, 80, 42)
    with sim_col2:
        rec_input = st.slider("Days Since Last Transaction (Recency)", 0, 180, 14)
        tx_input = st.slider("Monthly Transaction Counts", 0, 100, 22)
    with sim_col3:
        loan_input = st.selectbox("Active Account Debt Liability Status", ["No Active Loan", "Has Active Bank Loan"])
        loan_numeric = 1 if loan_input == "Has Active Bank Loan" else 0
        
    churn_prob = (rec_input / 180) * 0.75 + (1 - (score_input / 850)) * 0.25 - (loan_numeric * 0.15)
    churn_prob = max(0.0, min(1.0, churn_prob))
    
    if churn_prob > 0.55:
        st.error(f"🔴 CRITICAL ALERT: HIGH RISK OF CUSTOMER CHURN ({churn_prob*100:.1f}%) — Dispatch immediate retention vouchers.")
    else:
        st.success(f"🟢 HEALTHY ACCOUNT FRAME: ACTIVE PROFILE ({churn_prob*100:.1f}%) — Standard background monitoring metrics hold safe.")
