# =====================================================================
# BANKING ANALYTICS PLATFORM - MAIN APPLICATION
# =====================================================================
# UI-focused Streamlit application using modular components

import streamlit as st
import config
from data_processor import (
    load_and_process_banking_data,
    calculate_fraud_metrics,
    prepare_loan_risk_analysis,
    prepare_tier_spend_analysis
)
from analytics_engine import (
    get_customer_kpi,
    get_transaction_kpi,
    get_fraud_kpi,
    get_capital_exposure_kpi,
    create_feature_importance_chart,
    create_loan_default_chart,
    create_fraud_scatter_chart,
    create_tier_spending_chart,
    calculate_churn_probability,
    get_churn_alert_message
)


# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT
)

st.title("🚀 Executive Banking Performance Analytics Platform")
st.markdown("---")


# =====================================================================
# DATA LOADING
# =====================================================================
df_cust, df_tx, df_loans = load_and_process_banking_data()
df_tx, mean_amt, std_amt, fraud_limit, anomalies_count = calculate_fraud_metrics(df_tx)


# =====================================================================
# KPI METRICS SECTION
# =====================================================================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="Total Active Customers", 
        value=f"{get_customer_kpi(df_cust)}"
    )

with kpi2:
    st.metric(
        label="Total Audited Transactions", 
        value=f"{get_transaction_kpi(df_tx)}"
    )

with kpi3:
    st.metric(
        label="Flagged Fraud Anomalies", 
        value=get_fraud_kpi(anomalies_count),
        delta="Outlier Threshold >3 SD", 
        delta_color="inverse"
    )

with kpi4:
    st.metric(
        label="Active Capital Exposure", 
        value=f"${get_capital_exposure_kpi(df_loans):,.0f}"
    )

st.markdown("---")


# =====================================================================
# ANALYTICS DASHBOARD (2x2 GRID)
# =====================================================================
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# PANEL A: Feature Importance
with row1_col1:
    st.subheader("💡 Churn Predictive Feature Drivers")
    fig_a = create_feature_importance_chart()
    st.plotly_chart(fig_a, use_container_width=True)

# PANEL B: Loan Default Analysis
with row1_col2:
    st.subheader("📉 Asset Delinquency: Loan Default Rates")
    loan_summary = prepare_loan_risk_analysis(df_loans, df_cust)
    fig_b = create_loan_default_chart(loan_summary)
    st.plotly_chart(fig_b, use_container_width=True)

# PANEL C: Fraud Detection
with row2_col1:
    st.subheader("🚨 Transaction Auditing: Fraud Anomaly Plot")
    fig_c = create_fraud_scatter_chart(df_tx, fraud_limit)
    st.plotly_chart(fig_c, use_container_width=True)

# PANEL D: Tier Spending
with row2_col2:
    st.subheader("💎 Capital Contribution: Share per Card Tier")
    tier_spend = prepare_tier_spend_analysis(df_tx, df_cust)
    fig_d = create_tier_spending_chart(tier_spend)
    st.plotly_chart(fig_d, use_container_width=True)


# =====================================================================
# DATA MANAGEMENT CONTROLS
# =====================================================================
st.markdown("---")
col_refresh, col_space = st.columns([1, 9])

with col_refresh:
    if st.button("🔄 Refresh Data", key="refresh_button"):
        st.cache_data.clear()
        st.rerun()


# =====================================================================
# CUSTOMER CHURN RISK SIMULATOR
# =====================================================================
with st.expander("🔮 Open Interactive Real-Time Customer Churn Risk Simulator"):
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    with sim_col1:
        score_input = st.slider(
            "Customer Credit Score Metric", 
            300, 850, 680,
            help="Range: 300 (poorest) to 850 (excellent)"
        )
        age_input = st.slider(
            "Customer Age", 
            18, 80, 42,
            help="Customer's current age in years"
        )
    
    with sim_col2:
        rec_input = st.slider(
            "Days Since Last Transaction (Recency)", 
            0, 180, 14,
            help="More recent transactions = lower churn risk"
        )
        tx_input = st.slider(
            "Monthly Transaction Counts", 
            0, 100, 22,
            help="Higher transaction frequency = more engaged customer"
        )
    
    with sim_col3:
        loan_input = st.selectbox(
            "Active Account Debt Liability Status",
            ["No Active Loan", "Has Active Bank Loan"],
            help="Active loans indicate stronger account commitment"
        )
        has_loan = 1 if loan_input == "Has Active Bank Loan" else 0
    
    # Calculate and display churn probability
    churn_prob = calculate_churn_probability(rec_input, score_input, has_loan)
    alert_type, message = get_churn_alert_message(churn_prob)
    
    if alert_type == 'error':
        st.error(message)
    else:
        st.success(message)
    
    # Display detailed metrics
    st.markdown("#### Churn Risk Factors Breakdown")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        recency_factor = (rec_input / 180) * config.CHURN_WEIGHTS['recency_weight']
        st.metric(
            "Recency Factor", 
            f"{recency_factor:.3f}",
            help="Impact from days since last transaction"
        )
    
    with metrics_col2:
        score_factor = (1 - (score_input / 850)) * config.CHURN_WEIGHTS['score_weight']
        st.metric(
            "Score Factor", 
            f"{score_factor:.3f}",
            help="Impact from credit score (inverse)"
        )
    
    with metrics_col3:
        loan_factor = has_loan * config.CHURN_WEIGHTS['loan_discount']
        st.metric(
            "Loan Discount", 
            f"-{loan_factor:.3f}",
            help="Reduction in risk from active loan"
        )


# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "Banking Risk Predictive Analytics Platform | "
    "Data refreshes on app startup or via Refresh button"
    "</div>",
    unsafe_allow_html=True
)
