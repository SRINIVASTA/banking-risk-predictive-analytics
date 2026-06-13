# =====================================================================
# ANALYTICS ENGINE MODULE
# =====================================================================
# Handles all chart generation and KPI calculations

import streamlit as st
import pandas as pd
import plotly.express as px
import config


# =====================================================================
# CHART STYLING HELPER
# =====================================================================

def create_styled_chart(fig, height=config.CHART_HEIGHT):
    """Apply consistent styling to charts."""
    fig.update_layout(
        height=height,
        margin=config.CHART_MARGIN,
        coloraxis_showscale=False,
        font=dict(size=11)
    )
    return fig


# =====================================================================
# KPI CALCULATIONS
# =====================================================================

def get_customer_kpi(df_cust):
    """Get total active customers KPI."""
    return len(df_cust)


def get_transaction_kpi(df_tx):
    """Get total audited transactions KPI."""
    return len(df_tx)


def get_fraud_kpi(anomalies_count):
    """Get flagged fraud anomalies KPI."""
    return anomalies_count


def get_capital_exposure_kpi(df_loans):
    """Get active capital exposure KPI."""
    return df_loans['LoanAmount'].sum()


# =====================================================================
# CHART GENERATION FUNCTIONS
# =====================================================================

def create_feature_importance_chart():
    """
    Create feature importance bar chart for churn prediction.
    
    Returns:
        plotly.graph_objects.Figure: Feature importance chart
    """
    feature_data = pd.DataFrame(config.FEATURE_IMPORTANCE, columns=['Feature', 'Importance'])
    feature_data = feature_data.sort_values(by='Importance', ascending=True)
    
    fig = px.bar(
        feature_data, x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale='Blues',
        labels={'Importance': 'Predictive Weight Value', 'Feature': ''}
    )
    
    fig = create_styled_chart(fig)
    return fig


def create_loan_default_chart(loan_summary):
    """
    Create loan default rates bar chart by credit tier.
    
    Args:
        loan_summary (pd.DataFrame): Loan summary data with default rates
        
    Returns:
        plotly.graph_objects.Figure: Loan default rates chart
    """
    fig = px.bar(
        loan_summary, x='Credit_Tier', y='Default_Rate',
        color='Default_Rate', color_continuous_scale='Oranges',
        text=loan_summary['Default_Rate'].apply(lambda x: f"{x:.1f}%"),
        labels={'Credit_Tier': 'Customer Credit Risk Group', 'Default_Rate': 'Default Rate (%)'}
    )
    
    fig = create_styled_chart(fig)
    fig.update_traces(textposition='outside', textfont_size=12)
    return fig


def create_fraud_scatter_chart(df_tx, fraud_limit):
    """
    Create fraud anomaly detection scatter plot.
    
    Args:
        df_tx (pd.DataFrame): Transaction data with fraud flags
        fraud_limit (float): Z-score fraud threshold value
        
    Returns:
        plotly.graph_objects.Figure: Fraud scatter plot
    """
    fig = px.scatter(
        df_tx.reset_index(), x='index', y='Amount', color='Status',
        color_discrete_map=config.FRAUD_COLORS,
        labels={'index': 'Sequential Transaction Reference ID', 'Amount': 'Volume Magnitude ($)'}
    )
    
    fig.add_hline(
        y=fraud_limit, line_dash="dash", line_color=config.FRAUD_THRESHOLD_COLOR, 
        line_width=2, annotation_text="Z-Score Threshold"
    )
    
    fig = create_styled_chart(fig)
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig


def create_tier_spending_chart(tier_spend):
    """
    Create account tier spending distribution pie chart.
    
    Args:
        tier_spend (pd.DataFrame): Spending data by account tier
        
    Returns:
        plotly.graph_objects.Figure: Tier spending pie chart
    """
    fig = px.pie(
        tier_spend, values='Amount', names='AccountTier',
        color='AccountTier', color_discrete_map=config.TIER_COLORS,
        hole=0.3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=13)
    fig = create_styled_chart(fig)
    fig.update_layout(showlegend=False)
    return fig


# =====================================================================
# CHURN SIMULATION FUNCTIONS
# =====================================================================

def calculate_churn_probability(recency, score, has_loan):
    """
    Calculate customer churn probability based on input factors.
    
    Args:
        recency (int): Days since last transaction (0-180)
        score (int): Customer credit score (300-850)
        has_loan (int): 1 if has active loan, 0 otherwise
        
    Returns:
        float: Churn probability (0.0-1.0)
    """
    churn_prob = (
        (recency / config.CHURN_RECENCY_MAX) * config.CHURN_WEIGHTS['recency_weight'] + 
        (1 - (score / config.CHURN_SCORE_MAX)) * config.CHURN_WEIGHTS['score_weight'] - 
        (has_loan * config.CHURN_WEIGHTS['loan_discount'])
    )
    
    # Clamp probability between 0 and 1
    return max(0.0, min(1.0, churn_prob))


def get_churn_alert_message(churn_prob):
    """
    Generate alert message based on churn probability.
    
    Args:
        churn_prob (float): Calculated churn probability
        
    Returns:
        tuple: (alert_type, message) where alert_type is 'error' or 'success'
    """
    critical_threshold = config.CHURN_WEIGHTS['critical_threshold']
    
    if churn_prob > critical_threshold:
        return (
            'error',
            f"🔴 CRITICAL ALERT: HIGH RISK OF CUSTOMER CHURN ({churn_prob * 100:.1f}%) — "
            f"Dispatch immediate retention vouchers."
        )
    else:
        return (
            'success',
            f"🟢 HEALTHY ACCOUNT FRAME: ACTIVE PROFILE ({churn_prob * 100:.1f}%) — "
            f"Standard background monitoring metrics hold safe."
        )
