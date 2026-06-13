# =====================================================================
# DATA PROCESSING MODULE
# =====================================================================
# Handles data loading, generation, validation, and transformations

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config


# =====================================================================
# DATA GENERATION FUNCTIONS
# =====================================================================

@st.cache_data
def generate_customer_profiles():
    """Generate synthetic customer profile data."""
    customer_ids = [f"CUST_{i:04d}" for i in range(1, config.NUM_CUSTOMERS + 1)]
    df_cust = pd.DataFrame({
        'CustomerID': customer_ids,
        'Age': np.random.randint(*config.AGE_RANGE, size=config.NUM_CUSTOMERS),
        'Income': np.random.randint(*config.INCOME_RANGE, size=config.NUM_CUSTOMERS),
        'CreditScore': np.random.randint(*config.CREDIT_SCORE_RANGE, size=config.NUM_CUSTOMERS),
        'AccountTier': np.random.choice(
            config.ACCOUNT_TIERS, 
            size=config.NUM_CUSTOMERS, 
            p=config.ACCOUNT_TIER_DISTRIBUTION
        )
    })
    return df_cust, customer_ids


@st.cache_data
def generate_transactions(customer_ids):
    """Generate synthetic transaction data with fraud spikes."""
    # Generate base transaction amounts with exponential distribution
    tx_amounts = np.round(
        np.random.exponential(scale=config.FRAUD_SPIKE_MEAN, size=config.NUM_TRANSACTIONS) + config.FRAUD_SPIKE_MIN, 
        2
    )
    
    # Inject fraud spikes
    fraud_indices = np.random.choice(
        range(config.NUM_TRANSACTIONS), 
        size=int(config.NUM_TRANSACTIONS * config.FRAUD_SPIKE_PERCENTAGE), 
        replace=False
    )
    tx_amounts[fraud_indices] = np.round(
        np.random.uniform(*config.FRAUD_SPIKE_RANGE, size=len(fraud_indices)), 
        2
    )
    
    # Generate timestamps
    timestamps = [
        datetime(2026, 1, 1) + timedelta(days=int(np.random.randint(0, config.DATE_RANGE_DAYS)))
        for _ in range(config.NUM_TRANSACTIONS)
    ]
    
    df_tx = pd.DataFrame({
        'TransactionID': [f"TX_{i:06d}" for i in range(1, config.NUM_TRANSACTIONS + 1)],
        'CustomerID': np.random.choice(customer_ids, size=config.NUM_TRANSACTIONS),
        'Timestamp': pd.to_datetime(timestamps),
        'Amount': tx_amounts,
        'Category': np.random.choice(config.TRANSACTION_CATEGORIES, size=config.NUM_TRANSACTIONS),
        'Channel': np.random.choice(config.TRANSACTION_CHANNELS, size=config.NUM_TRANSACTIONS),
        'IsFlaggedFraud': [1 if i in fraud_indices else 0 for i in range(config.NUM_TRANSACTIONS)]
    })
    
    return df_tx


def get_default_probability(credit_score):
    """Return probability distribution for loan statuses based on credit score."""
    if credit_score < 580:
        return config.DEFAULT_PROBABILITIES['poor']
    elif credit_score < 670:
        return config.DEFAULT_PROBABILITIES['fair']
    else:
        return config.DEFAULT_PROBABILITIES['good']


@st.cache_data
def generate_loan_records(customer_ids, df_cust):
    """Generate loan records with credit-score-based default probabilities."""
    loan_cust_ids = np.random.choice(customer_ids, size=config.NUM_LOANS, replace=False)
    loan_status = []
    
    for cid in loan_cust_ids:
        score = df_cust.loc[df_cust['CustomerID'] == cid, 'CreditScore'].values[0]
        p_dist = get_default_probability(score)
        loan_status.append(np.random.choice(config.LOAN_STATUSES, p=p_dist))
    
    df_loans = pd.DataFrame({
        'LoanID': [f"LN_{i:04d}" for i in range(1, config.NUM_LOANS + 1)],
        'CustomerID': loan_cust_ids,
        'LoanAmount': np.round(np.random.randint(*config.LOAN_AMOUNT_RANGE, size=config.NUM_LOANS), -2),
        'InterestRate': np.round(np.random.uniform(*config.INTEREST_RATE_RANGE, size=config.NUM_LOANS), 2),
        'TermMonths': np.random.choice(config.LOAN_TERMS, size=config.NUM_LOANS),
        'CurrentStatus': loan_status
    })
    
    return df_loans


@st.cache_data
def generate_synthetic_data():
    """Generate all synthetic banking data."""
    np.random.seed(config.RANDOM_SEED)
    
    df_cust, customer_ids = generate_customer_profiles()
    df_tx = generate_transactions(customer_ids)
    df_loans = generate_loan_records(customer_ids, df_cust)
    
    return df_cust, df_tx, df_loans


# =====================================================================
# DATA LOADING AND VALIDATION
# =====================================================================

def validate_dataframes(df_cust, df_tx, df_loans):
    """Validate that dataframes are not empty and have required columns."""
    if df_cust is None or df_tx is None or df_loans is None:
        return False
    
    if df_cust.empty or df_tx.empty or df_loans.empty:
        return False
    
    required_cust_cols = {'CustomerID', 'Age', 'Income', 'CreditScore', 'AccountTier'}
    required_tx_cols = {'TransactionID', 'CustomerID', 'Timestamp', 'Amount', 'Category', 'Channel'}
    required_loan_cols = {'LoanID', 'CustomerID', 'LoanAmount', 'InterestRate', 'CurrentStatus'}
    
    return (
        required_cust_cols.issubset(df_cust.columns) and
        required_tx_cols.issubset(df_tx.columns) and
        required_loan_cols.issubset(df_loans.columns)
    )


@st.cache_data
def load_and_process_banking_data():
    """Load banking data from CSV files or generate synthetic data if missing."""
    try:
        df_cust = pd.read_csv('customer_profiles.csv')
        df_tx = pd.read_csv('transaction_history.csv')
        df_loans = pd.read_csv('loan_records.csv')
        df_tx['Timestamp'] = pd.to_datetime(df_tx['Timestamp'])
        
        if not validate_dataframes(df_cust, df_tx, df_loans):
            st.warning("Data validation failed. Generating synthetic data...")
            df_cust, df_tx, df_loans = generate_synthetic_data()
            # Save generated data
            save_data_to_csv(df_cust, df_tx, df_loans)
        
        return df_cust, df_tx, df_loans
        
    except FileNotFoundError:
        st.info("Data files not found. Generating synthetic data...")
        df_cust, df_tx, df_loans = generate_synthetic_data()
        save_data_to_csv(df_cust, df_tx, df_loans)
        return df_cust, df_tx, df_loans
        
    except Exception as e:
        st.error(f"Error loading data: {e}. Generating synthetic data...")
        df_cust, df_tx, df_loans = generate_synthetic_data()
        save_data_to_csv(df_cust, df_tx, df_loans)
        return df_cust, df_tx, df_loans


def save_data_to_csv(df_cust, df_tx, df_loans):
    """Save dataframes to CSV files for persistence."""
    try:
        df_cust.to_csv('customer_profiles.csv', index=False)
        df_tx.to_csv('transaction_history.csv', index=False)
        df_loans.to_csv('loan_records.csv', index=False)
    except Exception as e:
        st.warning(f"Could not save data to CSV: {e}")


# =====================================================================
# DATA TRANSFORMATION FUNCTIONS
# =====================================================================

@st.cache_data
def calculate_fraud_metrics(df_tx):
    """Calculate fraud detection metrics and add flags to dataframe."""
    mean_amt = df_tx['Amount'].mean()
    std_amt = df_tx['Amount'].std()
    fraud_limit = mean_amt + (config.FRAUD_Z_THRESHOLD * std_amt)
    
    df_tx = df_tx.copy()
    df_tx['Z_Score'] = (df_tx['Amount'] - mean_amt) / std_amt
    df_tx['Status'] = np.where(
        df_tx['Z_Score'] > config.FRAUD_Z_THRESHOLD, 
        'Flagged Outlier (>3 SD)', 
        'Normal Core Process'
    )
    
    anomalies_count = len(df_tx[df_tx['Amount'] > fraud_limit])
    
    return df_tx, mean_amt, std_amt, fraud_limit, anomalies_count


def score_to_credit_tier(score):
    """Convert credit score to tier label."""
    for tier, (low, high) in config.CREDIT_BUCKETS.items():
        if low <= score < high:
            return f"{tier} ({low}-{high-1})"
    return "Excellent (740+)"


def prepare_loan_risk_analysis(df_loans, df_cust):
    """Prepare loan risk data grouped by credit tier."""
    df_loan_risk = pd.merge(df_loans, df_cust, on='CustomerID')
    df_loan_risk['Credit_Tier'] = df_loan_risk['CreditScore'].apply(score_to_credit_tier)
    
    tier_order = [f"{tier} ({low}-{high-1})" for tier, (low, high) in config.CREDIT_BUCKETS.items()]
    
    loan_summary = df_loan_risk.groupby('Credit_Tier').agg(
        Total_Loans=('LoanID', 'count'),
        Defaults=('CurrentStatus', lambda x: (x == 'Defaulted').sum())
    ).reindex(tier_order).fillna(0).reset_index()
    
    loan_summary['Default_Rate'] = (loan_summary['Defaults'] / loan_summary['Total_Loans']) * 100
    
    return loan_summary


def prepare_tier_spend_analysis(df_tx, df_cust):
    """Prepare transaction volume by account tier."""
    tier_spend = df_tx.merge(df_cust, on='CustomerID').groupby('AccountTier')['Amount'].sum().reset_index()
    return tier_spend
