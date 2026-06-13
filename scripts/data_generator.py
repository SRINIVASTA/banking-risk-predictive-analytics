import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_relational_banking_data():
    # Force create data directory
    os.makedirs('data', exist_ok=True)
    
    np.random.seed(42)
    NUM_CUSTOMERS, NUM_TRANSACTIONS, NUM_LOANS = 200, 1500, 80

    # 1. Customer Profiles
    customer_ids = [f"CUST_{i:04d}" for i in range(1, NUM_CUSTOMERS + 1)]
    df_cust = pd.DataFrame({
        'CustomerID': customer_ids,
        'Age': np.random.randint(18, 75, size=NUM_CUSTOMERS),
        'Income': np.random.randint(20000, 160000, size=NUM_CUSTOMERS),
        'CreditScore': np.random.randint(450, 850, size=NUM_CUSTOMERS),
        'AccountTier': np.random.choice(['Silver', 'Gold', 'Platinum'], size=NUM_CUSTOMERS, p=[0.5, 0.3, 0.2])
    })

    # 2. Transaction History
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

    # 3. Loan Records
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

    # Save to CSV files
    df_cust.to_csv('data/customer_profiles.csv', index=False)
    df_tx.to_csv('data/transaction_history.csv', index=False)
    df_loans.to_csv('data/loan_records.csv', index=False)
    print("Files created successfully.")

if __name__ == "__main__":
    generate_relational_banking_data()
