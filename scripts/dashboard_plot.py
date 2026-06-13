import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("📊 Compiling Executive Visual Dashboard Grid Layout scaled for laptop screens...")
sns.set_theme(style="whitegrid")

# Optimized layout configurations for responsive laptop monitor dimensions
plt.rcParams.update({
    'font.family': 'sans-serif', 
    'axes.edgecolor': '#cccccc',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11
})

# Corrected paths pointing directly to the workspace data folder
df_cust = pd.read_csv('data/customer_profiles.csv')
df_tx = pd.read_csv('data/transaction_history.csv')
df_loans = pd.read_csv('data/loan_records.csv')

# Fixed dimension adjustments tailored for 16:9/16:10 laptop screens
fig, axes = plt.subplots(2, 2, figsize=(14, 7.5))
fig.suptitle('CareerDream.in - Executive Banking Performance Analytics Platform', fontsize=15, fontweight='bold', color='#003366', y=0.97)

# Panel A: Feature Importance Drivers
features = ['Recency', 'Tx_Count', 'CreditScore', 'Total_Spend', 'Income', 'Age', 'Avg_Tx_Value', 'Has_Active_Loan']
importances = [0.555690, 0.259590, 0.045125, 0.044271, 0.034610, 0.028467, 0.027646, 0.004600]
feature_data = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_data, ax=axes[0, 0], palette='Blues_r', hue='Feature', legend=False)
axes[0, 0].set_title('Churn Predictive Feature Drivers (Random Forest Weight)', fontweight='bold')
axes[0, 0].set_xlabel('Weight Value')
axes[0, 0].set_ylabel('')

# Panel B: Loan Defaults Per Tier
df_loan_risk = pd.merge(df_loans, df_cust, on='CustomerID')
def score_to_bucket(s): return 'Poor (<580)' if s<580 else ('Fair (580-669)' if s<670 else ('Good (670-739)' if s<740 else 'Excellent (740+)'))
df_loan_risk['Credit_Tier'] = df_loan_risk['CreditScore'].apply(score_to_bucket)
loan_summary = df_loan_risk.groupby('Credit_Tier').agg(Total=('LoanID', 'count'), Defaults=('CurrentStatus', lambda x: (x == 'Defaulted').sum())).reindex(['Poor (<580)', 'Fair (580-669)', 'Good (670-739)', 'Excellent (740+)']).fillna(0).reset_index()
loan_summary['Rate'] = (loan_summary['Defaults'] / loan_summary['Total']) * 100
sns.barplot(x='Credit_Tier', y='Rate', data=loan_summary, ax=axes[0, 1], palette='Oranges_r', hue='Credit_Tier', legend=False)
axes[0, 1].set_title('Asset Delinquency: Loan Default Rates per Credit Tier', fontweight='bold')
axes[0, 1].set_ylabel('Default Rate (%)')
axes[0, 1].set_xlabel('')
for i, r in loan_summary.iterrows(): 
    axes[0, 1].text(i, r['Rate'] + 0.5, f"{r['Rate']:.1f}%", ha="center", fontsize=9, fontweight='bold')

# Panel C: Fraud Outliers Plot
mean_amt, std_amt = df_tx['Amount'].mean(), df_tx['Amount'].std()
df_tx['Z_Score'] = (df_tx['Amount'] - mean_amt) / std_amt
normal, anomalous = df_tx[df_tx['Z_Score'] <= 3], df_tx[df_tx['Z_Score'] > 3]
axes[1, 0].scatter(normal.index, normal['Amount'], color='#cccccc', alpha=0.4, s=10)
axes[1, 0].scatter(anomalous.index, anomalous['Amount'], color='#cc0000', alpha=0.9, s=25, label='Outliers (>3 SD)')
axes[1, 0].axhline(mean_amt + (3 * std_amt), color='#cc0000', linestyle='--', linewidth=1)
axes[1, 0].set_title('Transaction Auditing: Fraud Anomaly Outlier Plot', fontweight='bold')
axes[1, 0].set_xlabel('Sequential Transaction Reference ID')
axes[1, 0].set_ylabel('Volume Magnitude ($)')
axes[1, 0].legend(loc='upper right', fontsize=8)

# Panel D: Revenue Volume Share
tier_spend = df_tx.merge(df_cust, on='CustomerID').groupby('AccountTier')['Amount'].sum().reset_index()
axes[1, 1].pie(tier_spend['Amount'], labels=tier_spend['AccountTier'], autopct='%1.1f%%', startangle=140, colors=['#cfd8dc', '#ffd54f', '#90caf9'], textprops={'fontsize': 9})
axes[1, 1].set_title('Capital Contribution: Revenue Volume Share per Card Tier', fontweight='bold')

# Tighter bounding box mapping optimizations
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('banking_powerbi_dashboard.png', dpi=150)
print("💾 Saved optimized screen image copy out as 'banking_powerbi_dashboard.png'")
plt.show()
