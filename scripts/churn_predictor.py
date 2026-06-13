import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_churn_risk_model():
    print("🧠 Initializing Core Machine Learning Analytics Engine...")
    
    # 1. Verify existence of structural relational dataset inputs
    cust_path = 'data/customer_profiles.csv'
    tx_path = 'data/transaction_history.csv'
    
    if not (os.path.exists(cust_path) and os.path.exists(tx_path)):
        raise FileNotFoundError("❌ Operational Error: Base relational tables missing. Run scripts/data_generator.py first.")
        
    df_cust = pd.read_csv(cust_path)
    df_tx = pd.read_csv(tx_path)
    df_tx['Timestamp'] = pd.to_datetime(df_tx['Timestamp'])

    # 2. Advanced Feature Engineering Engine
    print("⚙️ Engineering behavioral variables from streaming chronological data arrays...")
    
    # Calculate tracking reference bounds for transactional inactivity metrics
    max_date = df_tx['Timestamp'].max()
    
    # Extract Recency (Days since last active transaction) per profile group
    df_recency = df_tx.groupby('CustomerID')['Timestamp'].max().reset_index()
    df_recency['Recency'] = (max_date - df_recency['Timestamp']).dt.days

    # Extract Transaction Density (Frequency profiles) and Total Volumes
    df_tx_stats = df_tx.groupby('CustomerID').agg(
        Tx_Count=('TransactionID', 'count'),
        Total_Spend=('Amount', 'sum'),
        Avg_Tx_Value=('Amount', 'mean')
    ).reset_index()

    # Consolidate feature branches back to unified master demographic data frame matrix
    df_features = pd.merge(df_cust, df_recency[['CustomerID', 'Recency']], on='CustomerID', how='left')
    df_features = pd.merge(df_features, df_tx_stats, on='CustomerID', how='left')
    
    # Handle low-density missing data profiles securely via fallback initialization operations
    df_features['Recency'] = df_features['Recency'].fillna(180) # Flag highly inactive profiles
    df_features['Tx_Count'] = df_features['Tx_Count'].fillna(0)
    df_features['Total_Spend'] = df_features['Total_Spend'].fillna(0)
    df_features['Avg_Tx_Value'] = df_features['Avg_Tx_Value'].fillna(0)

    # 3. Formulate Deterministic Synthetic Churn Targets (Synthetic Ground Truth Rules Engine)
    # Replicates realistic systemic churn thresholds to address low-density imbalance states
    np.random.seed(42)
    risk_factor = (df_features['Recency'] * 0.4) - (df_features['Tx_Count'] * 0.3) + (df_features['Age'] * 0.05)
    risk_factor += np.where(df_features['CreditScore'] < 580, 15, 0)
    
    # Apply a dynamic threshold mapping to isolate high-risk user profiles
    threshold = np.percentile(risk_factor, 80) 
    df_features['Target_Churn'] = np.where(risk_factor >= threshold, 1, 0)

    # 4. Model Training Matrix Processing Layout Setup
    X = df_features[['Recency', 'Tx_Count', 'CreditScore', 'Total_Spend', 'Income', 'Age', 'Avg_Tx_Value']]
    y = df_features['Target_Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # 5. Execute Ensembled Random Forest Training Phase
    print("🚀 Execution: Training Scikit-Learn Classifier Weights...")
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    # 6. Quantify Performance Metrics Against Unseen Out-of-Sample Records
    y_pred = model.predict(X_test)
    print(f"🎯 Model Performance Metrics Baseline Accuracy Score: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("\n📋 Detailed Structural Classification Operational Report Breakdown:")
    print(classification_report(y_test, y_pred))

    # 7. Extract Feature Importance Arrays
    importances = model.feature_importances_
    feature_names = X.columns
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n💡 Calculated Predictive Model Weights Output Table:")
    for _, row in df_importance.iterrows():
        print(f"   -> {row['Feature']}: {row['Importance']:.6f}")

    print("\n✅ Success Engine: Optimization scripts complete.")

if __name__ == "__main__":
    train_churn_risk_model()
