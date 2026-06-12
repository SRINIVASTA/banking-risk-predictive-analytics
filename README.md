# 🚀 End-to-End Banking Analytics & Predictive Risk Platform

An enterprise-grade banking data intelligence project that leverages relational databases, statistical analysis, and machine learning to optimize customer retention, flag fraudulent transactions, and mitigate credit portfolio risk.

---

## 🎯 Project Overview & Core Goals
Modern financial institutions process massive transactional streams daily. This multi-layered analytical engine transforms raw relational database structures into actionable business frameworks. 

The primary business objectives achieved are:
1. **Customer Segmentation & Retention**: Tracking behavioral decay to predict user churn.
2. **Fraud Detection & Auditing**: Isolating anomalous spending behaviors in real time.
3. **Credit Risk Management**: Quantifying structural capital default exposure.
4. **Revenue Distribution Mapping**: Visualizing transactional volume contributions across premium card tiers.

---

## 🧰 Tech Stack & Architecture

*   **Language**: Python 3.10+
*   **Core Libraries**: Pandas, NumPy, Scikit-Learn
*   **Visualizations**: Matplotlib, Seaborn
*   **Database Training**: Relational SQL Query Workflows
*   **Application Deployment**: Streamlit Web Framework

### 🏗️ Relational System Blueprint
The platform operates on a robust, three-table relational database architecture joined together by the `CustomerID` primary key string:

```text
   [ Customer Profiles ]
         (CustomerID)
              |
      +-------+-------+

      |               |
      v               v
[ Loan Records ]  [ Transaction History ]
```

---

## 📁 Repository Directory Structure

Organize your project workspace folder using this standardized directory mapping layout:

```text
├── data/
│   ├── customer_profiles.csv     # Demographic attributes & Credit tiers
│   ├── transaction_history.csv    # 1,500 continuous chronological streams
│   └── loan_records.csv          # Active institutional debt liabilities
├── scripts/
│   ├── data_generator.py         # Relational database synthesizer (Block 1)
│   ├── churn_predictor.py        # Scikit-Learn training script (Block 2)
│   └── dashboard_plot.py         # Power BI visual matrix output engine (Block 3)
├── database/
│   └── banking_queries.sql       # Advanced CTE/Windowing script (Block 4)
├── streamlit_app.py              # Interactive Front-End user portal (Block 5)
├── banking_powerbi_dashboard.png # High-Resolution Executive Dashboard image
└── README.md                     # Comprehensive project portfolio manual
```

---

## 🧠 Core Features & Implementations

### 1. Data Synthesizer Engine
*   Simulates realistic customer age arrays and income metrics.
*   Models everyday consumer habits using an **exponential distribution** function curve.
*   Injects mathematical fraud spikes (1.5% frequency) mimicking high-magnitude capital theft events ($3,000–$12,000 outliers).

### 2. Predictive Churn Classification (Scikit-Learn)
*   Engineers behavioral features like **Recency** (days since active transaction) and **Transaction Counts**.
*   Utilizes a **Random Forest Classifier** to assess churn risks.
*   Addresses low-density class imbalance through a targeted, in-memory synthetic minority oversampling strategy.

### 3. Power BI-Style Executive Visualization Grid
*   Combines four comprehensive analytical reports inside a unified, widescreen workspace canvas plot.
*   **Panel A**: Machine Learning Feature Importance Drivers bar chart.
*   **Panel B**: Asset Delinquency default rate metrics per credit group.
*   **Panel C**: Transaction Auditing outlier scatter plot tracking standard deviation barriers ($Z$-Score $> 3$).
*   **Panel D**: Revenue share breakdown pie visual per account category.

---

## 💡 Key Analytical Insights Documented

*   **Behavioral Churn Weights**: Customer **Recency (55.6%)** and **Transaction Frequency (26.0%)** dominate the machine learning model's predictive importance. Customers consistently reduce their transaction velocity long before closing an account.
*   **Risk Profiles**: Loan accounts associated with a *Poor Credit Classification (<580)* carry significantly higher default probabilities than *Excellent Tiers (740+)*, confirming a clear statistical link between credit ratings and debt risk.

---

## 🚀 How to Set Up and Run the Platform

Follow these steps to deploy the complete environment pipeline on your local computer:

### 1. Clone the Repository Workspace
```bash
git clone https://github.com
cd banking-analytics-platform
```

### 2. Install Required Package Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### 3. Run the Sequential Pipeline
```bash
# Step A: Instantiate the relational database files
python scripts/data_generator.py

# Step B: Execute the Scikit-Learn machine learning model code
python scripts/churn_predictor.py

# Step C: Save out the High-Resolution dashboard matrix image
python scripts/dashboard_plot.py
```

### 4. Boot Up the Interactive Streamlit Web Interface
```bash
streamlit run streamlit_app.py
```

---

## 📊 Sample Executive Dashboard Output
Upon running the dashboard script, a publication-grade graphics layout file titled `banking_powerbi_dashboard.png` will be generated inside your workspace directory, displaying the 4-panel business grid.

---
💡 *Project developed for professional portfolio assessment and advanced data engineering training via [CareerDream.in](https://careerdream.in).*
