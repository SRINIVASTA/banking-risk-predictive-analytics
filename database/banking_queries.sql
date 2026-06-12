-- CareerDream.in: Banking Relational Database Analytics Schema
-- Run these queries to cross-verify dashboard metrics inside standard SQL Engines

-- Query 1. CUSTOMER SEGMENTATION: CALCULATE RFM METRICS
WITH CustomerRFM AS (
    SELECT 
        CustomerID,
        DATEDIFF('2026-06-10', MAX(Timestamp)) AS Recency,
        COUNT(TransactionID) AS Frequency,
        SUM(Amount) AS Monetary
    FROM transaction_history
    GROUP BY CustomerID
)
SELECT 
    CustomerID, Recency, Frequency, Monetary,
    NTILE(4) OVER (ORDER BY Recency DESC) AS R_Score,
    NTILE(4) OVER (ORDER BY Frequency ASC) AS F_Score,
    NTILE(4) OVER (ORDER BY Monetary ASC) AS M_Score
FROM CustomerRFM;

-- Query 2. RISK ANALYSIS: DEFAULT RATES PER RISK PROFILE
SELECT 
    CASE 
        WHEN c.CreditScore < 580 THEN 'Poor (<580)'
        WHEN c.CreditScore < 670 THEN 'Fair (580-669)'
        WHEN c.CreditScore < 740 THEN 'Good (670-739)'
        ELSE 'Excellent (740+)'
    END AS Credit_Tier,
    COUNT(l.LoanID) AS Total_Active_Loans,
    SUM(CASE WHEN l.CurrentStatus = 'Defaulted' THEN 1 ELSE 0 END) * 100.0 / COUNT(l.LoanID) AS Default_Rate_Percentage
FROM loan_records l
INNER JOIN customer_profiles c ON l.CustomerID = c.CustomerID
GROUP BY 1
ORDER BY Total_Active_Loans DESC;

-- Query 3. FRAUD ENGINE: EXTRACT ANOMALOUS OVERSPENDING VIA STANDARDISED STANDARD DEVIATIONS
SELECT TransactionID, CustomerID, Amount, Category, Channel
FROM transaction_history
WHERE Amount > (SELECT AVG(Amount) + (3 * STDDEV(Amount)) FROM transaction_history)
ORDER BY Amount DESC;
