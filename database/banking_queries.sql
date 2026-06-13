-- =====================================================================
-- Advanced Banking Relational Database Analytics Schema
-- Optimized for production-grade PostgreSQL / Snowflake / BigQuery engines
-- =====================================================================

-- 🚀 ADVANCED QUERY 1: DYNAMIC ROLLING RFM BEHAVIORAL DECAY MATRIX
WITH MaxSystemTimeline AS (
    SELECT MAX(Timestamp) AS Anchor_Max_Date 
    FROM transaction_history
),
CustomerBaseMetrics AS (
    SELECT 
        t.CustomerID,
        -- FIXED: Replaced EXTRACT with universally compliant standard DATE subtraction metrics
        CAST(m.Anchor_Max_Date AS DATE) - CAST(MAX(t.Timestamp) AS DATE) AS Behavioral_Recency_Days,
        COUNT(t.TransactionID) AS Transactional_Frequency_Count,
        ROUND(CAST(SUM(t.Amount) AS NUMERIC), 2) AS Aggregate_Monetary_Volume
    FROM transaction_history t
    CROSS JOIN MaxSystemTimeline m
    GROUP BY t.CustomerID, m.Anchor_Max_Date
)
SELECT 
    CustomerID,
    Behavioral_Recency_Days,
    Transactional_Frequency_Count,
    Aggregate_Monetary_Volume,
    NTILE(4) OVER (ORDER BY Behavioral_Recency_Days ASC) AS Recency_Tier_Score,   
    NTILE(4) OVER (ORDER BY Transactional_Frequency_Count DESC) AS Frequency_Tier_Score, 
    NTILE(4) OVER (ORDER BY Aggregate_Monetary_Volume DESC) AS Monetary_Tier_Score     
FROM CustomerBaseMetrics
ORDER BY Transactional_Frequency_Count DESC;


-- 📉 ADVANCED QUERY 2: CREDIT PORTFOLIO CAPITAL DEFAULT EXPOSURE MATRIX
SELECT 
    CASE 
        WHEN c.CreditScore < 580 THEN 'Poor Credit Tier (<580)'
        WHEN c.CreditScore < 670 THEN 'Fair Credit Tier (580-669)'
        WHEN c.CreditScore < 740 THEN 'Good Credit Tier (670-739)'
        ELSE 'Excellent Premium Tier (740+)'
    END AS Credit_Risk_Classification,
    COUNT(l.LoanID) AS Active_Portfolio_Loan_Count,
    ROUND(CAST(SUM(l.LoanAmount) AS NUMERIC), 2) AS Total_Institutional_Capital_Exposure,
    ROUND(
        CAST(SUM(CASE WHEN l.CurrentStatus = 'Defaulted' THEN 1 ELSE 0 END) AS NUMERIC) * 100.0 / 
        NULLIF(COUNT(l.LoanID), 0), 2
    ) AS Calculated_Default_Rate_Percentage
FROM loan_records l
INNER JOIN customer_profiles c ON l.CustomerID = c.CustomerID
GROUP BY 
    CASE 
        WHEN c.CreditScore < 580 THEN 'Poor Credit Tier (<580)'
        WHEN c.CreditScore < 670 THEN 'Fair Credit Tier (580-669)'
        WHEN c.CreditScore < 740 THEN 'Good Credit Tier (670-739)'
        ELSE 'Excellent Premium Tier (740+)'
    END
ORDER BY Total_Institutional_Capital_Exposure DESC;


-- 🚨 ADVANCED QUERY 3: FORENSIC FRAUD DETECTOR WITH Z-SCORE INTENSITY RANGE
WITH OperationalSystemMetrics AS (
    SELECT 
        AVG(Amount) AS Core_Mean_Volume,
        STDDEV(Amount) AS Core_Standard_Deviation_Value
    FROM transaction_history
)
SELECT 
    t.TransactionID,
    t.CustomerID,
    t.Category AS Transaction_Channel_Category,
    t.Channel AS Operational_Delivery_Channel,
    ROUND(CAST(t.Amount AS NUMERIC), 2) AS High_Magnitude_Outlier_Amount,
    ROUND(
        CAST(((t.Amount - m.Core_Mean_Volume) / m.Core_Standard_Deviation_Value) AS NUMERIC), 2
    ) AS Calculated_Statistical_Z_Score
FROM transaction_history t
CROSS JOIN OperationalSystemMetrics m
WHERE t.Amount > (m.Core_Mean_Volume + (3 * m.Core_Standard_Deviation_Value))
ORDER BY t.Amount DESC;
