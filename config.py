# =====================================================================
# CONFIGURATION MODULE
# =====================================================================
# Centralized configuration for all constants, thresholds, and styling

# =====================================================================
# DATA GENERATION PARAMETERS
# =====================================================================
NUM_CUSTOMERS = 200
NUM_TRANSACTIONS = 1500
NUM_LOANS = 80
DATE_RANGE_DAYS = 160
RANDOM_SEED = 42

# =====================================================================
# FRAUD DETECTION PARAMETERS
# =====================================================================
FRAUD_Z_THRESHOLD = 3
FRAUD_SPIKE_PERCENTAGE = 0.015
FRAUD_SPIKE_RANGE = (3000, 12000)
FRAUD_SPIKE_MEAN = 80
FRAUD_SPIKE_MIN = 2

# =====================================================================
# CREDIT SCORE BUCKETS
# =====================================================================
CREDIT_BUCKETS = {
    'Poor': (0, 580),
    'Fair': (580, 670),
    'Good': (670, 740),
    'Excellent': (740, 851)
}

# Default probability distributions by credit score
DEFAULT_PROBABILITIES = {
    'poor': [0.4, 0.3, 0.3],
    'fair': [0.7, 0.2, 0.1],
    'good': [0.95, 0.04, 0.01]
}

# =====================================================================
# CHURN SIMULATOR PARAMETERS
# =====================================================================
CHURN_WEIGHTS = {
    'recency_weight': 0.75,
    'score_weight': 0.25,
    'loan_discount': 0.15,
    'critical_threshold': 0.55
}

CHURN_RECENCY_MAX = 180  # days
CHURN_SCORE_MAX = 850

# =====================================================================
# CHART STYLING CONSTANTS
# =====================================================================
CHART_HEIGHT = 350
CHART_MARGIN = dict(l=10, r=10, t=10, b=10)

# Color palettes
TIER_COLORS = {
    'Silver': '#90caf9',
    'Gold': '#ffd54f',
    'Platinum': '#cfd8dc'
}

FRAUD_COLORS = {
    'Normal Core Process': '#cccccc',
    'Flagged Outlier (>3 SD)': '#cc0000'
}

FRAUD_THRESHOLD_COLOR = '#cc0000'

# =====================================================================
# TRANSACTION PARAMETERS
# =====================================================================
TRANSACTION_CATEGORIES = ['Groceries', 'Utilities', 'Travel', 'Dining Out']
TRANSACTION_CHANNELS = ['Mobile App', 'ATM', 'Branch']
ACCOUNT_TIERS = ['Silver', 'Gold', 'Platinum']
ACCOUNT_TIER_DISTRIBUTION = [0.5, 0.3, 0.2]  # [Silver, Gold, Platinum]

# =====================================================================
# CUSTOMER DATA RANGES
# =====================================================================
AGE_RANGE = (18, 75)
INCOME_RANGE = (20000, 160000)
CREDIT_SCORE_RANGE = (450, 850)
INTEREST_RATE_RANGE = (5.0, 18.0)
LOAN_AMOUNT_RANGE = (5000, 60000)
LOAN_TERMS = [12, 24, 36, 48, 60]

# =====================================================================
# LOAN STATUS OPTIONS
# =====================================================================
LOAN_STATUSES = ['Current', 'Late', 'Defaulted']

# =====================================================================
# STREAMLIT PAGE CONFIG
# =====================================================================
PAGE_TITLE = "Banking Analytics Platform"
PAGE_ICON = "🚀"
LAYOUT = "wide"

# =====================================================================
# FEATURE IMPORTANCE DATA (Churn Model)
# =====================================================================
FEATURE_IMPORTANCE = [
    ('Recency', 0.555690),
    ('Tx_Count', 0.259590),
    ('CreditScore', 0.045125),
    ('Total_Spend', 0.044271),
    ('Income', 0.034610),
    ('Age', 0.028467),
    ('Avg_Tx_Value', 0.027646),
    ('Has_Active_Loan', 0.004600)
]
