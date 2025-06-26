import numpy as np
import pandas as pd



def simulate_data():
    np.random.seed(42)
    n = 100

    # 1️⃣ Realistic Resolution Time Data
    teams = np.random.choice(['A', 'B', 'C', 'D'], n, p=[0.3, 0.3, 0.2, 0.2])
    priorities = np.random.choice([1, 2, 3], n, p=[0.5, 0.3, 0.2])  # more low-priority tickets
    categories = np.random.choice(['Billing', 'Tech', 'Account'], n, p=[0.4, 0.4, 0.2])
    resolution_base = np.random.normal(loc=24, scale=5, size=n)
    resolution_time = resolution_base + (priorities * 2)  # Priority adds time
    resolution_time += np.where(teams == 'B', 3, 0)  # Team B slower
    q1_data = pd.DataFrame({
        'Team': teams,
        'Priority': priorities,
        'Category': categories,
        'ResolutionTime': resolution_time
    })

    # 2️⃣ Realistic Churn Data
    tenure = np.random.randint(1, 36, n)
    churn_probs = np.where(tenure < 6, 0.5, 0.2)
    churn = np.random.binomial(1, churn_probs)
    support_experience = np.where(tenure < 6, 
                                  np.random.choice(['Good', 'Bad'], p=[0.4, 0.6], size=n), 
                                  np.random.choice(['Good', 'Bad'], p=[0.7, 0.3], size=n))
    q2_data = pd.DataFrame({
        'Tenure': tenure,
        'SupportExperience': support_experience,
        'Churn': churn
    })

    # 3️⃣ Realistic Campaign Effectiveness Data
    support_positive = np.random.choice([0, 1], n, p=[0.3, 0.7])
    upsell_success = np.where(support_positive == 1, 
                              np.random.choice([0, 1], n, p=[0.5, 0.5]),
                              np.random.choice([0, 1], n, p=[0.8, 0.2]))
    q3_data = pd.DataFrame({
        'SupportPositive': support_positive,
        'UpsellSuccess': upsell_success,
        'WinbackSuccess': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'LoyaltyUpgrade': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'RetentionOffer': np.random.choice([0, 1], n, p=[0.5, 0.5])
    })

    return q1_data, q2_data, q3_data