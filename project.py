# project.py
"""
Data Mining CRM Optimization for TelcoMax
"""

# 1. Imports and Setup
from simulate import simulate_data
from analyze_q1_resolution import analyze_resolution_time
from analyze_q2_churn import analyze_churn
from analyze_q3_campaigns import analyze_campaign_effectiveness

# 2. Main execution
if __name__ == "__main__":
    q1_data, q2_data, q3_data = simulate_data()
    analyze_resolution_time(q1_data)
    analyze_churn(q2_data)
    analyze_campaign_effectiveness(q3_data)
    print("Visuals generated and saved as .png files.")
    print("Sample of Q1 data:\n", q1_data.head())
    print("Sample of Q2 data:\n", q2_data.head())
    print("Sample of Q3 data:\n", q3_data.head())
