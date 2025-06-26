# Data Mining CRM Optimization for TelcoMax

This project demonstrates how data mining techniques can be applied to optimize Customer Relationship Management (CRM) processes at a fictional telecom provider, **TelcoMax**. It was developed as part of an academic portfolio submission at IU International University of Applied Sciences.

## 🚀 Objective
The goal is to simulate CRM data and apply suitable machine learning algorithms to address three business-critical questions:

1. **What causes delays in resolving support tickets?**
2. **Can we predict churn based on the first support experience?**
3. **Do positive support interactions increase the success of follow-up marketing campaigns?**

---

## 📊 Models and Techniques Used

| Question | Model Used                    | Output                       |
|----------|-------------------------------|------------------------------|
| Q1       | Linear Regression              | Bar chart, Coefficient plot  |
| Q2       | Random Forest Classification   | Confusion matrix, Feature importance |
| Q3       | Association Rule Mining (Apriori) | Top 5 rules (lift > 1)     |

---

## 🛠️ Tech Stack

- **Language**: Python 3.11+
- **Libraries**:  
  - `pandas`, `numpy`, `matplotlib`, `seaborn`  
  - `scikit-learn` for regression/classification  
  - `mlxtend` for association rules
 
---

## 📁 Folder Structure

![Screenshot 2025-06-26 at 6 24 17 PM](https://github.com/user-attachments/assets/697e8699-ee01-4d92-bbe6-195fe1ecf120)

---

## 📸 Sample Outputs

- `q1_resolution_time.png`: Average resolution time by support team 
- `q1_regression_coefficients.png`: Regression weights indicating ticket resolution drivers
- `q2_confusion_matrix.png`: Churn prediction confusion matrix
- `q2_feature_importance.png`: Feature importance from the churn prediction model
- `q3_association_rules.png`: Top 5 marketing lift rules from support interactions
---

## 📂 How to Run
1. **Create and activate a virtual environment
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

2. **Install required packages**  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn mlxtend

3. Run the script
    ```bash
    python project.py

4. Check generated charts
   
  The script saves *.png images for each model’s output in the *output/ directory.
  
---

## 📚 References
Breiman, L. (2001). Random Forests. Machine Learning.

Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules.

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning.
