from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def plot_feature_importance(model, X):
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance')

    plt.figure(figsize=(6, 4))
    sns.barplot(x='Importance', y='Feature', data=importance_df, color='cornflowerblue')
    plt.title('Feature Importance - Random Forest')
    plt.tight_layout()
    plt.figtext(0.99, 0.01, "Source: Author's own work", horizontalalignment='right', fontsize=8)
    plt.savefig('output/q2_feature_importance.png')
    plt.close()

def plot_regression_coefficients(model, X):
    coeff_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient')
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coeff_df, color='skyblue')
    plt.title('Linear Regression Coefficients')
    plt.tight_layout()
    plt.figtext(0.99, 0.01, "Source: Author's own work", horizontalalignment='right', fontsize=8)
    plt.savefig('q1_regression_coefficients.png')
    plt.close()