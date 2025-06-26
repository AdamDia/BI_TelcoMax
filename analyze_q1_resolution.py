from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
from utils import plot_regression_coefficients


def analyze_resolution_time(data):
    data_encoded = pd.get_dummies(data, columns=['Team', 'Category'], drop_first=True)
    X = data_encoded.drop('ResolutionTime', axis=1)
    y = data_encoded['ResolutionTime']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Calculate regression metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(f"R²: {r2:.2f}, MAE: {mae:.2f}")

    # Plot average resolution time by team
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Team', y='ResolutionTime', data=data, errorbar=None, color='skyblue')
    plt.title('Average Resolution Time by Support Team')
    plt.xlabel('Support Team')
    plt.ylabel('Resolution Time (hours)')
    plt.tight_layout()
    plt.figtext(0.99, 0.01, "Source: Author's own work", horizontalalignment='right', fontsize=8)
    plt.savefig('output/q1_resolution_time.png')
    plt.close()


    # Plot regression feature importance (coefficients)
    coef = pd.Series(model.coef_, index=X.columns)
    plt.figure(figsize=(8, 5))
    coef.sort_values().plot(kind='barh', color='coral')
    plt.title('Regression Coefficients - Feature Impact')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.figtext(0.99, 0.01, "Source: Author's own work", horizontalalignment='right', fontsize=8)
    plt.savefig('output/q1_regression_metrics.png')
    plt.close()

    plot_regression_coefficients(model, X)
    print("Q1 Regression R² Score:", r2_score(y, model.predict(X)))
    print("Q1 Mean Absolute Error (MAE):", mean_absolute_error(y, model.predict(X)))
