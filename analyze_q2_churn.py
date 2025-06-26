# analysis_q2.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from utils import plot_feature_importance  # ✅ use the shared utility function

def analyze_churn(data):
    data['SupportExperience'] = data['SupportExperience'].map({'Good': 1, 'Bad': 0})
    X = data[['Tenure', 'SupportExperience']]
    y = data['Churn']

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix - Churn Prediction')
    plt.tight_layout()
    plt.figtext(0.99, 0.01, "Source: Author's own work", horizontalalignment='right', fontsize=8)
    plt.savefig('output/q2_confusion_matrix.png')
    plt.close()

    # Feature importance
    plot_feature_importance(model, X)

    # Output
    print("Q2 Classification Report:\n", classification_report(y, y_pred))
