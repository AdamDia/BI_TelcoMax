import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

def analyze_campaign_effectiveness(data):
    basket = data.astype(bool)
    frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

    # Convert 'lift' to numeric
    rules['lift'] = pd.to_numeric(rules['lift'], errors='coerce')
    rules = rules.dropna(subset=['lift'])

    if not rules.empty:
        top_rules = rules.nlargest(5, 'lift')
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_rules['lift'], y=top_rules['consequents'].astype(str), color='green')
        plt.xlabel('Lift', fontsize=12)
        plt.ylabel('Consequents', fontsize=12)
        plt.title('Top 5 Association Rules', fontsize=14)
        plt.tight_layout()
        plt.figtext(0.99, 0.01, "Source: Author's own work", horizontalalignment='right', fontsize=8)
        plt.savefig('output/q3_association_rules.png', dpi=300)
        plt.close()
    else:
        print("No association rules found, even after lowering support threshold.")