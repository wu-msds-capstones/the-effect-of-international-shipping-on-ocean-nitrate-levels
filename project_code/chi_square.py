import pandas as pd
from scipy.stats import chi2_contingency


df = pd.read_csv("clustered_output_with_distance.csv")
low_thresh = df["shipping_density"].quantile(0.33)
high_thresh = df["shipping_density"].quantile(0.66)
def categorize_shipping(val):
    if val < low_thresh:
        return "Low"
    elif val > high_thresh:
        return "High"
    else:
        return "Average"

df["shipping_cat"] = df["shipping_density"].apply(categorize_shipping)

low_thresh = df["nitrate_value"].quantile(0.33)
high_thresh = df["nitrate_value"].quantile(0.66)

def categorize_nitrate(val):
    if val < low_thresh:
        return "Low"
    elif val > high_thresh:
        return "High"
    else:
        return "Average"

df["nitrate_cat"] = df["nitrate_value"].apply(categorize_nitrate)
contingency = pd.crosstab(df['shipping_cat'], df['nitrate_cat'])
print("Contingency Table (Counts):\n", contingency, "\n")

chi2, p, dof, expected = chi2_contingency(contingency)
print("Contingency Table:\n", contingency)
print("\nChi-square Statistic:", chi2)
print("p-value:", p)
print("Degrees of Freedom:", dof)
