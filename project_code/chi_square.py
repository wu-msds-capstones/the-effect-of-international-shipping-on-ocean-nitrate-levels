import pandas as pd
from scipy.stats import chi2_contingency


df = pd.read_csv("clustered_output_with_distance.csv")
min_lon, max_lon = -82.2484, 4.3238
min_lat, max_lat = 13.7150, 60.1331
df = df[
    (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon) &
    (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)]
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
chi2, p, dof, expected = chi2_contingency(contingency)
print("Contingency Table:\n", contingency)
print("\nChi-square Statistic:", chi2)
print("p-value:", p)
print("Degrees of Freedom:", dof)
