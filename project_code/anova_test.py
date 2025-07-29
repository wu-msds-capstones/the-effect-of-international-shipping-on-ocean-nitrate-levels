import pandas as pd
from scipy.stats import f_oneway


df = pd.read_csv("clustered_output_with_distance.csv")
min_lon, max_lon = -82.2484, 4.3238
min_lat, max_lat = 13.7150, 60.1331
df = df[(df["longitude"] >= min_lon) & (df["longitude"] <= max_lon) &
        (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)]


df = df.dropna(subset=["nitrate_value", "cluster"])
groups = [group["nitrate_value"].values for _, group in df.groupby("cluster")]
f_stat, p_val = f_oneway(*groups)


print("ANOVA Results (nitrate_value ~ cluster):")
print(f"F statistic: {f_stat:.4f}")
print(f"p value: {p_val:.4e}")
print("\nObservations per cluster:")
print(df["cluster"].value_counts().sort_index())
