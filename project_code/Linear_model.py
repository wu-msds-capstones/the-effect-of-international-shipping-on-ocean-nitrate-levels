import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv("clustered_output_with_distance.csv")


min_lon, max_lon = -82.2484, 4.3238
min_lat, max_lat = 13.7150, 60.1331
atlantic_df = df[
    (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon) &
    (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)].copy()


atlantic_df = atlantic_df[
    (atlantic_df["nitrate_value"] > 0) &
    (atlantic_df["shipping_density"] > 0)].copy()
atlantic_df["log_nitrate"] = np.log1p(atlantic_df["nitrate_value"])
atlantic_df["log_shipping"] = np.log1p(atlantic_df["shipping_density"])


X = sm.add_constant(atlantic_df["log_shipping"])
y = atlantic_df["log_nitrate"]
model = sm.OLS(y, X).fit()


print(model.summary())
plt.figure(figsize=(10, 6))
plt.scatter(atlantic_df["log_shipping"], atlantic_df["log_nitrate"], alpha=0.3, s=5, label="Data")
plt.plot(atlantic_df["log_shipping"], model.predict(X), color="red", label="Regression Line")
plt.xlabel("Log(Shipping Density + 1)")
plt.ylabel("Log(Nitrate Value + 1)")
plt.title("Log-Log Regression: Shipping vs. Nitrate (Atlantic)")
plt.legend()
plt.tight_layout()
plt.show()
