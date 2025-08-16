import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


df = pd.read_csv("clustered_output_with_distance.csv")


for col in ["cluster", "shipping_density", "nitrate_value"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


df = df.dropna(subset=["cluster", "shipping_density", "nitrate_value"])
df = df[df["shipping_density"] >= 0]


df["log_ship"] = np.log1p(df["shipping_density"])  


df["cluster"] = df["cluster"].astype(int)


clusters = sorted(df["cluster"].unique().tolist())[:5]
print("Clusters detected:", clusters)
print("Counts per cluster:\n", df["cluster"].value_counts().sort_index())




include_stats_in_titles = True


max_scatter_points = 150_000


def get_param_by_pos(res, i):

    return float(res.params[i])

def get_pvalue_by_pos(res, i):
    pv = res.pvalues
    return float(pv[i]) if np.ndim(pv) else float(pv)

def get_confint_by_pos(res, i):
    ci = res.conf_int()
    if hasattr(ci, "iloc"):     
        return float(ci.iloc[i, 0]), float(ci.iloc[i, 1])
    else:   
        return float(ci[i, 0]), float(ci[i, 1])


n = len(clusters)
ncols = 3 if n > 4 else min(3, n)
nrows = int(np.ceil(n / ncols))
fig, axs = plt.subplots(nrows, ncols, figsize=(6.6*ncols, 5.2*nrows))
axs = np.atleast_1d(axs).flatten()

for idx, c in enumerate(clusters):
    ax = axs[idx]
    cdf = df.loc[df["cluster"] == c, ["nitrate_value", "log_ship"]].dropna()


    if cdf.empty or np.unique(cdf["log_ship"]).size < 2:
        ax.scatter([], [])  
        ax.set_title(f"Cluster {c} | No variance in log(x) or no data")
        ax.set_xlabel("log(1 + shipping density)")
        ax.set_ylabel("Nitrate value")
        print(f"Cluster {c}: insufficient distinct x to fit OLS. Skipping line.")
        continue

    X = sm.add_constant(cdf[["log_ship"]])
    y = cdf["nitrate_value"]


    m = sm.OLS(y, X).fit()
    rob = m.get_robustcov_results(cov_type="HC3")


    names = list(m.model.exog_names) 

    i_const = names.index("const") if "const" in names else 0
    if "log_ship" in names:
        i_slope = names.index("log_ship")
    else:

        i_slope = len(names) - 1


    slope = get_param_by_pos(m, i_slope)
    intercept = get_param_by_pos(m, i_const)
    p = get_pvalue_by_pos(rob, i_slope)
    ci_lo, ci_hi = get_confint_by_pos(rob, i_slope)
    r2 = float(m.rsquared)
    N = len(cdf)

    print(f"\n=== Cluster {c} ===")
    print(f"Slope (HC3) on log1p(ship): {slope:.3e}   95% CI [{ci_lo:.3e}, {ci_hi:.3e}]   p={p:.3g}")
    print(f"R²={r2:.3f}   N={N:,}   Δnitrate per Δlog1p(ship)=1 ≈ {slope:.3f}")
    if len(cdf) > max_scatter_points:
        cdf_sample = cdf.sample(max_scatter_points, random_state=42)
    else:
        cdf_sample = cdf
    ax.scatter(cdf_sample["log_ship"], cdf_sample["nitrate_value"], s=5, alpha=0.3)
    x_min, x_max = cdf["log_ship"].min(), cdf["log_ship"].max()
    if np.isfinite(slope) and (x_max > x_min):
        xline = np.linspace(x_min, x_max, 200)
        yline = intercept + slope * xline
        ax.plot(xline, yline, lw=2, zorder=5)

    ax.set_xlabel("log(1 + shipping density)")
    ax.set_ylabel("Nitrate value")
    if include_stats_in_titles and np.isfinite(slope):
        ax.set_title(f"Cluster {c} | R²={r2:.2f}, slope={slope:.2e} (p={p:.3g}), N={N:,}")
    else:
        ax.set_title(f"Cluster {c} | N={N:,}")
    ax.grid(True, alpha=0.2)


for j in range(len(clusters), len(axs)):
    fig.delaxes(axs[j])


plt.subplots_adjust(hspace=0.45, wspace=0.3, bottom=0.16, top=0.9)
fig.text(0.02, 0.02, "Figure(5): Nitrate vs. log(1+shipping density) by cluster with OLS (HC3) trend lines. Associations are weak overall, the clearest positive slope is in Cluster 2.", ha="left", va="bottom", fontsize=20, wrap=True)

fig.suptitle("Nitrate vs. log(Shipping Density) per Cluster", fontsize=30)
plt.show()
