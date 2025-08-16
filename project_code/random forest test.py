import numpy as np, pandas as pd, matplotlib.pyplot as plt
from math import ceil
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay


SAMPLE_PER_CLUSTER = 0  
N_TREES = 200
MAX_COLS = 3

df = pd.read_csv("clustered_output_with_distance.csv")
for c in ["cluster","nitrate_value","shipping_density","avg_chlor_a","distance_from_land_km","latitude","longitude"]:
    if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["cluster","nitrate_value","shipping_density"])
df = df[df["shipping_density"] >= 0]
df["log_ship"] = np.log1p(df["shipping_density"])
df["cluster"] = df["cluster"].astype(int)

clusters = sorted(df["cluster"].unique().tolist())
feat_cols = [c for c in ["log_ship","avg_chlor_a","distance_from_land_km","latitude","longitude"] if c in df.columns]


cols = min(MAX_COLS, max(1, len(clusters)))
rows = int(ceil(len(clusters) / cols))

fig_pi,  axs_pi  = plt.subplots(rows, cols, figsize=(6*cols, 3.8*rows))
fig_pdp, axs_pdp = plt.subplots(rows, cols, figsize=(6*cols, 3.8*rows))
axs_pi  = np.atleast_1d(axs_pi).flatten()
axs_pdp = np.atleast_1d(axs_pdp).flatten()

axes_to_remove_pi  = []
axes_to_remove_pdp = []

for i, c in enumerate(clusters):
    ax_pi, ax_pdp = axs_pi[i], axs_pdp[i]
    cdf = df[df["cluster"] == c].dropna(subset=["nitrate_value"] + feat_cols)
    if cdf.empty:
        axes_to_remove_pi.append(ax_pi)
        axes_to_remove_pdp.append(ax_pdp)
        continue

    X, y = cdf[feat_cols], cdf["nitrate_value"]
    if SAMPLE_PER_CLUSTER and len(X) > SAMPLE_PER_CLUSTER:
        idx = X.sample(SAMPLE_PER_CLUSTER, random_state=42).index
        X, y = X.loc[idx], y.loc[idx]

    rf = RandomForestRegressor(
        n_estimators=N_TREES, max_depth=20, min_samples_leaf=5,
        max_features="sqrt", bootstrap=True, max_samples=0.5,
        n_jobs=-1, random_state=42
    )
    r2 = cross_val_score(rf, X, y, cv=5, scoring="r2", n_jobs=-1).mean()
    rf.fit(X, y)

  
    pi = permutation_importance(rf, X, y, n_repeats=3, random_state=42, n_jobs=-1, scoring="r2")
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
    ax_pi.barh(imp.index, imp.values)
    ax_pi.set_title(f"Cluster {c} | CV R²={r2:.2f} (n={len(X):,})", fontsize=11)
    ax_pi.set_xlabel("Mean ΔR² when permuted")
    ax_pi.grid(axis="x", alpha=0.2)
    ax_pi.tick_params(axis="y", labelsize=9)  

 
    if "log_ship" in X.columns and X["log_ship"].nunique() > 1:
        PartialDependenceDisplay.from_estimator(
            rf, X, ["log_ship"], kind="average", method="recursion",
            grid_resolution=30, ax=ax_pdp
        )
        ax_pdp.set_title(f"Cluster {c} — PDP: nitrate vs log(1+ship)", fontsize=11)
    else:
        axes_to_remove_pdp.append(ax_pdp)


for j in range(len(clusters), len(axs_pi)):
    axes_to_remove_pi.append(axs_pi[j])
for j in range(len(clusters), len(axs_pdp)):
    axes_to_remove_pdp.append(axs_pdp[j])

for ax in axes_to_remove_pi:
    fig_pi.delaxes(ax)
for ax in axes_to_remove_pdp:
    fig_pdp.delaxes(ax)


fig_pi.subplots_adjust(hspace=0.5, wspace=0.55, top=0.88, bottom=0.22)
fig_pi.suptitle("Permutation importance by cluster (RF)", fontsize=14)
fig_pi.text(0.02, 0.06, "Figure(6):Permutation importance by cluster (RF): spatial features longitude, latitude, and distance from land dominate; shipping density is secondary.", ha="left", va="bottom", fontsize=17, wrap=True,
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=3))

fig_pdp.subplots_adjust(hspace=0.5, wspace=0.35, top=0.88, bottom=0.22)
fig_pdp.suptitle("PDP: nitrate vs log(1+shipping density) by cluster (RF)", fontsize=30)
fig_pdp.text(0.02, 0.06, "Figure(7):Mild positive trend in cluster 0 and Cluster 3, near-flat in cluster 1, sharp rise at high values in cluster 2, and non-monotonic in cluster 4", ha="left", va="bottom", fontsize=17,wrap=True,
             bbox=dict(fc="white", ec="none", alpha=0.9, pad=3))

plt.show()
