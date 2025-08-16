




#-----------------------------------------------------------------------------------------


import pandas as pd
import geopandas as gpd
from shapely.strtree import STRtree
from shapely.geometry import Point


df = pd.read_csv("cluster_ready.csv")


gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326") 
gdf = gdf.to_crs(epsg=3857)


coastline = gpd.read_file("ne_50m_coastline/ne_50m_coastline.shp").to_crs(epsg=3857)

coast_geoms = list(coastline.geometry)
tree = STRtree(coast_geoms)
def compute_nearest_distance(geom):
    if not isinstance(geom, Point):
        return None
    if geom.is_empty or not geom.is_valid:
        return None
    try:
        nearest_index = tree.nearest(geom)
        nearest_geom = coast_geoms[nearest_index]

        if nearest_geom is None or nearest_geom.is_empty or not nearest_geom.is_valid:
            return None

        return geom.distance(nearest_geom) / 1000  
    except Exception as e:
        return None
gdf["distance_from_land_km"] = gdf["geometry"].apply(compute_nearest_distance)
gdf.drop(columns="geometry").to_csv("cluster_ready_with_distance_fast.csv", index=False)

#------------------------------------------------------------------------------------------

#elbow plot for clustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df = pd.read_csv("cluster_ready_with_distance_fast.csv")


features = [
    "latitude", "longitude", "nitrate_value",
    "distance_from_land_km", "avg_chlor_a", "shipping_density"]
X = df[features].fillna(0)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertia = []
k_range = range(2, 15)  

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters(k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_plot.png")
plt.show()


#------------------------------------------------------------------------------------------

#automatic elbow method using Kneed


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt


df = pd.read_csv("cluster_ready_with_distance_fast.csv")
features = ["latitude", "longitude", "nitrate_value", "distance_from_land_km", "avg_chlor_a", "shipping_density"]
X = df[features].fillna(0)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertia = []
k_range = range(2, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)


kl = KneeLocator(k_range, inertia, curve="convex", direction="decreasing")
print(f"Optimal number of clusters (elbow): k = {kl.elbow}")


plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.vlines(kl.elbow, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='red')
plt.xlabel("Number of Clusters(k)", fontsize=20)
plt.ylabel("Inertia", fontsize=20)
plt.title("Elbow Method", fontsize=30)
plt.tight_layout()

fig = plt.gcf()
fig.text(
    0.02, 0.02,
    "Figure(1): We use the Kneed library to automatically find the elbow point in the inertia plot. The best k is determined to be 5.",
    ha="left", va="center", fontsize=25, style="italic", wrap=True
)
plt.subplots_adjust(bottom=0.12)
plt.show()

    
#-----------------------------------------------------------------------------------------

#cluster attempt 2
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
df_cluster = pd.read_csv("cluster_ready_with_distance_fast.csv")


features = ["latitude", "longitude", "nitrate_value", "distance_from_land_km","avg_chlor_a", "shipping_density"]
X = df_cluster[features].fillna(0)  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
df_cluster["cluster"] = kmeans.fit_predict(X_scaled)


df_cluster.to_csv("clustered_output_with_distance.csv", index=False)
print(df_cluster["cluster"].unique())   



#-----------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm


df = pd.read_csv("clustered_output_with_distance.csv")


min_lon, max_lon = -82.2484, 4.3238
min_lat, max_lat = 13.7150, 60.1331


df = df[
    (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon) &
    (df["latitude"]  >= min_lat) & (df["latitude"]  <= max_lat)
].copy()


n_clusters = 5
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  
cmap = ListedColormap(colors[:n_clusters])


bounds = np.arange(-0.5, n_clusters + 0.5, 1)  
norm = BoundaryNorm(bounds, cmap.N)


fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')

gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
gl.top_labels = False
gl.right_labels = False


sc = ax.scatter(
    df["longitude"], df["latitude"],
    c=df["cluster"],
    cmap=cmap, norm=norm,
    s=5, alpha=0.6, transform=ccrs.PlateCarree())

cbar = plt.colorbar(sc, ax=ax, boundaries=bounds, ticks=np.arange(n_clusters))
cbar.set_label('Cluster')

fig = plt.gcf()
fig.text(
    0.02, 0.02,
    "Figure(2): Clustered Observation Points within the Atlantic bounding box showing all 5 clusters 0-4 seperated by color.",
    ha="left", va="center", fontsize=25, style="italic", wrap=True
)
plt.subplots_adjust(bottom=0.12)

plt.title("Clustered Observation Points (Atlantic Bounding Box)", fontsize=30)
plt.tight_layout()
plt.show()




#-----------------------------------------------------------------------------------------

#cluster maps seperate maps by cluster


import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


df = pd.read_csv("clustered_output_with_distance.csv")
min_lon, max_lon = -82.2484, 4.3238
min_lat, max_lat = 13.7150, 60.1331  

clusters_to_plot = [0, 1, 2, 3, 4]  
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig, axs = plt.subplots(3, 3, figsize=(16, 10),
                        subplot_kw={'projection': ccrs.PlateCarree()})
axs = axs.flatten()

for i, cluster_id in enumerate(clusters_to_plot):
    ax = axs[i]
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=False)
    cluster_data = df[df["cluster"] == cluster_id]
    sc = ax.scatter(
        cluster_data["longitude"],
        cluster_data["latitude"],
        color=colors[cluster_id],
        s=5, alpha=0.6,
        transform=ccrs.PlateCarree()
    )
    
    ax.set_title(f"Cluster {cluster_id}")
for j in range(len(clusters_to_plot), len(axs)):
    fig.delaxes(axs[j])
fig.text(
    0.02, 0.02,
    "Figure(3): Shows each cluster separated to make it easier to see where each cluster is localized.",
    ha="left", va="center", fontsize=25, style="italic", wrap=True
)
plt.subplots_adjust(bottom=0.12)
fig.suptitle("Clusters 0–4 Mapped Separately (Atlantic Focus)", fontsize=30)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()



#-----------------------------------------------------------------------------------------

#eda per cluster



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("clustered_output_with_distance.csv")

vars_of_interest = ["nitrate_value", "shipping_density", "avg_chlor_a"]

summary = df.groupby("cluster")[vars_of_interest].describe()
print(summary)


for c in sorted(df["cluster"].unique()):
    cluster_data = df[df["cluster"] == c]
    print(f"\n=== Cluster {c} ===")
    print(cluster_data[vars_of_interest].describe())
    cluster_data[vars_of_interest].hist(bins=30, figsize=(10, 6))
    plt.suptitle(f"Cluster {c} Distributions")
    plt.figtext(0.5, 0.01, "Caption: [Insert description for Cluster histogram here]", 
                ha="center", va="bottom", fontsize=9, style="italic", wrap=True)
    plt.show()
    sns.heatmap(cluster_data[vars_of_interest].corr(), annot=True, cmap="coolwarm")
    plt.title(f"Cluster {c} Correlation Matrix")
    plt.figtext(0.5, -0.05, "Caption: [Insert description for Cluster correlation matrix here]",
                ha="center", va="bottom", fontsize=9, style="italic", wrap=True)
    plt.show()
for var in vars_of_interest:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="cluster", y=var, data=df)
    plt.title(f"{var} by Cluster", fontsize=30)
    plt.figtext(0.02, 0.01, f"Figure(4): Boxplot of {var} by Cluster", 
                ha="left", va="bottom", fontsize=25, style="italic", wrap=True)
    plt.xlabel("Cluster", fontsize=20)
    plt.ylabel(var, fontsize=20)
    plt.show()


