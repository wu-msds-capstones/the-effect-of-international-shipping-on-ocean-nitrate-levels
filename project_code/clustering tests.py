import pandas as pd
env = pd.read_csv("cluster_input_aggregated.csv")
chlor = pd.read_csv("chlor_a_atlantic.csv")
env["lat_r"] = env["latitude"].round(2)
env["lon_r"] = env["longitude"].round(2)
chlor["lat_r"] = chlor["lat"].round(2)
chlor["lon_r"] = chlor["lon"].round(2)
merged = pd.merge(
    env, chlor,
    on=["lat_r", "lon_r", "year", "month"],
    how="outer")

merged = merged.drop(columns=["lat_r", "lon_r", "lat", "lon"], errors="ignore")

merged.to_csv("cluster_ready.csv", index=False)
print(merged.head())



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
    "latitude", "longitude", "nitrate_value", "plankton_value",
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
features = ["latitude", "longitude", "nitrate_value", "plankton_value", "distance_from_land_km", "avg_chlor_a", "shipping_density"]
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
plt.xlabel("Number of Clusters(k)")
plt.ylabel("Inertia")
plt.title("Elbow Method with kneed")
plt.tight_layout()
plt.show()


#-----------------------------------------------------------------------------------------

#cluster attempt 2
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
df_cluster = pd.read_csv("cluster_ready_with_distance_fast.csv")


features = ["latitude", "longitude", "nitrate_value", "plankton_value", "distance_from_land_km","avg_chlor_a", "shipping_density"]
X = df_cluster[features].fillna(0)  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
df_cluster["cluster"] = kmeans.fit_predict(X_scaled)


df_cluster.to_csv("clustered_output_with_distance.csv", index=False)
print(df_cluster["cluster"].unique())   



#-----------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


df = pd.read_csv("clustered_output_with_distance.csv")


fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_global()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.gridlines(draw_labels=True)


scatter = ax.scatter(
    df["longitude"], df["latitude"],
    c=df["cluster"], cmap='tab10', s=5, alpha=0.6,
    transform=ccrs.PlateCarree())


legend1 = ax.figure.colorbar(scatter, ax=ax, orientation='vertical', label='Cluster')
plt.title("Clustered Observation Points")
plt.tight_layout()
plt.show()



#-----------------------------------------------------------------------------------------

#cluster maps seperate maps by cluster


import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

df = pd.read_csv("clustered_output_with_distance.csv")


clusters_to_plot = [0, 1, 2, 3]

fig, axs = plt.subplots(2, 2, figsize=(16, 10),
                        subplot_kw={'projection': ccrs.PlateCarree()})
axs = axs.flatten()

for i, cluster_id in enumerate(clusters_to_plot):
    ax = axs[i]
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=False)


    cluster_data = df[df["cluster"] == cluster_id]

    sc = ax.scatter(
        cluster_data["longitude"],
        cluster_data["latitude"],
        c=[cluster_id]*len(cluster_data),
        cmap='tab10', s=5, alpha=0.6,
        transform=ccrs.PlateCarree()
    )
    ax.set_title(f"Cluster {cluster_id}")

fig.suptitle("Clusters 0–3 Mapped Separately", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

