import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, Point
import numpy as np
import geemap
import pandas as pd

gdf = gpd.read_file("data/delhi_ncr_region.geojson")



gdf_utm = gdf.to_crs("EPSG:32644")


minx, miny, maxx, maxy = gdf_utm.total_bounds
grid_size = 60000
cols = np.arange(minx, maxx, grid_size)
rows = np.arange(miny, maxy, grid_size)
grid_cells = []

for x in cols:
    for y in rows:
        cell = box(x, y, x + grid_size, y + grid_size)
        grid_cells.append(cell)
grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs='EPSG:32644')

# Plot static overview
ax = gdf_utm.plot(figsize=(10, 10), color='none', edgecolor='blue')
grid.boundary.plot(ax=ax, color=None, edgecolor='red', linewidth=1)
plt.title('Delhi NCR with 60x60km Grid (EPSG:32644)')
plt.show()



grid4326 = grid.to_crs("EPSG:4326")
Map = geemap.Map(center=[28.6,77.2], zoom=7)
Map.add_basemap("SATELLITE")
Map.add_gdf(gdf)
Map.add_gdf(grid4326, style={'fillColor':'none','color':'red'})

corner_pts = []
center_pts = []

for cell in grid.geometry:
    x, y = cell.centroid.x, cell.centroid.y
    center_pts.append(Point(x, y))
    coords = list(cell.exterior.coords)
    # four corners are the first four coords (works for boxes)
    for idx in [0,1,2,3]:
        corner_pts.append(Point(*coords[idx]))

corners_gdf = gpd.GeoDataFrame(geometry=corner_pts, crs=grid.crs)
centers_gdf = gpd.GeoDataFrame(geometry=center_pts, crs=grid.crs)
corners_gdf.plot(ax=ax, color='green', markersize=10, label='Corners')
centers_gdf.plot(ax=ax, color='magenta', markersize=10, label='Centers')
plt.show()


# Suppose you have a DataFrame `img_df` with ['filename','lat','lon']
img_df = pd.read_csv("image_coords.csv")  # filename mapped to center coords

# Convert to GeoDataFrame in 4326 and project
imgs_gdf = gpd.GeoDataFrame(
    img_df, geometry=gpd.points_from_xy(img_df.lon, img_df.lat), crs='EPSG:4326'
).to_crs('EPSG:32644')

# Spatial join: only keep images within at least one grid cell
imgs_within_grid = gpd.sjoin(imgs_gdf, grid, predicate='within', how='inner')
print(f'Before filtering: {len(imgs_gdf)}, After filtering: {len(imgs_within_grid)}')