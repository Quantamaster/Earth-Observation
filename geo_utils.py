# utils/geo_utils.py

import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

def mark_grid_centers_and_corners(grid: gpd.GeoDataFrame, output_path="grid_with_centers.png"):
    """Mark the corners and center of each grid cell."""
    corners = []
    centers = []

    for poly in grid.geometry:
        bounds = poly.bounds  # (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = bounds

        # Four corners
        corners.extend([
            Point(minx, miny),
            Point(minx, maxy),
            Point(maxx, miny),
            Point(maxx, maxy)
        ])

        # Center
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        centers.append(Point(center_x, center_y))

    corners_gdf = gpd.GeoDataFrame(geometry=corners, crs=grid.crs)
    centers_gdf = gpd.GeoDataFrame(geometry=centers, crs=grid.crs)

    # Plot grid, corners, centers
    ax = grid.boundary.plot(edgecolor='gray', figsize=(10, 10))
    corners_gdf.plot(ax=ax, color='blue', markersize=10, label='Corners')
    centers_gdf.plot(ax=ax, color='red', markersize=20, label='Centers')
    plt.legend()
    plt.title("Grid with Corners (Blue) and Centers (Red)")
    plt.savefig(output_path)
    plt.show()
