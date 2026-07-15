"""
STEP 2 (Google Colab): automatic fire mask generation via NASA FIRMS,
no manual annotation needed.

How to run:
    !pip install requests rasterio geopandas shapely numpy

Logic:
    1. NASA FIRMS gives active-fire hotspots (MODIS/VIIRS) with
       coordinates, date/time, and confidence.
    2. For each patch downloaded in step 1, take the FIRMS points within
       the same bounds and roughly the same time.
    3. Rasterize the points into a binary mask the same size as the patch
       (pixels near a point = 1 "fire", everything else = 0).
    4. VIIRS resolution is ~375m vs Sentinel-2's 10m, so each FIRMS point
       is buffered into a ~375x375m area on the mask.

NOTE: this is weak/noisy labeling, not pixel-perfect. Good enough for a
pilot study. For a final version, consider mixing in regions with known
precise masks (e.g. from the LAFD dataset) and/or manually verifying a
small control sample.
"""

import io
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point

FIRMS_MAP_KEY = "YOUR_FIRMS_MAP_KEY"

FIRMS_SOURCE = "VIIRS_SNPP_SP"
VIIRS_PIXEL_M = 375


def fetch_firms_points(lon_min, lat_min, lon_max, lat_max, date_str: str, days: int = 1) -> pd.DataFrame:
    """Downloads active-fire FIRMS points within a bbox for a given date."""
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/"
        f"{FIRMS_SOURCE}/{lon_min},{lat_min},{lon_max},{lat_max}/{days}/{date_str}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    return df


def rasterize_fire_mask(points_df: pd.DataFrame, reference_tif_path: str, buffer_m: int = VIIRS_PIXEL_M) -> np.ndarray:
    """Rasterizes FIRMS points into a binary mask matching a reference GeoTIFF."""
    with rasterio.open(reference_tif_path) as src:
        transform = src.transform
        out_shape = (src.height, src.width)
        crs = src.crs

    if points_df.empty:
        return np.zeros(out_shape, dtype=np.uint8)

    import geopandas as gpd
    gdf = gpd.GeoDataFrame(
        points_df,
        geometry=[Point(xy) for xy in zip(points_df.longitude, points_df.latitude)],
        crs="EPSG:4326",
    ).to_crs(crs)
    gdf["geometry"] = gdf.geometry.buffer(buffer_m)

    mask = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask


def save_mask(mask: np.ndarray, reference_tif_path: str, out_path: str):
    with rasterio.open(reference_tif_path) as src:
        profile = src.profile
    profile.update(count=1, dtype=rasterio.uint8)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)
    print(f"Mask saved: {out_path} (fire pixels: {mask.sum()})")
