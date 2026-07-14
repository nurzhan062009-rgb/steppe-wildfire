"""
ОДНА ЯЧЕЙКА: генерация масок пожара через NASA FIRMS для обоих случаев.
Вставь целиком в Colab и запусти. Предполагается, что:
  - GeoTIFF уже лежат в /content/drive/MyDrive/steppe_wildfire/
  - Google Drive уже примонтирован (drive.mount('/content/drive'))
  - data/images/*.npy уже созданы (шаг конвертации, который ты уже сделал)
"""

!pip install rasterio geopandas shapely --quiet

import io
import os
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point
import geopandas as gpd

# --- Настройки ---
FIRMS_MAP_KEY = "YOUR_FIRMS_MAP_KEY"
FIRMS_SOURCE = "VIIRS_SNPP_SP"
VIIRS_PIXEL_M = 375  # радиус буфера точки под разрешение VIIRS

os.makedirs("data/images", exist_ok=True)
os.makedirs("data/masks", exist_ok=True)


# --- Функции ---
def fetch_firms_points(lon_min, lat_min, lon_max, lat_max, date_str, days=1):
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/"
        f"{FIRMS_SOURCE}/{lon_min},{lat_min},{lon_max},{lat_max}/{days}/{date_str}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def rasterize_fire_mask(points_df, reference_tif_path, buffer_m=VIIRS_PIXEL_M):
    with rasterio.open(reference_tif_path) as src:
        transform = src.transform
        out_shape = (src.height, src.width)
        crs = src.crs

    if points_df.empty:
        return np.zeros(out_shape, dtype=np.uint8)

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


# --- Конвертация GeoTIFF -> .npy (на случай если ещё не сделано в этой сессии) ---
cases_info = [
    ("aktobe_2023_08_12", "2023-08-12", (57.30, 50.93, 57.40, 51.01)),
    ("karagandy_2023_08_02", "2023-08-02", (75.91, 49.94, 76.00, 50.02)),
]

for name, date_str, bbox in cases_info:
    tif_path = f"/content/drive/MyDrive/steppe_wildfire/{name}.tif"

    # images/*.npy
    with rasterio.open(tif_path) as src:
        arr = src.read().astype(np.float32)
    np.save(f"data/images/{name}.npy", arr)
    print(f"{name}: снимок сохранён, форма {arr.shape}")

    # masks/*.npy через FIRMS
    points = fetch_firms_points(*bbox, date_str=date_str)
    print(f"{name}: точек FIRMS = {len(points)}")

    mask = rasterize_fire_mask(points, tif_path)
    np.save(f"data/masks/{name}.npy", mask.astype(np.uint8))
    print(f"{name}: маска сохранена, пикселей огня = {mask.sum()} из {mask.size}")
    print("---")

print("Готово. Дальше: python scripts/train.py --data-dir data --in-channels 3")
