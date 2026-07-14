"""
ШАГ 2 (Google Colab): автоматическая разметка (маски пожаров) через NASA FIRMS,
без ручной обводки огня мышкой.

Как запустить:
    !pip install requests rasterio geopandas shapely numpy

Логика:
    1. NASA FIRMS даёт точки активного огня (hotspots) от спутников MODIS/VIIRS
       с координатами, датой/временем и confidence.
    2. Для каждого патча, скачанного на шаге 1, берём FIRMS-точки в тех же
       границах и в то же время (+-6 часов).
    3. Растеризуем точки в бинарную маску того же размера, что и патч
       (пиксели рядом с точкой = 1 "огонь", остальное = 0 "не огонь").
    4. VIIRS даёт разрешение ~375м, Sentinel-2 - 10м, поэтому одна точка FIRMS
       "разрастается" в маске в область ~375x375м (буфер).

ВАЖНО: это грубая (weak/noisy) разметка, а не пиксель-в-пиксель точная.
Для демо-версии этого достаточно. Для финальной версии стоит подмешать
регионы с известными точными масками (например, из LAFD dataset Pereira et al.)
и/или сделать ручную верификацию небольшой контрольной выборки.
"""

import io
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point

# --- Получи бесплатный MAP_KEY здесь: https://firms.modaps.eosdis.nasa.gov/api/ ---
FIRMS_MAP_KEY = "YOUR_FIRMS_MAP_KEY"

# VIIRS_SNPP_NRT / VIIRS_NOAA20_NRT - реалтайм; для архивных дат используй
# VIIRS_SNPP_SP (standard processing, доступно с задержкой ~2 месяца)
FIRMS_SOURCE = "VIIRS_SNPP_SP"
VIIRS_PIXEL_M = 375  # радиус буфера точки под разрешение VIIRS


def fetch_firms_points(lon_min, lat_min, lon_max, lat_max, date_str: str, days: int = 1) -> pd.DataFrame:
    """
    Скачивает точки активного огня FIRMS в заданном bbox за указанную дату.

    date_str: "YYYY-MM-DD"
    """
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/"
        f"{FIRMS_SOURCE}/{lon_min},{lat_min},{lon_max},{lat_max}/{days}/{date_str}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    return df


def rasterize_fire_mask(points_df: pd.DataFrame, reference_tif_path: str, buffer_m: int = VIIRS_PIXEL_M) -> np.ndarray:
    """
    Растеризует точки FIRMS в бинарную маску той же формы/проекции, что и
    референсный GeoTIFF (патч Sentinel-2, скачанный на шаге 1).
    """
    with rasterio.open(reference_tif_path) as src:
        transform = src.transform
        out_shape = (src.height, src.width)
        crs = src.crs

    if points_df.empty:
        return np.zeros(out_shape, dtype=np.uint8)

    # Переводим точки в проекцию патча и добавляем буфер под разрешение VIIRS
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
    print(f"Маска сохранена: {out_path} (пикселей огня: {mask.sum()})")


if __name__ == "__main__":
    # Пример использования для одного патча — повтори для каждого случая
    # из FIRE_CASES в 01_gee_download.py.
    example = dict(
        name="kostanay_2023_08",
        date_str="2023-08-15",
        bbox=(63.55, 53.15, 63.70, 53.28),  # lon_min, lat_min, lon_max, lat_max
        reference_tif="data/kostanay_2023_08.tif",  # скачан на шаге 1
    )

    points = fetch_firms_points(*example["bbox"], date_str=example["date_str"])
    print(f"Найдено точек FIRMS: {len(points)}")

    mask = rasterize_fire_mask(points, example["reference_tif"])
    save_mask(mask, example["reference_tif"], f"data/{example['name']}_mask.tif")
