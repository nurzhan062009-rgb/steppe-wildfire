"""
ОДНА ЯЧЕЙКА: ищет реальные случаи степных пожаров сразу за НЕСКОЛЬКО
пожароопасных периодов (несколько лет x несколько месяцев), а не за один
период вручную. Цель - набрать ~20 хороших кандидатов за один запуск.

Вставь целиком в Colab и запусти. Может занять несколько минут (много
запросов к FIRMS).
"""

import io
import time
import datetime
import numpy as np
import pandas as pd
import requests

FIRMS_MAP_KEY = "YOUR_FIRMS_MAP_KEY"
FIRMS_SOURCE = "VIIRS_SNPP_SP"  # архивные данные, есть с 2012 по начало 2024
KZ_BBOX = (46.0, 40.5, 87.5, 55.5)

# Нефтегазоносные регионы - исключаем (факелы)
OIL_REGIONS_BBOX = [
    (50.0, 42.0, 59.5, 46.5, "Мангистауская обл."),
    (46.5, 44.0, 54.5, 48.0, "Атырауская обл."),
    (48.0, 48.5, 55.0, 52.0, "ЗКО (Карачаганак)"),
]


def is_in_oil_region(lat, lon):
    return any(lon_min <= lon <= lon_max and lat_min <= lat <= lat_max
               for lon_min, lat_min, lon_max, lat_max, _ in OIL_REGIONS_BBOX)


def fetch_chunk(start_date: str, days: int) -> pd.DataFrame:
    lon_min, lat_min, lon_max, lat_max = KZ_BBOX
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/"
           f"{FIRMS_SOURCE}/{lon_min},{lat_min},{lon_max},{lat_max}/{days}/{start_date}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def fetch_period(start_date: str, total_days: int) -> pd.DataFrame:
    chunks = []
    current = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    remaining = total_days
    while remaining > 0:
        chunk_days = min(5, remaining)
        date_str = current.strftime("%Y-%m-%d")
        try:
            chunks.append(fetch_chunk(date_str, chunk_days))
        except requests.HTTPError as e:
            print(f"  ошибка {date_str}: {e}")
        current += datetime.timedelta(days=chunk_days)
        remaining -= chunk_days
        time.sleep(0.5)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# --- Периоды для сканирования: 5 лет x пожароопасные месяцы (июль-сентябрь) ---
PERIODS = [
    ("2019-07-01", 92),
    ("2020-07-01", 92),
    ("2021-07-01", 92),
    ("2022-07-01", 92),
    ("2023-07-01", 92),
]
# VIIRS_SNPP_SP архив доступен примерно с 2012 по начало 2024 - весь этот
# диапазон покрыт с запасом. Для дат позже начала 2024 понадобится
# VIIRS_SNPP_NRT / VIIRS_NOAA20_NRT вместо _SP.

TARGET_N_CASES = 80  # сколько кандидатов хотим получить на выходе

all_points = []
for start_date, total_days in PERIODS:
    print(f"Качаю период {start_date} + {total_days} дн...")
    df = fetch_period(start_date, total_days)
    print(f"  получено точек: {len(df)}")
    all_points.append(df)

df_all = pd.concat(all_points, ignore_index=True)
print(f"\nВсего точек за все периоды: {len(df_all)}")

# --- Фильтрация: убираем нефтяные регионы + стационарные точки (флаг вместо огня) ---
df_all["in_oil"] = df_all.apply(lambda r: is_in_oil_region(r["latitude"], r["longitude"]), axis=1)
df_all["lat_r"] = df_all["latitude"].round(1)
df_all["lon_r"] = df_all["longitude"].round(1)
n_dates = df_all.groupby(["lat_r", "lon_r"])["acq_date"].transform("nunique")
df_all["stationary"] = n_dates >= 2

clean = df_all[~df_all["in_oil"] & ~df_all["stationary"]].copy()
print(f"После фильтрации (без нефти/факелов): {len(clean)} точек")

# --- Сортируем по FRP, берём top-30, разносим по разным датам чтобы не было дублей ---
clean = clean.sort_values("frp", ascending=False)
clean = clean.drop_duplicates(subset=["acq_date"], keep="first")  # 1 лучшая точка на дату
topN = clean.head(TARGET_N_CASES)[["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp"]]

print(f"\n=== ТОП-{len(topN)} КАНДИДАТОВ (разные даты, вне нефтяных зон, не стационарные) ===")
print(topN.to_string(index=False))

print("\nГотовый список для FIRE_CASES (проверь координаты глазами перед использованием!):")
print("FIRE_CASES = [")
for _, row in topN.iterrows():
    name = f"kz_{row['acq_date'].replace('-', '_')}"
    print(f'    ("{name}", "{row["acq_date"]}", {row["longitude"]:.5f}, {row["latitude"]:.5f}),')
print("]")
