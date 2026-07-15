"""
ONE CELL: searches for real steppe fire cases across SEVERAL fire seasons
(multiple years x multiple months) in one run, instead of scanning one
period at a time by hand. Goal: gather ~80 good candidates in a single run.

Paste this whole thing into Colab and run. May take a few minutes (many
requests to FIRMS).
"""

import io
import time
import datetime
import numpy as np
import pandas as pd
import requests

FIRMS_MAP_KEY = "YOUR_FIRMS_MAP_KEY"
FIRMS_SOURCE = "VIIRS_SNPP_SP"  # archive data, available roughly 2012 - early 2024
KZ_BBOX = (46.0, 40.5, 87.5, 55.5)

OIL_REGIONS_BBOX = [
    (50.0, 42.0, 59.5, 46.5, "Mangystau region"),
    (46.5, 44.0, 54.5, 48.0, "Atyrau region"),
    (48.0, 48.5, 55.0, 52.0, "West Kazakhstan (Karachaganak)"),
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
            print(f"  error {date_str}: {e}")
        current += datetime.timedelta(days=chunk_days)
        remaining -= chunk_days
        time.sleep(0.5)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# Periods to scan: 5 years x fire-prone months (July-September)
PERIODS = [
    ("2019-07-01", 92),
    ("2020-07-01", 92),
    ("2021-07-01", 92),
    ("2022-07-01", 92),
    ("2023-07-01", 92),
]
# VIIRS_SNPP_SP archive is available roughly from 2012 to early 2024 -
# this range is well covered. For dates after early 2024, use
# VIIRS_SNPP_NRT / VIIRS_NOAA20_NRT instead of _SP.

TARGET_N_CASES = 80

all_points = []
for start_date, total_days in PERIODS:
    print(f"Fetching period {start_date} + {total_days} days...")
    df = fetch_period(start_date, total_days)
    print(f"  points received: {len(df)}")
    all_points.append(df)

df_all = pd.concat(all_points, ignore_index=True)
print(f"\nTotal points across all periods: {len(df_all)}")

df_all["in_oil"] = df_all.apply(lambda r: is_in_oil_region(r["latitude"], r["longitude"]), axis=1)
df_all["lat_r"] = df_all["latitude"].round(1)
df_all["lon_r"] = df_all["longitude"].round(1)
n_dates = df_all.groupby(["lat_r", "lon_r"])["acq_date"].transform("nunique")
df_all["stationary"] = n_dates >= 2

clean = df_all[~df_all["in_oil"] & ~df_all["stationary"]].copy()
print(f"After filtering (no oil/flares): {len(clean)} points")

clean = clean.sort_values("frp", ascending=False)
clean = clean.drop_duplicates(subset=["acq_date"], keep="first")  # 1 best point per date
topN = clean.head(TARGET_N_CASES)[["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp"]]

print(f"\n=== TOP-{len(topN)} CANDIDATES (different dates, no oil zones, not stationary) ===")
print(topN.to_string(index=False))

print("\nReady-to-use FIRE_CASES block (still eyeball-check coordinates first!):")
print("FIRE_CASES = [")
for _, row in topN.iterrows():
    name = f"kz_{row['acq_date'].replace('-', '_')}"
    print(f'    ("{name}", "{row["acq_date"]}", {row["longitude"]:.5f}, {row["latitude"]:.5f}),')
print("]")
