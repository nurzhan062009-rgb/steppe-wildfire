"""
Find real fire cases via the NASA FIRMS archive (instead of guessing from
news articles). Run in Google Colab.

Idea: instead of digging coordinates out of news articles (usually not
given), download ALL active-fire points from FIRMS over Kazakhstan for a
chosen period, then pick clusters located in open steppe (not forest, not
near towns - verifiable by eye on a map).

Get a free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/
"""

import io
import pandas as pd
import requests

FIRMS_MAP_KEY = "YOUR_FIRMS_MAP_KEY"

# Rough bounding box covering all of Kazakhstan
KZ_BBOX = (46.0, 40.5, 87.5, 55.5)  # lon_min, lat_min, lon_max, lat_max

# Main steppe/dry regions to look at on the map afterward:
# Kostanay, Akmola, Pavlodar, Karaganda, West Kazakhstan, Aktobe


def fetch_country_archive(source: str, start_date: str, days: int = 5) -> pd.DataFrame:
    """
    Downloads FIRMS points over the Kazakhstan bbox for a period.

    source: "VIIRS_SNPP_SP" (archive data, for dates older than ~2 months)
    start_date: "YYYY-MM-DD"
    days: how many days to pull. NOTE: FIRMS Area API has a hard limit of
          1-5 days per request (not 10!). For a longer period, make several
          sequential requests and concatenate (see fetch_multi_week below).
    """
    lon_min, lat_min, lon_max, lat_max = KZ_BBOX
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/"
        f"{source}/{lon_min},{lat_min},{lon_max},{lat_max}/{days}/{start_date}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    return df


def fetch_multi_week(source: str, start_date: str, total_days: int = 31) -> pd.DataFrame:
    """Pulls an arbitrary period (e.g. a full month) in 5-day chunks and concatenates."""
    import datetime
    import time

    chunks = []
    current = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    remaining = total_days

    while remaining > 0:
        chunk_days = min(5, remaining)
        date_str = current.strftime("%Y-%m-%d")
        print(f"  Fetching {date_str} + {chunk_days} days...")
        try:
            df_chunk = fetch_country_archive(source, date_str, days=chunk_days)
            chunks.append(df_chunk)
        except requests.HTTPError as e:
            print(f"  Error on {date_str}: {e}")
        current += datetime.timedelta(days=chunk_days)
        remaining -= chunk_days
        time.sleep(1)  # polite pause to avoid hitting the rate limit

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


# Known oil/gas regions in Kazakhstan (rough bboxes) where VIIRS mostly
# picks up gas-flaring, not wildfires. Not scientifically precise, just a
# fast practical geographic filter.
OIL_REGIONS_BBOX = [
    (50.0, 42.0, 59.5, 46.5, "Mangystau region (Uzen/Zhetybai/Buzachi/Karakiya)"),
    (46.5, 44.0, 54.5, 48.0, "Atyrau region (Tengiz/Kashagan/Prorva)"),
    (48.0, 48.5, 55.0, 52.0, "West Kazakhstan (Karachaganak)"),
]


def is_in_oil_region(lat: float, lon: float) -> bool:
    for lon_min, lat_min, lon_max, lat_max, _name in OIL_REGIONS_BBOX:
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return True
    return False


def filter_likely_flares(df: pd.DataFrame, round_decimals: int = 1) -> pd.DataFrame:
    """
    Filters out likely gas flares with two heuristics:
      1) Point falls in a known oil/gas region (see OIL_REGIONS_BBOX).
      2) Point is "stationary" - roughly the same coordinates (rounded to
         ~1km) show up on 2+ different dates. A real steppe fire flares up
         and dies out in 1-3 days rather than sitting in the same spot for
         weeks - a flare burns continuously.
    """
    df = df.copy()
    df["in_oil_region"] = df.apply(lambda r: is_in_oil_region(r["latitude"], r["longitude"]), axis=1)

    df["lat_round"] = df["latitude"].round(round_decimals)
    df["lon_round"] = df["longitude"].round(round_decimals)
    n_unique_dates = df.groupby(["lat_round", "lon_round"])["acq_date"].transform("nunique")
    df["is_stationary"] = n_unique_dates >= 2

    n_before = len(df)
    filtered = df[~df["in_oil_region"] & ~df["is_stationary"]].drop(
        columns=["in_oil_region", "is_stationary", "lat_round", "lon_round"]
    )
    print(f"Filtered: {n_before} -> {len(filtered)} points "
          f"(removed oil regions + stationary flares)")
    return filtered


def find_high_confidence_clusters(df: pd.DataFrame, min_confidence: str = "n") -> pd.DataFrame:
    """Filters by detection confidence and sorts by FRP (Fire Radiative Power)."""
    if "confidence" in df.columns:
        conf_order = {"l": 0, "n": 1, "h": 2}
        min_level = conf_order.get(min_confidence, 1)
        df = df[df["confidence"].map(lambda c: conf_order.get(c, 0)) >= min_level]

    df = df.sort_values("frp", ascending=False)
    return df[["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp"]]


if __name__ == "__main__":
    df = fetch_multi_week(source="VIIRS_SNPP_SP", start_date="2023-08-01", total_days=31)
    print(f"\nTotal FIRMS points over Kazakhstan for August 2023: {len(df)}")

    df_clean = filter_likely_flares(df)
    top_clusters = find_high_confidence_clusters(df_clean, min_confidence="n")
    print("\nTop 20 likely real fires (by FRP), with coordinates:")
    print(top_clusters.head(20).to_string(index=False))
    print(
        "\nThis is already filtered (no known oil regions, no stationary "
        "points). The heuristic is rough though - still worth a quick "
        "eyeball check on Google Maps before exporting imagery."
    )

    print(
        "\nNext steps:\n"
        "  1. Take 5-10 coordinates from top_clusters with the highest FRP.\n"
        "  2. Check them on Google Maps/Earth - confirm it's OPEN STEPPE\n"
        "     (grassland/field), not forest and not a populated area.\n"
        "  3. Add these (acq_date, longitude, latitude) to FIRE_CASES\n"
        "     in 01_gee_download.py.\n"
    )
