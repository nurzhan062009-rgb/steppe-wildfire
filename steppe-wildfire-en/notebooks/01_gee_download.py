"""
STEP 1 (Google Colab, NOT local): download satellite imagery via Google
Earth Engine over known steppe wildfire locations in Kazakhstan.

How to run:
    1. Open Google Colab (colab.research.google.com), new notebook.
    2. !pip install earthengine-api geemap
    3. Paste this file into a cell and run it.
    4. On first run, ee.Authenticate() opens a Google auth window - a
       regular Google account is enough (no special access needed for
       non-commercial research, though Google may ask for a stated purpose).

What the script does:
    - Takes a list of known steppe fires in Kazakhstan (dates + coordinates).
    - For each case, finds the closest-in-time, least-cloudy Sentinel-2 image.
    - Extracts the relevant bands: B12 (SWIR2), B11 (SWIR1), B8 (NIR), B4 (RED).
    - Saves a patch (crop) around the fire location as GeoTIFF to Google Drive.
"""

import ee
import geemap

ee.Authenticate()

# REPLACE with your own real Project ID from Google Cloud Console
# (get one by registering at https://code.earthengine.google.com/register,
# see the note at the end of the file).
ee.Initialize(project="YOUR_GCP_PROJECT_ID")

# List of known steppe fire cases in Kazakhstan.
# Format: (name, "YYYY-MM-DD", lon, lat)
FIRE_CASES = [
    ("kz_2022_09_03", "2022-09-03", 64.02033, 52.50582),
    ("kz_2022_09_02", "2022-09-02", 59.41358, 51.22836),
    ("kz_2022_09_01", "2022-09-01", 55.61998, 50.59920),
    ("kz_2022_07_03", "2022-07-03", 66.37196, 47.08612),
    ("kz_2022_07_04", "2022-07-04", 66.88747, 47.35342),
    ("kz_2022_09_07", "2022-09-07", 81.40237, 48.24867),
    ("kz_2023_07_12", "2023-07-12", 71.61528, 53.26375),
    ("kz_2022_08_24", "2022-08-24", 81.41451, 50.00890),
    ("kz_2023_07_15", "2023-07-15", 79.67358, 51.52944),
    ("kz_2023_07_29", "2023-07-29", 72.24783, 53.44491),
    ("kz_2022_09_22", "2022-09-22", 62.63229, 52.47319),
    ("kz_2022_07_07", "2022-07-07", 76.95801, 48.50639),
    ("kz_2022_07_21", "2022-07-21", 71.22227, 48.69585),
    ("kz_2022_07_11", "2022-07-11", 59.52668, 50.01511),
    ("kz_2023_08_12", "2023-08-12", 58.69904, 50.95348),
    ("kz_2023_07_14", "2023-07-14", 77.17655, 53.77640),
    ("kz_2022_07_17", "2022-07-17", 74.67937, 53.43013),
    ("kz_2023_07_13", "2023-07-13", 74.17462, 51.52138),
    ("kz_2023_07_16", "2023-07-16", 85.48930, 48.42489),
    ("aktobe_2023_08_12", "2023-08-12", 57.34701, 50.96963),
    ("karagandy_2023_08_02", "2023-08-02", 75.95662, 49.97959),
    ("kz_2021_07_02", "2021-07-02", 65.19592, 49.76354),
    ("kz_2021_09_17", "2021-09-17", 81.22962, 49.56231),
    ("kz_2019_07_18", "2019-07-18", 72.43948, 47.37563),
    ("kz_2020_08_06", "2020-08-06", 66.71121, 46.59685),
    ("kz_2020_09_20", "2020-09-20", 73.39212, 51.53109),
    ("kz_2019_07_29", "2019-07-29", 72.66211, 47.30689),
    ("kz_2019_07_20", "2019-07-20", 82.49545, 46.36676),
    ("kz_2019_08_13", "2019-08-13", 76.38600, 49.59104),
    ("kz_2019_08_30", "2019-08-30", 75.65635, 52.93439),
    ("kz_2019_08_11", "2019-08-11", 64.78418, 51.30782),
    ("kz_2019_07_19", "2019-07-19", 60.80383, 49.09750),
    ("kz_2021_07_01", "2021-07-01", 66.35249, 49.78928),
    ("kz_2019_07_23", "2019-07-23", 65.50791, 47.71703),
    ("kz_2020_07_20", "2020-07-20", 79.12323, 46.28012),
    ("kz_2020_08_05", "2020-08-05", 62.42289, 51.29292),
    ("kz_2020_07_15", "2020-07-15", 57.97184, 49.95998),
    ("kz_2020_08_20", "2020-08-20", 77.34579, 47.86561),
    ("kz_2022_09_07_b1", "2022-09-07", 66.16063, 49.93681),
    ("kz_2021_07_31", "2021-07-31", 53.17807, 53.26012),
    ("kz_2021_09_26", "2021-09-26", 66.46098, 48.53818),
    ("kz_2019_08_15", "2019-08-15", 76.64898, 50.82680),
    ("kz_2019_07_27", "2019-07-27", 55.65347, 48.39072),
    ("kz_2019_09_22", "2019-09-22", 75.90692, 50.40398),
    ("kz_2021_07_06", "2021-07-06", 74.25776, 47.16971),
    ("kz_2021_09_02", "2021-09-02", 59.76763, 51.64725),
    ("kz_2019_07_22", "2019-07-22", 72.05093, 46.57810),
    ("kz_2021_07_04", "2021-07-04", 66.33852, 48.32021),
    ("kz_2019_08_01", "2019-08-01", 72.85011, 53.11110),
    ("kz_2020_07_13", "2020-07-13", 61.36283, 49.85951),
    ("kz_2020_09_06", "2020-09-06", 61.04969, 45.96062),
    ("kz_2019_07_16", "2019-07-16", 55.06707, 49.58326),
    ("kz_2019_07_21", "2019-07-21", 59.33092, 49.11683),
    ("kz_2021_08_30", "2021-08-30", 63.02418, 52.98957),
    ("kz_2021_09_03", "2021-09-03", 60.58300, 50.32684),
    ("kz_2019_07_01", "2019-07-01", 60.80542, 49.54708),
    ("kz_2019_08_12", "2019-08-12", 76.34225, 50.70213),
    ("kz_2019_08_18", "2019-08-18", 59.03245, 50.22581),
    ("kz_2019_07_24", "2019-07-24", 64.57635, 50.56408),
    ("kz_2020_07_14", "2020-07-14", 55.11921, 49.74895),
    ("kz_2020_09_25", "2020-09-25", 55.05075, 52.35833),
    ("kz_2019_08_25", "2019-08-25", 60.88680, 51.36588),
    ("kz_2019_07_15", "2019-07-15", 71.99594, 43.60561),
    ("kz_2019_09_28", "2019-09-28", 64.26816, 44.58587),
    ("kz_2020_07_21", "2020-07-21", 80.07718, 46.79924),
    ("kz_2020_07_26", "2020-07-26", 65.99120, 50.25717),
    ("kz_2019_08_31", "2019-08-31", 69.66496, 44.94464),
    ("kz_2021_09_11", "2021-09-11", 79.39551, 48.16122),
    ("kz_2020_09_19", "2020-09-19", 71.65659, 52.92712),
    ("kz_2019_08_27", "2019-08-27", 73.67196, 46.87743),
    ("kz_2020_07_10", "2020-07-10", 59.09552, 52.55722),
    ("kz_2021_08_13", "2021-08-13", 63.48994, 52.02614),
    ("kz_2021_08_29", "2021-08-29", 57.05412, 49.85709),
    ("kz_2020_07_02", "2020-07-02", 75.60966, 47.81419),
    ("kz_2021_09_18", "2021-09-18", 81.64813, 48.13341),
    ("kz_2019_09_19", "2019-09-19", 69.06021, 44.80115),
    ("kz_2021_08_28", "2021-08-28", 63.32569, 52.37029),
    ("kz_2019_07_17", "2019-07-17", 58.97729, 50.08032),
    ("kz_2020_07_18", "2020-07-18", 58.75412, 52.19677),
    ("kz_2019_09_21", "2019-09-21", 57.93655, 48.78669),
    ("kz_2020_07_17", "2020-07-17", 77.53938, 51.31222),
    ("kz_2019_08_07", "2019-08-07", 57.00111, 50.14198),
    ("kz_2021_08_15", "2021-08-15", 67.35385, 55.23957),
]

PATCH_RADIUS_M = 5000
CLOUD_MAX_PCT = 20


def get_sentinel2_patch(lon: float, lat: float, date_str: str, radius_m: int = PATCH_RADIUS_M):
    """
    Finds the closest-in-time, least-cloudy Sentinel-2 image around a point.
    If nothing is found in a +-5 day window, widens the search to +-10, +-15
    days before giving up.
    """
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(radius_m).bounds()
    date = ee.Date(date_str)

    for window_days in (5, 10, 15):
        window_start = date.advance(-window_days, "day")
        window_end = date.advance(window_days, "day")

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(window_start, window_end)
            .filterBounds(region)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_MAX_PCT))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )

        n_found = collection.size().getInfo()
        if n_found > 0:
            if window_days > 5:
                print(f"    (found only after widening the window to +-{window_days} days)")
            image = collection.first()
            bands = image.select(["B12", "B11", "B8", "B4"])
            return bands.clip(region), region

    raise RuntimeError(
        f"No Sentinel-2 image with cloud cover <{CLOUD_MAX_PCT}% found "
        f"even within +-15 days of {date_str} for point ({lon}, {lat})."
    )


def export_patch(image: ee.Image, region: ee.Geometry, name: str, folder: str = "steppe_wildfire"):
    """Starts exporting a patch to Google Drive (async GEE task)."""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=name,
        folder=folder,
        region=region,
        scale=10,
        fileFormat="GeoTIFF",
        maxPixels=1e9,
    )
    task.start()
    print(f"Export started: {name} -> Google Drive/{folder}/. Check progress in the GEE Tasks tab.")
    return task


if __name__ == "__main__":
    if not FIRE_CASES:
        raise ValueError("FIRE_CASES is empty. Add fire cases (name, date, lon, lat) first.")

    # With a large number of cases (60-90+), don't export all at once - you
    # may hit GEE's concurrent-task limits. Export in batches of BATCH_SIZE.
    # Change BATCH_INDEX (0, 1, 2...) and rerun the cell for each next batch.
    BATCH_SIZE = 28
    BATCH_INDEX = 0  # <-- bump to 1, 2, etc. for the next batches

    batch_start = BATCH_INDEX * BATCH_SIZE
    batch_end = batch_start + BATCH_SIZE
    current_batch = FIRE_CASES[batch_start:batch_end]

    print(f"Total cases: {len(FIRE_CASES)}. Batch {BATCH_INDEX}: "
          f"{len(current_batch)} cases (indices {batch_start}-{batch_end-1})")

    for name, date_str, lon, lat in current_batch:
        try:
            patch, region = get_sentinel2_patch(lon, lat, date_str)
            export_patch(patch, region, name)
        except Exception as e:
            print(f"Failed to process {name}: {e}")

    print(
        "\nDone. Next:\n"
        "  1. Wait for tasks to finish in GEE Code Editor -> Tasks (or ee task list).\n"
        "  2. Download the GeoTIFF files from Google Drive.\n"
        "  3. Move on to notebooks/02_firms_labels.py to generate masks.\n"
    )

# NOTE on project ID:
# Since 2024, Earth Engine requires a linked Google Cloud Project (free for
# non-commercial/research use). Create one at https://console.cloud.google.com/
# -> New Project -> then register it for Earth Engine at
# https://code.earthengine.google.com/register
