# Reproducing this project from scratch

The full pipeline runs in Google Colab (GEE and Google Drive need a Google
account). Locally you can only run `models/`, `scripts/train.py --synthetic`
and `scripts/spectral_indices.py` — those are plain PyTorch/NumPy, no
internet required.

## What you need

- A Google account
- A Google Cloud project registered for Earth Engine (free):
  https://code.earthengine.google.com/register — choose Unpaid usage ->
  Academic/Individual -> Community Tier
- A free NASA FIRMS API key: https://firms.modaps.eosdis.nasa.gov/api/

## Running order in Colab

1. **`notebooks/00_find_real_cases.py`** — searches the NASA FIRMS archive
   for real fire cases, filtering out oil flares and stationary sources.
   Replace `YOUR_FIRMS_MAP_KEY` with your own key. For a larger batch of
   cases use `notebooks/05_bulk_find_cases.py` (scans multiple years at
   once).

2. **`notebooks/01_gee_download.py`** — downloads Sentinel-2 imagery via
   Earth Engine. You need to:
   - Set `ee.Initialize(project="...")` to your own GCP project ID
   - Fill `FIRE_CASES` with the coordinates from step 1
   - For a large number of cases (50+), export in batches via
     `BATCH_INDEX` (see the comment in the file), or you'll hit GEE's
     concurrent-task limits

3. **`notebooks/06_bulk_masks.py`** — converts the downloaded GeoTIFFs to
   `.npy` and generates fire masks via FIRMS (a buffer around each
   hotspot). Also needs a FIRMS key. `FIRE_CASES` must match step 2.

4. **`scripts/patchify.py --data-dir data --out-dir data_patches`** — cuts
   the large images into 256x256 patches. Runs locally, no internet
   needed.

5. **`scripts/train.py`** — training:
   ```
   python scripts/train.py --data-dir data_patches --in-channels 3 --crop-size 256 --epochs 50 --save-path model_baseline.pth
   python scripts/train.py --data-dir data_patches --in-channels 5 --crop-size 256 --epochs 50 --save-path model_ndvi_nbr.pth
   ```
   `--in-channels 3` is the baseline (SWIR2, SWIR1, NIR), `5` adds NDVI/NBR.
   Training on patches will be slow on CPU — switch the Colab runtime to
   GPU (Runtime -> Change runtime type -> T4 GPU).

## Quick check without real data

```
python scripts/train.py --synthetic --epochs 5 --in-channels 3
```
Runs the whole pipeline (Dataset -> model -> loss -> IoU) on synthetic data
— useful to confirm the code isn't broken before going to GEE/FIRMS.
