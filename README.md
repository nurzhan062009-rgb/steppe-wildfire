# Steppe Wildfire Segmentation (Kazakhstan)

Adapting active-fire segmentation models (U-Net on multispectral Sentinel-2
imagery) to steppe wildfires in Kazakhstan, where models trained on forest
fires (US/Europe) lose accuracy due to thin, fast-moving fire fronts and
false positives on sun-heated soil/salt flats.

## Final results

Dataset: **82 real cases** of steppe wildfires across Kazakhstan
(2019-2023), cut into **3097 patches** of 256x256 (Sentinel-2 + automatic
labeling via NASA FIRMS, with oil-flare filtering). Train/val split by
source image (66/16), no data leakage between patches of the same fire.

| Model | Parameters | Best val IoU |
|---|---|---|
| Baseline U-Net (from scratch) | 7.77M | **0.3454** |
| U-Net + ResNet34 (ImageNet, transfer learning) | 24.4M | **0.3624** |

Transfer learning with an ImageNet-pretrained encoder gives a small but
consistent improvement over training from scratch — consistent with the
expectation that pretrained representations help when remote-sensing data
is limited.

## Dataset regions

- **Western Kazakhstan** (Aktobe, West Kazakhstan, Atyrau regions) — dry
  steppe and semi-desert, the most frequent summer fire activity
- **Central Kazakhstan** (Karaganda, Ulytau regions) — open Sary-Arka
  grasslands
- **Northern/Pavlodar region** (Kostanay, Akmola, Pavlodar regions)
- **Eastern Kazakhstan** (Abai, East Kazakhstan regions) — including the
  relict pine-ribbon-forest zone
- **Southern Kazakhstan** (Kyzylorda, Turkestan regions) — steppe/semi-desert
  transition zone near Betpak-Dala

## Project structure

```
steppe-wildfire/
├── models/
│   └── unet.py              # U-Net architecture (3 or 5 input channels)
├── scripts/
│   ├── spectral_indices.py  # NDVI, NBR computation
│   ├── patchify.py          # slices large images into 256x256 patches
│   └── train.py             # training loop, cross-validation, synthetic sanity-check
├── notebooks/
│   ├── 00_find_real_cases.py   # find real fire cases via NASA FIRMS archive
│   ├── 05_bulk_find_cases.py   # same, scanning multiple years at once
│   ├── 01_gee_download.py      # Colab only: download Sentinel-2 via GEE
│   ├── 02_firms_labels.py      # Colab only: generate masks via NASA FIRMS
│   ├── 03_all_in_one_masks.py  # single-cell version for a couple of cases
│   ├── 06_bulk_masks.py        # mask generation for all collected cases
│   └── 04_upload_and_train.py  # upload zip + run training in one cell
├── SETUP.md                 # step-by-step reproduction guide
├── research_draft.md        # short research write-up with final results
└── data/                    # real image/mask patches go here (not included)
```

## Quick local check (no real data needed)

```
python scripts/train.py --synthetic --epochs 5 --in-channels 3
python scripts/train.py --synthetic --epochs 5 --in-channels 5
```

Runs the full pipeline (Dataset -> model -> loss -> IoU) on synthetic data
to confirm everything works end-to-end before touching GEE/FIRMS.

## Reproducing with real data

See `SETUP.md` for the full step-by-step guide (Google Earth Engine setup,
NASA FIRMS API key, running order for all scripts).

## Scientific framing

We are not claiming satellite-based fire detection as a new task — this is
an active research area (Pereira et al. 2021, Fusioka et al. 2024, and
others). Our contribution is a data collection + labeling pipeline for
Kazakhstani steppe wildfires (GEE + NASA FIRMS, with automatic oil-flare
filtering), and an empirical check on the benefit of transfer learning
under limited regional data.

## References

1. de Almeida Pereira, G. H., Fusioka, A. M., Nassu, B. T., & Minetto, R.
   (2021). Active fire detection in Landsat-8 imagery: A large-scale dataset
   and a deep-learning study. *ISPRS J. Photogrammetry and Remote Sensing*,
   178, 171-186.
2. Fusioka, A. M., et al. (2024). Active Fire Segmentation: A Transfer
   Learning Study From Landsat-8 to Sentinel-2. *IEEE JSTARS*, 17,
   14093-14108.
3. Chuvieco, E., et al. (2019). Historical background and current
   developments for mapping burned area from satellite Earth observation.
   *Remote Sensing of Environment*, 225, 45-64.
   DOI: 10.1016/j.rse.2019.02.013
4. Spatiotemporal Fire Risk Index for Kazakhstan integrating machine
   learning and remote sensing (2025). *Frontiers in Forests and Global
   Change*. https://doi.org/10.3389/ffgc.2025.1680856
5. Mapping fire hazard potential in Kazakhstan: a machine learning and
   remote sensing perspective (2025). *International Journal of Wildland
   Fire*. https://doi.org/10.1071/WF24232
