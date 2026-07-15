# Steppe Wildfire Segmentation: Adapting Active-Fire Models to Kazakhstan

## Abstract

Satellite-based active-fire segmentation models (U-Net on multispectral
Sentinel-2 imagery) are mostly developed and validated on forest fires
(US, Siberia). We collect a dataset of Kazakhstani steppe wildfires — 82
cases from 2019-2023, automatically labeled via NASA FIRMS, and cut them
into 3097 patches of 256x256 for training. We compare a baseline U-Net
trained from scratch against a version using a pretrained (ImageNet)
ResNet34 encoder — i.e. transfer learning. The baseline reaches val
IoU=0.3454, the transfer-learning version reaches val IoU=0.3624. The gap
is small but consistent, matching the expectation that pretrained
representations help under limited remote-sensing data.

## 1. Introduction

Satellite-based active-fire detection is an active research area (Pereira
et al. 2021, Fusioka et al. 2024). A known problem: models trained on
forest fires perform worse on grassland ecosystems — short burn duration,
thin fast-moving fire fronts, and false positives from sun-heated soil and
salt flats. As far as we know, no such validation has been done for
Kazakhstani steppes.

Our contribution: a data collection and labeling pipeline for Kazakhstani
steppe wildfires (GEE + NASA FIRMS, with automatic oil-flare filtering),
and an empirical check on the benefit of transfer learning under limited
regional data.

## 2. Data

| | |
|---|---|
| Imagery source | Sentinel-2 SR Harmonized (Google Earth Engine) |
| Bands | B12 (SWIR2), B11 (SWIR1), B8 (NIR), B4 (RED) |
| Labeling | NASA FIRMS VIIRS_SNPP_SP, 375m buffer around each hotspot |
| Cases (source images) | 82, after filtering |
| Patches after slicing (256x256) | 3097 (2001 with fire, 1096 without) |
| Period | July-September, 2019-2023 |
| Train/Val | 66/16 source images (2508/589 patches), split by source |

FIRMS points located in known oil/gas regions (Mangystau, Atyrau, West
Kazakhstan/Karachaganak) and points with coordinates repeating across
multiple dates (a signature of gas flares rather than wildfires) were
excluded during case selection.

Regions covered: western (Aktobe, West Kazakhstan, Atyrau), central
(Karaganda, Ulytau), northern/Pavlodar (Kostanay, Akmola, Pavlodar),
eastern (Abai, East Kazakhstan, including the pine-ribbon-forest zone),
and southern (Kyzylorda, Turkestan).

## 3. Method

Train/val splitting and the subsequent leave-one-source-out cross-
validation are done by source image, not by individual patch — otherwise
overlapping patches from the same fire would leak between train and val.

- **Baseline**: U-Net from scratch, 4 encoder/decoder levels,
  base_filters=32, 7.77M parameters.
- **Transfer learning**: U-Net with a ResNet34 encoder pretrained on
  ImageNet, 24.4M parameters.
- Input: 3 channels [SWIR2, SWIR1, NIR], normalized by /10000 (Sentinel-2
  SR digital numbers -> reflectance). The extended version adds NDVI and
  NBR as input channels.
- Augmentation: random 90-degree rotations/flips — valid for satellite
  imagery since there's no fixed "up".
- Training: Adam, ReduceLROnPlateau, weight decay, gradient clipping,
  BCEWithLogitsLoss, early stopping on val IoU.

## 4. Results

| Model | Parameters | Best val IoU |
|---|---|---|
| Baseline U-Net (from scratch) | 7.77M | 0.3454 |
| U-Net + ResNet34 (transfer learning) | 24.4M | 0.3624 |

Both models were trained on the same train/val split (66/16 source
images). Transfer learning gives a small but reproducible improvement. The
gap is modest — expected given the data volume — and we do not claim it as
statistically conclusive (see Limitations), but the direction of the
effect matches expectations from the literature.

## 5. Limitations

- **Weak-label supervision**: masks are a 375m buffer around each FIRMS
  point, not hand-drawn segmentation. This is a noisy but reproducible
  labeling method that requires no manual annotation.
- **Temporal gap**: a Sentinel-2 image may be captured up to 15 days after
  the FIRMS detection date — the peak of the fire itself is not always
  visible.
- **Dataset size**: 82 cases is already an order of magnitude above a
  typical pilot study, but still smaller than the tens of thousands of
  patches used in the original active-fire detection literature. The
  baseline-vs-transfer-learning gap may be sensitive to random
  initialization/split at this scale.

## 6. Future work

1. Manual verification of a subsample of masks instead of relying purely
   on automatic FIRMS labeling.
2. Expanding the dataset with additional seasons/regions.
3. Direct comparison against the zero-shot performance of a model
   pretrained on forest fires (without any fine-tuning on steppe data) —
   quantifying the original domain gap.

## References

1. de Almeida Pereira, G. H., Fusioka, A. M., Nassu, B. T., & Minetto, R.
   (2021). Active fire detection in Landsat-8 imagery: A large-scale dataset
   and a deep-learning study. *ISPRS Journal of Photogrammetry and Remote
   Sensing*, 178, 171-186.
2. Fusioka, A. M., et al. (2024). Active Fire Segmentation: A Transfer
   Learning Study From Landsat-8 to Sentinel-2. *IEEE JSTARS*, 17,
   14093-14108.
3. Chuvieco, E., et al. (2019). Historical background and current
   developments for mapping burned area from satellite Earth observation.
   *Remote Sensing of Environment*, 225, 45-64.
4. Spatiotemporal Fire Risk Index for Kazakhstan integrating machine
   learning and remote sensing (2025). *Frontiers in Forests and Global
   Change*. https://doi.org/10.3389/ffgc.2025.1680856
5. Mapping fire hazard potential in Kazakhstan: a machine learning and
   remote sensing perspective (2025). *International Journal of Wildland
   Fire*. https://doi.org/10.1071/WF24232
