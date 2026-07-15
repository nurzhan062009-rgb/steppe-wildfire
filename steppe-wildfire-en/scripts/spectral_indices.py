"""NDVI/NBR for the extended 5-channel model input."""

import numpy as np

EPS = 1e-8


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - RED) / (NIR + RED). Range [-1, 1]."""
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return (nir - red) / (nir + red + EPS)


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """NBR = (NIR - SWIR2) / (NIR + SWIR2). Range [-1, 1]."""
    nir = nir.astype(np.float32)
    swir2 = swir2.astype(np.float32)
    return (nir - swir2) / (nir + swir2 + EPS)


def build_extended_input(swir2: np.ndarray, swir1: np.ndarray, nir: np.ndarray,
                          red: np.ndarray) -> np.ndarray:
    """[SWIR2, SWIR1, NIR, NDVI, NBR] -> (5, H, W) array for WildfireUNet(in_channels=5)."""
    ndvi_band = ndvi(nir, red)
    nbr_band = nbr(nir, swir2)
    stacked = np.stack([swir2, swir1, nir, ndvi_band, nbr_band], axis=0)
    return stacked.astype(np.float32)


if __name__ == "__main__":
    cases = {
        "dry steppe":       dict(swir2=0.25, swir1=0.30, nir=0.35, red=0.28),
        "active fire":      dict(swir2=0.75, swir1=0.65, nir=0.30, red=0.20),
        "hot salt flat":    dict(swir2=0.40, swir1=0.42, nir=0.45, red=0.40),
    }

    for name, bands in cases.items():
        arrs = {k: np.full((4, 4), v) for k, v in bands.items()}
        stacked = build_extended_input(**arrs)
        print(f"{name:15s} -> NDVI={stacked[3,0,0]:.3f}  NBR={stacked[4,0,0]:.3f}")
