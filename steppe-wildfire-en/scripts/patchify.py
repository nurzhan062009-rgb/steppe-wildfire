"""Slices large satellite images (~1000x1000) into 256x256 training patches."""

import os
import numpy as np
from pathlib import Path

PATCH_SIZE = 256
STRIDE = 128
MIN_FIRE_PIXELS = 50
NEG_TO_POS_RATIO = 1.0  # keep roughly 1 no-fire patch per 1 fire patch, for balance


def extract_patches(img: np.ndarray, mask: np.ndarray, patch_size=PATCH_SIZE, stride=STRIDE):
    """img: (C, H, W), mask: (H, W). Returns a list of (patch_img, patch_mask)."""
    C, H, W = img.shape
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            p_img = img[:, y:y+patch_size, x:x+patch_size]
            p_mask = mask[y:y+patch_size, x:x+patch_size]
            patches.append((p_img, p_mask))
    return patches


def patchify_dataset(data_dir: str, out_dir: str, patch_size: int = PATCH_SIZE,
                      stride: int = STRIDE, seed: int = 42):
    rng = np.random.default_rng(seed)
    image_dir = Path(data_dir) / "images"
    mask_dir = Path(data_dir) / "masks"
    out_img_dir = Path(out_dir) / "images"
    out_mask_dir = Path(out_dir) / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f.stem for f in image_dir.glob("*.npy"))
    print(f"Source images: {len(files)}")

    total_pos, total_neg_kept, total_neg_dropped = 0, 0, 0

    for stem in files:
        img = np.load(image_dir / f"{stem}.npy")
        mask = np.load(mask_dir / f"{stem}.npy")

        raw_patches = extract_patches(img, mask, patch_size=patch_size, stride=stride)
        pos_patches = [(pi, pm) for pi, pm in raw_patches if pm.sum() >= MIN_FIRE_PIXELS]
        neg_patches = [(pi, pm) for pi, pm in raw_patches if pm.sum() < MIN_FIRE_PIXELS]

        n_neg_keep = min(len(neg_patches), int(len(pos_patches) * NEG_TO_POS_RATIO))
        if n_neg_keep > 0 and len(neg_patches) > 0:
            keep_idx = rng.choice(len(neg_patches), size=n_neg_keep, replace=False)
            neg_kept = [neg_patches[i] for i in keep_idx]
        else:
            neg_kept = []

        kept = pos_patches + neg_kept
        total_pos += len(pos_patches)
        total_neg_kept += len(neg_kept)
        total_neg_dropped += len(neg_patches) - len(neg_kept)

        # patch filenames start with the source name - train.py uses this
        # to group patches by source when splitting train/val
        for i, (p_img, p_mask) in enumerate(kept):
            name = f"{stem}_p{i:03d}"
            np.save(out_img_dir / f"{name}.npy", p_img)
            np.save(out_mask_dir / f"{name}.npy", p_mask.astype(np.uint8))

        print(f"{stem}: {len(raw_patches)} total patches -> "
              f"{len(pos_patches)} with fire + {len(neg_kept)} without = {len(kept)} saved")

    total = total_pos + total_neg_kept
    print(f"\n=== TOTAL ===")
    print(f"Patches with fire: {total_pos}")
    print(f"Patches without fire (kept): {total_neg_kept}")
    print(f"Patches without fire (dropped for balance): {total_neg_dropped}")
    print(f"Total patches in dataset: {total}")
    print(f"({len(files)} source images -> {total} patches, "
          f"growth x{total/max(len(files),1):.1f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="data_patches")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args()
    patchify_dataset(args.data_dir, args.out_dir, patch_size=args.patch_size, stride=args.stride)
