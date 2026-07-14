"""Нарезка снимков (~1000x1000) на патчи 256x256 для обучения."""

import os
import numpy as np
from pathlib import Path

PATCH_SIZE = 256
STRIDE = 128
MIN_FIRE_PIXELS = 50
NEG_TO_POS_RATIO = 1.0  # держим примерно 1 патч без огня на 1 патч с огнём, для баланса


def extract_patches(img: np.ndarray, mask: np.ndarray, patch_size=PATCH_SIZE, stride=STRIDE):
    """img: (C, H, W), mask: (H, W). Возвращает список (patch_img, patch_mask)."""
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
    print(f"Источников (крупных снимков): {len(files)}")

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

        # имя патча начинается с имени источника - train.py использует это
        # для группировки при разбиении train/val
        for i, (p_img, p_mask) in enumerate(kept):
            name = f"{stem}_p{i:03d}"
            np.save(out_img_dir / f"{name}.npy", p_img)
            np.save(out_mask_dir / f"{name}.npy", p_mask.astype(np.uint8))

        print(f"{stem}: {len(raw_patches)} патчей всего -> "
              f"{len(pos_patches)} с огнём + {len(neg_kept)} без огня = {len(kept)} сохранено")

    total = total_pos + total_neg_kept
    print(f"\n=== ИТОГО ===")
    print(f"Патчей с огнём: {total_pos}")
    print(f"Патчей без огня (оставлено): {total_neg_kept}")
    print(f"Патчей без огня (отброшено, для баланса): {total_neg_dropped}")
    print(f"Всего патчей в датасете: {total}")
    print(f"(было {len(files)} крупных снимков -> стало {total} патчей, "
          f"рост x{total/max(len(files),1):.1f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="data_patches")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args()
    patchify_dataset(args.data_dir, args.out_dir, patch_size=args.patch_size, stride=args.stride)
