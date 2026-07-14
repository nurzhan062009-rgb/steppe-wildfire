"""
Обучение U-Net на патчах Sentinel-2 для сегментации активного огня.

Демо: python scripts/train.py --synthetic --epochs 3 --in-channels 3
Реальные данные: python scripts/train.py --data-dir data_patches --epochs 50 --in-channels 5 --save-path model.pth
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.unet import WildfireUNet
from scripts.spectral_indices import build_extended_input


def get_source_group(patch_stem: str) -> str:
    """'kz_2022_09_03_p007' -> 'kz_2022_09_03', чтобы группировать патчи по исходному снимку."""
    return re.sub(r"_p\d{3}$", "", patch_stem)


class WildfirePatchDataset(Dataset):
    """
    data_dir/images/*.npy - (4, H, W) float32: SWIR2, SWIR1, NIR, RED
    data_dir/masks/*.npy  - (H, W) uint8, {0,1}
    """

    def __init__(self, data_dir: str, in_channels: int = 3, crop_size: int = 480, augment: bool = False):
        self.image_dir = Path(data_dir) / "images"
        self.mask_dir = Path(data_dir) / "masks"
        self.files = sorted(f.stem for f in self.image_dir.glob("*.npy"))
        self.in_channels = in_channels
        self.crop_size = crop_size
        self.augment = augment
        if not self.files:
            raise FileNotFoundError(f"Нет .npy файлов в {self.image_dir}")

    @staticmethod
    def _center_crop_or_pad(arr: np.ndarray, size: int) -> np.ndarray:
        *lead, h, w = arr.shape
        pad_h, pad_w = max(0, size - h), max(0, size - w)
        if pad_h or pad_w:
            pad_width = [(0, 0)] * len(lead) + [(pad_h // 2, pad_h - pad_h // 2),
                                                   (pad_w // 2, pad_w - pad_w // 2)]
            arr = np.pad(arr, pad_width, mode="constant", constant_values=0)
            h, w = arr.shape[-2], arr.shape[-1]
        start_h, start_w = (h - size) // 2, (w - size) // 2
        slicer = tuple([slice(None)] * len(lead) + [slice(start_h, start_h + size),
                                                       slice(start_w, start_w + size)])
        return arr[slicer]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        stem = self.files[idx]
        img = np.load(self.image_dir / f"{stem}.npy")
        mask = np.load(self.mask_dir / f"{stem}.npy")

        # Sentinel-2 SR отдаёт digital numbers 0-10000, приводим к отражательной способности
        img = np.clip(img / 10000.0, 0.0, 1.5)

        if self.in_channels == 5:
            swir2, swir1, nir, red = img[0], img[1], img[2], img[3]
            img = build_extended_input(swir2, swir1, nir, red)
        else:
            img = img[:3]

        img = self._center_crop_or_pad(img, self.crop_size)
        mask = self._center_crop_or_pad(mask, self.crop_size)

        if self.augment:
            k = np.random.randint(0, 4)
            img = np.rot90(img, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()

        return torch.from_numpy(img).float(), torch.from_numpy(mask).float().unsqueeze(0)


class SyntheticWildfireDataset(Dataset):
    """Синтетика для проверки пайплайна без реальных данных."""

    def __init__(self, n_samples: int = 64, size: int = 128, in_channels: int = 3, seed: int = 0):
        self.n_samples = n_samples
        self.size = size
        self.in_channels = in_channels

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s = self.size
        rng = np.random.default_rng(idx)

        swir2 = rng.normal(0.25, 0.03, (s, s)).clip(0, 1)
        swir1 = rng.normal(0.30, 0.03, (s, s)).clip(0, 1)
        nir = rng.normal(0.35, 0.03, (s, s)).clip(0, 1)
        red = rng.normal(0.28, 0.03, (s, s)).clip(0, 1)
        mask = np.zeros((s, s), dtype=np.float32)

        cy, cx = rng.integers(20, s - 20, size=2)
        r = rng.integers(5, 15)
        yy, xx = np.ogrid[:s, :s]
        fire_area = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2

        swir2[fire_area] = rng.normal(0.75, 0.05, fire_area.sum()).clip(0, 1)
        swir1[fire_area] = rng.normal(0.65, 0.05, fire_area.sum()).clip(0, 1)
        nir[fire_area] = rng.normal(0.30, 0.05, fire_area.sum()).clip(0, 1)
        mask[fire_area] = 1.0

        if self.in_channels == 5:
            img = build_extended_input(swir2, swir1, nir, red)
        else:
            img = np.stack([swir2, swir1, nir], axis=0).astype(np.float32)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).float().unsqueeze(0)


def iou_score(pred_logits, target, threshold=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) > 0).float().sum(dim=(1, 2, 3))
    return ((intersection + eps) / (union + eps)).mean().item()


def train_single_split(args, train_files, val_files, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = WildfirePatchDataset(args.data_dir, in_channels=args.in_channels, augment=True, crop_size=args.crop_size)
    val_ds = WildfirePatchDataset(args.data_dir, in_channels=args.in_channels, augment=False, crop_size=args.crop_size)
    train_ds.files = train_files
    val_ds.files = val_files

    train_loader = DataLoader(train_ds, batch_size=min(args.batch_size, len(train_files)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_files), shuffle=False)

    model = WildfireUNet(in_channels=args.in_channels, base_filters=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_iou = -1.0
    epochs_since_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                val_iou += iou_score(model(imgs), masks) * imgs.size(0)
        val_iou /= len(val_loader.dataset)
        scheduler.step(val_iou)

        if val_iou > best_iou + 1e-4:
            best_iou = val_iou
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        if args.patience > 0 and epochs_since_improvement >= args.patience:
            break

    if verbose:
        print(f"  val={val_files}  best_IoU={best_iou:.4f}")
    return best_iou


def cross_validate(args):
    """Leave-one-source-out: каждый исходный снимок по очереди в val, остальные в train."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_files = sorted(f.stem for f in (Path(args.data_dir) / "images").glob("*.npy"))
    groups = sorted(set(get_source_group(f) for f in all_files))
    print(f"Leave-one-source-out CV: {len(all_files)} патчей из {len(groups)} источников, in_channels={args.in_channels}")

    ious = []
    for i, val_group in enumerate(groups):
        train_files = [f for f in all_files if get_source_group(f) != val_group]
        val_files = [f for f in all_files if get_source_group(f) == val_group]
        iou = train_single_split(args, train_files, val_files, verbose=False)
        ious.append(iou)
        print(f"  [{i+1}/{len(groups)}] val_source={val_group} ({len(val_files)} патчей): IoU={iou:.4f}")

    ious = np.array(ious)
    print(f"\n=== Leave-one-source-out CV результат (in_channels={args.in_channels}) ===")
    print(f"Mean IoU: {ious.mean():.4f}  Std: {ious.std():.4f}")
    print(f"Median IoU: {np.median(ious):.4f}")
    return ious


def train(args):
    # seed фиксирован, иначе разброс между запусками на маленьком датасете
    # маскирует реальную разницу между baseline и +NDVI/NBR
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}  (seed={args.seed})")

    if args.synthetic:
        train_ds = SyntheticWildfireDataset(n_samples=64, in_channels=args.in_channels, seed=0)
        val_ds = SyntheticWildfireDataset(n_samples=16, in_channels=args.in_channels, seed=1000)
    else:
        # делим по источникам, а не по отдельным патчам - иначе патчи одного
        # пожара утекают между train и val
        full_files = sorted(f.stem for f in (Path(args.data_dir) / "images").glob("*.npy"))
        groups = sorted(set(get_source_group(f) for f in full_files))
        n_val_groups = max(1, int(0.2 * len(groups)))
        rng = np.random.default_rng(42)
        shuffled_groups = rng.permutation(groups)
        val_groups = set(shuffled_groups[:n_val_groups])

        train_files = [f for f in full_files if get_source_group(f) not in val_groups]
        val_files = [f for f in full_files if get_source_group(f) in val_groups]

        train_ds = WildfirePatchDataset(args.data_dir, in_channels=args.in_channels, augment=True, crop_size=args.crop_size)
        val_ds = WildfirePatchDataset(args.data_dir, in_channels=args.in_channels, augment=False, crop_size=args.crop_size)
        train_ds.files = train_files
        val_ds.files = val_files
        print(f"Train: {len(train_files)} патчей из {len(groups)-n_val_groups} источников, "
              f"Val: {len(val_files)} патчей из {n_val_groups} источников (seed=42)")
        print(f"Val источники: {sorted(val_groups)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = WildfireUNet(in_channels=args.in_channels, base_filters=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_iou = -1.0
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_iou += iou_score(logits, masks) * imgs.size(0)
        val_iou /= len(val_loader.dataset)
        scheduler.step(val_iou)

        marker = ""
        if val_iou > best_iou + 1e-4:
            best_iou = val_iou
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
            marker = "  <- лучший"
            if args.save_path:
                torch.save(best_state, args.save_path)
        else:
            epochs_since_improvement += 1

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_IoU={val_iou:.4f}  lr={cur_lr:.1e}{marker}")

        if args.patience > 0 and epochs_since_improvement >= args.patience:
            print(f"Early stopping: нет улучшения {args.patience} эпох подряд, останавливаюсь.")
            break

    print(f"\nЛучший val_IoU за всё обучение: {best_iou:.4f}")

    if args.save_path and best_state is not None:
        torch.save(best_state, args.save_path)
        print(f"Лучшая модель (не последняя!) сохранена: {args.save_path}")

    return model


def tif_to_npy(tif_path: str, out_path: str):
    import rasterio
    with rasterio.open(tif_path) as src:
        arr = src.read().astype(np.float32)
    np.save(out_path, arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--in-channels", type=int, default=3, choices=[3, 5])
    parser.add_argument("--crop-size", type=int, default=480,
                         help="256 для нарезанных патчей, 480 для цельных снимков")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cross-val", action="store_true")
    args = parser.parse_args()

    if not args.synthetic and not args.data_dir:
        parser.error("Укажи либо --synthetic, либо --data-dir")

    if args.cross_val:
        if args.synthetic or not args.data_dir:
            parser.error("--cross-val работает только с --data-dir")
        cross_validate(args)
    else:
        train(args)
