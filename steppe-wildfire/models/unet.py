"""
U-Net для сегментации активного огня по мультиспектральным Sentinel-2 снимкам.

3 канала (SWIR2, SWIR1, NIR) - baseline, как у Pereira et al. 2021.
5 каналов (+NDVI, +NBR) - расширенная версия, см. scripts/spectral_indices.py.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Два свёрточных слоя 3x3 + BatchNorm + ReLU + Dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool + ConvBlock (encoder step)."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsample + concat skip connection + ConvBlock (decoder step)."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # На случай нечётных размеров входа - выравниваем по skip-connection
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                                    diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class WildfireUNet(nn.Module):
    """
    U-Net для бинарной сегментации активного огня.

    in_channels: 3 = baseline (SWIR2, SWIR1, NIR), 5 = +NDVI/NBR
    base_filters: 32 по умолчанию, лёгкая версия для Colab
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 32, dropout: float = 0.1):
        super().__init__()
        f = base_filters

        self.inc = ConvBlock(in_channels, f, dropout)
        self.down1 = Down(f, f * 2, dropout)
        self.down2 = Down(f * 2, f * 4, dropout)
        self.down3 = Down(f * 4, f * 8, dropout)
        self.down4 = Down(f * 8, f * 16, dropout)

        self.up1 = Up(f * 16, f * 8, dropout)
        self.up2 = Up(f * 8, f * 4, dropout)
        self.up3 = Up(f * 4, f * 2, dropout)
        self.up4 = Up(f * 2, f, dropout)

        self.outc = nn.Conv2d(f, 1, kernel_size=1)  # 1 канал -> вероятность огня

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)  # логиты, сигмоиду применяем в лоссе (BCEWithLogits)


if __name__ == "__main__":
    for in_ch in (3, 5):
        model = WildfireUNet(in_channels=in_ch, base_filters=32)
        dummy = torch.randn(2, in_ch, 256, 256)
        out = model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"in_channels={in_ch}: input {tuple(dummy.shape)} -> output {tuple(out.shape)}, "
              f"params={n_params:,}")
