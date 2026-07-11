"""Autoencoder architectures for the CIFAR-10 rate--distortion experiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """A same-shape residual block with batch-independent normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.Identity(),
            nn.SiLU(inplace=True),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    """Resize then convolve, avoiding transposed-convolution checkerboards."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm = nn.Identity()
        self.activation = nn.SiLU(inplace=True)
        self.residual = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.activation(self.norm(self.conv(x)))
        return self.residual(x)


class AutoEncoder(nn.Module):
    """Configurable CIFAR-10 autoencoder with one intentional bottleneck.

    The convolutional representation adjacent to the latent tensor contains
    ``48 * 8 * 8 = 3072`` values. Thus, for every supported latent size, the
    configured latent vector is the smallest internal representation. There are
    deliberately no encoder-to-decoder skips, because those would bypass the
    compression bottleneck and invalidate the experiment.
    """

    input_values = 3 * 32 * 32
    pre_latent_values = 48 * 8 * 8

    def __init__(self, latent_size: int = 512):
        super().__init__()
        if not 1 <= latent_size <= self.input_values:
            raise ValueError(f"latent_size must be in [1, {self.input_values}]")
        if latent_size % 64:
            raise ValueError("latent_size must be divisible by 64 for the 8x8 spatial latent")
        self.latent_size = latent_size
        self.latent_channels = latent_size // 64

        self.exact_identity_path = latent_size == self.input_values
        if self.exact_identity_path:
            self.space_to_depth = nn.PixelUnshuffle(4)
            self.to_latent = nn.Conv2d(48, 48, 1, bias=False)
            self.from_latent = nn.Conv2d(48, 48, 1, bias=False)
            self.depth_to_space = nn.PixelShuffle(4)
            self._initialize_latent_projection()
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1, bias=False),
                nn.Identity(),
                nn.SiLU(inplace=True),
                ResidualBlock(64),
            )
            self.encoder_conv = nn.Sequential(
                DownBlock(64, 96),        # 96 x 16 x 16 = 24576 values
                DownBlock(96, 48),        # 48 x 8 x 8 = 3072 values
                ResidualBlock(48),
            )
            self.to_latent = nn.Conv2d(48, self.latent_channels, 1)
            self.from_latent = nn.Conv2d(self.latent_channels, 48, 1)
            self.decoder_conv = nn.Sequential(
                ResidualBlock(48),
                UpBlock(48, 96),          # 96 x 16 x 16
                UpBlock(96, 64),          # 64 x 32 x 32
                ResidualBlock(64),
                nn.Conv2d(64, 3, 3, padding=1),
                nn.Sigmoid(),
            )

    def _initialize_latent_projection(self) -> None:
        """Initialize the projections as a truncated identity channel map."""
        with torch.no_grad():
            self.to_latent.weight.zero_()
            self.from_latent.weight.zero_()
            for channel in range(self.latent_channels):
                self.to_latent.weight[channel, channel, 0, 0] = 1.0
                self.from_latent.weight[channel, channel, 0, 0] = 1.0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.exact_identity_path:
            return self.to_latent(self.space_to_depth(x))
        x = self.encoder_conv(self.stem(x))
        return self.to_latent(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.exact_identity_path:
            return self.depth_to_space(self.from_latent(z))
        x = self.from_latent(z)
        return self.decoder_conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @property
    def compression_ratio(self) -> float:
        return self.input_values / self.latent_size
