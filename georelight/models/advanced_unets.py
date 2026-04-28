"""Larger image-restoration models for de-lighting experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def match_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == reference.shape[-2:]:
        return x
    return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm(out_channels),
        )
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int = 2) -> None:
        super().__init__()
        blocks = [ResidualBlock(in_channels, out_channels)]
        blocks.extend(ResidualBlock(out_channels, out_channels) for _ in range(depth - 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ResidualUNet(nn.Module):
    """A deeper residual U-Net with one extra scale compared with TinyUNet."""

    def __init__(self, in_channels: int = 7, out_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = [base_channels * scale for scale in (1, 2, 4, 8, 16)]
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ResidualStack(in_channels, c1, depth=2)
        self.enc2 = ResidualStack(c1, c2, depth=2)
        self.enc3 = ResidualStack(c2, c3, depth=2)
        self.enc4 = ResidualStack(c3, c4, depth=2)
        self.bottleneck = ResidualStack(c4, c5, depth=2)

        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = ResidualStack(c4 + c4, c4, depth=2)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ResidualStack(c3 + c3, c3, depth=2)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ResidualStack(c2 + c2, c2, depth=2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ResidualStack(c1 + c1, c1, depth=2)
        self.out = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([match_size(d4, e4), e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([match_size(d3, e3), e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([match_size(d2, e2), e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([match_size(d1, e1), e1], dim=1))
        return torch.sigmoid(self.out(d1))


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_channels, inter_channels, kernel_size=1)
        self.gate_proj = nn.Conv2d(gate_channels, inter_channels, kernel_size=1)
        self.psi = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = match_size(gate, skip)
        attention = self.psi(self.skip_proj(skip) + self.gate_proj(gate))
        return skip * attention


class AttentionUNet(nn.Module):
    """Residual U-Net with attention gates on skip features."""

    def __init__(self, in_channels: int = 7, out_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = [base_channels * scale for scale in (1, 2, 4, 8, 16)]
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ResidualStack(in_channels, c1, depth=2)
        self.enc2 = ResidualStack(c1, c2, depth=2)
        self.enc3 = ResidualStack(c2, c3, depth=2)
        self.enc4 = ResidualStack(c3, c4, depth=2)
        self.bottleneck = ResidualStack(c4, c5, depth=2)

        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.att4 = AttentionGate(c4, c4, max(1, c4 // 2))
        self.dec4 = ResidualStack(c4 + c4, c4, depth=2)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.att3 = AttentionGate(c3, c3, max(1, c3 // 2))
        self.dec3 = ResidualStack(c3 + c3, c3, depth=2)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(c2, c2, max(1, c2 // 2))
        self.dec2 = ResidualStack(c2 + c2, c2, depth=2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.att1 = AttentionGate(c1, c1, max(1, c1 // 2))
        self.dec1 = ResidualStack(c1 + c1, c1, depth=2)
        self.out = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        s4 = self.att4(e4, d4)
        d4 = self.dec4(torch.cat([match_size(d4, s4), s4], dim=1))
        d3 = self.up3(d4)
        s3 = self.att3(e3, d3)
        d3 = self.dec3(torch.cat([match_size(d3, s3), s3], dim=1))
        d2 = self.up2(d3)
        s2 = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([match_size(d2, s2), s2], dim=1))
        d1 = self.up1(d2)
        s1 = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([match_size(d1, s1), s1], dim=1))
        return torch.sigmoid(self.out(d1))


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = norm(channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pointwise(x)
        return residual + self.gamma * x


class ConvNeXtStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int = 2) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*(ConvNeXtBlock(out_channels) for _ in range(depth)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))


class ConvNeXtUNet(nn.Module):
    """ConvNeXt-style restoration U-Net with larger local receptive fields."""

    def __init__(self, in_channels: int = 7, out_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = [base_channels * scale for scale in (1, 2, 4, 8)]
        self.down1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(c3, c4, kernel_size=4, stride=2, padding=1)

        self.enc1 = ConvNeXtStage(in_channels, c1, depth=3)
        self.enc2 = nn.Sequential(ConvNeXtBlock(c2), ConvNeXtBlock(c2), ConvNeXtBlock(c2))
        self.enc3 = nn.Sequential(ConvNeXtBlock(c3), ConvNeXtBlock(c3), ConvNeXtBlock(c3))
        self.bottleneck = nn.Sequential(ConvNeXtBlock(c4), ConvNeXtBlock(c4), ConvNeXtBlock(c4), ConvNeXtBlock(c4))

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvNeXtStage(c3 + c3, c3, depth=2)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvNeXtStage(c2 + c2, c2, depth=2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvNeXtStage(c1 + c1, c1, depth=2)
        self.out = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([match_size(d3, e3), e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([match_size(d2, e2), e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([match_size(d1, e1), e1], dim=1))
        return torch.sigmoid(self.out(d1))
