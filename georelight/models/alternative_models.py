"""Non-U-Net model families for de-lighting experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Simplified NAFNet block for low-level image restoration."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.gate = SimpleGate()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden // 2, hidden // 2, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Conv2d(hidden // 2, channels, kernel_size=1)
        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.ffn2 = nn.Conv2d(hidden // 2, channels, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.gate(y)
        y = y * self.channel_attention(y)
        y = self.conv2(y)
        x = x + self.beta * y

        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.gate(y)
        y = self.ffn2(y)
        return x + self.gamma * y


class NAFNet(nn.Module):
    """Full-resolution NAFNet-style restoration network.

    Unlike the U-Net variants, this model keeps a single feature scale and relies
    on repeated gated restoration blocks.
    """

    def __init__(self, in_channels: int = 7, out_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        width = base_channels * 6
        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.body = nn.Sequential(*(NAFBlock(width) for _ in range(18)))
        self.out = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.body(self.intro(x))
        return torch.sigmoid(self.out(features))


class MDTA(nn.Module):
    """Multi-DConv head transposed attention from Restormer, simplified."""

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dw = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, self.heads, c // self.heads, h * w)
        k = k.reshape(b, self.heads, c // self.heads, h * w)
        v = v.reshape(b, self.heads, c // self.heads, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).reshape(b, c, h, w)
        return self.project(out)


class GatedDconvFeedForward(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.66) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        hidden += hidden % 2
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class RestormerBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attn = MDTA(channels, heads=heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GatedDconvFeedForward(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class RestormerLite(nn.Module):
    """Restormer-style model without encoder-decoder skip topology."""

    def __init__(self, in_channels: int = 7, out_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        width = base_channels * 6
        self.embed = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*(RestormerBlock(width, heads=4) for _ in range(8)))
        self.out = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.out(self.blocks(self.embed(x))))


class RetinexBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class RetinexPhysicsNet(nn.Module):
    """Physics-inspired decomposition network.

    The network predicts illumination, shadow, and specular residuals, then uses
    a simple image-formation equation to estimate albedo before a small learned
    refinement head. It still returns the common `[albedo RGB, shadow]` tensor so
    it can share the same training and evaluation scripts.
    """

    def __init__(self, in_channels: int = 7, out_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        if out_channels != 4:
            raise ValueError("RetinexPhysicsNet expects out_channels=4")
        width = base_channels * 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(*(RetinexBlock(width) for _ in range(12)))
        self.illumination = nn.Conv2d(width, 1, kernel_size=3, padding=1)
        self.shadow = nn.Conv2d(width, 1, kernel_size=3, padding=1)
        self.specular = nn.Conv2d(width, 3, kernel_size=3, padding=1)
        self.refine = nn.Sequential(
            nn.Conv2d(width + 3, width, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shaded = x[:, :3].clamp(0.0, 1.0)
        features = self.body(self.stem(x))
        illumination = 0.25 + 1.75 * torch.sigmoid(self.illumination(features))
        shadow = torch.sigmoid(self.shadow(features))
        specular = 0.35 * torch.sigmoid(self.specular(features))

        denominator = (illumination * (1.0 - 0.85 * shadow)).clamp_min(0.08)
        albedo_physics = ((shaded - specular) / denominator).clamp(0.0, 1.0)
        albedo_delta = torch.tanh(self.refine(torch.cat([features, albedo_physics], dim=1))) * 0.25
        albedo = (albedo_physics + albedo_delta).clamp(0.0, 1.0)
        return torch.cat([albedo, shadow], dim=1)
