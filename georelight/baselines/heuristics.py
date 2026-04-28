"""Non-learned de-lighting baselines."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def predict_heuristic(name: str, batch: dict) -> torch.Tensor:
    shaded = batch["shaded"].float().clamp(0.0, 1.0)
    if name == "identity":
        albedo = shaded
    elif name == "gray_world":
        albedo = gray_world(shaded)
    elif name == "ao_divide":
        albedo = ao_divide(shaded, _extract_ao(batch, shaded))
    elif name == "retinex":
        albedo = multi_scale_retinex(shaded)
    else:
        raise ValueError(f"unknown heuristic baseline: {name}")

    shadow = estimate_shadow_residual(shaded, albedo)
    return torch.cat([albedo.clamp(0.0, 1.0), shadow.clamp(0.0, 1.0)], dim=1)


def gray_world(shaded: torch.Tensor) -> torch.Tensor:
    channel_mean = shaded.mean(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    balanced = shaded / channel_mean * channel_mean.mean(dim=1, keepdim=True)
    return normalize_exposure(balanced)


def ao_divide(shaded: torch.Tensor, ao: torch.Tensor) -> torch.Tensor:
    corrected = shaded / ao.clamp_min(0.2)
    return normalize_exposure(corrected)


def multi_scale_retinex(shaded: torch.Tensor) -> torch.Tensor:
    eps = 1e-4
    logs = []
    for kernel in (5, 15, 31):
        blur = F.avg_pool2d(shaded, kernel_size=kernel, stride=1, padding=kernel // 2)
        logs.append(torch.log(shaded + eps) - torch.log(blur + eps))
    retinex = torch.stack(logs, dim=0).mean(dim=0)
    return normalize_minmax(retinex)


def estimate_shadow_residual(shaded: torch.Tensor, albedo: torch.Tensor) -> torch.Tensor:
    shaded_luma = shaded.mean(dim=1, keepdim=True)
    albedo_luma = albedo.mean(dim=1, keepdim=True).clamp_min(1e-4)
    return (1.0 - shaded_luma / albedo_luma).clamp(0.0, 1.0)


def normalize_exposure(image: torch.Tensor) -> torch.Tensor:
    mean_luma = image.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
    return (image * (0.5 / mean_luma)).clamp(0.0, 1.0)


def normalize_minmax(image: torch.Tensor) -> torch.Tensor:
    flat = image.flatten(start_dim=1)
    minv = flat.min(dim=1).values[:, None, None, None]
    maxv = flat.max(dim=1).values[:, None, None, None]
    return ((image - minv) / (maxv - minv).clamp_min(1e-4)).clamp(0.0, 1.0)


def _extract_ao(batch: dict, shaded: torch.Tensor) -> torch.Tensor:
    if "input" in batch and batch["input"].shape[1] >= 7:
        return batch["input"][:, 6:7].float()
    return torch.ones((shaded.shape[0], 1, shaded.shape[2], shaded.shape[3]), dtype=shaded.dtype)
