"""Model registry for training and evaluation."""

from __future__ import annotations

from torch import nn

from georelight.models.alternative_models import NAFNet, RestormerLite, RetinexPhysicsNet
from georelight.models.advanced_unets import AttentionUNet, ConvNeXtUNet, ResidualUNet
from georelight.models.tiny_unet import TinyUNet

MODEL_REGISTRY = {
    "tiny_unet": TinyUNet,
    "residual_unet": ResidualUNet,
    "attention_unet": AttentionUNet,
    "convnext_unet": ConvNeXtUNet,
    "nafnet": NAFNet,
    "restormer_lite": RestormerLite,
    "retinex_physics": RetinexPhysicsNet,
}


def model_names() -> list[str]:
    return sorted(MODEL_REGISTRY)


def build_model(
    name: str,
    in_channels: int,
    out_channels: int = 4,
    base_channels: int = 32,
) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}; expected one of {model_names()}")
    return MODEL_REGISTRY[name](
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
