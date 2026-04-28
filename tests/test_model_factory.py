from __future__ import annotations

import torch

from georelight.models.factory import build_model, count_parameters, model_names


def test_model_factory_builds_all_registered_models():
    names = model_names()
    assert {
        "tiny_unet",
        "residual_unet",
        "attention_unet",
        "convnext_unet",
        "nafnet",
        "restormer_lite",
        "retinex_physics",
    }.issubset(set(names))

    for name in names:
        model = build_model(name, in_channels=7, out_channels=4, base_channels=8)
        y = model(torch.rand(2, 7, 32, 32))
        assert y.shape == (2, 4, 32, 32)
        assert torch.all(y >= 0.0)
        assert torch.all(y <= 1.0)


def test_larger_models_have_more_parameters_than_tiny_unet():
    tiny = build_model("tiny_unet", in_channels=7, out_channels=4, base_channels=16)
    residual = build_model("residual_unet", in_channels=7, out_channels=4, base_channels=16)
    attention = build_model("attention_unet", in_channels=7, out_channels=4, base_channels=16)
    convnext = build_model("convnext_unet", in_channels=7, out_channels=4, base_channels=16)
    nafnet = build_model("nafnet", in_channels=7, out_channels=4, base_channels=16)
    restormer = build_model("restormer_lite", in_channels=7, out_channels=4, base_channels=16)
    retinex = build_model("retinex_physics", in_channels=7, out_channels=4, base_channels=16)

    tiny_params = count_parameters(tiny)
    assert count_parameters(residual) > tiny_params
    assert count_parameters(attention) > tiny_params
    assert count_parameters(convnext) > tiny_params
    assert count_parameters(nafnet) > tiny_params
    assert count_parameters(restormer) > tiny_params
    assert count_parameters(retinex) > tiny_params
