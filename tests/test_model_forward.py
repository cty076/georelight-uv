from __future__ import annotations

import torch

from georelight.models.tiny_unet import TinyUNet


def test_tiny_unet_forward_shape_and_range():
    model = TinyUNet(in_channels=7, out_channels=4, base_channels=8)
    x = torch.rand(2, 7, 32, 32)

    y = model(x)

    assert y.shape == (2, 4, 32, 32)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)
