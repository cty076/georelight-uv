from __future__ import annotations

import torch

from georelight.baselines.heuristics import predict_heuristic
from georelight.dataset.synthetic import SyntheticDatasetConfig, generate_dataset
from georelight.dataset.torch_dataset import GeoRelightDataset


def test_dataset_input_modes_change_channel_count(tmp_path):
    cfg = SyntheticDatasetConfig(resolution=32, num_materials=3, lights_per_material=2, seed=22)
    generate_dataset(tmp_path, cfg)

    full = GeoRelightDataset(tmp_path, split="train", input_mode="full")[0]
    rgb = GeoRelightDataset(tmp_path, split="train", input_mode="rgb")[0]
    rgb_ao = GeoRelightDataset(tmp_path, split="train", input_mode="rgb_ao")[0]

    assert full["input"].shape[0] == 7
    assert rgb["input"].shape[0] == 3
    assert rgb_ao["input"].shape[0] == 4


def test_heuristic_baselines_return_albedo_and_shadow():
    batch = {
        "shaded": torch.full((2, 3, 8, 8), 0.25),
        "input": torch.cat(
            [
                torch.full((2, 3, 8, 8), 0.25),
                torch.full((2, 3, 8, 8), 0.5),
                torch.full((2, 1, 8, 8), 0.8),
            ],
            dim=1,
        ),
    }

    for name in ["identity", "gray_world", "ao_divide", "retinex"]:
        pred = predict_heuristic(name, batch)
        assert pred.shape == (2, 4, 8, 8)
        assert torch.all(pred >= 0.0)
        assert torch.all(pred <= 1.0)
