"""PyTorch dataset for GeoRelight-UV samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from georelight.dataset.schema import SampleRecord, read_manifest


def load_image_tensor(path: str | Path, channels: int) -> torch.Tensor:
    mode = "L" if channels == 1 else "RGB"
    image = Image.open(path).convert(mode)
    array = np.asarray(image).astype(np.float32) / 255.0
    if channels == 1:
        array = array[None, ...]
    else:
        array = array.transpose(2, 0, 1)
    return torch.from_numpy(array)


INPUT_CHANNELS = {
    "full": 7,
    "rgb": 3,
    "rgb_ao": 4,
}


class GeoRelightDataset(Dataset):
    """Loads tensors for the first de-lighting baseline.

    Input channels are `[shaded RGB, normal RGB, AO]`.
    Target channels are `[albedo RGB, shadow residual]`.
    """

    def __init__(self, root: str | Path, split: str = "train", input_mode: str = "full") -> None:
        self.root = Path(root)
        if input_mode not in INPUT_CHANNELS:
            raise ValueError(f"unknown input_mode {input_mode!r}; expected one of {sorted(INPUT_CHANNELS)}")
        self.input_mode = input_mode
        self.input_channels = INPUT_CHANNELS[input_mode]
        self.records: list[SampleRecord] = read_manifest(self.root, split=split)
        if not self.records:
            raise ValueError(f"no samples found for split {split!r} in {self.root}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        paths = record.paths

        shaded = load_image_tensor(self.root / paths["shaded"], 3)
        normal = load_image_tensor(self.root / paths["normal"], 3)
        ao = load_image_tensor(self.root / paths["ao"], 1)
        albedo = load_image_tensor(self.root / paths["albedo"], 3)
        shadow = load_image_tensor(self.root / paths["shadow"], 1)

        if self.input_mode == "full":
            model_input = torch.cat([shaded, normal, ao], dim=0)
        elif self.input_mode == "rgb_ao":
            model_input = torch.cat([shaded, ao], dim=0)
        else:
            model_input = shaded

        return {
            "input": model_input,
            "target": torch.cat([albedo, shadow], dim=0),
            "shaded": shaded,
            "normal": normal,
            "ao": ao,
            "albedo": albedo,
            "shadow": shadow,
            "sample_id": record.sample_id,
            "material_id": record.material_id,
            "light_id": record.light_id,
        }
