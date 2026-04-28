from __future__ import annotations

from argparse import Namespace

from georelight.dataset.synthetic import SyntheticDatasetConfig, generate_dataset
from georelight.train import train


def test_train_saves_best_checkpoint(tmp_path):
    data_root = tmp_path / "data"
    out_root = tmp_path / "run"
    cfg = SyntheticDatasetConfig(resolution=32, num_materials=10, lights_per_material=2, seed=44)
    generate_dataset(data_root, cfg)

    train(
        Namespace(
            data=str(data_root),
            out=str(out_root),
            epochs=2,
            batch_size=2,
            base_channels=8,
            input_mode="full",
            model="tiny_unet",
            lr=2e-4,
            weight_decay=1e-4,
            shadow_weight=0.1,
            cpu=True,
        )
    )

    assert (out_root / "checkpoint.pt").exists()
    assert (out_root / "best_checkpoint.pt").exists()
