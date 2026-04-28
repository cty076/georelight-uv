from __future__ import annotations

from georelight.dataset.schema import read_manifest, validate_dataset
from georelight.dataset.synthetic import SyntheticDatasetConfig, generate_dataset


def test_generate_dataset_writes_manifest_and_files(tmp_path):
    cfg = SyntheticDatasetConfig(
        resolution=32,
        num_materials=4,
        lights_per_material=2,
        seed=11,
        train_fraction=0.5,
        val_fraction=0.25,
    )

    records = generate_dataset(tmp_path, cfg)
    counts = validate_dataset(tmp_path)
    manifest = read_manifest(tmp_path)

    assert len(records) == 8
    assert len(manifest) == 8
    assert counts == {"train": 4, "val": 2, "test": 2}
    assert (tmp_path / "samples" / "mat000000_l00" / "shaded.png").exists()
    assert (tmp_path / "samples" / "mat000000_l00" / "albedo.png").exists()
