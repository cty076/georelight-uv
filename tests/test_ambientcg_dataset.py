from __future__ import annotations

import numpy as np
from PIL import Image

from georelight.dataset.ambientcg import (
    AmbientCGConfig,
    discover_material_maps,
    generate_from_material_dirs,
)
from georelight.dataset.schema import read_manifest, validate_dataset


def _write_rgb(path, value):
    arr = np.full((32, 32, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_gray(path, value):
    arr = np.full((32, 32), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def test_discover_material_maps_matches_common_ambientcg_names(tmp_path):
    _write_rgb(tmp_path / "Rock064_1K-JPG_Color.jpg", 120)
    _write_rgb(tmp_path / "Rock064_1K-JPG_NormalGL.jpg", 128)
    _write_gray(tmp_path / "Rock064_1K-JPG_Roughness.jpg", 180)
    _write_gray(tmp_path / "Rock064_1K-JPG_AmbientOcclusion.jpg", 220)

    maps = discover_material_maps(tmp_path)

    assert maps.asset_id == "Rock064"
    assert maps.color.name.endswith("Color.jpg")
    assert maps.normal.name.endswith("NormalGL.jpg")
    assert maps.roughness.name.endswith("Roughness.jpg")
    assert maps.ao is not None


def test_generate_from_material_dirs_writes_schema_dataset(tmp_path):
    material_dir = tmp_path / "raw" / "Rock064"
    material_dir.mkdir(parents=True)
    _write_rgb(material_dir / "Rock064_1K-JPG_Color.jpg", 120)
    _write_rgb(material_dir / "Rock064_1K-JPG_NormalGL.jpg", 128)
    _write_gray(material_dir / "Rock064_1K-JPG_Roughness.jpg", 180)
    _write_gray(material_dir / "Rock064_1K-JPG_AmbientOcclusion.jpg", 220)

    out_dir = tmp_path / "dataset"
    cfg = AmbientCGConfig(resolution=32, lights_per_material=2, seed=12)
    records = generate_from_material_dirs([material_dir], out_dir, cfg)

    assert len(records) == 2
    assert validate_dataset(out_dir) == {"train": 2, "val": 0, "test": 0}
    assert len(read_manifest(out_dir)) == 2
    assert (out_dir / "samples" / "Rock064_l00" / "shaded.png").exists()
