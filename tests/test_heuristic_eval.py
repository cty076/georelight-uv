from __future__ import annotations

from argparse import Namespace

from georelight.baselines.evaluate_heuristics import evaluate_heuristics
from georelight.dataset.synthetic import SyntheticDatasetConfig, generate_dataset


def test_evaluate_heuristics_writes_metrics(tmp_path):
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    cfg = SyntheticDatasetConfig(resolution=32, num_materials=10, lights_per_material=2, seed=33)
    generate_dataset(data_root, cfg)

    metrics = evaluate_heuristics(
        Namespace(
            data=str(data_root),
            out=str(out_root),
            split="val",
            batch_size=2,
            names=["identity", "ao_divide"],
            max_visuals=0,
        )
    )

    assert set(metrics["baselines"]) == {"identity", "ao_divide"}
    assert (out_root / "metrics.json").exists()
    assert metrics["baselines"]["identity"]["num_samples"] > 0
