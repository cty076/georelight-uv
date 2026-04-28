"""Dataset schema helpers for GeoRelight-UV."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SAMPLE_KEYS = ("shaded", "albedo", "normal", "ao", "shadow", "specular", "meta")
IMAGE_KEYS = ("shaded", "albedo", "normal", "ao", "shadow", "specular")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    material_id: str
    light_id: str
    split: str
    paths: dict[str, str]

    @classmethod
    def from_dict(cls, row: dict) -> "SampleRecord":
        missing = {"sample_id", "material_id", "light_id", "split", "paths"} - set(row)
        if missing:
            raise ValueError(f"manifest row is missing fields: {sorted(missing)}")
        if row["split"] not in SPLITS:
            raise ValueError(f"unknown split {row['split']!r}")
        for key in SAMPLE_KEYS:
            if key not in row["paths"]:
                raise ValueError(f"sample {row['sample_id']} is missing path key {key!r}")
        return cls(
            sample_id=str(row["sample_id"]),
            material_id=str(row["material_id"]),
            light_id=str(row["light_id"]),
            split=str(row["split"]),
            paths={str(k): str(v) for k, v in row["paths"].items()},
        )

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "material_id": self.material_id,
            "light_id": self.light_id,
            "split": self.split,
            "paths": self.paths,
        }


def manifest_path(root: str | Path) -> Path:
    return Path(root) / "manifest.jsonl"


def read_manifest(root: str | Path, split: str | None = None) -> list[SampleRecord]:
    path = manifest_path(root)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    records: list[SampleRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = SampleRecord.from_dict(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_no}") from exc
            if split is None or record.split == split:
                records.append(record)
    return records


def write_manifest(root: str | Path, records: Iterable[SampleRecord]) -> None:
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def validate_dataset(root: str | Path) -> dict[str, int]:
    root_path = Path(root)
    records = read_manifest(root_path)
    counts = {split: 0 for split in SPLITS}
    missing_files: list[str] = []

    for record in records:
        counts[record.split] += 1
        for key in SAMPLE_KEYS:
            candidate = root_path / record.paths[key]
            if not candidate.exists():
                missing_files.append(str(candidate))

    if missing_files:
        preview = "\n".join(missing_files[:10])
        suffix = "" if len(missing_files) <= 10 else f"\n... and {len(missing_files) - 10} more"
        raise FileNotFoundError(f"dataset has missing files:\n{preview}{suffix}")

    return counts
