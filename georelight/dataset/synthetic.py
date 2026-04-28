"""Procedural paired dataset generation for de-lighting experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from georelight.dataset.schema import SAMPLE_KEYS, SPLITS, SampleRecord, write_manifest


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    resolution: int = 128
    num_materials: int = 64
    lights_per_material: int = 4
    seed: int = 7
    train_fraction: float = 0.8
    val_fraction: float = 0.1

    @classmethod
    def from_json(cls, path: str | Path) -> "SyntheticDatasetConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(**data)


@dataclass
class ProceduralMaterial:
    albedo: np.ndarray
    normal: np.ndarray
    ao: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(array, 0.0, 1.0)
    if clipped.ndim == 2:
        image = Image.fromarray((clipped * 255.0 + 0.5).astype(np.uint8), mode="L")
    else:
        image = Image.fromarray((clipped * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    image.save(path)


def _smooth_noise(rng: np.random.Generator, size: int, blur: float) -> np.ndarray:
    low = rng.random((max(4, size // 8), max(4, size // 8)), dtype=np.float32)
    image = Image.fromarray((low * 255.0).astype(np.uint8), mode="L")
    image = image.resize((size, size), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(image).astype(np.float32) / 255.0
    return arr


def _normalize01(array: np.ndarray) -> np.ndarray:
    amin = float(array.min())
    amax = float(array.max())
    if amax - amin < 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - amin) / (amax - amin)).astype(np.float32)


def _make_albedo(rng: np.random.Generator, size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    xx /= max(1, size - 1)
    yy /= max(1, size - 1)

    base = rng.uniform(0.12, 0.85, size=(1, 1, 3)).astype(np.float32)
    accent = rng.uniform(0.08, 0.95, size=(1, 1, 3)).astype(np.float32)
    noise = _smooth_noise(rng, size, blur=rng.uniform(0.5, 2.5))

    pattern_type = int(rng.integers(0, 4))
    if pattern_type == 0:
        pattern = noise
    elif pattern_type == 1:
        freq = rng.uniform(5.0, 16.0)
        angle = rng.uniform(0.0, np.pi)
        pattern = 0.5 + 0.5 * np.sin(freq * (np.cos(angle) * xx + np.sin(angle) * yy) * np.pi)
        pattern = 0.65 * pattern + 0.35 * noise
    elif pattern_type == 2:
        cells = rng.integers(4, 12)
        pattern = (((xx * cells).astype(np.int32) + (yy * cells).astype(np.int32)) % 2).astype(np.float32)
        pattern = 0.75 * pattern + 0.25 * noise
    else:
        veins = np.sin((xx + 0.25 * noise) * rng.uniform(18.0, 40.0))
        pattern = _normalize01(veins + 0.75 * noise)

    pattern = pattern[..., None]
    albedo = base * (1.0 - pattern) + accent * pattern
    return np.clip(albedo, 0.02, 0.98).astype(np.float32)


def _normal_from_height(height: np.ndarray, strength: float) -> np.ndarray:
    gy, gx = np.gradient(height.astype(np.float32))
    normal = np.stack((-gx * strength, -gy * strength, np.ones_like(height)), axis=-1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    return ((normal + 1.0) * 0.5).astype(np.float32)


def make_material(rng: np.random.Generator, size: int) -> ProceduralMaterial:
    height = _smooth_noise(rng, size, blur=rng.uniform(0.8, 2.8))
    detail = _smooth_noise(rng, size, blur=rng.uniform(0.2, 1.0))
    height = _normalize01(0.75 * height + 0.25 * detail)
    normal = _normal_from_height(height, strength=rng.uniform(1.2, 4.0))

    albedo = _make_albedo(rng, size)
    ao = np.clip(0.72 + 0.28 * height, 0.0, 1.0).astype(np.float32)
    rough_base = rng.uniform(0.25, 0.85)
    roughness = np.clip(rough_base + 0.25 * (_smooth_noise(rng, size, blur=1.5) - 0.5), 0.05, 0.95)

    metal_chance = rng.random()
    if metal_chance < 0.25:
        metallic = np.clip(0.75 + 0.25 * _smooth_noise(rng, size, blur=2.0), 0.0, 1.0)
    else:
        metallic = np.clip(0.08 * _smooth_noise(rng, size, blur=2.0), 0.0, 0.25)

    return ProceduralMaterial(
        albedo=albedo,
        normal=normal,
        ao=ao.astype(np.float32),
        roughness=roughness.astype(np.float32),
        metallic=metallic.astype(np.float32),
    )


def _make_shadow_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    xx = (xx / max(1, size - 1)) * 2.0 - 1.0
    yy = (yy / max(1, size - 1)) * 2.0 - 1.0
    mask = np.zeros((size, size), dtype=np.float32)

    for _ in range(int(rng.integers(1, 4))):
        cx = rng.uniform(-0.8, 0.8)
        cy = rng.uniform(-0.8, 0.8)
        sx = rng.uniform(0.25, 0.85)
        sy = rng.uniform(0.15, 0.75)
        angle = rng.uniform(0.0, np.pi)
        xr = np.cos(angle) * (xx - cx) + np.sin(angle) * (yy - cy)
        yr = -np.sin(angle) * (xx - cx) + np.cos(angle) * (yy - cy)
        blob = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
        mask = np.maximum(mask, blob.astype(np.float32))

    return np.clip(mask, 0.0, 1.0)


def render_material(
    material: ProceduralMaterial,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    normal = material.normal * 2.0 - 1.0
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)

    phi = rng.uniform(0.0, 2.0 * np.pi)
    z = rng.uniform(0.35, 0.95)
    r = np.sqrt(max(0.0, 1.0 - z * z))
    light_dir = np.array([r * np.cos(phi), r * np.sin(phi), z], dtype=np.float32)
    view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half_dir = light_dir + view_dir
    half_dir /= np.linalg.norm(half_dir)

    ndotl = np.maximum(np.sum(normal * light_dir[None, None, :], axis=-1), 0.0)
    ndoth = np.maximum(np.sum(normal * half_dir[None, None, :], axis=-1), 0.0)

    ambient = rng.uniform(0.22, 0.45)
    diffuse_strength = rng.uniform(0.55, 1.15)
    light_color = rng.uniform(0.82, 1.12, size=(1, 1, 3)).astype(np.float32)
    illumination = ambient + diffuse_strength * ndotl[..., None]

    shadow = _make_shadow_mask(rng, material.albedo.shape[0])
    shadow_strength = rng.uniform(0.25, 0.65)
    shadow_residual = (shadow * shadow_strength).astype(np.float32)
    shadow_multiplier = 1.0 - shadow_residual

    roughness = material.roughness
    metallic = material.metallic
    shininess = 8.0 + (1.0 - roughness) * 90.0
    specular_shape = ndoth ** shininess
    f0 = 0.04 * (1.0 - metallic[..., None]) + material.albedo * metallic[..., None]
    specular = f0 * specular_shape[..., None] * (1.0 - roughness[..., None]) * rng.uniform(0.15, 0.75)

    shaded = material.albedo * illumination * material.ao[..., None] * shadow_multiplier[..., None] * light_color
    shaded = np.clip(shaded + specular, 0.0, 1.0).astype(np.float32)

    meta = {
        "ambient": float(ambient),
        "diffuse_strength": float(diffuse_strength),
        "light_dir": [float(v) for v in light_dir],
        "light_color": [float(v) for v in light_color.reshape(-1)],
        "shadow_strength": float(shadow_strength),
    }
    return shaded, shadow_residual, np.clip(specular, 0.0, 1.0).astype(np.float32), meta


def _split_for_material(index: int, total: int, cfg: SyntheticDatasetConfig) -> str:
    if total <= 0:
        return "train"
    ratio = index / total
    if ratio < cfg.train_fraction:
        return "train"
    if ratio < cfg.train_fraction + cfg.val_fraction:
        return "val"
    return "test"


def generate_dataset(root: str | Path, cfg: SyntheticDatasetConfig) -> list[SampleRecord]:
    root_path = Path(root)
    if root_path.exists() and any(root_path.iterdir()):
        raise FileExistsError(f"output directory already exists and is not empty: {root_path}")

    rng = np.random.default_rng(cfg.seed)
    records: list[SampleRecord] = []
    split_ids = {split: [] for split in SPLITS}

    for material_index in range(cfg.num_materials):
        split = _split_for_material(material_index, cfg.num_materials, cfg)
        material_id = f"mat{material_index:06d}"
        material = make_material(rng, cfg.resolution)

        for light_index in range(cfg.lights_per_material):
            light_id = f"l{light_index:02d}"
            sample_id = f"{material_id}_{light_id}"
            sample_dir = root_path / "samples" / sample_id
            shaded, shadow, specular, light_meta = render_material(material, rng)

            paths = {
                "shaded": f"samples/{sample_id}/shaded.png",
                "albedo": f"samples/{sample_id}/albedo.png",
                "normal": f"samples/{sample_id}/normal.png",
                "ao": f"samples/{sample_id}/ao.png",
                "shadow": f"samples/{sample_id}/shadow.png",
                "specular": f"samples/{sample_id}/specular.png",
                "meta": f"samples/{sample_id}/meta.json",
            }

            save_png(sample_dir / "shaded.png", shaded)
            save_png(sample_dir / "albedo.png", material.albedo)
            save_png(sample_dir / "normal.png", material.normal)
            save_png(sample_dir / "ao.png", material.ao)
            save_png(sample_dir / "shadow.png", shadow)
            save_png(sample_dir / "specular.png", specular)

            meta = {
                "sample_id": sample_id,
                "material_id": material_id,
                "light_id": light_id,
                "split": split,
                "light": light_meta,
                "generator": "procedural_v1",
            }
            with (sample_dir / "meta.json").open("w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2, sort_keys=True)

            record = SampleRecord(sample_id, material_id, light_id, split, paths)
            records.append(record)
            split_ids[split].append(sample_id)

    write_manifest(root_path, records)
    (root_path / "splits").mkdir(parents=True, exist_ok=True)
    for split, ids in split_ids.items():
        (root_path / "splits" / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    metadata = {
        "config": asdict(cfg),
        "sample_keys": list(SAMPLE_KEYS),
        "num_samples": len(records),
        "splits": {split: len(ids) for split, ids in split_ids.items()},
    }
    with (root_path / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    return records
