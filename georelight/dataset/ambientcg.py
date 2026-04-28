"""Real PBR material dataset generation from ambientCG assets.

The source textures are real CC0 PBR material maps. Lighting variants are
rendered locally so the dataset keeps clean albedo supervision.
"""

from __future__ import annotations

import json
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

from georelight.dataset.schema import SAMPLE_KEYS, SPLITS, SampleRecord, write_manifest
from georelight.dataset.synthetic import ProceduralMaterial, render_material, save_png

AMBIENTCG_API = "https://ambientCG.com/api/v3/assets"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class AmbientCGConfig:
    resolution: int = 256
    num_assets: int = 4
    lights_per_material: int = 4
    seed: int = 20260427
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    download_attributes: str = "1K-JPG"
    sort: str = "popular"
    max_zip_size_mb: float = 64.0

    @classmethod
    def from_json(cls, path: str | Path) -> "AmbientCGConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


@dataclass(frozen=True)
class AmbientCGAsset:
    asset_id: str
    title: str
    url: str
    download_url: str
    size: int
    maps: list[str]
    tags: list[str]
    technique: str | None


@dataclass(frozen=True)
class MaterialMaps:
    asset_id: str
    root: Path
    color: Path
    normal: Path
    roughness: Path
    ao: Path | None
    metallic: Path | None


def fetch_ambientcg_assets(cfg: AmbientCGConfig, limit_multiplier: int = 5) -> list[AmbientCGAsset]:
    query = {
        "type": "material",
        "sort": cfg.sort,
        "limit": max(cfg.num_assets * limit_multiplier, cfg.num_assets),
        "include": "downloads,maps,title,url,technique,tags",
    }
    request = urllib.request.Request(
        AMBIENTCG_API + "?" + urllib.parse.urlencode(query),
        headers={"User-Agent": "GeoRelight-UV/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)

    assets: list[AmbientCGAsset] = []
    max_size = int(cfg.max_zip_size_mb * 1024 * 1024)
    for row in payload.get("assets", []):
        maps = [str(item).lower() for item in row.get("maps", [])]
        if not {"color", "normal", "roughness"}.issubset(set(maps)):
            continue
        download = _select_download(row.get("downloads", []), cfg.download_attributes, max_size)
        if download is None:
            continue
        assets.append(
            AmbientCGAsset(
                asset_id=str(row["id"]),
                title=str(row.get("title", row["id"])),
                url=str(row.get("url", "")),
                download_url=str(download["url"]),
                size=int(download.get("size", 0)),
                maps=list(row.get("maps", [])),
                tags=list(row.get("tags", [])),
                technique=row.get("technique"),
            )
        )
        if len(assets) >= cfg.num_assets:
            break

    if len(assets) < cfg.num_assets:
        raise RuntimeError(f"ambientCG query returned only {len(assets)} usable assets")
    return assets


def _select_download(downloads: list[dict], attributes: str, max_size: int) -> dict | None:
    for item in downloads:
        if item.get("attributes") == attributes and item.get("extension") == "zip":
            if int(item.get("size", 0)) <= max_size:
                return item
    return None


def download_and_extract_assets(
    assets: Iterable[AmbientCGAsset],
    raw_root: str | Path,
    cfg: AmbientCGConfig,
) -> list[Path]:
    raw_path = Path(raw_root)
    zip_dir = raw_path / "zips"
    asset_root = raw_path / "assets"
    zip_dir.mkdir(parents=True, exist_ok=True)
    asset_root.mkdir(parents=True, exist_ok=True)

    material_dirs: list[Path] = []
    for asset in assets:
        zip_path = zip_dir / f"{asset.asset_id}_{cfg.download_attributes}.zip"
        out_dir = asset_root / asset.asset_id / cfg.download_attributes
        if not zip_path.exists():
            _download_file(asset.download_url, zip_path)
        if not out_dir.exists() or not any(out_dir.rglob("*")):
            out_dir.mkdir(parents=True, exist_ok=True)
            _safe_extract_zip(zip_path, out_dir)
        material_dirs.append(out_dir)
    return material_dirs


def _download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "GeoRelight-UV/0.1"})
    with urllib.request.urlopen(request, timeout=180) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(path)


def _safe_extract_zip(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        root = out_dir.resolve()
        for member in archive.infolist():
            target = (out_dir / member.filename).resolve()
            if not str(target).startswith(str(root)):
                raise ValueError(f"unsafe zip member path: {member.filename}")
        archive.extractall(out_dir)


def discover_material_maps(root: str | Path) -> MaterialMaps:
    root_path = Path(root)
    files = [path for path in root_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"no image maps found under {root_path}")

    color = _find_first(files, _is_color)
    normal = _find_first(files, _is_normal)
    roughness = _find_first(files, lambda name: "roughness" in name)
    ao = _find_first(files, _is_ao, required=False)
    metallic = _find_first(files, _is_metallic, required=False)
    if color is None or normal is None or roughness is None:
        names = [path.name for path in files]
        raise FileNotFoundError(f"required color/normal/roughness maps not found in {root_path}: {names}")

    return MaterialMaps(
        asset_id=_infer_asset_id(color, root_path),
        root=root_path,
        color=color,
        normal=normal,
        roughness=roughness,
        ao=ao,
        metallic=metallic,
    )


def _find_first(files: list[Path], predicate, required: bool = True) -> Path | None:
    matches = [path for path in files if predicate(_normalized_name(path))]
    if not matches:
        if required:
            return None
        return None
    matches.sort(key=lambda path: len(path.name))
    return matches[0]


def _normalized_name(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "", path.stem.lower())


def _is_color(name: str) -> bool:
    return any(token in name for token in ("color", "basecolor", "albedo", "diffuse")) and "ambient" not in name


def _is_normal(name: str) -> bool:
    return "normal" in name


def _is_ao(name: str) -> bool:
    return "ambientocclusion" in name or name.endswith("ao") or "occlusion" in name


def _is_metallic(name: str) -> bool:
    return "metallic" in name or "metalness" in name


def _infer_asset_id(path: Path, root: Path) -> str:
    match = re.match(r"([A-Za-z]+[0-9]+)", path.name)
    if match:
        return match.group(1)
    if root.name:
        return root.name
    return path.stem.split("_")[0]


def generate_from_ambientcg(
    out_root: str | Path,
    raw_root: str | Path,
    cfg: AmbientCGConfig,
) -> list[SampleRecord]:
    assets = fetch_ambientcg_assets(cfg)
    material_dirs = download_and_extract_assets(assets, raw_root, cfg)
    records = generate_from_material_dirs(material_dirs, out_root, cfg)
    _write_source_metadata(out_root, cfg, assets)
    return records


def generate_from_material_dirs(
    material_dirs: Iterable[str | Path],
    out_root: str | Path,
    cfg: AmbientCGConfig,
) -> list[SampleRecord]:
    out_path = Path(out_root)
    if out_path.exists() and any(out_path.iterdir()):
        raise FileExistsError(f"output directory already exists and is not empty: {out_path}")

    rng = np.random.default_rng(cfg.seed)
    material_maps = [discover_material_maps(path) for path in material_dirs]
    records: list[SampleRecord] = []
    split_ids = {split: [] for split in SPLITS}

    for material_index, maps in enumerate(material_maps):
        split = _split_for_material(material_index, len(material_maps), cfg)
        material = load_material(maps, cfg.resolution)
        for light_index in range(cfg.lights_per_material):
            light_id = f"l{light_index:02d}"
            sample_id = f"{maps.asset_id}_{light_id}"
            sample_dir = out_path / "samples" / sample_id
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
                "material_id": maps.asset_id,
                "light_id": light_id,
                "split": split,
                "source": "ambientCG",
                "license": "CC0",
                "map_paths": {
                    "color": str(maps.color),
                    "normal": str(maps.normal),
                    "roughness": str(maps.roughness),
                    "ao": str(maps.ao) if maps.ao else None,
                    "metallic": str(maps.metallic) if maps.metallic else None,
                },
                "light": light_meta,
                "generator": "ambientcg_pbr_v1",
            }
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

            record = SampleRecord(sample_id, maps.asset_id, light_id, split, paths)
            records.append(record)
            split_ids[split].append(sample_id)

    write_manifest(out_path, records)
    (out_path / "splits").mkdir(parents=True, exist_ok=True)
    for split, ids in split_ids.items():
        (out_path / "splits" / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    metadata = {
        "config": asdict(cfg),
        "source": "ambientCG",
        "license": "CC0",
        "sample_keys": list(SAMPLE_KEYS),
        "num_materials": len(material_maps),
        "num_samples": len(records),
        "splits": {split: len(ids) for split, ids in split_ids.items()},
    }
    (out_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return records


def load_material(maps: MaterialMaps, resolution: int) -> ProceduralMaterial:
    albedo = _load_image(maps.color, resolution, mode="RGB")
    normal = _load_image(maps.normal, resolution, mode="RGB")
    roughness = _load_image(maps.roughness, resolution, mode="L")
    ao = _load_image(maps.ao, resolution, mode="L") if maps.ao else np.ones((resolution, resolution), dtype=np.float32)
    metallic = _load_image(maps.metallic, resolution, mode="L") if maps.metallic else np.zeros((resolution, resolution), dtype=np.float32)
    return ProceduralMaterial(
        albedo=albedo.astype(np.float32),
        normal=normal.astype(np.float32),
        ao=ao.astype(np.float32),
        roughness=roughness.astype(np.float32),
        metallic=metallic.astype(np.float32),
    )


def _load_image(path: str | Path, resolution: int, mode: str) -> np.ndarray:
    image = Image.open(path).convert(mode)
    image = ImageOps.fit(image, (resolution, resolution), method=Image.Resampling.BICUBIC)
    arr = np.asarray(image).astype(np.float32) / 255.0
    return arr


def _split_for_material(index: int, total: int, cfg: AmbientCGConfig) -> str:
    ratio = index / max(1, total)
    if ratio < cfg.train_fraction:
        return "train"
    if ratio < cfg.train_fraction + cfg.val_fraction:
        return "val"
    return "test"


def _write_source_metadata(out_root: str | Path, cfg: AmbientCGConfig, assets: list[AmbientCGAsset]) -> None:
    path = Path(out_root) / "source_assets.json"
    payload = {
        "source": "ambientCG",
        "api": AMBIENTCG_API,
        "license": "CC0",
        "download_attributes": cfg.download_attributes,
        "assets": [asdict(asset) for asset in assets],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
