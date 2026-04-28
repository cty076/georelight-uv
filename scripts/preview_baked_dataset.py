"""Create montage previews and simple statistics for UV-baked ABO datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_COLUMNS = (
    "baked_lit",
    "clean_albedo",
    "normal",
    "ao",
    "shadow",
    "specular",
    "illumination",
)


def load_manifest(root: Path) -> list[dict]:
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def summarize(records: list[dict], root: Path, columns: tuple[str, ...]) -> dict:
    stats: dict[str, list[list[float]]] = {column: [] for column in columns}
    stats["lit_albedo_absdiff"] = []

    for record in records:
        sample_dir = root / "samples" / record["sample_id"]
        for column in columns:
            array = read_rgb(sample_dir / f"{column}.png")
            stats[column].append(
                [
                    float(array.mean()),
                    float(array.std()),
                    float(array.min()),
                    float(array.max()),
                ]
            )
        lit = read_rgb(sample_dir / "baked_lit.png")
        albedo = read_rgb(sample_dir / "clean_albedo.png")
        diff = np.abs(lit - albedo)
        stats["lit_albedo_absdiff"].append(
            [
                float(diff.mean()),
                float(diff.std()),
                float(np.quantile(diff, 0.95)),
                float(diff.max()),
            ]
        )

    summary = {}
    for key, values in stats.items():
        array = np.asarray(values, dtype=np.float32)
        names = ("mean", "std", "p95", "max") if key == "lit_albedo_absdiff" else ("mean", "std", "min", "max")
        summary[key] = {
            name: {
                "mean": float(array[:, index].mean()),
                "min": float(array[:, index].min()),
                "max": float(array[:, index].max()),
            }
            for index, name in enumerate(names)
        }
    return summary


def select_preview_records(records: list[dict], max_per_split: int) -> list[dict]:
    preview = []
    for split in ("train", "val", "test"):
        selected = [record for record in records if record.get("split") == split]
        preview.extend(selected[:max_per_split])
    return preview


def make_montage(record: dict, root: Path, columns: tuple[str, ...], thumb: int) -> Image.Image:
    pad = 10
    label_h = 22
    font = ImageFont.load_default()
    sample_dir = root / "samples" / record["sample_id"]

    thumbs = []
    for column in columns:
        image = Image.open(sample_dir / f"{column}.png").convert("RGB")
        image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb, thumb), (28, 28, 28))
        tile.paste(image, ((thumb - image.width) // 2, (thumb - image.height) // 2))
        thumbs.append(tile)

    width = len(columns) * thumb + (len(columns) + 1) * pad
    height = thumb + label_h + pad * 3 + 20
    canvas = Image.new("RGB", (width, height), (245, 245, 242))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), f"{record['sample_id']}  {record.get('split', '')}", fill=(20, 20, 20), font=font)
    y = pad + 20
    for index, (column, image) in enumerate(zip(columns, thumbs)):
        x = pad + index * (thumb + pad)
        draw.text((x, y), column, fill=(20, 20, 20), font=font)
        canvas.paste(image, (x, y + label_h))
    return canvas


def preview_dataset(args: argparse.Namespace) -> None:
    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    columns = tuple(args.columns)
    records = load_manifest(root)

    summary = summarize(records, root, columns)
    (out / "summary_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for record in select_preview_records(records, args.max_per_split):
        montage = make_montage(record, root, columns, args.thumb)
        montage.save(out / f"{record['sample_id']}_montage.png")

    print(f"wrote previews and summary to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview UV-baked ABO de-lighting datasets.")
    parser.add_argument("--data", required=True, help="Dataset root containing manifest.jsonl.")
    parser.add_argument("--out", required=True, help="Output directory for montages and summary_stats.json.")
    parser.add_argument("--max-per-split", type=int, default=2)
    parser.add_argument("--thumb", type=int, default=140)
    parser.add_argument("--columns", nargs="+", default=list(DEFAULT_COLUMNS))
    return parser


def main() -> None:
    preview_dataset(build_parser().parse_args())


if __name__ == "__main__":
    main()
