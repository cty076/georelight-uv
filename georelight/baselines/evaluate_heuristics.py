"""Evaluate non-learned heuristic baselines."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from georelight.baselines.heuristics import predict_heuristic
from georelight.dataset.torch_dataset import GeoRelightDataset
from georelight.evaluate import make_visual_grid


@torch.no_grad()
def evaluate_heuristics(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = GeoRelightDataset(args.data, split=args.split, input_mode="full")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    metrics = {"split": args.split, "baselines": {}}
    for name in args.names:
        metrics["baselines"][name] = _evaluate_one(name, loader, out_dir, args.max_visuals)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def _evaluate_one(name: str, loader: DataLoader, out_dir: Path, max_visuals: int) -> dict:
    total_albedo = 0.0
    total_shadow = 0.0
    count = 0
    by_material: dict[str, list[torch.Tensor]] = defaultdict(list)
    visual_dir = out_dir / name
    visual_dir.mkdir(parents=True, exist_ok=True)
    visual_count = 0

    for batch in loader:
        pred = predict_heuristic(name, batch)
        target = batch["target"]
        albedo_mae = (pred[:, :3] - target[:, :3]).abs().mean(dim=(1, 2, 3))
        shadow_mae = (pred[:, 3:4] - target[:, 3:4]).abs().mean(dim=(1, 2, 3))
        total_albedo += float(albedo_mae.sum())
        total_shadow += float(shadow_mae.sum())
        count += pred.shape[0]

        for i, material_id in enumerate(batch["material_id"]):
            by_material[str(material_id)].append(pred[i, :3].detach().cpu())

        if visual_count < max_visuals:
            for i in range(pred.shape[0]):
                if visual_count >= max_visuals:
                    break
                sample_id = str(batch["sample_id"][i])
                grid = make_visual_grid(
                    shaded=batch["shaded"][i],
                    pred_albedo=pred[i, :3],
                    target_albedo=batch["albedo"][i],
                    pred_shadow=pred[i, 3:4],
                    target_shadow=batch["shadow"][i],
                    title=f"{name}:{sample_id}",
                )
                grid.save(visual_dir / f"{visual_count:03d}_{sample_id}.png")
                visual_count += 1

    consistency_values = []
    for preds in by_material.values():
        if len(preds) < 2:
            continue
        stack = torch.stack(preds, dim=0)
        mean = stack.mean(dim=0, keepdim=True)
        consistency_values.append(float((stack - mean).abs().mean()))

    return {
        "num_samples": count,
        "albedo_mae": total_albedo / max(1, count),
        "shadow_mae": total_shadow / max(1, count),
        "consistency_mae": float(np.mean(consistency_values)) if consistency_values else None,
        "num_consistency_groups": len(consistency_values),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate heuristic de-lighting baselines.")
    parser.add_argument("--data", required=True, help="Dataset root.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--names", nargs="+", default=["identity", "gray_world", "ao_divide", "retinex"])
    parser.add_argument("--max-visuals", type=int, default=8)
    return parser


def main() -> None:
    evaluate_heuristics(build_parser().parse_args())


if __name__ == "__main__":
    main()
