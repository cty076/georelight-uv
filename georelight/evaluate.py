"""Evaluation entry point for GeoRelight-UV checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from georelight.dataset.torch_dataset import GeoRelightDataset
from georelight.models.factory import build_model, count_parameters


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    input_mode = args.input_mode or checkpoint.get("data", {}).get("input_mode", "full")
    dataset = GeoRelightDataset(args.data, split=args.split, input_mode=input_mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model_cfg = checkpoint.get("model", {"name": "tiny_unet", "in_channels": 7, "out_channels": 4, "base_channels": 32})
    model_name = model_cfg.get("name", "tiny_unet")
    model = build_model(
        model_name,
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg.get("out_channels", 4),
        base_channels=model_cfg.get("base_channels", 32),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    total_albedo = 0.0
    total_shadow = 0.0
    count = 0
    by_material: dict[str, list[torch.Tensor]] = defaultdict(list)
    visual_count = 0

    for batch in loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        pred = model(x)

        albedo_mae = (pred[:, :3] - y[:, :3]).abs().mean(dim=(1, 2, 3))
        shadow_mae = (pred[:, 3:4] - y[:, 3:4]).abs().mean(dim=(1, 2, 3))
        total_albedo += float(albedo_mae.sum().cpu())
        total_shadow += float(shadow_mae.sum().cpu())
        count += x.shape[0]

        for i, material_id in enumerate(batch["material_id"]):
            by_material[str(material_id)].append(pred[i, :3].detach().cpu())

        if visual_count < args.max_visuals:
            for i in range(x.shape[0]):
                if visual_count >= args.max_visuals:
                    break
                sample_id = str(batch["sample_id"][i])
                grid = make_visual_grid(
                    shaded=batch["shaded"][i],
                    pred_albedo=pred[i, :3].detach().cpu(),
                    target_albedo=batch["albedo"][i],
                    pred_shadow=pred[i, 3:4].detach().cpu(),
                    target_shadow=batch["shadow"][i],
                    title=sample_id,
                )
                grid.save(out_dir / f"{visual_count:03d}_{sample_id}.png")
                visual_count += 1

    consistency_values = []
    for preds in by_material.values():
        if len(preds) < 2:
            continue
        stack = torch.stack(preds, dim=0)
        mean = stack.mean(dim=0, keepdim=True)
        consistency_values.append(float((stack - mean).abs().mean()))

    metrics = {
        "device": str(device),
        "split": args.split,
        "input_mode": input_mode,
        "model": model_name,
        "parameters": count_parameters(model),
        "num_samples": count,
        "albedo_mae": total_albedo / max(1, count),
        "shadow_mae": total_shadow / max(1, count),
        "consistency_mae": float(np.mean(consistency_values)) if consistency_values else None,
        "num_consistency_groups": len(consistency_values),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    array = tensor.permute(1, 2, 0).numpy()
    return (array * 255.0 + 0.5).astype(np.uint8)


def make_visual_grid(
    shaded: torch.Tensor,
    pred_albedo: torch.Tensor,
    target_albedo: torch.Tensor,
    pred_shadow: torch.Tensor,
    target_shadow: torch.Tensor,
    title: str,
) -> Image.Image:
    panels = [
        ("shaded", shaded),
        ("pred_albedo", pred_albedo),
        ("target_albedo", target_albedo),
        ("pred_shadow", pred_shadow),
        ("target_shadow", target_shadow),
    ]
    images = [Image.fromarray(tensor_to_uint8(tensor)) for _, tensor in panels]
    w, h = images[0].size
    label_h = 18
    canvas = Image.new("RGB", (w * len(images), h + label_h * 2), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index, ((label, _), image) in enumerate(zip(panels, images)):
        x = index * w
        canvas.paste(image, (x, label_h))
        draw.text((x + 3, 2), label, fill=(0, 0, 0))
    draw.text((3, h + label_h + 2), title, fill=(0, 0, 0))
    return canvas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a GeoRelight-UV checkpoint.")
    parser.add_argument("--data", required=True, help="Dataset root.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt.")
    parser.add_argument("--out", required=True, help="Evaluation output directory.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--input-mode", default=None, choices=["full", "rgb", "rgb_ao"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-visuals", type=int, default=8)
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation.")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
