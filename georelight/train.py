"""Training entry point for the Tiny U-Net de-lighting baseline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from georelight.dataset.torch_dataset import GeoRelightDataset
from georelight.models.factory import build_model, count_parameters, model_names


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> dict:
    seed = int(getattr(args, "seed", 7))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = GeoRelightDataset(args.data, split="train", input_mode=args.input_mode)
    val_set = GeoRelightDataset(args.data, split="val", input_mode=args.input_mode)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(
        args.model,
        in_channels=train_set.input_channels,
        out_channels=4,
        base_channels=args.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.L1Loss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history: list[dict] = []
    best_albedo_mae = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in train_loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(x)
                albedo_loss = loss_fn(pred[:, :3], y[:, :3])
                shadow_loss = loss_fn(pred[:, 3:4], y[:, 3:4])
                loss = albedo_loss + args.shadow_weight * shadow_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = x.shape[0]
            train_loss += float(loss.detach().cpu()) * batch_size
            train_count += batch_size

        val_metrics = evaluate_loss(model, val_loader, device, args.shadow_weight)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_count),
            **val_metrics,
        }
        history.append(row)
        if row["val_albedo_mae"] < best_albedo_mae:
            best_albedo_mae = row["val_albedo_mae"]
            best_epoch = epoch
            torch.save(
                make_checkpoint(model, args, train_set.input_channels, history),
                out_dir / "best_checkpoint.pt",
            )
        print(
            f"epoch {epoch:03d} "
            f"train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} "
            f"val_albedo_mae={row['val_albedo_mae']:.6f}"
        )

    checkpoint = make_checkpoint(model, args, train_set.input_channels, history)
    checkpoint["best"] = {
        "epoch": best_epoch,
        "val_albedo_mae": best_albedo_mae,
    }
    torch.save(checkpoint, out_dir / "checkpoint.pt")
    metrics = {
        "device": str(device),
        "seed": seed,
        "model": args.model,
        "parameters": count_parameters(model),
        "best": checkpoint["best"],
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def make_checkpoint(
    model: nn.Module,
    args: argparse.Namespace,
    input_channels: int,
    history: list[dict],
) -> dict:
    return {
        "state_dict": model.state_dict(),
        "model": {
            "name": args.model,
            "in_channels": input_channels,
            "out_channels": 4,
            "base_channels": args.base_channels,
        },
        "data": {
            "input_mode": args.input_mode,
        },
        "args": vars(args),
        "history": history,
    }


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    shadow_weight: float,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_albedo = 0.0
    total_shadow = 0.0
    count = 0
    loss_fn = nn.L1Loss(reduction="none")

    for batch in loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        pred = model(x)
        albedo_mae = loss_fn(pred[:, :3], y[:, :3]).mean(dim=(1, 2, 3))
        shadow_mae = loss_fn(pred[:, 3:4], y[:, 3:4]).mean(dim=(1, 2, 3))
        loss = albedo_mae + shadow_weight * shadow_mae

        batch_size = x.shape[0]
        total_loss += float(loss.sum().cpu())
        total_albedo += float(albedo_mae.sum().cpu())
        total_shadow += float(shadow_mae.sum().cpu())
        count += batch_size

    return {
        "val_loss": total_loss / max(1, count),
        "val_albedo_mae": total_albedo / max(1, count),
        "val_shadow_mae": total_shadow / max(1, count),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Tiny U-Net for GeoRelight-UV.")
    parser.add_argument("--data", required=True, help="Dataset root.")
    parser.add_argument("--out", required=True, help="Run output directory.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--model", default="tiny_unet", choices=model_names())
    parser.add_argument("--input-mode", default="full", choices=["full", "rgb", "rgb_ao"])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--shadow-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
