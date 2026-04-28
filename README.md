# GeoRelight-UV

Lightweight research scaffold for 3D texture de-lighting and PBR cleanup.

The project explores a focused research question:

> Can a small geometry-aware model remove baked shadows/highlights from generated or scanned 3D textures, producing cleaner albedo maps that relight more reliably?

This repository starts with a reproducible synthetic dataset generator and a Tiny U-Net baseline that can run on a single 16 GB GPU.

## Current Scope

- Generate paired synthetic samples:
  - shaded RGB render
  - clean albedo target
  - normal map condition
  - AO condition
  - shadow residual target
- Train a lightweight de-lighting baseline:
  - input: shaded RGB + normal + AO
  - output: clean albedo + shadow residual
- Evaluate:
  - albedo MAE
  - shadow residual MAE
  - multi-light albedo consistency
  - saved visual predictions

The first version uses procedural material patches so the full pipeline works even without Blender. The next data step is adding MatSynth / ambientCG / Poly Haven assets and Blender/Cycles rendering.

## Quick Start

Generate a smoke dataset:

```powershell
python scripts/generate_synthetic_dataset.py --config configs/synth_smoke.json --out data/synth_smoke
```

Generate a small real-material dataset from ambientCG CC0 PBR assets:

```powershell
python scripts/generate_ambientcg_dataset.py --config configs/ambientcg_real_small.json --out data/ambientcg_real_small --raw data/raw/ambientcg
```

Train for one quick epoch:

```powershell
python scripts/train_tiny_unet.py --data data/synth_smoke --out runs/smoke --epochs 1 --batch-size 4 --base-channels 16
```

Evaluate:

```powershell
python scripts/evaluate_model.py --data data/synth_smoke --checkpoint runs/smoke/checkpoint.pt --out runs/smoke_eval --split val
```

Run heuristic baselines:

```powershell
python scripts/evaluate_heuristics.py --data data/synth_smoke --out runs/smoke_heuristics --split val
```

Train an RGB-only ablation:

```powershell
python scripts/train_tiny_unet.py --data data/synth_smoke --out runs/smoke_rgb --epochs 1 --batch-size 4 --base-channels 16 --input-mode rgb
```

Train a larger model:

```powershell
python scripts/train_tiny_unet.py --data data/synth_smoke --out runs/smoke_residual --epochs 1 --batch-size 4 --base-channels 16 --model residual_unet
```

Available model names:

```text
tiny_unet
residual_unet
attention_unet
convnext_unet
nafnet
restormer_lite
retinex_physics
```

Run tests:

```powershell
pytest -q
```

## Project Layout

```text
configs/                         Small experiment configs
docs/                            Research notes and dataset spec
georelight/dataset/              Dataset schema, generator, PyTorch dataset
georelight/models/               Tiny baseline models
georelight/train.py              Training entry point
georelight/evaluate.py           Evaluation entry point
scripts/                         Thin CLI wrappers
tests/                           Smoke tests
```
