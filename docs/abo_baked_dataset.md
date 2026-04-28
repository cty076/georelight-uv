# ABO UV-Baked De-Lighting Dataset

This project now includes a Blender pipeline for creating paired UV-space de-lighting data from ABO PBR GLB assets.

## Data Definition

Each sample is one model under one light setup:

- `baked_lit.png`: input UV texture baked from the full PBR material under Blender lights.
- `clean_albedo.png` and `albedo.png`: target base-color texture without lighting.
- `normal.png`: UV-space normal bake.
- `ao.png`: ambient occlusion bake.
- `shadow.png`: estimated shadow/darkening residual derived from `baked_lit / clean_albedo`.
- `specular.png`: estimated highlight residual.
- `illumination.png`: normalized illumination estimate for diagnostics.
- `mask.png`: UV occupancy-style mask. For current ABO assets this is often all white because the atlas is densely filled.
- `meta.json`: model, split, light, license, and auxiliary-map metadata.

The `manifest.jsonl` is compatible with `GeoRelightDataset` and keeps all lights from the same model in the same split to reduce train/validation leakage.

## Current 200-Model Dataset

Generated dataset:

```powershell
data\abo_baked_512_200
```

Generation command:

```powershell
& 'D:\software\blender\blender.exe' --background --python blender\bake_abo_textures.py -- --manifest data\abo\pbr_subset_200\manifest.json --out data\abo_baked_512_200 --limit 200 --lights 4 --resolution 512 --samples 64 --margin 16 --device GPU
```

Current split:

- train: 640 samples
- val: 80 samples
- test: 80 samples

Size: about 1.56 GB.

Preview:

```powershell
python scripts\preview_baked_dataset.py --data data\abo_baked_512_200 --out runs\abo_baked_512_200_preview --max-per-split 4 --thumb 140
```

## Pilot Small Dataset

Generated dataset:

```powershell
data\abo_baked_small
```

Generation command:

```powershell
& 'D:\software\blender\blender.exe' --background --python blender\bake_abo_textures.py -- --manifest data\abo\pbr_subset_200\manifest.json --out data\abo_baked_small --limit 20 --lights 4 --resolution 512 --samples 64 --margin 16
```

Current split:

- train: 64 samples
- val: 8 samples
- test: 8 samples

## Preview

```powershell
python scripts\preview_baked_dataset.py --data data\abo_baked_small --out runs\abo_baked_small_preview
```

## Baselines

Heuristic validation baselines:

```powershell
python -m georelight.baselines.evaluate_heuristics --data data\abo_baked_small --out runs\abo_baked_small_heuristics --split val --batch-size 2 --max-visuals 4
```

Short learned baseline:

```powershell
python -m georelight.train --data data\abo_baked_small --out runs\abo_baked_small_retinex_physics --model retinex_physics --input-mode full --epochs 3 --batch-size 1 --base-channels 16 --lr 2e-4 --shadow-weight 0.1
```

Evaluate:

```powershell
python -m georelight.evaluate --data data\abo_baked_small --checkpoint runs\abo_baked_small_retinex_physics\best_checkpoint.pt --out runs\abo_baked_small_retinex_physics_eval --split val --batch-size 1 --max-visuals 6
```

## License Note

ABO assets are distributed under CC-BY-4.0. Any paper, dataset release, or model trained on these generated samples should attribute Amazon.com and the Amazon Berkeley Objects dataset builders.
