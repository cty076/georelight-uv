# Baseline Experiments

## Experiment: `synth_compare_64`

Dataset:

- 80 procedural materials
- 4 lights per material
- 320 total samples
- split: 256 train / 32 val / 32 test
- resolution: 64 x 64

Metrics:

- lower is better for all columns
- `consistency_mae` measures albedo variation among the same material under different lights

| Method | Input | Trainable | Albedo MAE | Shadow MAE | Consistency MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| Identity | shaded RGB | no | 0.1965 | 0.1750 | 0.0595 |
| Gray-world | shaded RGB | no | 0.1648 | 0.3060 | 0.0582 |
| AO divide | shaded RGB + AO | no | 0.0919 | 0.3068 | 0.0625 |
| Retinex | shaded RGB | no | 0.2089 | 0.1942 | 0.0222 |
| Tiny U-Net | RGB only | yes | 0.0696 | 0.0943 | 0.0327 |
| Tiny U-Net | RGB + AO | yes | 0.0798 | 0.0973 | 0.0274 |
| Tiny U-Net | RGB + normal + AO | yes | 0.0692 | 0.0836 | 0.0345 |

## Initial Reading

The strongest non-learned albedo baseline is AO divide. The learned models beat it on albedo and shadow prediction after a longer 30-epoch run with `base_channels=32` and `shadow_weight=0.1`.

The geometry-aware full model gives the best albedo and shadow numbers in this run, but its consistency is not yet best. That suggests the next research step should add an explicit multi-light consistency loss or best-checkpoint selection instead of only optimizing per-sample reconstruction.

## Model Capacity Experiment: `compare_64_best`

This run compares larger models with the same full input, `base_channels=32`, `shadow_weight=0.1`, 30 epochs, and evaluation from `best_checkpoint.pt` selected by validation albedo MAE.

| Method | Parameters | Albedo MAE | Shadow MAE | Consistency MAE |
| --- | ---: | ---: | ---: | ---: |
| Tiny U-Net | 1.93M | **0.0620** | 0.0874 | 0.0343 |
| Residual U-Net | 15.98M | 0.0658 | 0.0740 | **0.0340** |
| Attention U-Net | 16.07M | 0.0727 | **0.0733** | 0.0357 |
| ConvNeXt U-Net | 4.33M | 0.0659 | 0.0745 | 0.0377 |

### Reading

Increasing capacity did not improve clean albedo on this small synthetic dataset. Tiny U-Net remains best on albedo MAE, while larger residual/attention models improve shadow residual prediction by roughly 15-16% relative to Tiny.

This points to a useful research direction: larger models are not the missing ingredient by themselves. The next likely gains should come from better supervision and training objectives:

- save and evaluate best checkpoints, not final checkpoints;
- add multi-light consistency loss for albedo;
- add residual-specific losses so shadow/specular maps absorb lighting contamination;
- scale from procedural data to MatSynth/Blender-rendered PBR materials.

## Non-U-Net Experiment: `compare_64_best`

This run adds three non-traditional U-Net candidates while keeping the same dataset and training settings:

- `nafnet`: full-resolution NAFNet-style gated restoration blocks;
- `restormer_lite`: Restormer-style transposed attention blocks;
- `retinex_physics`: physics-inspired decomposition that predicts illumination, shadow, and specular before estimating albedo.

All rows use full input, `base_channels=32`, `shadow_weight=0.1`, 30 epochs, and `best_checkpoint.pt`.

| Method | Parameters | Albedo MAE | Shadow MAE | Consistency MAE | Score = Albedo + 0.1 Shadow |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny U-Net | 1.93M | **0.0620** | 0.0874 | 0.0343 | 0.0707 |
| Residual U-Net | 15.98M | 0.0658 | 0.0740 | **0.0340** | 0.0732 |
| Attention U-Net | 16.07M | 0.0727 | 0.0733 | 0.0357 | 0.0800 |
| ConvNeXt U-Net | 4.33M | 0.0659 | 0.0745 | 0.0377 | 0.0733 |
| NAFNet-style | 4.78M | 0.0654 | 0.1030 | 0.0350 | 0.0757 |
| Restormer-lite | 3.67M | 0.0620 | 0.0768 | 0.0367 | 0.0697 |
| Retinex-physics | 3.72M | 0.0626 | **0.0628** | 0.0351 | **0.0689** |

### Reading

If the only target is clean albedo MAE, Tiny U-Net remains narrowly best, but the margin over Restormer-lite is only `0.000054`, which is too small to treat as a meaningful architectural win from one run.

If the target is de-lighting as a decomposition problem, Retinex-physics is currently the best candidate. It keeps albedo nearly tied with Tiny U-Net while reducing shadow MAE from `0.0874` to `0.0628`. It also wins the same weighted score used during training.

The next best research direction is therefore not simply "make a bigger network." It is to make the model more decomposition-aware:

- keep Retinex-physics as the current overall winner;
- add explicit specular supervision to the training loss;
- add shaded reconstruction loss using predicted albedo/shadow/specular;
- add multi-light consistency loss so albedo stays stable across lights.

## Reproduction Commands

```powershell
python scripts/generate_synthetic_dataset.py --config configs/synth_compare_64.json --out data/synth_compare_64
python scripts/evaluate_heuristics.py --data data/synth_compare_64 --out runs/compare_64/heuristics --split test --batch-size 16 --max-visuals 4
python scripts/train_tiny_unet.py --data data/synth_compare_64 --out runs/compare_64/tiny_rgb_b32_e30_sw01 --epochs 30 --batch-size 16 --base-channels 32 --input-mode rgb --shadow-weight 0.1
python scripts/evaluate_model.py --data data/synth_compare_64 --checkpoint runs/compare_64/tiny_rgb_b32_e30_sw01/checkpoint.pt --out runs/compare_64/tiny_rgb_b32_e30_sw01_eval --split test --batch-size 16 --max-visuals 4
python scripts/train_tiny_unet.py --data data/synth_compare_64 --out runs/compare_64/tiny_rgb_ao_b32_e30_sw01 --epochs 30 --batch-size 16 --base-channels 32 --input-mode rgb_ao --shadow-weight 0.1
python scripts/evaluate_model.py --data data/synth_compare_64 --checkpoint runs/compare_64/tiny_rgb_ao_b32_e30_sw01/checkpoint.pt --out runs/compare_64/tiny_rgb_ao_b32_e30_sw01_eval --split test --batch-size 16 --max-visuals 4
python scripts/train_tiny_unet.py --data data/synth_compare_64 --out runs/compare_64/tiny_full_b32_e30_sw01 --epochs 30 --batch-size 16 --base-channels 32 --input-mode full --shadow-weight 0.1
python scripts/evaluate_model.py --data data/synth_compare_64 --checkpoint runs/compare_64/tiny_full_b32_e30_sw01/checkpoint.pt --out runs/compare_64/tiny_full_b32_e30_sw01_eval --split test --batch-size 16 --max-visuals 4
```
