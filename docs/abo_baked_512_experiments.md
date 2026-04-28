# ABO 512 UV-Baked Model Sweep

Dataset: `data/abo_baked_small`

- Source: ABO PBR GLB assets
- Resolution: 512 x 512
- Models: 20
- Lights per model: 4
- Samples: 80
- Split: train 64 / val 8 / test 8
- Split policy: all lights from the same model stay in the same split

Learned models were trained with:

- epochs: 5
- batch size: 1
- base channels: 16
- learning rate: 2e-4
- shadow weight: 0.1
- seed: 123

## Results

| Rank(val) | Method | Type | Params | Best epoch | Val albedo MAE | Val shadow MAE | Test albedo MAE | Test shadow MAE |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `restormer_lite` | learned | 955,044 | 5 | 0.1864 | 0.1498 | 0.3531 | 0.1252 |
| 2 | `retinex_physics` | learned | 936,840 | 5 | 0.2028 | 0.1800 | 0.4068 | 0.1473 |
| 3 | `residual_unet` | learned | 3,999,988 | 5 | 0.2269 | 0.3477 | 0.4543 | 0.3400 |
| 4 | `attention_unet` | learned | 4,022,112 | 5 | 0.2307 | 0.3129 | 0.4413 | 0.3310 |
| 5 | `convnext_unet` | learned | 1,114,852 | 2 | 0.2374 | 0.2524 | 0.5024 | 0.2192 |
| 6 | `tiny_unet` | learned | 484,068 | 3 | 0.2384 | 0.2970 | 0.4387 | 0.2815 |
| 7 | `nafnet` | learned | 1,227,844 | 1 | 0.2557 | 0.2500 | 0.5753 | 0.2086 |
| 8 | `retinex` | heuristic | 0 | - | 0.3147 | 0.1580 | 0.2442 | 0.0797 |
| 9 | `ao_divide` | heuristic | 0 | - | 0.3399 | 0.1901 | 0.5136 | 0.1356 |
| 10 | `identity` | heuristic | 0 | - | 0.3480 | 0.4618 | 0.6129 | 0.3373 |
| 11 | `gray_world` | heuristic | 0 | - | 0.3675 | 0.2534 | 0.5310 | 0.1992 |

Raw outputs:

- `runs/abo_baked_512_sweep/summary.md`
- `runs/abo_baked_512_sweep/summary.csv`
- `runs/abo_baked_512_sweep/summary.json`

## Interpretation

On validation, `restormer_lite` is the strongest learned model, followed by `retinex_physics`.

On test, the hand-crafted `retinex` baseline is still strongest. This is a useful negative result: on the current 20-model pilot dataset, learned networks are already fitting the validation models but do not generalize as well to unseen model categories.

The next experiment should expand the 512 dataset before increasing model size or training epochs. With the current data generator, moving from 20 to 200 ABO models should be much more valuable than pushing 4K or training a larger network on the same 20 models.
