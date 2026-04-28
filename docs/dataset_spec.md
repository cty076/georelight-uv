# GeoRelight-UV Dataset Spec

## Goal

The dataset supports supervised and consistency-based research for de-lighting 3D texture inputs.

Each sample represents one material under one lighting condition. Multiple samples can share the same `material_id`, which enables multi-light consistency evaluation.

## Sample Files

Each sample directory contains:

```text
shaded.png        RGB input with lighting, AO, shadow, and specular effects
albedo.png        lighting-free RGB target
normal.png        tangent/view-space normal encoded to RGB
ao.png            single-channel ambient occlusion condition
shadow.png        single-channel shadow residual target
specular.png      RGB specular residual target
meta.json         sample-level metadata
```

## Root Files

```text
manifest.jsonl    one JSON object per sample
metadata.json     dataset generation settings
splits/train.txt  sample ids for training
splits/val.txt    sample ids for validation
splits/test.txt   sample ids for held-out testing
```

## Manifest Fields

```json
{
  "sample_id": "mat000001_l00",
  "material_id": "mat000001",
  "light_id": "l00",
  "split": "train",
  "paths": {
    "shaded": "samples/mat000001_l00/shaded.png",
    "albedo": "samples/mat000001_l00/albedo.png",
    "normal": "samples/mat000001_l00/normal.png",
    "ao": "samples/mat000001_l00/ao.png",
    "shadow": "samples/mat000001_l00/shadow.png",
    "specular": "samples/mat000001_l00/specular.png",
    "meta": "samples/mat000001_l00/meta.json"
  }
}
```

## Baseline Tensor Contract

The first baseline uses:

```text
input tensor:  [shaded RGB, normal RGB, AO] = 7 channels
output tensor: [clean albedo RGB, shadow residual] = 4 channels
```

All tensors are normalized to `[0, 1]`.

## Evaluation Metrics

- `albedo_mae`: pixel L1 error against clean albedo.
- `shadow_mae`: pixel L1 error against shadow residual.
- `consistency_mae`: mean absolute deviation of predicted albedo among samples with the same material id under different lights.
- Visual check: saved prediction grids for shaded input, predicted albedo, target albedo, predicted shadow, target shadow.

## Real-Material Source Variant

The `ambientCG` generator keeps the same schema but replaces procedural albedo/normal/roughness/AO with real CC0 PBR material maps downloaded from ambientCG. The lighting variants are still rendered locally so each sample keeps paired supervision:

```text
real albedo/normal/roughness/AO maps
+
controlled local lighting/shadow/specular rendering
=
paired shaded input and clean albedo target
```

This variant is more realistic than the procedural dataset because the material appearance comes from real scanned or authored PBR maps. It is still not a real-camera shadow-removal dataset; the shadows are controlled renders so the clean albedo target remains available.
