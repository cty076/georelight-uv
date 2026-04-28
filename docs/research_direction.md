# GeoRelight-UV Research Direction

## Motivation

Current PBR texture generation methods increasingly acknowledge that baked lighting is a core failure mode. If shadows, highlights, or environment color leak into albedo, the asset may look good in the paper render but fail under new lighting. Large systems address this with light-aware diffusion, multi-view PBR diffusion, deferred shading losses, or UV refinement. Those methods are valuable but expensive to reproduce on a single workstation.

GeoRelight-UV targets a smaller and reusable module: a geometry-aware de-lighting refiner that takes a shaded or baked texture plus geometric conditions and predicts a cleaner albedo map.

## Research Gap

The practical gap is not "no one has noticed lighting contamination." The gap is that most solutions are embedded inside large end-to-end generation systems. That makes them hard to train, hard to evaluate in isolation, and hard to attach to outputs from other generators.

This project studies whether a compact model can solve the cleanup layer well enough to be useful:

- as a post-process for AI-generated 3D textures,
- as a cleanup stage for scanned or baked assets,
- as a benchmark component for albedo contamination and relighting stability.

## Proposed Contribution

GeoRelight-UV focuses on three contributions:

1. A lightweight geometry-aware de-lighting baseline that runs on a single 16 GB GPU.
2. A controllable paired dataset recipe for shaded input, clean albedo target, normal/AO conditions, and shadow residual labels.
3. An evaluation protocol that measures not only image error but also multi-light albedo consistency.

## First Milestone

The first milestone avoids Blender dependency and creates a synthetic procedural dataset. It is not the final dataset, but it validates the training and evaluation loop:

- generate material-like albedo, height, normal, AO, roughness, metallic;
- render multiple shaded variants under different lights;
- train Tiny U-Net to recover albedo and shadow residual;
- evaluate consistency across different lighting for the same material.

## Dataset Expansion

After the smoke pipeline works, the dataset should be expanded with real PBR sources:

- MatSynth for CC0 PBR materials and multi-light rendered supervision;
- ambientCG / Poly Haven for extra CC0 materials and HDRIs;
- Objaverse meshes after filtering for usable UVs and materials;
- Blender/Cycles renders for curved surfaces and UV texture atlases.

The first real-material implementation uses ambientCG 1K JPG PBR packages. It downloads real albedo/normal/roughness/AO maps and renders controlled multi-light variants into the existing paired dataset schema. This is the right intermediate step before full Blender/Cycles mesh rendering.
