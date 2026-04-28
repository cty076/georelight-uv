# GeoRelight-UV Design

## Goal

Build a single-machine research prototype for geometry-aware de-lighting of 3D textures. The first version should produce a paired dataset, train a compact baseline, and measure whether predicted albedo remains stable across different lighting conditions.

## Scope

The prototype does not attempt full text-to-PBR generation. It focuses on the cleanup layer:

- input: shaded/baked RGB plus geometry conditions;
- output: clean albedo and shadow residual;
- evaluation: reconstruction quality and multi-light consistency.

## Architecture

The project has four independent units:

- Dataset schema and validation define a durable file layout.
- Synthetic procedural generation creates paired training data without requiring Blender.
- Tiny U-Net baseline tests whether small models can learn de-lighting.
- Evaluation computes albedo/shadow error and consistency across light variants.

## Hardware Assumption

The initial target is a single RTX 5080 Laptop GPU with 16 GB VRAM and 32 GB system RAM. Training uses small crops, mixed precision when CUDA is available, and a compact model.

## Future Dataset Direction

After the procedural pipeline is stable, the same schema will accept MatSynth, ambientCG, Poly Haven, and Blender/Cycles-rendered mesh texture atlases.
