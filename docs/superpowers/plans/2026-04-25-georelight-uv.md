# GeoRelight-UV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local research scaffold for lightweight geometry-aware de-lighting and dataset creation.

**Architecture:** Use a durable dataset schema, a procedural synthetic data generator, a compact Tiny U-Net baseline, and CLI scripts for generation/training/evaluation. Keep Blender and real PBR sources as the next expansion layer.

**Tech Stack:** Python, NumPy, Pillow, PyTorch, pytest.

---

### Task 1: Dataset Schema And Docs

**Files:**
- Create: `README.md`
- Create: `docs/research_direction.md`
- Create: `docs/dataset_spec.md`
- Create: `georelight/dataset/schema.py`

- [ ] Define sample file names and manifest fields.
- [ ] Add dataset validation for missing files and split counts.
- [ ] Document the first baseline tensor contract.

### Task 2: Synthetic Dataset Generator

**Files:**
- Create: `configs/synth_smoke.json`
- Create: `georelight/dataset/synthetic.py`
- Create: `scripts/generate_synthetic_dataset.py`
- Test: `tests/test_synthetic_dataset.py`

- [ ] Generate procedural albedo, normal, AO, roughness, and metallic maps.
- [ ] Render multiple shaded variants per material with sampled lights.
- [ ] Save sample directories, metadata, manifest, and split files.
- [ ] Verify generation through a pytest smoke test.

### Task 3: Tiny Baseline

**Files:**
- Create: `georelight/models/tiny_unet.py`
- Create: `georelight/dataset/torch_dataset.py`
- Test: `tests/test_model_forward.py`

- [ ] Load baseline tensors as `[shaded, normal, ao]`.
- [ ] Predict `[albedo, shadow]`.
- [ ] Verify forward pass shape and output range.

### Task 4: Train And Evaluate

**Files:**
- Create: `georelight/train.py`
- Create: `georelight/evaluate.py`
- Create: `scripts/train_tiny_unet.py`
- Create: `scripts/evaluate_model.py`

- [ ] Train Tiny U-Net with albedo and shadow L1 losses.
- [ ] Save checkpoint and metrics.
- [ ] Evaluate albedo MAE, shadow MAE, and multi-light consistency.
- [ ] Save visual prediction grids.

### Task 5: Smoke Verification

**Commands:**
- `pytest -q`
- `python scripts/generate_synthetic_dataset.py --config configs/synth_smoke.json --out data/synth_smoke`
- `python scripts/train_tiny_unet.py --data data/synth_smoke --out runs/smoke --epochs 1 --batch-size 4 --base-channels 16`
- `python scripts/evaluate_model.py --data data/synth_smoke --checkpoint runs/smoke/checkpoint.pt --out runs/smoke_eval --split val`

- [ ] Confirm all tests pass.
- [ ] Confirm dataset generation writes a manifest.
- [ ] Confirm training writes `checkpoint.pt`.
- [ ] Confirm evaluation writes `metrics.json` and visual predictions.
