"""Bake lit and clean UV textures from ABO PBR GLB models.

This creates paired texture-space supervision:

  input:  baked_lit.png       (PBR material with light/shadow baked into UV)
  target: clean_albedo.png    (diffuse color/base-color only in the same UV)

Run:
  blender --background --python blender/bake_abo_textures.py -- --manifest data/abo/pbr_subset_200/manifest.json --out data/abo_baked_smoke --limit 1 --lights 2 --resolution 512
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import bpy
from mathutils import Vector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--lights", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--ambient", type=float, default=0.015)
    parser.add_argument("--light-energy-scale", type=float, default=1.25)
    return parser.parse_args(argv)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def setup_cycles(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.bake.target = "IMAGE_TEXTURES"
    scene.render.bake.margin = args.margin
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    if args.device == "GPU":
        scene.cycles.device = "GPU"


def import_glb(path: Path) -> list[bpy.types.Object]:
    bpy.ops.import_scene.gltf(filepath=str(path))
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError(f"no mesh objects imported from {path}")
    return objects


def normalize_objects(objects: list[bpy.types.Object]) -> None:
    bpy.context.view_layer.update()
    min_corner = Vector((float("inf"), float("inf"), float("inf")))
    max_corner = Vector((float("-inf"), float("-inf"), float("-inf")))
    for obj in objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, world.x)
            min_corner.y = min(min_corner.y, world.y)
            min_corner.z = min(min_corner.z, world.z)
            max_corner.x = max(max_corner.x, world.x)
            max_corner.y = max(max_corner.y, world.y)
            max_corner.z = max(max_corner.z, world.z)
    center = (min_corner + max_corner) * 0.5
    extent = max(max_corner.x - min_corner.x, max_corner.y - min_corner.y, max_corner.z - min_corner.z)
    scale = 2.0 / max(extent, 1e-6)
    for obj in objects:
        obj.location = (obj.location - center) * scale
        obj.scale = obj.scale * scale
    bpy.context.view_layer.update()


def ensure_material_slot(obj: bpy.types.Object) -> None:
    if not obj.material_slots:
        obj.data.materials.append(None)


def selected_meshes(objects: list[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    active = None
    for obj in objects:
        obj.select_set(True)
        if active is None:
            active = obj
    bpy.context.view_layer.objects.active = active


def create_bake_image(name: str, resolution: int, color=(0.0, 0.0, 0.0, 1.0)) -> bpy.types.Image:
    image = bpy.data.images.new(name, width=resolution, height=resolution, alpha=True, float_buffer=False)
    image.generated_color = color
    return image


def attach_bake_image(objects: list[bpy.types.Object], image: bpy.types.Image) -> list[bpy.types.Node]:
    nodes: list[bpy.types.Node] = []
    for obj in objects:
        ensure_material_slot(obj)
        for slot in obj.material_slots:
            if slot.material is None:
                slot.material = bpy.data.materials.new(f"{obj.name}_material")
            mat = slot.material
            mat.use_nodes = True
            for node in mat.node_tree.nodes:
                node.select = False
            tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
            tex.image = image
            tex.select = True
            mat.node_tree.nodes.active = tex
            nodes.append(tex)
    return nodes


def remove_nodes(nodes: list[bpy.types.Node]) -> None:
    for node in nodes:
        tree = getattr(node, "id_data", None)
        if tree is not None:
            try:
                tree.nodes.remove(node)
            except ReferenceError:
                pass


def save_image(image: bpy.types.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 1.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * q))))
    return float(sorted_values[index])


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def bake_to_image(
    objects: list[bpy.types.Object],
    bake_type: str,
    image: bpy.types.Image,
    pass_filter: set[str] | None = None,
) -> None:
    scene = bpy.context.scene
    bake = scene.render.bake
    bake.target = "IMAGE_TEXTURES"
    nodes = attach_bake_image(objects, image)
    selected_meshes(objects)
    kwargs = {"type": bake_type}
    if pass_filter is not None:
        kwargs["pass_filter"] = pass_filter
    bpy.ops.object.bake(**kwargs)
    remove_nodes(nodes)


def add_light(kind: str, location: Vector, energy: float, color: tuple[float, float, float], size: float) -> dict:
    bpy.ops.object.light_add(type=kind, location=location)
    light = bpy.context.object
    light.data.energy = energy
    light.data.color = color
    if hasattr(light.data, "size"):
        light.data.size = size
    return {
        "type": kind,
        "location": [float(location.x), float(location.y), float(location.z)],
        "energy": float(energy),
        "color": [float(v) for v in color],
        "size": float(size),
    }


def create_light(light_index: int, total_lights: int, args: argparse.Namespace) -> dict:
    for obj in list(bpy.context.scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    angle = (2.0 * math.pi * light_index) / max(1, total_lights)
    radius = 2.6 + 0.35 * (light_index % 3)
    height = [2.4, 1.1, 3.2, 1.7][light_index % 4]
    key_location = Vector((math.cos(angle) * radius, math.sin(angle) * radius, height))
    fill_angle = angle + math.pi * (0.78 + 0.07 * (light_index % 2))
    fill_location = Vector((math.cos(fill_angle) * 3.5, math.sin(fill_angle) * 3.5, 2.0))
    highlight_angle = angle + math.pi * 0.42
    highlight_location = Vector((math.cos(highlight_angle) * 1.8, math.sin(highlight_angle) * 1.8, 2.8))

    key_size = [0.35, 0.75, 1.4, 2.4][light_index % 4]
    key_energy = (620.0 + 160.0 * (light_index % 3)) * args.light_energy_scale
    colors = [
        (1.00, 0.94, 0.86),
        (0.88, 0.96, 1.00),
        (1.00, 1.00, 0.96),
        (0.96, 0.90, 1.00),
    ]
    key_color = colors[light_index % len(colors)]

    ambient = max(0.0, args.ambient)
    bpy.context.scene.world.color = (ambient, ambient, ambient)
    lights = [
        add_light("AREA", key_location, key_energy, key_color, key_size),
        add_light("AREA", fill_location, key_energy * 0.08, (0.82, 0.88, 1.0), 3.5),
        add_light("POINT", highlight_location, key_energy * 0.10, (1.0, 0.96, 0.90), 0.2),
    ]
    return {
        "world_color": [float(ambient), float(ambient), float(ambient)],
        "lights": lights,
    }


def bake_static_targets(
    objects: list[bpy.types.Object],
    asset_dir: Path,
    resolution: int,
) -> tuple[dict[str, Path], bpy.types.Image]:
    clean = create_bake_image("clean_albedo", resolution)
    bake_to_image(objects, "DIFFUSE", clean, pass_filter={"COLOR"})
    clean_path = asset_dir / "clean_albedo.png"
    save_image(clean, clean_path)

    normal = create_bake_image("normal_uv", resolution)
    bake_to_image(objects, "NORMAL", normal)
    normal_path = asset_dir / "normal.png"
    save_image(normal, normal_path)

    ao = create_bake_image("ao_uv", resolution, color=(1.0, 1.0, 1.0, 1.0))
    bake_to_image(objects, "AO", ao)
    ao_path = asset_dir / "ao.png"
    save_image(ao, ao_path)

    mask = create_bake_image("mask_uv", resolution)
    mask_materials = override_with_emission(objects, (1.0, 1.0, 1.0, 1.0))
    bake_to_image(objects, "EMIT", mask)
    restore_materials(mask_materials)
    mask_path = asset_dir / "mask.png"
    save_image(mask, mask_path)

    return {
        "clean_albedo": clean_path,
        "normal": normal_path,
        "ao": ao_path,
        "mask": mask_path,
    }, clean


def override_with_emission(objects: list[bpy.types.Object], color) -> dict[bpy.types.Object, list[bpy.types.Material]]:
    originals: dict[bpy.types.Object, list[bpy.types.Material]] = {}
    material = bpy.data.materials.new("override_emission")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    for obj in objects:
        originals[obj] = [slot.material for slot in obj.material_slots]
        ensure_material_slot(obj)
        for slot in obj.material_slots:
            slot.material = material
    return originals


def restore_materials(originals: dict[bpy.types.Object, list[bpy.types.Material]]) -> None:
    for obj, materials in originals.items():
        ensure_material_slot(obj)
        for index, material in enumerate(materials):
            if index < len(obj.material_slots):
                obj.material_slots[index].material = material


def copy_static_targets(paths: dict[str, Path], sample_dir: Path) -> None:
    mapping = {
        "clean_albedo": "clean_albedo.png",
        "normal": "normal.png",
        "ao": "ao.png",
        "mask": "mask.png",
    }
    for key, name in mapping.items():
        shutil.copy2(paths[key], sample_dir / name)
    shutil.copy2(paths["clean_albedo"], sample_dir / "albedo.png")


def derive_auxiliary_maps(
    lit: bpy.types.Image,
    clean: bpy.types.Image,
    resolution: int,
    shadow: bpy.types.Image,
    specular: bpy.types.Image,
    illumination: bpy.types.Image,
) -> dict:
    lit_pixels = list(lit.pixels[:])
    clean_pixels = list(clean.pixels[:])
    ratios: list[float] = []
    for i in range(0, len(lit_pixels), 4):
        lr, lg, lb = lit_pixels[i], lit_pixels[i + 1], lit_pixels[i + 2]
        ar, ag, ab = clean_pixels[i], clean_pixels[i + 1], clean_pixels[i + 2]
        lit_luma = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb
        albedo_luma = 0.2126 * ar + 0.7152 * ag + 0.0722 * ab
        if albedo_luma > 0.025:
            ratios.append(lit_luma / max(albedo_luma, 1e-4))

    diffuse_reference = max(0.15, percentile(ratios, 0.90))
    shadow_pixels: list[float] = []
    specular_pixels: list[float] = []
    illumination_pixels: list[float] = []
    shadow_sum = 0.0
    specular_sum = 0.0

    for i in range(0, len(lit_pixels), 4):
        lr, lg, lb = lit_pixels[i], lit_pixels[i + 1], lit_pixels[i + 2]
        ar, ag, ab = clean_pixels[i], clean_pixels[i + 1], clean_pixels[i + 2]
        lit_luma = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb
        albedo_luma = 0.2126 * ar + 0.7152 * ag + 0.0722 * ab
        ratio = lit_luma / max(albedo_luma, 1e-4) if albedo_luma > 0.025 else diffuse_reference
        normalized_light = ratio / diffuse_reference
        shadow_value = clamp01(1.0 - normalized_light)
        shadow_sum += shadow_value

        sr = max(0.0, lr - ar * diffuse_reference)
        sg = max(0.0, lg - ag * diffuse_reference)
        sb = max(0.0, lb - ab * diffuse_reference)
        specular_sum += (sr + sg + sb) / 3.0

        illum_value = clamp01(normalized_light * 0.5)
        shadow_pixels.extend((shadow_value, shadow_value, shadow_value, 1.0))
        specular_pixels.extend((clamp01(sr), clamp01(sg), clamp01(sb), 1.0))
        illumination_pixels.extend((illum_value, illum_value, illum_value, 1.0))

    shadow.pixels.foreach_set(shadow_pixels)
    specular.pixels.foreach_set(specular_pixels)
    illumination.pixels.foreach_set(illumination_pixels)
    shadow.update()
    specular.update()
    illumination.update()

    pixel_count = max(1, resolution * resolution)
    return {
        "diffuse_reference": float(diffuse_reference),
        "shadow_mean": float(shadow_sum / pixel_count),
        "specular_mean": float(specular_sum / pixel_count),
    }


def split_for_model(index: int, total: int, args: argparse.Namespace) -> str:
    if total <= 1:
        return "train"
    ratio = index / total
    if ratio < args.train_fraction:
        return "train"
    if ratio < args.train_fraction + args.val_fraction:
        return "val"
    return "test"


def bake_model(item: dict, args: argparse.Namespace, out_root: Path, split: str) -> list[dict]:
    reset_scene()
    setup_cycles(args)
    glb_path = resolve_project_path(item["local_path"]).resolve()
    objects = import_glb(glb_path)
    normalize_objects(objects)

    model_id = item["3dmodel_id"]
    asset_dir = out_root / "assets" / model_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    static_paths, clean_image = bake_static_targets(objects, asset_dir, args.resolution)

    records = []
    for light_index in range(args.lights):
        light_meta = create_light(light_index, args.lights, args)
        sample_id = f"{model_id}_l{light_index:02d}"
        sample_dir = out_root / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        lit = create_bake_image(f"{sample_id}_baked_lit", args.resolution)
        bake_to_image(objects, "COMBINED", lit)
        baked_lit_path = sample_dir / "baked_lit.png"
        save_image(lit, baked_lit_path)
        shutil.copy2(baked_lit_path, sample_dir / "shaded.png")

        shadow = create_bake_image(f"{sample_id}_shadow_residual", args.resolution)
        specular = create_bake_image(f"{sample_id}_specular_estimate", args.resolution)
        illumination = create_bake_image(f"{sample_id}_illumination", args.resolution)
        aux_meta = derive_auxiliary_maps(lit, clean_image, args.resolution, shadow, specular, illumination)
        save_image(shadow, sample_dir / "shadow.png")
        save_image(specular, sample_dir / "specular.png")
        save_image(illumination, sample_dir / "illumination.png")

        copy_static_targets(static_paths, sample_dir)
        meta = {
            "sample_id": sample_id,
            "material_id": model_id,
            "model_id": model_id,
            "light_id": f"l{light_index:02d}",
            "split": split,
            "source": "ABO",
            "license": "CC-BY-4.0",
            "task": "uv_baked_texture_delighting",
            "glb": item["local_path"],
            "light": light_meta,
            "auxiliary_maps": aux_meta,
            "resolution": args.resolution,
        }
        (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        records.append(
            {
                "sample_id": sample_id,
                "material_id": model_id,
                "model_id": model_id,
                "light_id": f"l{light_index:02d}",
                "split": split,
                "paths": {
                    "baked_lit": f"samples/{sample_id}/baked_lit.png",
                    "clean_albedo": f"samples/{sample_id}/clean_albedo.png",
                    "shaded": f"samples/{sample_id}/shaded.png",
                    "albedo": f"samples/{sample_id}/albedo.png",
                    "normal": f"samples/{sample_id}/normal.png",
                    "ao": f"samples/{sample_id}/ao.png",
                    "mask": f"samples/{sample_id}/mask.png",
                    "shadow": f"samples/{sample_id}/shadow.png",
                    "specular": f"samples/{sample_id}/specular.png",
                    "illumination": f"samples/{sample_id}/illumination.png",
                    "meta": f"samples/{sample_id}/meta.json",
                },
            }
        )
    return records


def main() -> None:
    args = parse_args()
    out_root = resolve_project_path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(resolve_project_path(args.manifest).read_text(encoding="utf-8"))
    selected = manifest[: args.limit]
    all_records = []
    split_ids = {split: [] for split in SPLITS}
    for index, item in enumerate(selected):
        split = split_for_model(index, len(selected), args)
        print(f"Baking {item['3dmodel_id']} ({split})")
        records = bake_model(item, args, out_root, split)
        all_records.extend(records)
        for record in records:
            split_ids[split].append(record["sample_id"])
    metadata = {
        "source": "ABO",
        "license": "CC-BY-4.0",
        "task": "uv_baked_texture_delighting",
        "num_models": len(selected),
        "lights_per_model": args.lights,
        "num_samples": len(all_records),
        "resolution": args.resolution,
        "sample_keys": ["shaded", "albedo", "normal", "ao", "shadow", "specular", "meta"],
        "extra_keys": ["baked_lit", "clean_albedo", "mask", "illumination"],
        "splits": {split: len(ids) for split, ids in split_ids.items()},
        "generator": "abo_blender_uv_bake_v2",
        "blender": bpy.app.version_string,
    }
    (out_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (out_root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    splits_dir = out_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in split_ids.items():
        (splits_dir / f"{split}.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    print(f"Baked {len(all_records)} texture samples to {out_root}")


if __name__ == "__main__":
    main()
