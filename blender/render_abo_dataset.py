"""Render ABO GLB models into GeoRelight-style supervision with Blender.

Run with:
  blender --background --python blender/render_abo_dataset.py -- --manifest data/abo/pbr_subset_200/manifest.json --out data/abo_blender_smoke --limit 1 --views 2 --resolution 256
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--engine", default="BLENDER_EEVEE", choices=["BLENDER_EEVEE", "CYCLES"])
    parser.add_argument("--cycles-samples", type=int, default=32)
    return parser.parse_args(argv)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def setup_render(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.render.engine = args.engine
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    if args.engine == "CYCLES":
        scene.cycles.samples = args.cycles_samples
        scene.cycles.use_denoising = True


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


def add_camera_and_light(view_index: int, total_views: int) -> None:
    angle = (2.0 * math.pi * view_index) / max(1, total_views)
    radius = 4.0
    cam_location = Vector((math.cos(angle) * radius, -math.sin(angle) * radius, 1.7))
    bpy.ops.object.camera_add(location=cam_location)
    camera = bpy.context.object
    direction = Vector((0.0, 0.0, 0.2)) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 55
    bpy.context.scene.camera = camera

    light_location = Vector((math.cos(angle + 0.9) * 3.0, -math.sin(angle + 0.9) * 3.0, 4.0))
    bpy.ops.object.light_add(type="AREA", location=light_location)
    light = bpy.context.object
    light.data.energy = 550
    light.data.size = 3.0

    bpy.context.scene.world.color = (0.03, 0.03, 0.03)


def make_albedo_materials(objects: list[bpy.types.Object]) -> dict[bpy.types.Object, list[bpy.types.Material]]:
    originals: dict[bpy.types.Object, list[bpy.types.Material]] = {}
    for obj in objects:
        originals[obj] = [slot.material for slot in obj.material_slots]
        ensure_material_slot(obj)
        for slot in obj.material_slots:
            slot.material = make_flat_albedo_material(slot.material)
    return originals


def restore_materials(originals: dict[bpy.types.Object, list[bpy.types.Material]]) -> None:
    for obj, materials in originals.items():
        ensure_material_slot(obj)
        for index, material in enumerate(materials):
            if index < len(obj.material_slots):
                obj.material_slots[index].material = material


def ensure_material_slot(obj: bpy.types.Object) -> None:
    if not obj.material_slots:
        obj.data.materials.append(None)


def make_flat_albedo_material(source: bpy.types.Material | None) -> bpy.types.Material:
    mat = bpy.data.materials.new((source.name if source else "mat") + "_flat_albedo")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    color_linked = False
    if source and source.use_nodes:
        bsdf = next((node for node in source.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
        if bsdf and bsdf.inputs["Base Color"].is_linked:
            from_socket = bsdf.inputs["Base Color"].links[0].from_socket
            tex_node = from_socket.node
            if tex_node.type == "TEX_IMAGE" and tex_node.image:
                tex = nodes.new("ShaderNodeTexImage")
                tex.image = tex_node.image
                mat.node_tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
                color_linked = True
        elif bsdf:
            emission.inputs["Color"].default_value = bsdf.inputs["Base Color"].default_value
            color_linked = True
    if not color_linked:
        emission.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    mat.node_tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_normal_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("normal_pass_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    geometry = nodes.new("ShaderNodeNewGeometry")
    add = nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (1.0, 1.0, 1.0)
    mul = nodes.new("ShaderNodeVectorMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = (0.5, 0.5, 0.5)
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    mat.node_tree.links.new(geometry.outputs["Normal"], add.inputs[0])
    mat.node_tree.links.new(add.outputs["Vector"], mul.inputs[0])
    mat.node_tree.links.new(mul.outputs["Vector"], emission.inputs["Color"])
    mat.node_tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_mask_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("mask_pass_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    mat.node_tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_override_materials(objects: list[bpy.types.Object], material: bpy.types.Material) -> dict[bpy.types.Object, list[bpy.types.Material]]:
    originals: dict[bpy.types.Object, list[bpy.types.Material]] = {}
    for obj in objects:
        originals[obj] = [slot.material for slot in obj.material_slots]
        ensure_material_slot(obj)
        for slot in obj.material_slots:
            slot.material = material
    return originals


def render_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_model(item: dict, args: argparse.Namespace, out_root: Path) -> list[dict]:
    reset_scene()
    setup_render(args)
    glb_path = resolve_project_path(item["local_path"]).resolve()
    objects = import_glb(glb_path)
    normalize_objects(objects)
    records = []
    for view in range(args.views):
        for obj in list(bpy.context.scene.objects):
            if obj.type in {"CAMERA", "LIGHT"}:
                bpy.data.objects.remove(obj, do_unlink=True)
        add_camera_and_light(view, args.views)
        sample_id = f"{item['3dmodel_id']}_v{view:02d}"
        sample_dir = out_root / "samples" / sample_id
        render_png(sample_dir / "shaded.png")
        originals = make_albedo_materials(objects)
        render_png(sample_dir / "albedo.png")
        restore_materials(originals)
        normal_originals = make_override_materials(objects, make_normal_material())
        render_png(sample_dir / "normal.png")
        restore_materials(normal_originals)
        mask_originals = make_override_materials(objects, make_mask_material())
        render_png(sample_dir / "mask.png")
        restore_materials(mask_originals)
        meta = {
            "sample_id": sample_id,
            "model_id": item["3dmodel_id"],
            "source": "ABO",
            "license": "CC-BY-4.0",
            "glb": item["local_path"],
            "view": view,
        }
        (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        records.append(meta)
    return records


def main() -> None:
    args = parse_args()
    out_root = resolve_project_path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    all_records = []
    for item in manifest[: args.limit]:
        print(f"Rendering {item['3dmodel_id']}")
        all_records.extend(render_model(item, args, out_root))
    (out_root / "render_manifest.json").write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    print(f"Rendered {len(all_records)} samples to {out_root}")


if __name__ == "__main__":
    main()
