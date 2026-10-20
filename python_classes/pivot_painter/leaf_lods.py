from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .tree_asset import link_exclusively, owning_collection

if TYPE_CHECKING:
    import bpy

LOD_LEVEL_PROPERTY = "mtree_lod_level"
SOURCE_VERTEX_ATTRIBUTE = "lod0_vertex"
COARSE_LEVELS = (1, 2)


def lod_object_name(obj: bpy.types.Object, level: int) -> str:
    return f"{obj.name}_LOD{level}"


def is_lod_object(obj: bpy.types.Object) -> bool:
    return obj.get(LOD_LEVEL_PROPERTY) is not None


def find_lod_objects(obj: bpy.types.Object) -> list:
    import bpy

    found = []
    for level in COARSE_LEVELS:
        lod = bpy.data.objects.get(lod_object_name(obj, level))
        if lod is not None and lod.type == "MESH" and is_lod_object(lod):
            found.append(lod)
    return found


def source_vertices(bark_vertex_count: int, kept_cards, vertices_per_card: int) -> np.ndarray:
    bark = np.arange(bark_vertex_count, dtype=np.int32)
    cards = (
        bark_vertex_count
        + np.asarray(kept_cards, dtype=np.int32).reshape(-1, 1) * vertices_per_card
        + np.arange(vertices_per_card, dtype=np.int32).reshape(1, -1)
    ).reshape(-1)
    return np.concatenate([bark, cards]).astype(np.int32)


def write_lod_object(obj, level, cpp_mesh, sources, material_slots):
    import bpy

    from ..mesh_utils import create_mesh_from_cpp

    name = lod_object_name(obj, level)
    lod = bpy.data.objects.get(name)
    if lod is not None and (lod.type != "MESH" or not is_lod_object(lod)):
        raise ValueError(f"'{name}' exists and is not a leaf LOD of {obj.name}; rename it")
    if lod is None:
        lod = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        lod[LOD_LEVEL_PROPERTY] = level

    link_exclusively(lod, owning_collection(obj))
    unbind_lod(lod)
    attach_to_tree(lod, obj)

    lod.data.clear_geometry()
    create_mesh_from_cpp(lod.data, cpp_mesh, flat_shading=True, material_slots=material_slots)
    for index in range(material_slots):
        lod.data.materials[index] = obj.data.materials[index]

    attribute = lod.data.attributes.get(SOURCE_VERTEX_ATTRIBUTE)
    if attribute is not None:
        lod.data.attributes.remove(attribute)
    attribute = lod.data.attributes.new(SOURCE_VERTEX_ATTRIBUTE, "INT", "POINT")
    attribute.data.foreach_set("value", sources)
    return lod


def read_source_vertices(lod) -> np.ndarray:
    sources = np.zeros(len(lod.data.vertices), dtype=np.int32)
    lod.data.attributes[SOURCE_VERTEX_ATTRIBUTE].data.foreach_get("value", sources)
    return sources


def remove_lod_objects(obj) -> None:
    import bpy

    for lod in find_lod_objects(obj):
        mesh = lod.data
        bpy.data.objects.remove(lod, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def attach_to_tree(lod, obj) -> None:
    lod.parent = obj
    lod.matrix_parent_inverse.identity()
    lod.matrix_basis.identity()


def unbind_lod(lod) -> None:
    for modifier in [m for m in lod.modifiers if m.type == "ARMATURE"]:
        lod.modifiers.remove(modifier)
    lod.vertex_groups.clear()


def bind_lods(obj, armature_object, rig) -> None:
    from .wind_animation import write_vertex_groups

    for lod in find_lod_objects(obj):
        unbind_lod(lod)
        write_vertex_groups(lod, armature_object.name, rig, read_source_vertices(lod))
        modifier = lod.modifiers.new(name="FoliageRig", type="ARMATURE")
        modifier.object = armature_object
        modifier.use_vertex_groups = True

        lod.parent = armature_object
        lod.matrix_parent_inverse = obj.matrix_parent_inverse.copy()
        lod.matrix_basis = obj.matrix_basis.copy()


def unbind_lods(obj) -> None:
    for lod in find_lod_objects(obj):
        unbind_lod(lod)
        attach_to_tree(lod, obj)


def per_vertex_uvs(mesh, layer_index: int) -> np.ndarray:
    loops = np.zeros(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loops)
    uvs = np.zeros(len(mesh.loops) * 2, dtype=np.float32)
    mesh.uv_layers[layer_index].data.foreach_get("uv", uvs)
    per_vertex = np.zeros((len(mesh.vertices), 2), dtype=np.float32)
    per_vertex[loops] = uvs.reshape(-1, 2)
    return per_vertex


def copy_foliage_streams(obj) -> None:
    from .formats.foliage import FoliageExporter

    source = obj.data
    stream_layers = [
        index
        for index, layer in enumerate(source.uv_layers)
        if layer.name in (FoliageExporter.WIND_UV_NAME, FoliageExporter.HEIGHT_UV_NAME)
    ]
    last_slot = max(stream_layers, default=0)

    for lod in find_lod_objects(obj):
        target = lod.data
        sources = read_source_vertices(lod)
        loops = np.zeros(len(target.loops), dtype=np.int32)
        target.loops.foreach_get("vertex_index", loops)

        for index in range(1, last_slot + 1):
            name = source.uv_layers[index].name
            if len(target.uv_layers) <= index:
                target.uv_layers.new(name=name)
            if target.uv_layers[index].name != name:
                raise ValueError(f"{lod.name}: UV slot {index} is not '{name}'")
            values = per_vertex_uvs(source, index)[sources][loops]
            target.uv_layers[index].data.foreach_set("uv", values.ravel())

        colors = source.color_attributes.get(FoliageExporter.COLOR_NAME)
        if colors is None or colors.domain != "POINT":
            continue
        values = np.zeros(len(source.vertices) * 4, dtype=np.float32)
        colors.data.foreach_get("color", values)
        if FoliageExporter.COLOR_NAME not in target.color_attributes:
            target.color_attributes.new(FoliageExporter.COLOR_NAME, "FLOAT_COLOR", "POINT")
        target.color_attributes[FoliageExporter.COLOR_NAME].data.foreach_set(
            "color", values.reshape(-1, 4)[sources].ravel()
        )


def show_lod(obj, level: int) -> None:
    obj.display_type = "TEXTURED" if level == 0 else "BOUNDS"
    for lod in find_lod_objects(obj):
        lod.hide_viewport = lod[LOD_LEVEL_PROPERTY] != level
        lod.hide_render = True
