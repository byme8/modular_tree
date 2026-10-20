"""Utilities for creating Blender meshes from C++ mesh data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import bpy


# Attribute definitions for tree meshes
FLOAT_ATTRIBUTES = [
    "radius",
    "stem_id",
    "hierarchy_depth",
    "branch_extent",
    "vein_distance",
    "phyllotaxis_angle",
]

VECTOR3_ATTRIBUTES = [
    "direction",
    "pivot_position",
    "leaf_attachment",
]


def create_mesh_from_cpp(
    mesh: bpy.types.Mesh,
    cpp_mesh,
    flat_shading: bool = False,
    material_slots: int = 0,
) -> None:
    _add_geometry(mesh, cpp_mesh, flat_shading)
    _add_attributes(mesh, cpp_mesh)
    _add_uvs(mesh, cpp_mesh)
    _assign_tree_materials(mesh, material_slots)
    mesh.update(calc_edges=True)

    # The debug overlay caches a GPU batch built from these attributes; without
    # this it would keep drawing the previous tree's colours.
    from .viewport.foliage_debug_drawer import invalidate_debug_cache

    invalidate_debug_cache()


TREE_MATERIALS = (
    ("Bark", (0.27, 0.17, 0.10, 1.0)),
    ("Leaf", (0.18, 0.40, 0.12, 1.0)),
)


def shared_tree_material(name: str, color) -> bpy.types.Material:
    import bpy

    material = bpy.data.materials.get(name)
    if material is not None:
        return material

    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    if material.node_tree is None:
        material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is not None:
        shader.inputs["Base Color"].default_value = color
    return material


def _assign_tree_materials(mesh: bpy.types.Mesh, slot_count: int) -> None:
    while len(mesh.materials) < slot_count:
        mesh.materials.append(None)

    for index, (name, color) in enumerate(TREE_MATERIALS[:slot_count]):
        if mesh.materials[index] is None:
            mesh.materials[index] = shared_tree_material(name, color)


def _add_geometry(mesh: bpy.types.Mesh, cpp_mesh, flat_shading: bool = False) -> None:
    verts = cpp_mesh.get_vertices()
    faces = np.copy(cpp_mesh.get_polygons()[::-1])
    materials = np.copy(cpp_mesh.get_material_indices()[::-1]).astype(np.int32)

    mesh.vertices.add(int(len(verts) / 3))
    mesh.vertices.foreach_set("co", verts)

    mesh.loops.add(len(faces))
    mesh.loops.foreach_set("vertex_index", faces)

    polygon_count = int(len(faces) / 4)
    loop_start = np.arange(0, len(faces), 4, dtype=np.int32)
    loop_total = np.ones(polygon_count, dtype=np.int32) * 4
    mesh.polygons.add(polygon_count)
    mesh.polygons.foreach_set("loop_start", loop_start)
    mesh.polygons.foreach_set("loop_total", loop_total)
    mesh.polygons.foreach_set("use_smooth", np.full(polygon_count, not flat_shading, dtype=bool))
    mesh.polygons.foreach_set("material_index", materials)


def _add_attributes(mesh: bpy.types.Mesh, cpp_mesh) -> None:
    """Add custom attributes to the mesh."""
    for attr_name in FLOAT_ATTRIBUTES:
        if not cpp_mesh.has_float_attribute(attr_name):
            continue
        data = cpp_mesh.get_float_attribute(attr_name)
        if attr_name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[attr_name])
        mesh.attributes.new(name=attr_name, type="FLOAT", domain="POINT")
        mesh.attributes[attr_name].data.foreach_set("value", data)

    for attr_name in VECTOR3_ATTRIBUTES:
        if not cpp_mesh.has_vector3_attribute(attr_name):
            continue
        data = cpp_mesh.get_vector3_attribute(attr_name)
        if attr_name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[attr_name])
        mesh.attributes.new(name=attr_name, type="FLOAT_VECTOR", domain="POINT")
        mesh.attributes[attr_name].data.foreach_set("vector", data)


def _add_uvs(mesh: bpy.types.Mesh, cpp_mesh) -> None:
    """Add UV coordinates to the mesh."""
    uv_data = cpp_mesh.get_uvs()
    uv_data.shape = (len(uv_data) // 2, 2)
    uv_loops = np.copy(cpp_mesh.get_uv_loops()[::-1])  # Reverse since faces are reversed
    uvs = uv_data[uv_loops].flatten()

    uv_layer = mesh.uv_layers.new() if len(mesh.uv_layers) == 0 else mesh.uv_layers[0]
    uv_layer.data.foreach_set("uv", uvs)


def create_leaf_mesh_from_cpp(
    mesh: bpy.types.Mesh,
    cpp_mesh,
) -> None:
    """Populate a Blender mesh from C++ leaf mesh data.

    Handles triangle meshes stored as degenerate quads (4th vertex = 3rd vertex)
    by the LeafShapeGenerator's ear clipping triangulation.

    Args:
        mesh: The Blender mesh to populate.
        cpp_mesh: The C++ Mesh object from LeafShapeGenerator.generate().
    """
    _add_leaf_geometry(mesh, cpp_mesh)
    _add_leaf_attributes(mesh, cpp_mesh)
    _add_leaf_uvs(mesh, cpp_mesh)
    mesh.update(calc_edges=True)


def _add_leaf_geometry(mesh: bpy.types.Mesh, cpp_mesh) -> None:
    """Add vertices and triangle polygons to the leaf mesh."""
    verts = cpp_mesh.get_vertices()
    raw_faces = cpp_mesh.get_polygons()  # Stride 4, degenerate quads

    mesh.vertices.add(len(verts) // 3)
    mesh.vertices.foreach_set("co", verts)

    # Convert degenerate quads to proper triangles (take first 3 of each quad)
    raw_faces = raw_faces.reshape(-1, 4)
    tri_faces = np.ascontiguousarray(raw_faces[:, :3])
    tri_faces_flat = np.copy(tri_faces[:, ::-1].flatten())  # Reverse winding to flip normals

    num_tris = len(tri_faces)
    mesh.loops.add(num_tris * 3)
    mesh.loops.foreach_set("vertex_index", tri_faces_flat)

    loop_start = np.arange(0, num_tris * 3, 3, dtype=np.int32)
    loop_total = np.ones(num_tris, dtype=np.int32) * 3
    mesh.polygons.add(num_tris)
    mesh.polygons.foreach_set("loop_start", loop_start)
    mesh.polygons.foreach_set("loop_total", loop_total)
    mesh.polygons.foreach_set("use_smooth", np.ones(num_tris, dtype=bool))


def _add_leaf_attributes(mesh: bpy.types.Mesh, cpp_mesh) -> None:
    """Add custom attributes to the leaf mesh (only those that exist)."""
    for attr_name in FLOAT_ATTRIBUTES:
        if not cpp_mesh.has_float_attribute(attr_name):
            continue
        data = cpp_mesh.get_float_attribute(attr_name)
        if attr_name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[attr_name])
        mesh.attributes.new(name=attr_name, type="FLOAT", domain="POINT")
        mesh.attributes[attr_name].data.foreach_set("value", data)


def _add_leaf_uvs(mesh: bpy.types.Mesh, cpp_mesh) -> None:
    """Add UV coordinates to the leaf mesh."""
    uv_data = cpp_mesh.get_uvs()
    if len(uv_data) == 0:
        return

    uv_data = uv_data.reshape(-1, 2)
    raw_uv_loops = cpp_mesh.get_uv_loops()  # Stride 4

    # Convert degenerate quad UV loops to triangle UV loops
    raw_uv_loops = raw_uv_loops.reshape(-1, 4)
    tri_uv_loops = np.ascontiguousarray(raw_uv_loops[:, :3])
    tri_uv_loops_flat = np.copy(
        tri_uv_loops[:, ::-1].flatten()
    )  # Reverse winding to match face reversal

    uvs = uv_data[tri_uv_loops_flat].flatten()

    uv_layer = mesh.uv_layers.new() if len(mesh.uv_layers) == 0 else mesh.uv_layers[0]
    uv_layer.data.foreach_set("uv", uvs)
