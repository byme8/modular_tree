from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bpy

from .foliage_bones import build_rig, pose_rig
from .formats.foliage_rig import (
    SOURCE_FLOAT_ATTRIBUTES,
    SOURCE_VECTOR_ATTRIBUTES,
    FoliageRigExporter,
)
from .tree_asset import armature_name_for, find_armature, owning_collection

MORPH_PREFIX = "Wind_"
DEFAULT_DIRECTIONS = 8


def clear_wind_morphs(obj: bpy.types.Object) -> None:
    shape_keys = obj.data.shape_keys
    if shape_keys is None:
        return

    for key in [
        block for block in shape_keys.key_blocks if block.name.startswith(MORPH_PREFIX)
    ]:
        obj.shape_key_remove(key)

    remaining = obj.data.shape_keys
    if remaining is not None and len(remaining.key_blocks) == 1:
        obj.shape_key_clear()


def count_wind_morphs(obj: bpy.types.Object) -> int:
    shape_keys = obj.data.shape_keys
    if shape_keys is None:
        return 0
    return sum(1 for key in shape_keys.key_blocks if key.name.startswith(MORPH_PREFIX))


def apply_wind_pose(armature: bpy.types.Object, rig, angle_degrees: float, strength: float) -> None:
    from mathutils import Matrix

    matrices = pose_rig(rig, angle_degrees, strength)

    by_name = {
        f"stem{bone.stem_id}_seg{bone.segment}": index
        for index, bone in enumerate(rig.bones)
    }

    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"

        index = by_name.get(pose_bone.name)
        if index is None:
            pose_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            continue

        world = Matrix([list(row) for row in matrices[index]])

        parent = pose_bone.parent
        if parent is not None:
            parent_index = by_name.get(parent.name)
            if parent_index is not None:
                world = Matrix([list(row) for row in matrices[parent_index]]).inverted() @ world

        pose_bone.rotation_quaternion = world.to_3x3().to_quaternion()


_rig_cache = {}


def _cache_key(mesh, segments_per_chain, stiffness):
    return (
        mesh.name,
        len(mesh.vertices),
        segments_per_chain,
        tuple(sorted(stiffness.items())),
    )


def clear_rig_cache():
    _rig_cache.clear()


def build_rig_for(mesh, segments_per_chain, stiffness, use_cache=False):
    """Build the bone rig for a mesh.

    The rig depends only on the mesh and the stiffness settings, never on the
    wind being applied to it, so posing the same tree repeatedly rebuilds an
    identical structure. That dominates the cost of a pose: on a 54k-vertex tree
    the rig takes ~626ms to build against ~92ms to pose. Interactive callers
    pass use_cache; bake paths do not, so a bake always reads current geometry.
    """
    key = _cache_key(mesh, segments_per_chain, stiffness) if use_cache else None
    if key is not None:
        cached = _rig_cache.get(key)
        if cached is not None:
            return cached

    reader = FoliageRigExporter(
        mesh=mesh,
        segments_per_chain=segments_per_chain,
        create_armature=False,
    )
    source = reader._read_source_attributes()

    rig = build_rig(
        vertex_positions=source["positions"],
        stem_ids=source["stem_id"],
        branch_extents=source["branch_extent"],
        radii=source["radius"],
        pivot_positions=source["pivot_position"],
        directions=source["direction"],
        hierarchy_depths=source["hierarchy_depth"],
        segments_per_chain=segments_per_chain,
        **stiffness,
    )

    if key is not None:
        _rig_cache.clear()
        _rig_cache[key] = rig

    return rig


def _write_pose_keys_from_rest(obj, mesh, rig, directions, step):
    import numpy as np

    from .foliage_bones import deform_vertices

    vertex_count = len(mesh.vertices)
    rest = np.empty(vertex_count * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", rest)
    rest = rest.reshape(-1, 3).astype(np.float64)

    largest = 0.0
    for index in range(directions):
        angle = index * step
        posed = deform_vertices(rig, rest, pose_rig(rig, angle, 1.0))
        largest = max(largest, float(np.abs(posed - rest).max()))

        key = obj.shape_key_add(
            name=f"{MORPH_PREFIX}{int(round(angle)):03d}", from_mix=False
        )
        key.data.foreach_set("co", np.ascontiguousarray(posed, dtype=np.float32).ravel())
        key.slider_min = 0.0
        key.slider_max = 1.0
        key.value = 0.0

    mesh.update()
    return largest


def _bake_without_armature(obj, mesh, directions, segments_per_chain, stiffness):
    import numpy as np

    from .foliage_bones import deform_vertices, validate_rig

    rig = build_rig_for(mesh, segments_per_chain, stiffness)

    if rig.bone_count == 0:
        return 0, "No branch chains found; nothing to bake."

    problems = validate_rig(rig)
    if problems:
        return 0, "Rig failed validation: " + "; ".join(problems[:2])

    vertex_count = len(mesh.vertices)
    if len(rig.bone_indices_a) != vertex_count:
        return 0, (
            f"Rig binds {len(rig.bone_indices_a)} vertices but the mesh has {vertex_count}"
        )

    clear_wind_morphs(obj)
    if mesh.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)

    largest = _write_pose_keys_from_rest(obj, mesh, rig, directions, 360.0 / directions)
    return directions, f"{directions} morphs, max offset {largest:.3f}m"


def bake_wind_morphs(
    obj: bpy.types.Object,
    directions: int = DEFAULT_DIRECTIONS,
    segments_per_chain: int = 4,
    trunk_base_stiffness: float = 1.0,
    trunk_tip_stiffness: float = 0.6,
    branch_stiffness: float = 1.0,
    junction_falloff: float = 0.6,
    keep_armature: bool = False,
) -> tuple[int, str]:
    import bpy

    mesh = obj.data

    missing = [
        name
        for name in SOURCE_FLOAT_ATTRIBUTES + SOURCE_VECTOR_ATTRIBUTES
        if name not in mesh.attributes
    ]
    if missing:
        return 0, f"Missing attribute '{missing[0]}'; generate the tree first."

    stiffness = {
        "trunk_base_stiffness": trunk_base_stiffness,
        "trunk_tip_stiffness": trunk_tip_stiffness,
        "branch_stiffness": branch_stiffness,
        "junction_falloff": junction_falloff,
    }

    if not keep_armature:
        return _bake_without_armature(obj, mesh, directions, segments_per_chain, stiffness)

    placement = obj.matrix_world.copy()

    stale = find_armature(obj)
    if stale is not None:
        remove_bake_rig(obj, stale)

    exporter = FoliageRigExporter(
        mesh=mesh,
        mesh_object=obj,
        segments_per_chain=segments_per_chain,
        create_armature=True,
        armature_name=armature_name_for(obj),
        collection=owning_collection(obj),
        **stiffness,
    )

    result = exporter.export()
    if not result.success:
        return 0, result.message

    armature = exporter.armature_object
    if armature is None:
        return 0, "Rig built but no armature was created."

    rig = exporter.rig
    clear_wind_morphs(obj)
    if mesh.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)

    step = 360.0 / directions
    largest = 0.0
    completed = False

    try:
        largest = _write_pose_keys_from_rest(obj, mesh, rig, directions, step)

        apply_wind_pose(armature, rig, 0.0, 0.0)
        completed = True

    finally:
        if not completed:
            clear_wind_morphs(obj)
            remove_bake_rig(obj, armature)
            obj.matrix_world = placement

    return directions, f"{directions} morphs, max offset {largest:.3f}m"


def remove_bake_rig(obj: bpy.types.Object, armature: bpy.types.Object) -> None:
    import bpy

    for modifier in [m for m in obj.modifiers if m.type == "ARMATURE" and m.object == armature]:
        obj.modifiers.remove(modifier)

    if obj.parent == armature:
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world

    obj.vertex_groups.clear()

    data = armature.data
    bpy.data.objects.remove(armature, do_unlink=True)
    if data.users == 0:
        bpy.data.armatures.remove(data)
