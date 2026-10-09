from __future__ import annotations

from .foliage_bones import build_rig, pose_rig
from .formats.foliage_rig import FoliageRigExporter
from .tree_asset import armature_name_for, owning_collection

ACTION_PREFIX = "Wind_"
DEFAULT_DIRECTIONS = 8
STRENGTH_STOPS = (0.0, 0.25, 1.0)
MIN_DISPLAY_BONE_LENGTH = 1e-3

# PERF: on a 54k-vertex tree the rig costs ~626ms to build against ~92ms to pose,
# so interactive callers pass use_cache. Bake paths do not, and so always read
# current geometry.
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
    key = _cache_key(mesh, segments_per_chain, stiffness) if use_cache else None
    if key is not None:
        cached = _rig_cache.get(key)
        if cached is not None:
            return cached

    source = FoliageRigExporter(
        mesh=mesh,
        segments_per_chain=segments_per_chain,
    )._read_source_attributes()

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


def bone_name(bone) -> str:
    return f"stem{bone.stem_id}_seg{bone.segment}"


def build_armature(obj, rig, collection=None):
    import bpy
    from mathutils import Vector

    name = armature_name_for(obj)
    armature_data = bpy.data.armatures.new(name)
    armature_object = bpy.data.objects.new(name, armature_data)

    (collection or owning_collection(obj)).objects.link(armature_object)

    previous_active = bpy.context.view_layer.objects.active
    previous_mode = bpy.context.mode

    if previous_active is not None and previous_active.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.context.view_layer.objects.active = armature_object

    try:
        bpy.ops.object.mode_set(mode="EDIT")

        edit_bones = []
        for bone in rig.bones:
            edit_bone = armature_data.edit_bones.new(bone_name(bone))

            head = Vector(bone.rest_position.tolist())
            length = max(bone.length, MIN_DISPLAY_BONE_LENGTH)

            edit_bone.head = head
            edit_bone.tail = head + Vector(bone.rest_direction.tolist()) * length
            edit_bone.use_connect = False

            edit_bones.append(edit_bone)

        for index, bone in enumerate(rig.bones):
            if bone.parent >= 0:
                edit_bones[index].parent = edit_bones[bone.parent]

    finally:
        if bpy.context.view_layer.objects.active is armature_object:
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.objects.active = previous_active
        if previous_active is not None and previous_mode.startswith("EDIT"):
            bpy.ops.object.mode_set(mode="EDIT")

    return armature_object


def bind_to_armature(obj, armature_object, rig) -> None:
    write_vertex_groups(obj, rig)

    modifier = next((m for m in obj.modifiers if m.type == "ARMATURE"), None)
    if modifier is None:
        modifier = obj.modifiers.new(name="FoliageRig", type="ARMATURE")

    modifier.object = armature_object
    modifier.use_vertex_groups = True

    armature_object.matrix_world = obj.matrix_world.copy()
    obj.parent = armature_object
    obj.matrix_parent_inverse = armature_object.matrix_world.inverted()


def write_vertex_groups(obj, rig) -> None:
    for bone in rig.bones:
        name = bone_name(bone)
        if name not in obj.vertex_groups:
            obj.vertex_groups.new(name=name)

    for vertex in range(len(rig.bone_indices_a)):
        index_a = int(rig.bone_indices_a[vertex])
        index_b = int(rig.bone_indices_b[vertex])
        weight_a = float(rig.weights_a[vertex])

        group_a = obj.vertex_groups[bone_name(rig.bones[index_a])]
        group_b = obj.vertex_groups[bone_name(rig.bones[index_b])]

        if index_b == index_a:
            group_a.add([vertex], 1.0, "REPLACE")
            continue

        group_a.add([vertex], weight_a, "REPLACE")
        group_b.add([vertex], 1.0 - weight_a, "REPLACE")


def apply_pose(armature_object, rig, angle_degrees: float, strength: float) -> None:
    from mathutils import Matrix

    matrices = pose_rig(rig, angle_degrees, strength)

    by_name = {bone_name(bone): index for index, bone in enumerate(rig.bones)}

    for pose_bone in armature_object.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"

        index = by_name.get(pose_bone.name)
        if index is None:
            pose_bone.matrix_basis = Matrix.Identity(4)
            continue

        bone = rig.bones[index]
        world = Matrix([list(row) for row in matrices[index]])

        parent_world = Matrix.Identity(4)
        if bone.parent >= 0:
            parent_world = Matrix([list(row) for row in matrices[bone.parent]])

        rest = pose_bone.bone.matrix_local
        pose_bone.matrix_basis = rest.inverted() @ parent_world.inverted() @ world @ rest


def clear_pose(armature_object) -> None:
    from mathutils import Matrix

    for pose_bone in armature_object.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.matrix_basis = Matrix.Identity(4)


def count_wind_clips(obj) -> int:
    import bpy

    from .tree_asset import armature_name_for

    prefix = f"{armature_name_for(obj)}_{ACTION_PREFIX}"
    return sum(1 for action in bpy.data.actions if action.name.startswith(prefix))


def action_prefix_for(armature_object) -> str:
    return f"{armature_object.name}_{ACTION_PREFIX}"


def action_name(armature_object, angle_degrees: float) -> str:
    return f"{action_prefix_for(armature_object)}{int(round(angle_degrees)):03d}"


def clear_wind_actions(armature_object) -> None:
    import bpy

    if armature_object.animation_data is not None:
        armature_object.animation_data.action = None

    prefix = action_prefix_for(armature_object)
    for action in [a for a in bpy.data.actions if a.name.startswith(prefix)]:
        bpy.data.actions.remove(action)


def bake_direction_clip(armature_object, rig, angle_degrees: float, stops=STRENGTH_STOPS):
    import bpy

    name = action_name(armature_object, angle_degrees)

    existing = bpy.data.actions.get(name)
    if existing is not None:
        bpy.data.actions.remove(existing)

    action = bpy.data.actions.new(name)

    if armature_object.animation_data is None:
        armature_object.animation_data_create()
    armature_object.animation_data.action = action

    for frame, strength in enumerate(stops, start=1):
        apply_pose(armature_object, rig, angle_degrees, strength)

        for pose_bone in armature_object.pose.bones:
            pose_bone.keyframe_insert("location", frame=frame, group=pose_bone.name)
            pose_bone.keyframe_insert(
                "rotation_quaternion", frame=frame, group=pose_bone.name
            )

    clear_pose(armature_object)
    return action


def bake_wind_clips(
    obj,
    rig,
    directions: int = DEFAULT_DIRECTIONS,
    stops=STRENGTH_STOPS,
) -> tuple[int, str]:
    from .tree_asset import strip_rig

    if rig.bone_count == 0:
        return 0, "No branch chains found; nothing to rig."

    strip_rig(obj)

    armature_object = build_armature(obj, rig)
    bind_to_armature(obj, armature_object, rig)

    step = 360.0 / directions
    for index in range(directions):
        bake_direction_clip(armature_object, rig, index * step, stops)

    armature_object.animation_data.action = None

    return directions, (
        f"{directions} wind clips, {len(stops)} strength stops, {rig.bone_count} bones"
    )
