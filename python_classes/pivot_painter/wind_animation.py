from __future__ import annotations

import contextlib

from .foliage_bones import build_rig, pose_rig
from .formats.foliage_rig import FoliageRigExporter
from .tree_asset import armature_name_for, authored_world_matrix, owning_collection

ACTION_PREFIX = "Wind_"
DEFAULT_DIRECTIONS = 8
STRENGTH_STOPS = (0.0, 0.25, 1.0)
FIRST_CLIP_FRAME = 1
FRAMES_PER_UNIT_STRENGTH = 4
MIN_DISPLAY_BONE_LENGTH = 1e-3
MAX_NAME_BYTES = 63

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


def bone_name(armature_name: str, bone) -> str:
    name = f"{armature_name}_stem{bone.stem_id}_seg{bone.segment}"
    if len(name.encode()) > MAX_NAME_BYTES:
        raise ValueError(f"Bone name '{name}' exceeds Blender's limit; shorten the tree's name.")
    return name


def build_armature(obj, rig, collection=None):
    import bpy
    from mathutils import Vector

    name = armature_name_for(obj)
    bone_names = [bone_name(name, bone) for bone in rig.bones]
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
        for bone, name_of_bone in zip(rig.bones, bone_names, strict=True):
            edit_bone = armature_data.edit_bones.new(name_of_bone)

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
    write_vertex_groups(obj, armature_object.name, rig)

    modifier = next((m for m in obj.modifiers if m.type == "ARMATURE"), None)
    if modifier is None:
        modifier = obj.modifiers.new(name="FoliageRig", type="ARMATURE")

    modifier.object = armature_object
    modifier.use_vertex_groups = True

    armature_object.matrix_world = authored_world_matrix(obj)
    obj.parent = armature_object
    obj.matrix_parent_inverse = armature_object.matrix_world.inverted()

    from .leaf_lods import bind_lods

    bind_lods(obj, armature_object, rig)


def write_vertex_groups(obj, armature_name: str, rig, source_vertices=None) -> None:
    names = [bone_name(armature_name, bone) for bone in rig.bones]
    for name in names:
        if name not in obj.vertex_groups:
            obj.vertex_groups.new(name=name)

    if source_vertices is None:
        source_vertices = range(len(rig.bone_indices_a))

    for vertex, source in enumerate(source_vertices):
        index_a = int(rig.bone_indices_a[source])
        index_b = int(rig.bone_indices_b[source])
        weight_a = float(rig.weights_a[source])

        group_a = obj.vertex_groups[names[index_a]]
        group_b = obj.vertex_groups[names[index_b]]

        if index_b == index_a:
            group_a.add([vertex], 1.0, "REPLACE")
            continue

        group_a.add([vertex], weight_a, "REPLACE")
        group_b.add([vertex], 1.0 - weight_a, "REPLACE")


def apply_pose(armature_object, rig, angle_degrees: float, strength: float) -> None:
    from mathutils import Matrix

    matrices = pose_rig(rig, angle_degrees, strength)

    by_name = {bone_name(armature_object.name, bone): index for index, bone in enumerate(rig.bones)}

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

    for strength in stops:
        apply_pose(armature_object, rig, angle_degrees, strength)

        frame = stop_frame(strength)
        for pose_bone in armature_object.pose.bones:
            pose_bone.keyframe_insert("location", frame=frame, group=pose_bone.name)
            pose_bone.keyframe_insert("rotation_quaternion", frame=frame, group=pose_bone.name)

    for fcurve in action_fcurves(action):
        for point in fcurve.keyframe_points:
            point.interpolation = "LINEAR"

    clear_pose(armature_object)
    return action


def stop_frame(strength: float) -> int:
    return FIRST_CLIP_FRAME + round(strength * FRAMES_PER_UNIT_STRENGTH)


def action_fcurves(action) -> list:
    layers = getattr(action, "layers", None)
    if not layers:
        return list(action.fcurves)
    return [
        fcurve
        for layer in layers
        for strip in layer.strips
        for channelbag in strip.channelbags
        for fcurve in channelbag.fcurves
    ]


@contextlib.contextmanager
def own_clips_on_nla(armature_names):
    import bpy

    created_animation_data = []
    previous_actions = {}
    added_tracks = []

    try:
        for armature_name in armature_names:
            armature = bpy.data.objects[armature_name]
            if armature.animation_data is None:
                armature.animation_data_create()
                created_animation_data.append(armature_name)

            animation_data = armature.animation_data
            previous = animation_data.action
            previous_actions[armature_name] = None if previous is None else previous.name
            animation_data.action = None

            prefix = action_prefix_for(armature)
            for action in sorted(
                (a for a in bpy.data.actions if a.name.startswith(prefix)), key=lambda a: a.name
            ):
                track = animation_data.nla_tracks.new()
                track.name = action.name
                strip = track.strips.new(action.name, int(action.frame_range[0]), action)
                strip.name = action.name
                added_tracks.append((armature_name, track.name))

        yield

    finally:
        for armature_name, track_name in added_tracks:
            armature = bpy.data.objects.get(armature_name)
            if armature is None or armature.animation_data is None:
                continue
            track = armature.animation_data.nla_tracks.get(track_name)
            if track is not None:
                armature.animation_data.nla_tracks.remove(track)

        for armature_name, action_name_before in previous_actions.items():
            armature = bpy.data.objects.get(armature_name)
            if armature is not None and armature.animation_data is not None:
                armature.animation_data.action = (
                    None if action_name_before is None else bpy.data.actions.get(action_name_before)
                )

        for armature_name in created_animation_data:
            armature = bpy.data.objects.get(armature_name)
            if armature is not None:
                armature.animation_data_clear()


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
