from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bpy


def armature_name_for(obj: bpy.types.Object) -> str:
    return f"{obj.name}_rig"


def find_armature(obj: bpy.types.Object) -> bpy.types.Object | None:
    import bpy

    armature = bpy.data.objects.get(armature_name_for(obj))
    if armature is None or armature.type != "ARMATURE":
        return None
    return armature


def strip_rig(obj: bpy.types.Object, remove_armature: bool = True) -> None:
    """Removing objects from bpy.data during a node-tree update crashes Blender,
    so callers running inside one pass remove_armature=False to unbind without
    deleting.
    """
    import bpy

    from .wind_animation import clear_rig_cache, clear_wind_actions

    clear_rig_cache()

    if obj.data.shape_keys is not None:
        obj.shape_key_clear()

    for modifier in [m for m in obj.modifiers if m.type == "ARMATURE"]:
        obj.modifiers.remove(modifier)

    if obj.parent is not None and obj.parent.type == "ARMATURE":
        world_placement = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world_placement

    obj.vertex_groups.clear()

    armature = find_armature(obj)
    if armature is not None:
        clear_wind_actions(armature)

        if remove_armature:
            armature_data = armature.data
            bpy.data.objects.remove(armature, do_unlink=True)
            if armature_data.users == 0:
                bpy.data.armatures.remove(armature_data)


def owning_collection(obj: bpy.types.Object) -> bpy.types.Collection:
    import bpy

    if obj.users_collection:
        return obj.users_collection[0]
    return bpy.context.scene.collection


def link_exclusively(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    if obj.name in collection.objects:
        return
    for existing in list(obj.users_collection):
        existing.objects.unlink(obj)
    collection.objects.link(obj)


def resolve_mesh_object(object_name: str) -> bpy.types.Object | None:
    import bpy

    obj = bpy.data.objects.get(object_name)
    if obj is None or not isinstance(obj.data, bpy.types.Mesh):
        return None
    return obj
