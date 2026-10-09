"""Node that bakes foliage wind streams and the optional bone rig onto the mesh.

Writes onto the mesh rather than to a file. Mesh Export writes the file
afterwards, and its payload check exists because running it first produces a
file silently missing this data.
"""

from __future__ import annotations

import bpy

from ...pivot_painter.tree_asset import find_armature, resolve_mesh_object
from ..base_types.node import MtreeNode
from ..debounce import schedule_preview


def on_update_preview(node, context):
    schedule_preview(node, "apply_pose_preview")


class MTreeFoliageExport(bpy.types.Node, MtreeNode):
    """Bake foliage wind and shading streams onto the tree mesh"""

    bl_idname = "mt_FoliageExportNode"
    bl_label = "Foliage Export"

    preview_wind_strength: bpy.props.FloatProperty(
        name="Wind Strength",
        description=(
            "Blend the baked wind morphs as if wind of this strength were "
            "blowing. These are the same shape keys the engine reads, so a "
            "branch that shears away from its limb here will shear there too"
        ),
        default=0.0,
        min=0.0,
        max=1.0,
        update=on_update_preview,
    )
    preview_wind_angle: bpy.props.FloatProperty(
        name="Wind Angle",
        description="Direction the preview wind blows from, in degrees",
        default=0.0,
        min=0.0,
        max=360.0,
        update=on_update_preview,
    )

    status_message: bpy.props.StringProperty(default="")
    status_is_error: bpy.props.BoolProperty(default=False)

    def init(self, context):
        self.add_input("mt_TreeSocket", "Tree", is_property=False)

    def draw(self, context, layout):
        if self.status_message:
            status_row = layout.row()
            status_row.alert = self.status_is_error
            icon = "ERROR" if self.status_is_error else "CHECKMARK"
            status_row.label(text=self.status_message, icon=icon)

        from .foliage_wind_node import (
            MTreeFoliageWind,
            describe_rig_layout,
            find_foliage_wind_settings,
        )

        settings = find_foliage_wind_settings(self.get_node_tree())

        has_wind_node = any(
            node.bl_idname == MTreeFoliageWind.bl_idname
            for node in self.get_node_tree().nodes
        )
        if not has_wind_node:
            layout.label(text="No Foliage Wind node; using defaults", icon="INFO")

        if settings["build_bone_rig"]:
            rig_box = layout.box()
            rig_box.label(text=f"Bone Rig: {describe_rig_layout(settings)}")

            preview_box = layout.box()
            preview_box.label(text="Wind Preview", icon="ARMATURE_DATA")

            if self._armature() is None:
                preview_box.label(text="Generate the tree to build the rig", icon="INFO")
            else:
                preview_box.prop(self, "preview_wind_strength", slider=True)
                preview_box.prop(self, "preview_wind_angle", slider=True)

                reset = preview_box.operator("mtree.node_function", text="Rest Pose")
                reset.node_tree_name = self.get_node_tree().name
                reset.node_name = self.name
                reset.function_name = "clear_pose_preview"

        debug_box = layout.box()
        debug_box.label(text="Debug View")
        debug_box.prop(context.scene, "mtree_foliage_debug_channel", text="")

        export_row = layout.row()
        export_row.enabled = self._has_valid_input()
        props = export_row.operator("mtree.node_function", text="Bake")
        props.node_tree_name = self.get_node_tree().name
        props.node_name = self.name
        props.function_name = "do_export"

    def _has_valid_input(self):
        return self.get_mesher() is not None

    def _get_tree_object(self):
        mesher = self.get_mesher()
        if mesher is None:
            return None
        if not hasattr(mesher, "tree_object") or not mesher.tree_object:
            return None
        return resolve_mesh_object(mesher.tree_object)

    def _armature(self):
        obj = self._get_tree_object()
        if obj is None:
            return None
        return find_armature(obj)

    def clear_pose_preview(self):
        from ...pivot_painter.wind_animation import clear_pose

        armature = self._armature()
        if armature is None:
            return

        clear_pose(armature)

        self.preview_wind_strength = 0.0
        self.status_message = "Rest pose"
        self.status_is_error = False

    def apply_pose_preview(self):
        from ...pivot_painter.wind_animation import apply_pose, build_rig_for

        obj = self._get_tree_object()
        armature = self._armature()
        if obj is None or armature is None:
            return

        from .foliage_wind_node import find_foliage_wind_settings, rig_settings

        settings = find_foliage_wind_settings(self.get_node_tree())

        rig = build_rig_for(
            obj.data,
            settings["segments_per_chain"],
            rig_settings(settings),
            use_cache=True,
        )

        apply_pose(armature, rig, self.preview_wind_angle, self.preview_wind_strength)

    def do_export(self):
        self.status_message = ""
        self.status_is_error = False

        obj = self._get_tree_object()
        if obj is None:
            self.status_message = "No tree object found. Generate tree first."
            self.status_is_error = True
            return

        from .foliage_wind_node import find_foliage_wind_settings

        settings = find_foliage_wind_settings(self.get_node_tree())

        result = bpy.ops.mtree.export_foliage(
            "EXEC_DEFAULT",
            object_name=obj.name,
            **settings,
        )

        if result == {"FINISHED"}:
            if settings["build_bone_rig"] and self._armature() is not None:
                self.preview_wind_strength = 0.0
                self.status_message = "Baked. Drag Wind Strength to preview."
            else:
                self.status_message = "Baked"
            self.status_is_error = False
        else:
            self.status_message = "Bake failed"
            self.status_is_error = True
