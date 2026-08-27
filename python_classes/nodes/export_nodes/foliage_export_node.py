"""Node that bakes foliage wind streams and the optional bone rig onto the mesh.

Writes onto the mesh rather than to a file. Mesh Export writes the file
afterwards, and its payload check exists because running it first produces a
file silently missing this data.
"""

from __future__ import annotations

import bpy

from ...pivot_painter.tree_asset import find_armature, resolve_mesh_object
from ..base_types.node import MtreeNode


class MTreeFoliageExport(bpy.types.Node, MtreeNode):
    """Bake foliage wind and shading streams onto the tree mesh"""

    bl_idname = "mt_FoliageExportNode"
    bl_label = "Foliage Export"

    create_armature: bpy.props.BoolProperty(
        name="Create Armature",
        description=(
            "Create a Blender armature from the rig and bind the mesh to it. "
            "The engine reads the bone records and the binding stream, not this "
            "armature, but without it the rig cannot be seen or posed"
        ),
        default=True,
    )

    preview_wind_strength: bpy.props.FloatProperty(
        name="Wind Strength",
        description=(
            "Pose the armature as if wind of this strength were blowing. Drives "
            "the same stiffness curve the engine uses, so a branch that shears "
            "away from its limb here will shear in the engine too"
        ),
        default=0.0,
        min=0.0,
        max=1.0,
        update=lambda self, context: self.apply_pose_preview(),
    )
    preview_wind_angle: bpy.props.FloatProperty(
        name="Wind Angle",
        description="Direction the preview wind blows from, in degrees",
        default=0.0,
        min=0.0,
        max=360.0,
        update=lambda self, context: self.apply_pose_preview(),
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

        from .foliage_wind_node import MTreeFoliageWind, find_foliage_wind_settings

        settings = find_foliage_wind_settings(self.get_node_tree())

        has_wind_node = any(
            node.bl_idname == MTreeFoliageWind.bl_idname
            for node in self.get_node_tree().nodes
        )
        if not has_wind_node:
            layout.label(text="No Foliage Wind node; using defaults", icon="INFO")

        if settings["build_bone_rig"]:
            rig_box = layout.box()
            rig_box.label(text=f"Bone Rig: {settings['segments_per_chain']} per branch")
            rig_box.prop(self, "create_armature")

            if self.create_armature:
                preview_box = layout.box()
                preview_box.label(text="Pose Preview", icon="ARMATURE_DATA")

                if self._get_armature() is None:
                    preview_box.label(text="Bake first to create the rig", icon="INFO")
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

    def _get_armature(self):
        obj = self._get_tree_object()
        if obj is None:
            return None
        return find_armature(obj)

    def clear_pose_preview(self):
        armature = self._get_armature()
        if armature is None:
            return

        for bone in armature.pose.bones:
            bone.rotation_mode = "QUATERNION"
            bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)

        self.preview_wind_strength = 0.0
        self.status_message = "Rest pose"
        self.status_is_error = False

    def apply_pose_preview(self):
        from ...pivot_painter.formats.foliage_rig import FoliageRigExporter
        from ...pivot_painter.wind_morphs import apply_wind_pose, build_rig_for

        armature = self._get_armature()
        obj = self._get_tree_object()
        if armature is None or obj is None:
            return

        from .foliage_wind_node import find_foliage_wind_settings

        settings = find_foliage_wind_settings(self.get_node_tree())
        if not settings["build_bone_rig"]:
            return

        probe = FoliageRigExporter(
            mesh=obj.data,
            segments_per_chain=settings["segments_per_chain"],
            create_armature=False,
        )
        if probe.validate() is not None:
            return

        rig = build_rig_for(
            obj.data,
            settings["segments_per_chain"],
            {
                key: settings[key]
                for key in (
                    "trunk_base_stiffness",
                    "trunk_tip_stiffness",
                    "branch_stiffness",
                    "junction_falloff",
                )
            },
            use_cache=True,
        )

        apply_wind_pose(
            armature, rig, self.preview_wind_angle, self.preview_wind_strength
        )

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
            create_armature=self.create_armature,
            **settings,
        )

        if result == {"FINISHED"}:
            if settings["build_bone_rig"] and self.create_armature:
                self.preview_wind_strength = 0.0
                self.status_message = "Baked. Drag Wind Strength to preview."
            else:
                self.status_message = "Baked"
            self.status_is_error = False
        else:
            self.status_message = "Bake failed"
            self.status_is_error = True
