from __future__ import annotations

import contextlib
import os

import bpy

from ...pivot_painter.tree_asset import resolve_mesh_object
from ..base_types.node import MtreeNode
from ..pipeline import VARIANTS_ID, mesher_upstream, upstream_node

ENGINE_AXES = {
    "UNITY": {"axis_up": "Y", "axis_forward": "-Z"},
    "UNREAL": {"axis_up": "Z", "axis_forward": "-Y"},
    "BLENDER": {"axis_up": "Z", "axis_forward": "-Y"},
}


@contextlib.contextmanager
def selectable(objects):
    hidden = [obj.name for obj in objects if obj.hide_viewport]
    for name in hidden:
        bpy.data.objects[name].hide_viewport = False
    bpy.context.view_layer.update()
    try:
        yield
    finally:
        for name in hidden:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.hide_viewport = True


@contextlib.contextmanager
def rest_pose(armature_names):
    previous = {}
    for name in armature_names:
        armature = bpy.data.objects[name].data
        previous[name] = armature.pose_position
        armature.pose_position = "REST"
    try:
        yield
    finally:
        for name, position in previous.items():
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.data.pose_position = position


class MTreeMeshExport(bpy.types.Node, MtreeNode):
    bl_idname = "mt_MeshExportNode"
    bl_label = "Mesh Export"

    export_format: bpy.props.EnumProperty(
        name="Format",
        items=[
            ("FBX", "FBX", "Export as FBX; the usual choice for Unity and Unreal"),
            ("GLTF", "glTF 2.0", "Export as glTF/glb"),
            ("BLEND", "Blend File", "Save a copy of the .blend file"),
        ],
        default="FBX",
    )
    target_engine: bpy.props.EnumProperty(
        name="Target",
        description="Axis convention to convert into",
        items=[
            ("UNITY", "Unity", "Y-up: converts so the tree stands upright in Unity"),
            ("UNREAL", "Unreal", "Z-up: matches Unreal's convention"),
            ("BLENDER", "Blender", "No conversion; keeps Blender's Z-up axes"),
        ],
        default="UNITY",
    )
    export_path: bpy.props.StringProperty(
        name="Path",
        subtype="DIR_PATH",
        default="//export/",
    )
    separate_files: bpy.props.BoolProperty(
        name="One File Each",
        description=(
            "Write one file per object. Off writes every object into a single "
            "file, keeping their relative placement"
        ),
        default=True,
    )
    apply_modifiers: bpy.props.BoolProperty(
        name="Apply Modifiers",
        description="Evaluate modifiers into the exported mesh",
        default=True,
    )
    reset_transforms: bpy.props.BoolProperty(
        name="Zero Transforms",
        description=(
            "Export each object at the origin rather than at its placed location. "
            "Turn this on for prefabs, off to preserve a laid-out scene"
        ),
        default=True,
    )

    status_message: bpy.props.StringProperty(default="")
    status_is_error: bpy.props.BoolProperty(default=False)

    payload_summary: bpy.props.StringProperty(default="")

    chain_input = ("mt_MeshSocket", "Source")

    def init(self, context):
        self.add_chain_sockets()

    def draw(self, context, layout):
        if self.status_message:
            status_row = layout.row()
            status_row.alert = self.status_is_error
            icon = "ERROR" if self.status_is_error else "CHECKMARK"
            status_row.label(text=self.status_message, icon=icon)

        layout.prop(self, "export_format")
        layout.prop(self, "target_engine")

        if self.target_engine == "UNITY" and self.export_format == "FBX":
            info = layout.row()
            info.label(text="Import at Scale 1, no rotation fix needed", icon="INFO")

        layout.prop(self, "export_path")

        source_box = layout.box()
        source_box.label(text="Source")
        source_box.label(text=self._describe_source(), icon="LINKED")
        source_box.prop(self, "separate_files")

        options_box = layout.box()
        options_box.label(text="Options")
        options_box.prop(self, "apply_modifiers")
        options_box.prop(self, "reset_transforms")

        payload_box = layout.box()
        payload_box.label(text="Payload")
        if self.payload_summary:
            payload_box.label(text=self.payload_summary, icon="CHECKMARK")
        else:
            payload_box.label(text="Not checked", icon="QUESTION")
        check = payload_box.operator("mtree.node_function", text="Check")
        check.node_tree_name = self.get_node_tree().name
        check.node_name = self.name
        check.function_name = "check_payload"

        props = layout.operator("mtree.node_function", text="Export")
        props.node_tree_name = self.get_node_tree().name
        props.node_name = self.name
        props.function_name = "do_export"

    def check_payload(self):
        objects = self._gather_objects()
        if not objects:
            self.payload_summary = ""
            self.status_message = "No mesh objects found"
            self.status_is_error = True
            return

        uv_layers, color_layers = self._describe_payload(objects)

        if not uv_layers and not color_layers:
            self.payload_summary = f"{len(objects)} object(s), no UV or colour layers"
            self.status_message = "Nothing to export: bake foliage or pivot painter data first"
            self.status_is_error = True
            return

        rigged, bones, clips = self._describe_rig(objects)
        rig_part = (
            f"Rig: {rigged}/{len(objects)} rigged, {bones} bones, {clips} clips"
            if rigged
            else "Rig: none"
        )

        self.payload_summary = (
            f"{len(objects)} obj | UV: {', '.join(uv_layers) or 'none'} | "
            f"Colour: {', '.join(color_layers) or 'none'} | {rig_part}"
        )
        self.status_message = ""
        self.status_is_error = False

    def _describe_payload(self, objects):
        uv_layers, color_layers = set(), set()
        for obj in objects:
            mesh = obj.data
            if not isinstance(mesh, bpy.types.Mesh):
                continue
            uv_layers.update(layer.name for layer in mesh.uv_layers)
            color_layers.update(attr.name for attr in mesh.color_attributes)
        return sorted(uv_layers), sorted(color_layers)

    def _describe_rig(self, objects):
        from ...pivot_painter.tree_asset import find_armature
        from ...pivot_painter.wind_animation import count_wind_clips

        bones, clips, rigged = 0, 0, 0
        for obj in objects:
            armature = find_armature(obj)
            if armature is None:
                continue
            rigged += 1
            bones += len(armature.data.bones)
            clips += count_wind_clips(obj)
        return rigged, bones, clips

    def _describe_source(self):
        source = upstream_node(self)
        if source is None:
            return "Connect a tree or Tree Variants"
        if source.bl_idname == VARIANTS_ID:
            return f"Variants: {source.output_collection_name()}"
        return f"Tree: {self._source_name()}"

    def _gather_objects(self):
        source = upstream_node(self)
        if source is None:
            return []
        if source.bl_idname == VARIANTS_ID:
            return source.variant_objects()

        mesher = mesher_upstream(source)
        obj = None if mesher is None else resolve_mesh_object(mesher.tree_object)
        return [] if obj is None else [obj]

    def _source_name(self):
        source = upstream_node(self)
        if source is not None and source.bl_idname == VARIANTS_ID:
            return source.output_collection_name()
        mesher = None if source is None else mesher_upstream(source)
        if mesher is None or not mesher.tree_object:
            return "trees"
        return mesher.tree_object

    def do_export(self):
        self.status_message = ""
        self.status_is_error = False

        objects = self._gather_objects()
        if not objects:
            self.status_message = "No mesh objects found to export"
            self.status_is_error = True
            return

        export_dir = bpy.path.abspath(self.export_path)
        os.makedirs(export_dir, exist_ok=True)

        baked = self._ensure_wind_clips(objects)

        previous_selection = list(bpy.context.selected_objects)
        previous_active = bpy.context.view_layer.objects.active

        try:
            levels = self._lod_levels()
            if self.export_format == "BLEND":
                count = self._export_blend(export_dir)
            elif levels:
                count = self._export_lods(objects, export_dir, levels)
            elif self.separate_files:
                count = self._export_separate(objects, export_dir)
            else:
                count = self._export_combined(objects, export_dir)

            uv_layers, color_layers = self._describe_payload(objects)
            rigged, bones, clips = self._describe_rig(objects)
            rig_part = f", {bones} bones, {clips} clips" if rigged else ", no rig"
            self.status_message = (
                f"Exported {count} file(s): {len(uv_layers)} UV, "
                f"{len(color_layers)} colour layer(s){rig_part}"
                + (f" (baked {baked})" if baked else "")
            )
            self.status_is_error = False

        except Exception as e:
            self.status_message = f"Error: {str(e)}"
            self.status_is_error = True

        finally:
            bpy.ops.object.select_all(action="DESELECT")
            for obj in previous_selection:
                with contextlib.suppress(ReferenceError):
                    obj.select_set(True)
            bpy.context.view_layer.objects.active = previous_active

    def _ensure_wind_clips(self, objects):
        from ...pivot_painter.tree_asset import find_armature
        from ...pivot_painter.wind_animation import (
            bake_wind_clips,
            build_rig_for,
            count_wind_clips,
        )
        from .foliage_wind_node import foliage_wind_settings_for

        mesher = self.get_mesher()
        if mesher is None:
            return 0

        settings = foliage_wind_settings_for(mesher)
        if not settings["build_bone_rig"]:
            return 0

        wanted = mesher.morph_directions
        baked = 0

        for obj in objects:
            if not isinstance(obj.data, bpy.types.Mesh):
                continue
            if "stem_id" not in obj.data.attributes:
                continue
            if find_armature(obj) is not None and count_wind_clips(obj) == wanted:
                continue

            rig = build_rig_for(obj.data, mesher.segments_per_chain, mesher._stiffness_settings())
            count, _ = bake_wind_clips(obj, rig, directions=wanted)
            baked += count

        return baked

    def _select_only(self, objects):
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0] if objects else None

    def _rigged_armatures(self, objects):
        from ...pivot_painter.tree_asset import find_armature

        armatures = []
        for obj in objects:
            if obj.type != "MESH":
                continue
            armature = find_armature(obj)
            if armature is not None and armature not in armatures:
                armatures.append(armature)

        return armatures

    def _write_file(self, filepath, objects, owners=None, clips=True):
        axes = ENGINE_AXES[self.target_engine]
        armatures = self._rigged_armatures(objects if owners is None else owners)

        if armatures:
            self._select_only(list(objects) + armatures)

        armature_names = [armature.name for armature in armatures]
        if clips:
            self._write_selection(filepath, axes, armature_names)
            return

        with rest_pose(armature_names):
            self._write_selection(filepath, axes, armature_names, clips=False)

    def _write_selection(self, filepath, axes, armature_names, clips=True):
        from ...pivot_painter.wind_animation import own_clips_on_nla

        rigged = bool(armature_names)
        animated = rigged and clips
        if self.export_format == "FBX":
            bpy.ops.export_scene.fbx(
                filepath=filepath,
                use_selection=True,
                apply_unit_scale=True,
                bake_space_transform=False,
                apply_scale_options="FBX_SCALE_UNITS",
                mesh_smooth_type="FACE",
                object_types={"MESH", "ARMATURE"} if rigged else {"MESH"},
                bake_anim=animated,
                bake_anim_use_all_actions=animated,
                bake_anim_use_nla_strips=False,
                bake_anim_simplify_factor=0.0,
                add_leaf_bones=False,
                use_mesh_modifiers=self.apply_modifiers,
                colors_type="LINEAR",
                prioritize_active_color=True,
                use_triangles=True,
                use_custom_props=True,
                axis_up=axes["axis_up"],
                axis_forward=axes["axis_forward"],
            )
        else:
            clip_tracks = own_clips_on_nla(armature_names) if clips else contextlib.nullcontext()
            with clip_tracks:
                bpy.ops.export_scene.gltf(
                    filepath=filepath,
                    use_selection=True,
                    export_yup=(axes["axis_up"] == "Y"),
                    export_apply=self.apply_modifiers,
                    export_morph=False,
                    export_skins=rigged,
                    export_animations=animated,
                    export_animation_mode="ACTIONS",
                    export_anim_single_armature=False,
                    export_texcoords=True,
                    export_normals=True,
                    export_all_vertex_colors=True,
                    export_attributes=True,
                )

    def _export_separate(self, objects, export_dir):
        extension = "fbx" if self.export_format == "FBX" else "glb"
        count = 0

        for obj in objects:
            anchor = self._hierarchy_root(obj)
            original_location = tuple(anchor.location)
            if self.reset_transforms:
                anchor.location = (0.0, 0.0, 0.0)

            try:
                self._select_only([obj])
                self._write_file(os.path.join(export_dir, f"{obj.name}.{extension}"), [obj])
                count += 1
            finally:
                anchor.location = original_location

        return count

    @staticmethod
    def _hierarchy_root(obj):
        while obj.parent is not None:
            obj = obj.parent
        return obj

    def _export_combined(self, objects, export_dir):
        extension = "fbx" if self.export_format == "FBX" else "glb"
        name = self._source_name()

        self._select_only(objects)
        self._write_file(os.path.join(export_dir, f"{name}.{extension}"), objects)
        return 1

    def _lod_levels(self):
        from ...pivot_painter.leaf_lods import COARSE_LEVELS

        mesher = self.get_mesher()
        leaf_stage = None if mesher is None else mesher.leaf_stage()
        if leaf_stage is None or not leaf_stage.build_lods:
            return ()
        return (0, *COARSE_LEVELS)

    def _objects_at_level(self, trees, level):
        from ...pivot_painter.leaf_lods import lod_object_name

        if level == 0:
            return list(trees)
        found = []
        for tree in trees:
            lod = resolve_mesh_object(lod_object_name(tree, level))
            if lod is None:
                raise ValueError(f"{tree.name} has no LOD{level}; generate the tree again")
            found.append(lod)
        return found

    def _export_lods(self, trees, export_dir, levels):
        extension = "fbx" if self.export_format == "FBX" else "glb"
        by_level = {level: self._objects_at_level(trees, level) for level in levels}
        everything = [obj for objects in by_level.values() for obj in objects]
        count = 0

        with selectable(everything):
            if not self.separate_files:
                name = self._source_name()
                for level, objects in by_level.items():
                    filepath = os.path.join(export_dir, f"{name}_LOD{level}.{extension}")
                    self._select_only(objects)
                    self._write_file(filepath, objects, owners=trees, clips=level == 0)
                    count += 1
                return count

            for index, tree in enumerate(trees):
                anchor = self._hierarchy_root(tree)
                original_location = tuple(anchor.location)
                if self.reset_transforms:
                    anchor.location = (0.0, 0.0, 0.0)

                try:
                    for level, objects in by_level.items():
                        obj = objects[index]
                        filepath = os.path.join(export_dir, f"{tree.name}_LOD{level}.{extension}")
                        self._select_only([obj])
                        self._write_file(filepath, [obj], owners=[tree], clips=level == 0)
                        count += 1
                finally:
                    anchor.location = original_location

        return count

    def _export_blend(self, export_dir):
        name = self._source_name()
        bpy.ops.wm.save_as_mainfile(
            filepath=os.path.join(export_dir, f"{name}.blend"),
            copy=True,
        )
        return 1
