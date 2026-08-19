"""Node for exporting generated tree meshes to FBX or glTF.

Blender is Z-up right-handed; Unity is Y-up left-handed. Blender's FBX exporter
can perform that conversion, but its defaults do not: a tree exported with stock
settings arrives in Unity lying on its side or rotated ninety degrees, and the
usual workaround — rotating the object in Blender before exporting — bakes a
non-identity rotation into the transform that then fights every downstream tool.

This node exports with the axis settings that make a Z-up Blender tree stand
upright in Unity, with `apply_unit_scale` and `bake_space_transform` set so the
resulting object carries an identity rotation rather than a corrective one.
"""

from __future__ import annotations

import contextlib
import os

import bpy

from ..base_types.node import MtreeNode

# Axis conventions of the supported engines, as Blender's FBX exporter expects
# them. Blender's own axes are Z-up / -Y-forward.
ENGINE_AXES = {
    "UNITY": {"axis_up": "Y", "axis_forward": "-Z"},
    "UNREAL": {"axis_up": "Z", "axis_forward": "-Y"},
    "BLENDER": {"axis_up": "Z", "axis_forward": "-Y"},
}


class MTreeMeshExport(bpy.types.Node, MtreeNode):
    """Export generated tree meshes with engine-correct axis conversion"""

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
    export_selection: bpy.props.EnumProperty(
        name="Objects",
        items=[
            ("COLLECTION", "Collection", "Every mesh in the named collection"),
            ("SELECTED", "Selected", "Currently selected objects"),
        ],
        default="COLLECTION",
    )
    collection_name: bpy.props.StringProperty(
        name="Collection",
        description="Collection to export when Objects is set to Collection",
        default="tree_variants",
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

    # Cached result of the last payload check. Stored rather than recomputed
    # because draw() must stay cheap; see the note in draw().
    payload_summary: bpy.props.StringProperty(default="")

    def init(self, context):
        self.add_input("mt_TreeSocket", "Tree", is_property=False)

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
        source_box.prop(self, "export_selection")
        if self.export_selection == "COLLECTION":
            source_box.prop(self, "collection_name")
        source_box.prop(self, "separate_files")

        options_box = layout.box()
        options_box.label(text="Options")
        options_box.prop(self, "apply_modifiers")
        options_box.prop(self, "reset_transforms")

        # The payload summary is deliberately NOT computed here: draw() runs on
        # every UI redraw, and scanning every object's layers from it makes the
        # whole interface stall. It is filled in by the Check button and by the
        # export itself, and cached in payload_summary.
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
        """Scan the export set and cache what will travel with it.

        Run on demand rather than from draw(), because this walks every object's
        layers and draw() is called on every redraw.
        """
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

        self.payload_summary = (
            f"{len(objects)} obj | UV: {', '.join(uv_layers) or 'none'} | "
            f"Colour: {', '.join(color_layers) or 'none'}"
        )
        self.status_message = ""
        self.status_is_error = False

    def _describe_payload(self, objects):
        """Summarise the UV and colour layers that will be written.

        FBX carries UV layers and colour attributes, but **not** arbitrary named
        float attributes: ``stem_id``, ``branch_extent`` and the rest exist only
        inside Blender. That is why the foliage and pivot painter exports bake
        their data into UVs and vertex colours first. Running Mesh Export before
        those bakes produces a file that is missing the data, with nothing in the
        export itself to say so.
        """
        uv_layers, color_layers = set(), set()
        for obj in objects:
            mesh = obj.data
            if not isinstance(mesh, bpy.types.Mesh):
                continue
            uv_layers.update(layer.name for layer in mesh.uv_layers)
            color_layers.update(attr.name for attr in mesh.color_attributes)
        return sorted(uv_layers), sorted(color_layers)

    def _gather_objects(self):
        """Collect the mesh objects to export."""
        if self.export_selection == "SELECTED":
            return [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]

        collection = bpy.data.collections.get(self.collection_name)
        if collection is None:
            return []
        return [obj for obj in collection.objects if obj.type == "MESH"]

    def do_export(self):
        """Run the export."""
        self.status_message = ""
        self.status_is_error = False

        objects = self._gather_objects()
        if not objects:
            self.status_message = "No mesh objects found to export"
            self.status_is_error = True
            return

        export_dir = bpy.path.abspath(self.export_path)
        os.makedirs(export_dir, exist_ok=True)

        # Selection state is UI state the user owns; restore whatever it was.
        previous_selection = list(bpy.context.selected_objects)
        previous_active = bpy.context.view_layer.objects.active

        try:
            if self.export_format == "BLEND":
                count = self._export_blend(export_dir)
            elif self.separate_files:
                count = self._export_separate(objects, export_dir)
            else:
                count = self._export_combined(objects, export_dir)

            uv_layers, color_layers = self._describe_payload(objects)
            self.status_message = (
                f"Exported {count} file(s): {len(uv_layers)} UV, "
                f"{len(color_layers)} colour layer(s)"
            )
            self.status_is_error = False

        except Exception as e:
            self.status_message = f"Error: {str(e)}"
            self.status_is_error = True

        finally:
            bpy.ops.object.select_all(action="DESELECT")
            for obj in previous_selection:
                # An object exported and then removed mid-run leaves a stale
                # reference; restoring selection must not fail on it.
                with contextlib.suppress(ReferenceError):
                    obj.select_set(True)
            bpy.context.view_layer.objects.active = previous_active

    def _select_only(self, objects):
        """Make exactly the given objects selected and active."""
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0] if objects else None

    def _write_file(self, filepath):
        """Invoke the format-specific exporter for the current selection."""
        axes = ENGINE_AXES[self.target_engine]

        if self.export_format == "FBX":
            bpy.ops.export_scene.fbx(
                filepath=filepath,
                use_selection=True,
                apply_unit_scale=True,
                # Bakes the axis conversion into the mesh data so the exported
                # object carries an identity rotation, instead of a -90 X that
                # Unity then has to undo.
                bake_space_transform=True,
                apply_scale_options="FBX_SCALE_ALL",
                mesh_smooth_type="FACE",
                use_mesh_modifiers=self.apply_modifiers,
                # The colour layers carry four independent scalars, not colour.
                # SRGB would gamma-encode them and silently corrupt every value.
                colors_type="LINEAR",
                # Some importers keep only the first colour attribute and discard
                # the rest; with both PivotPainterMask and FoliageShading present
                # that would drop one of them without an error.
                prioritize_active_color=True,
                # Triangulate here rather than letting the engine do it on
                # import: a different triangulation splits UVs differently at
                # seams, which moves texture coordinates that were correct.
                use_triangles=True,
                # Pivot painter and foliage data ride along as custom properties.
                use_custom_props=True,
                axis_up=axes["axis_up"],
                axis_forward=axes["axis_forward"],
            )
        else:
            bpy.ops.export_scene.gltf(
                filepath=filepath,
                use_selection=True,
                export_yup=(axes["axis_up"] == "Y"),
                export_apply=self.apply_modifiers,
                # All of these default to dropping data. UVs and colours are on
                # by default; generic attributes are not, and the foliage
                # streams live there.
                export_texcoords=True,
                export_normals=True,
                export_all_vertex_colors=True,
                export_attributes=True,
            )

    def _export_separate(self, objects, export_dir):
        """Write one file per object."""
        extension = "fbx" if self.export_format == "FBX" else "glb"
        count = 0

        for obj in objects:
            original_location = tuple(obj.location)
            if self.reset_transforms:
                obj.location = (0.0, 0.0, 0.0)

            try:
                self._select_only([obj])
                self._write_file(os.path.join(export_dir, f"{obj.name}.{extension}"))
                count += 1
            finally:
                obj.location = original_location

        return count

    def _export_combined(self, objects, export_dir):
        """Write every object into a single file."""
        extension = "fbx" if self.export_format == "FBX" else "glb"
        name = self.collection_name if self.export_selection == "COLLECTION" else "trees"

        self._select_only(objects)
        self._write_file(os.path.join(export_dir, f"{name}.{extension}"))
        return 1

    def _export_blend(self, export_dir):
        """Save a copy of the current blend file."""
        name = self.collection_name if self.export_selection == "COLLECTION" else "trees"
        bpy.ops.wm.save_as_mainfile(
            filepath=os.path.join(export_dir, f"{name}.blend"),
            copy=True,
        )
        return 1
