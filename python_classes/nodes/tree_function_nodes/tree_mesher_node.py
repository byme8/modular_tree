
from __future__ import annotations

import time

import bpy

from ... import diagnostics
from ...m_tree_wrapper import lazy_m_tree as m_tree
from ...mesh_utils import create_mesh_from_cpp
from ...pivot_painter.tree_asset import (
    armature_name_for,
    find_armature,
    owning_collection,
    resolve_mesh_object,
    strip_rig,
)
from ..base_types.node import MtreeNode
from ..debounce import schedule_build


def on_update_prop(node, context):
    schedule_build(node)


class TreeMesherNode(bpy.types.Node, MtreeNode):

    bl_idname = "mt_MesherNode"
    bl_label = "Tree Mesher"

    auto_update: bpy.props.BoolProperty(
        name="Auto Update",
        description="Automatically rebuild tree when parameters or connections change",
        default=True,
    )
    radial_resolution: bpy.props.IntProperty(
        name="Radial Resolution", default=32, min=3, update=on_update_prop
    )
    smoothness: bpy.props.IntProperty(name="smoothness", default=4, min=0, update=on_update_prop)
    tree_object: bpy.props.StringProperty(default="")

    build_bone_rig: bpy.props.BoolProperty(
        name="Build Bone Rig",
        description=(
            "Build a bone rig on generate and bind the mesh to an armature, so "
            "the tree can be posed in the viewport and travels rigged into the "
            "export. Bones compose rotations down the hierarchy, so a branch "
            "follows the trunk's lean instead of shearing against it"
        ),
        default=False,
    )
    @property
    def segments_per_chain(self):
        """Read from the Foliage Wind node, which owns the rig settings.

        Storing a second copy here let the preview build one rig while the bake
        built another: the value shown on the wind node was ignored and this
        one silently won.
        """
        from ...nodes.export_nodes.foliage_wind_node import find_foliage_wind_settings

        return find_foliage_wind_settings(self.get_node_tree())["segments_per_chain"]

    morph_directions: bpy.props.IntProperty(
        name="Wind Directions",
        description=(
            "How many compass directions to bake a wind pose for. The runtime "
            "blends the two nearest, which cuts the corner between them, so more "
            "directions bend more accurately at one vertex buffer each"
        ),
        default=8,
        min=2,
        max=16,
    )

    preview_wind_strength: bpy.props.FloatProperty(
        name="Wind Strength",
        description=(
            "Pose the armature as if wind of this strength were blowing, using "
            "the same stiffness curve the engine does. Preview only; it does "
            "not change what is exported"
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

    hide_armature: bpy.props.BoolProperty(
        name="Hide Armature",
        description=(
            "Hide the rig in the viewport. It still poses the mesh and still "
            "drives the wind preview; only its own bones stop being drawn"
        ),
        default=False,
        update=lambda self, context: self.apply_armature_visibility(),
    )

    status_message: bpy.props.StringProperty(default="")
    status_is_error: bpy.props.BoolProperty(default=False)

    log_pipeline: bpy.props.BoolProperty(
        name="Log Pipeline",
        description=(
            "Write every build stage and its result to the system console. "
            "Blender reduces a failure to one status line, which hides which "
            "stage actually failed"
        ),
        default=False,
        update=lambda self, context: diagnostics.set_enabled(self.log_pipeline),
    )

    def init(self, context):
        self.add_output("mt_TreeSocket", "Tree", is_property=False)

    def draw(self, context, layout):
        """Blender aborts a node's draw at the first exception and renders only
        what was emitted before it, so a truncated node looks like a missing
        feature. Each step is logged and failures are drawn into the node.
        """
        step = "start"
        try:
            step = "get_tree_validity"
            valid_tree = self.get_tree_validity()
            diagnostics.log("draw: %s -> %s", step, valid_tree)

            if self.status_message:
                step = "status"
                status_row = layout.row()
                status_row.alert = self.status_is_error
                icon = "ERROR" if self.status_is_error else "CHECKMARK"
                status_row.label(text=self.status_message, icon=icon)
                diagnostics.log("draw: %s", step)

            step = "generate_button"
            generate_row = layout.row()
            generate_row.enabled = valid_tree
            self._draw_generate_button(generate_row)
            diagnostics.log("draw: %s", step)

            step = "tree_object_selector"
            self._draw_tree_object_selector(layout, context)
            diagnostics.log("draw: %s", step)

            step = "properties"
            self._draw_properties(layout)
            diagnostics.log("draw: %s", step)

            step = "leaves_button"
            leaves_row = layout.row()
            leaves_row.enabled = valid_tree
            self._draw_leaves_button(leaves_row)
            diagnostics.log("draw: %s", step)

            step = "rig"
            self._draw_rig(layout)
            diagnostics.log("draw: %s complete", step)

        except Exception as error:
            diagnostics.exception("draw FAILED at step %r", step)
            failure = layout.box()
            failure.alert = True
            failure.label(text=f"UI error in {step}", icon="ERROR")
            failure.label(text=str(error)[:60])

    def _draw_rig(self, container):
        diagnostics.log("draw_rig: enter")

        rig_box = container.box()
        rig_box.label(text="Wind Rig", icon="ARMATURE_DATA")
        rig_box.prop(self, "build_bone_rig")
        diagnostics.log("draw_rig: box + build_bone_rig=%s", self.build_bone_rig)

        if not self.build_bone_rig:
            diagnostics.log("draw_rig: stop, build_bone_rig is off")
            return

        segments = self.segments_per_chain
        diagnostics.log("draw_rig: segments_per_chain=%s", segments)

        rig_box.label(
            text=f"Bones per branch: {segments} (Foliage Wind)",
            icon="INFO",
        )
        if segments == 1:
            warning = rig_box.row()
            warning.label(text="1 bone per branch bends nothing", icon="ERROR")

        has_armature = self._get_armature() is not None
        diagnostics.log("draw_rig: has_armature=%s", has_armature)

        visibility_row = rig_box.row()
        visibility_row.enabled = has_armature
        visibility_row.prop(self, "hide_armature", icon="HIDE_ON")

        preview_column = rig_box.column()
        preview_column.enabled = has_armature
        preview_column.prop(self, "preview_wind_strength", slider=True)
        preview_column.prop(self, "preview_wind_angle", slider=True)

        reset = preview_column.operator("mtree.node_function", text="Rest Pose")
        reset.node_tree_name = self.get_node_tree().name
        reset.node_name = self.name
        reset.function_name = "clear_pose_preview"

        if not has_armature:
            hint = rig_box.row()
            hint.label(text="Press Generate Tree to build the rig", icon="INFO")

        diagnostics.log("draw_rig: preview controls drawn")

        morph_box = container.box()
        morph_box.label(text="Wind Morphs", icon="SHAPEKEY_DATA")
        morph_box.prop(self, "morph_directions")

        baked = self._morph_count()
        diagnostics.log("draw_rig: morph_count=%s", baked)
        morph_box.label(text=f"{baked} baked", icon="INFO")

        bake_row = morph_box.row()
        bake_row.enabled = has_armature
        bake = bake_row.operator("mtree.node_function", text="Bake Wind Morphs")
        bake.node_tree_name = self.get_node_tree().name
        bake.node_name = self.name
        bake.function_name = "bake_wind_morphs"

        diagnostics.log("draw_rig: complete")

    def _morph_count(self):
        from ...pivot_painter.wind_morphs import count_wind_morphs

        tree_obj = resolve_mesh_object(self.tree_object)
        if tree_obj is None:
            return 0
        return count_wind_morphs(tree_obj)

    def _draw_generate_button(self, container):
        props = container.operator("mtree.node_function", text="Generate Tree")
        props.node_tree_name = self.get_node_tree().name
        props.node_name = self.name
        props.function_name = "build_tree"

    def _draw_tree_object_selector(self, container, context):
        container.prop_search(
            self,
            property="tree_object",
            search_data=context.scene,
            search_property="objects",
            text="",
        )

    def _draw_properties(self, container):
        container.prop(self, "auto_update")
        container.prop(self, "radial_resolution")
        container.prop(self, "smoothness")
        container.prop(self, "log_pipeline", icon="CONSOLE")

    def _draw_leaves_button(self, container):
        """Draw must not call get_current_tree_object: it links and selects, and
        Blender forbids writing to ID data while drawing. The exception aborts
        the whole node's draw, hiding every panel below this one.
        """
        tree_obj = resolve_mesh_object(self.tree_object)
        if tree_obj is None or not tree_obj.users_collection:
            return

        props = container.operator("mtree.add_leaves", text="Add leaves")
        props.object_id = tree_obj.name

    def _has_valid_tree_object(self):
        tree_obj = resolve_mesh_object(self.tree_object)
        return tree_obj is not None and bool(tree_obj.users_collection)

    def build_into(self, obj, seed_offset=0):
        """The one build path. Every generated tree goes through here, so the
        mesher's resolution and rig settings cannot drift from a variant's.
        """
        with diagnostics.stage(
            "build_into",
            target=obj.name,
            seed_offset=seed_offset,
            resolution=self.radial_resolution,
            smoothness=self.smoothness,
            rig=self.build_bone_rig,
        ):
            diagnostics.log("  graph %s", diagnostics.describe_graph(self))

            trunk_node = self._trunk_link()
            if trunk_node is None:
                raise ValueError("No connected trunk node")

            with diagnostics.stage("construct_function", trunk=trunk_node.name):
                trunk_function = trunk_node.construct_function(seed_offset)
            if trunk_function is None:
                raise ValueError("Connected node returned no tree function")

            with diagnostics.stage("execute_functions"):
                tree = m_tree.Tree()
                tree.set_trunk_function(trunk_function)
                tree.execute_functions()

            with diagnostics.stage("mesh_tree") as meshing:
                cpp_mesh = self._mesh_tree(tree)
                coordinates = len(cpp_mesh.get_vertices())
                meshing.record(
                    verts=int(coordinates / 3),
                    polys=len(cpp_mesh.get_polygons()),
                )

            strip_rig(obj)
            if obj.data.shape_keys is not None:
                obj.shape_key_clear()
            obj.data.clear_geometry()

            create_mesh_from_cpp(obj.data, cpp_mesh)
            diagnostics.log("  built %s", diagnostics.describe_mesh(obj.data))

            if len(obj.data.vertices) == 0:
                raise ValueError(self._empty_mesh_reason(trunk_node))

            if self.build_bone_rig:
                return self._build_rig(obj)
            return ""

    MIN_TRUNK_SEGMENTS = 1.0

    def _empty_mesh_reason(self, trunk_node):
        """Name the values that produced nothing.

        The trunk is divided into length * resolution segments, so the mesh is
        empty whenever that product rounds below one — neither value has to be
        zero, and reporting them separately sends the reader looking for a zero
        that is not there.
        """
        values = {
            socket.property_name: socket.property_value
            for socket in trunk_node.inputs
            if getattr(socket, "property_name", "") in ("length", "resolution")
        }

        length = values.get("length")
        resolution = values.get("resolution")

        if length is None or resolution is None:
            return "Tree built with no geometry."

        return (
            f"Tree built with no geometry: trunk length {length:.4g} x resolution "
            f"{resolution:.4g} = {length * resolution:.4g} segments, which rounds "
            f"to none. Raise either until their product exceeds "
            f"{self.MIN_TRUNK_SEGMENTS:.0f}."
        )

    def build_tree(self):
        if not self.get_tree_validity():
            self.status_message = "Connect a Trunk node to generate"
            self.status_is_error = True
            return

        self.status_message = ""
        self.status_is_error = False

        try:
            start_time = time.time()

            tree_obj = self.get_current_tree_object()
            rig_message = self.build_into(tree_obj)
            bpy.context.view_layer.objects.active = tree_obj

            elapsed = time.time() - start_time
            self.status_message = f"Generated in {elapsed:.2f}s {rig_message}".strip()
            self.status_is_error = False

        except Exception as e:
            diagnostics.exception("build_tree failed on %s", self.name)
            self.status_message = f"Error: {str(e)}"
            self.status_is_error = True

    def _get_armature(self):
        tree_obj = resolve_mesh_object(self.tree_object)
        if tree_obj is None:
            return None
        return find_armature(tree_obj)

    def _build_rig(self, tree_obj):
        from ...pivot_painter.formats.foliage_rig import FoliageRigExporter

        with diagnostics.stage(
            "build_rig",
            target=tree_obj.name,
            segments_per_chain=self.segments_per_chain,
        ) as rigging:
            diagnostics.log("  source %s", diagnostics.describe_mesh(tree_obj.data))

            result = FoliageRigExporter(
                mesh=tree_obj.data,
                mesh_object=tree_obj,
                segments_per_chain=self.segments_per_chain,
                create_armature=True,
                armature_name=armature_name_for(tree_obj),
                collection=owning_collection(tree_obj),
                **self._stiffness_settings(),
            ).export()

            rigging.record(success=result.success, message=result.message)

        if not result.success:
            raise ValueError(f"Rig failed: {result.message}")

        self.preview_wind_strength = 0.0
        self.apply_armature_visibility()
        return result.message

    def apply_pose_preview(self):
        from ...pivot_painter.wind_morphs import apply_wind_pose

        armature = self._get_armature()
        if armature is None:
            return

        rig = self._current_rig()
        if rig is None:
            return

        apply_wind_pose(
            armature, rig, self.preview_wind_angle, self.preview_wind_strength
        )

    def bake_wind_morphs(self):
        from ...pivot_painter.wind_morphs import bake_wind_morphs

        tree_obj = resolve_mesh_object(self.tree_object)
        if tree_obj is None:
            self.status_message = "No tree object; generate first"
            self.status_is_error = True
            return

        settings = self._stiffness_settings()
        count, message = bake_wind_morphs(
            tree_obj,
            directions=self.morph_directions,
            segments_per_chain=self.segments_per_chain,
            keep_armature=True,
            **settings,
        )

        self.status_message = message
        self.status_is_error = count == 0

    def apply_armature_visibility(self):
        """Toggle the rig's viewport visibility.

        hide_viewport rather than hide_set: the latter is per-view-layer state
        that a saved file or a second view layer will not carry. Neither stops
        the armature deforming the mesh, so the wind preview keeps working
        while the bones are hidden.
        """
        armature = self._get_armature()
        if armature is None:
            return

        armature.hide_viewport = self.hide_armature

    def clear_pose_preview(self):
        armature = self._get_armature()
        if armature is None:
            return

        for bone in armature.pose.bones:
            bone.rotation_mode = "QUATERNION"
            bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)

        self.preview_wind_strength = 0.0

    def _stiffness_settings(self):
        from ...nodes.export_nodes.foliage_wind_node import find_foliage_wind_settings

        settings = find_foliage_wind_settings(self.get_node_tree())
        return {
            "trunk_base_stiffness": settings["trunk_base_stiffness"],
            "trunk_tip_stiffness": settings["trunk_tip_stiffness"],
            "branch_stiffness": settings["branch_stiffness"],
            "junction_falloff": settings["junction_falloff"],
        }

    def _current_rig(self):
        from ...pivot_painter.formats.foliage_rig import FoliageRigExporter
        from ...pivot_painter.wind_morphs import build_rig_for

        tree_obj = resolve_mesh_object(self.tree_object)
        if tree_obj is None:
            return None

        probe = FoliageRigExporter(
            mesh=tree_obj.data,
            segments_per_chain=self.segments_per_chain,
            create_armature=False,
        )
        if probe.validate() is not None:
            return None

        return build_rig_for(
            tree_obj.data,
            self.segments_per_chain,
            self._stiffness_settings(),
            use_cache=True,
        )

    def _mesh_tree(self, tree):
        mesher = m_tree.ManifoldMesher()
        mesher.radial_n_points = self.radial_resolution
        mesher.smooth_iterations = self.smoothness
        return mesher.mesh_tree(tree)

    def get_current_tree_object(self):
        """An object unlinked from every collection still exists in bpy.data, so
        it is relinked rather than abandoned: creating a replacement would take
        the name `tree.001` and orphan the node's reference on the next lookup.
        """
        tree_obj = resolve_mesh_object(self.tree_object)
        if tree_obj is None:
            tree_mesh = bpy.data.meshes.new("tree")
            tree_obj = bpy.data.objects.new("tree", tree_mesh)
            self.tree_object = tree_obj.name

        if not tree_obj.users_collection:
            bpy.context.collection.objects.link(tree_obj)

        bpy.context.view_layer.objects.active = tree_obj
        tree_obj.select_set(True)
        return tree_obj

    def get_tree_validity(self):
        """Non-function nodes such as Tree Variants also attach to this output,
        so validity is "a trunk is connected", not "exactly one link".
        """
        return self._trunk_link() is not None and not self._detect_loop_rec(self)

    def _trunk_link(self):
        from ..base_types.node import MtreeFunctionNode

        for link in self.outputs[0].links or []:
            if isinstance(link.to_node, MtreeFunctionNode):
                return link.to_node
        return None

    def _detect_loop_rec(self, node=None, seen_nodes=None):
        if node is None:
            node = self
        if seen_nodes is None:
            seen_nodes = set()

        for output in node.outputs:
            if output.links is None:
                continue
            for link in output.links:
                destination_node = link.to_node
                if destination_node in seen_nodes:
                    return True
                seen_nodes.add(destination_node)
                self._detect_loop_rec(destination_node, seen_nodes)

        return False
