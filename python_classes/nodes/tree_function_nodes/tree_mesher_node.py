
from __future__ import annotations

import time

import bpy

from ... import diagnostics
from ...m_tree_wrapper import lazy_m_tree as m_tree
from ...mesh_utils import create_mesh_from_cpp
from ...pivot_painter.tree_asset import find_armature, resolve_mesh_object, strip_rig
from ...pivot_painter.wind_animation import STRENGTH_STOPS
from ..base_types.node import MtreeNode
from ..debounce import schedule_build, schedule_preview
from ..surface_nodes.linking import (
    BLOCK_FILL_ID,
    LEAF_CARDS_ID,
    acting_surface_node,
    linked_surface_nodes,
)


def on_update_prop(node, context):
    schedule_build(node)


def on_update_preview(node, context):
    schedule_preview(node, "apply_pose_preview")


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
            "Build the bone chains wind is computed from, so the tree can be "
            "baked to wind morphs and travels rigged into the export. Bones "
            "compose rotations down the hierarchy, so a branch follows the "
            "trunk's lean instead of shearing against it"
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
            "How many compass directions to bake a wind clip for. The engine "
            "blends the two nearest, which cuts the corner between them, so "
            "more directions bend more accurately at one short clip each"
        ),
        default=8,
        min=2,
        max=16,
    )

    preview_wind_strength: bpy.props.FloatProperty(
        name="Wind Strength",
        description=(
            "Pose the rig as if wind of this strength were blowing. Preview "
            "only; it runs the same solver the clips are baked from, so it "
            "cannot disagree with what is exported"
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

        from ...nodes.export_nodes.foliage_wind_node import (
            describe_rig_layout,
            find_foliage_wind_settings,
        )

        settings = find_foliage_wind_settings(self.get_node_tree())
        diagnostics.log("draw_rig: segments_per_chain=%s", settings["segments_per_chain"])

        rig_box.label(
            text=f"Bones: {describe_rig_layout(settings)} (Foliage Wind)",
            icon="INFO",
        )
        if settings["segments_per_chain"] == 1:
            warning = rig_box.row()
            warning.label(text="1 trunk bone bends nothing", icon="ERROR")

        clip_box = container.box()
        clip_box.label(text="Wind Clips", icon="ACTION")
        clip_box.prop(self, "morph_directions")

        baked = self._clip_count()
        diagnostics.log("draw_rig: clip_count=%s", baked)
        clip_box.label(
            text=f"{baked} clips, {len(STRENGTH_STOPS)} strength stops", icon="INFO"
        )

        bake_row = clip_box.row()
        bake_row.enabled = self._has_valid_tree_object()
        bake = bake_row.operator("mtree.node_function", text="Bake Wind Clips")
        bake.node_tree_name = self.get_node_tree().name
        bake.node_name = self.name
        bake.function_name = "bake_wind_clips"

        preview_column = clip_box.column()
        preview_column.enabled = self._has_valid_tree_object()
        preview_column.prop(self, "preview_wind_strength", slider=True)
        preview_column.prop(self, "preview_wind_angle", slider=True)

        reset = preview_column.operator("mtree.node_function", text="Rest Pose")
        reset.node_tree_name = self.get_node_tree().name
        reset.node_name = self.name
        reset.function_name = "clear_pose_preview"

        diagnostics.log("draw_rig: complete")

    def _clip_count(self):
        from ...pivot_painter.wind_animation import count_wind_clips

        tree_obj = resolve_mesh_object(self.tree_object)
        if tree_obj is None:
            return 0
        return count_wind_clips(tree_obj)

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
        if acting_surface_node(self, LEAF_CARDS_ID) is not None:
            return

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

            block_fill = acting_surface_node(self, BLOCK_FILL_ID)
            leaf_cards = acting_surface_node(self, LEAF_CARDS_ID)
            surface_acted = block_fill is not None or leaf_cards is not None
            if surface_acted:
                cpp_mesh = self._apply_surface_nodes(
                    tree, cpp_mesh, block_fill, leaf_cards, seed_offset
                )

            if leaf_cards is not None:
                self._remove_leaves_modifier(obj)

            strip_rig(obj)
            if obj.data.shape_keys is not None:
                obj.shape_key_clear()
            obj.data.clear_geometry()

            create_mesh_from_cpp(
                obj.data,
                cpp_mesh,
                flat_shading=surface_acted,
                material_slots=2 if surface_acted else 0,
            )
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

    def _build_rig(self, tree_obj):
        from ...pivot_painter.formats.foliage_rig import FoliageRigExporter
        from ...pivot_painter.wind_animation import (
            bind_to_armature,
            build_armature,
            build_rig_for,
        )

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
                **self._stiffness_settings(),
            ).export()

            rigging.record(success=result.success, message=result.message)

            if result.success:
                rig = build_rig_for(
                    tree_obj.data,
                    self.segments_per_chain,
                    self._stiffness_settings(),
                )
                armature = build_armature(tree_obj, rig)
                bind_to_armature(tree_obj, armature, rig)

        if not result.success:
            raise ValueError(f"Rig failed: {result.message}")

        self.preview_wind_strength = 0.0
        return result.message

    def apply_pose_preview(self):
        from ...pivot_painter.wind_animation import apply_pose, build_rig_for

        tree_obj = resolve_mesh_object(self.tree_object)
        if tree_obj is None:
            return

        armature = find_armature(tree_obj)
        if armature is None:
            return

        rig = build_rig_for(
            tree_obj.data,
            self.segments_per_chain,
            self._stiffness_settings(),
            use_cache=True,
        )

        apply_pose(armature, rig, self.preview_wind_angle, self.preview_wind_strength)

    def bake_wind_clips(self):
        from ...pivot_painter.wind_animation import bake_wind_clips, build_rig_for

        tree_obj = resolve_mesh_object(self.tree_object)
        if tree_obj is None:
            self.status_message = "No tree object; generate first"
            self.status_is_error = True
            return

        with diagnostics.stage("bake_wind_clips", target=tree_obj.name) as baking:
            rig = build_rig_for(
                tree_obj.data, self.segments_per_chain, self._stiffness_settings()
            )
            count, message = bake_wind_clips(
                tree_obj, rig, directions=self.morph_directions
            )
            baking.record(clips=count)

        self.preview_wind_strength = 0.0
        self.status_message = message
        self.status_is_error = count == 0

    def clear_pose_preview(self):
        from ...pivot_painter.wind_animation import clear_pose

        tree_obj = resolve_mesh_object(self.tree_object)
        if tree_obj is None:
            return

        armature = find_armature(tree_obj)
        if armature is not None:
            clear_pose(armature)

        self.preview_wind_strength = 0.0

    def _stiffness_settings(self):
        from ...nodes.export_nodes.foliage_wind_node import (
            find_foliage_wind_settings,
            rig_settings,
        )

        return rig_settings(find_foliage_wind_settings(self.get_node_tree()))

    def _mesh_tree(self, tree):
        mesher = m_tree.ManifoldMesher()
        mesher.radial_n_points = self.radial_resolution
        mesher.smooth_iterations = self.smoothness
        return mesher.mesh_tree(tree)

    def _apply_surface_nodes(self, tree, cpp_mesh, block_fill, leaf_cards, seed_offset):
        for node, kind in ((block_fill, BLOCK_FILL_ID), (leaf_cards, LEAF_CARDS_ID)):
            if node is None:
                continue
            for ignored in linked_surface_nodes(self, kind)[1:]:
                ignored.set_status(f"Ignored: {node.name} already acts on {self.name}", True)

        blocks = None if block_fill is None else block_fill.native(seed_offset)
        cards = None if leaf_cards is None else leaf_cards.native(seed_offset)

        with diagnostics.stage(
            "surface_nodes",
            block_fill=None if block_fill is None else block_fill.name,
            leaf_cards=None if leaf_cards is None else leaf_cards.name,
        ) as stage:
            polygons = 0
            if blocks is not None:
                polygons += blocks.count_polygons(tree)
            if cards is not None:
                polygons += cards.count_polygons(tree)

            if block_fill is not None and polygons > block_fill.max_polygons:
                message = (
                    f"Block Fill needs {polygons:,} polygons, over the cap of "
                    f"{block_fill.max_polygons:,}"
                )
                block_fill.set_status(f"Error: {message}", True)
                raise ValueError(message)

            if blocks is not None:
                cpp_mesh = blocks.mesh_tree(tree)
                stage.record(
                    density=block_fill.density,
                    scale=block_fill.scale,
                    blocks=blocks.block_count,
                    polygons=blocks.polygon_count,
                )
                block_fill.set_status(
                    f"{blocks.block_count:,} blocks, {blocks.polygon_count:,} polygons"
                )

            if cards is not None:
                cards.append(tree, cpp_mesh)
                stage.record(cards=cards.card_count)
                leaf_cards.set_status(f"{cards.card_count:,} cards")

        return cpp_mesh

    def _remove_leaves_modifier(self, obj):
        from ...resources.node_groups import LEAVES_MODIFIER_NAME

        modifier = obj.modifiers.get(LEAVES_MODIFIER_NAME)
        if modifier is not None:
            obj.modifiers.remove(modifier)

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
