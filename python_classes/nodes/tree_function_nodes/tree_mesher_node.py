from __future__ import annotations

import time

import bpy
import numpy as np

from ... import diagnostics
from ...m_tree_wrapper import lazy_m_tree as m_tree
from ...mesh_utils import create_mesh_from_cpp
from ...pivot_painter.tree_asset import find_armature, resolve_mesh_object, strip_rig
from ...pivot_painter.wind_animation import STRENGTH_STOPS
from ..base_types.node import MtreeNode
from ..debounce import schedule_build, schedule_preview
from ..pipeline import LEAF_CARDS_ID, LEAF_CLUMPS_ID, chain_stage, resolve_trunk

MANIFOLD = "MANIFOLD"
BLOCKS = "BLOCKS"


def on_update_prop(node, context):
    schedule_build(node)


def on_update_preview(node, context):
    schedule_preview(node, "apply_pose_preview")


class TreeMesherNode(bpy.types.Node, MtreeNode):
    bl_idname = "mt_MesherNode"
    bl_label = "Tree Mesher"

    chain_input = ("mt_TreeSocket", "Skeleton")
    chain_output = ("mt_MeshSocket", "Mesh")

    auto_update: bpy.props.BoolProperty(
        name="Auto Update",
        description="Automatically rebuild tree when parameters or connections change",
        default=True,
    )
    mesh_mode: bpy.props.EnumProperty(
        name="Mesh",
        items=[
            (MANIFOLD, "Manifold", "One continuous skin swept along every branch", 0),
            (BLOCKS, "Blocks", "Loose blocks strung along every branch chain", 1),
        ],
        default=MANIFOLD,
        update=on_update_prop,
    )
    radial_resolution: bpy.props.IntProperty(
        name="Radial Resolution", default=32, min=3, update=on_update_prop
    )
    smoothness: bpy.props.IntProperty(name="smoothness", default=4, min=0, update=on_update_prop)

    block_density: bpy.props.FloatProperty(
        name="Density",
        description="Blocks per metre along every branch chain",
        default=10.0,
        min=0.1,
        max=1000.0,
        update=on_update_prop,
    )
    block_scale: bpy.props.FloatProperty(
        name="Scale",
        description="Grows or shrinks every block; above 1 they overlap in every direction",
        default=1.0,
        min=0.1,
        max=5.0,
        update=on_update_prop,
    )
    block_trunk_length: bpy.props.FloatProperty(
        name="Trunk Block Length",
        description="Length of each trunk block in metres; 0 uses Density like every other chain",
        default=0.0,
        min=0.0,
        max=100.0,
        update=on_update_prop,
    )
    block_roll_variation: bpy.props.FloatProperty(
        name="Roll Variation",
        description="Each block twists about its branch by a random angle up to this, in degrees",
        default=15.0,
        min=0.0,
        max=180.0,
        update=on_update_prop,
    )
    block_seed: bpy.props.IntProperty(
        name="Seed",
        description="Seed for the block rolls; the variants offset is added to it",
        default=0,
        update=on_update_prop,
    )
    block_max_polygons: bpy.props.IntProperty(
        name="Max Polygons",
        description="A build that would exceed this fails and leaves the tree as it was",
        default=200000,
        min=1000,
        max=10000000,
        update=on_update_prop,
    )
    blocks_status_message: bpy.props.StringProperty(default="")
    blocks_status_is_error: bpy.props.BoolProperty(default=False)

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
        from ...nodes.export_nodes.foliage_wind_node import foliage_wind_settings_for

        return foliage_wind_settings_for(self)["segments_per_chain"]

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
        self.add_chain_sockets()

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
            foliage_wind_settings_for,
        )

        settings = foliage_wind_settings_for(self)
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
        clip_box.label(text=f"{baked} clips, {len(STRENGTH_STOPS)} strength stops", icon="INFO")

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
        container.prop(self, "mesh_mode", expand=True)

        if self.mesh_mode == BLOCKS:
            blocks_box = container.box()
            if self.blocks_status_message:
                status_row = blocks_box.row()
                status_row.alert = self.blocks_status_is_error
                icon = "ERROR" if self.blocks_status_is_error else "CHECKMARK"
                status_row.label(text=self.blocks_status_message, icon=icon)
            for name in (
                "block_density",
                "block_scale",
                "block_trunk_length",
                "block_roll_variation",
                "block_seed",
                "block_max_polygons",
            ):
                blocks_box.prop(self, name)
        else:
            container.prop(self, "radial_resolution")
            container.prop(self, "smoothness")

        container.prop(self, "log_pipeline", icon="CONSOLE")

    def _draw_leaves_button(self, container):
        """Draw must not call get_current_tree_object: it links and selects, and
        Blender forbids writing to ID data while drawing. The exception aborts
        the whole node's draw, hiding every panel below this one.
        """
        if self.leaf_cards_node() is not None or self.leaf_clumps_node() is not None:
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
            mesh_mode=self.mesh_mode,
            resolution=self.radial_resolution,
            smoothness=self.smoothness,
            rig=self.build_bone_rig,
        ):
            diagnostics.log("  graph %s", diagnostics.describe_graph(self))

            trunk_node, reason = resolve_trunk(self)
            if trunk_node is None:
                raise ValueError(reason)

            with diagnostics.stage("construct_function", trunk=trunk_node.name):
                trunk_function = trunk_node.construct_function(seed_offset)
            if trunk_function is None:
                raise ValueError("Connected node returned no tree function")

            with diagnostics.stage("execute_functions"):
                tree = m_tree.Tree()
                tree.set_trunk_function(trunk_function)
                tree.execute_functions()

            blocks = self.mesh_mode == BLOCKS
            cpp_mesh = None
            if not blocks:
                with diagnostics.stage("mesh_tree") as meshing:
                    cpp_mesh = self._mesh_tree(tree)
                    coordinates = len(cpp_mesh.get_vertices())
                    meshing.record(
                        verts=int(coordinates / 3),
                        polys=len(cpp_mesh.get_polygons()),
                    )

            leaf_cards = self.leaf_cards_node()
            leaf_clumps = self.leaf_clumps_node()
            has_leaves = leaf_cards is not None or leaf_clumps is not None
            surface_acted = blocks or has_leaves
            if surface_acted:
                cpp_mesh = self._apply_surface(tree, cpp_mesh, leaf_cards, leaf_clumps, seed_offset)

            if has_leaves:
                self._remove_leaves_modifier(obj)

            if leaf_cards is not None:
                lods = self._leaf_lods(tree, leaf_cards, seed_offset)
            elif leaf_clumps is not None:
                lods = self._clump_lods(tree, leaf_clumps, seed_offset)
            else:
                lods = []

            strip_rig(obj)
            if obj.data.shape_keys is not None:
                obj.shape_key_clear()
            obj.data.clear_geometry()

            create_mesh_from_cpp(
                obj.data,
                cpp_mesh,
                flat_shading=surface_acted,
                material_slots=2 if has_leaves else 1,
            )
            diagnostics.log("  built %s", diagnostics.describe_mesh(obj.data))

            if len(obj.data.vertices) == 0:
                raise ValueError(self._empty_mesh_reason(trunk_node))

            self._write_leaf_lods(obj, lods, leaf_cards or leaf_clumps)

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
        trunk_node, reason = resolve_trunk(self)
        if trunk_node is None:
            self.status_message = reason
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
            rig = build_rig_for(tree_obj.data, self.segments_per_chain, self._stiffness_settings())
            count, message = bake_wind_clips(tree_obj, rig, directions=self.morph_directions)
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
            foliage_wind_settings_for,
            rig_settings,
        )

        return rig_settings(foliage_wind_settings_for(self))

    def leaf_cards_node(self):
        return chain_stage(self, LEAF_CARDS_ID)

    def leaf_clumps_node(self):
        return chain_stage(self, LEAF_CLUMPS_ID)

    def leaf_stage(self):
        return self.leaf_cards_node() or self.leaf_clumps_node()

    def _block_fill(self, seed_offset):
        fill = m_tree.BlockFill()
        fill.density = self.block_density
        fill.scale = self.block_scale
        fill.trunk_block_length = self.block_trunk_length
        fill.roll_variation_degrees = self.block_roll_variation
        fill.seed = self.block_seed + seed_offset
        return fill

    def _set_blocks_status(self, message, is_error=False):
        self.blocks_status_message = message
        self.blocks_status_is_error = is_error

    def _mesh_tree(self, tree):
        mesher = m_tree.ManifoldMesher()
        mesher.radial_n_points = self.radial_resolution
        mesher.smooth_iterations = self.smoothness
        return mesher.mesh_tree(tree)

    def _apply_surface(self, tree, cpp_mesh, leaf_cards, leaf_clumps, seed_offset):
        blocks = self._block_fill(seed_offset) if self.mesh_mode == BLOCKS else None
        cards = None if leaf_cards is None else leaf_cards.native(seed_offset)
        clumps = None if leaf_clumps is None else leaf_clumps.native(seed_offset)

        with diagnostics.stage(
            "surface",
            mesh_mode=self.mesh_mode,
            leaf_cards=None if leaf_cards is None else leaf_cards.name,
            leaf_clumps=None if leaf_clumps is None else leaf_clumps.name,
        ) as stage:
            polygons = 0
            if blocks is not None:
                polygons += blocks.count_polygons(tree)
            if cards is not None:
                polygons += cards.count_polygons(tree)

            if blocks is not None and polygons > self.block_max_polygons:
                message = (
                    f"Block Fill needs {polygons:,} polygons, over the cap of "
                    f"{self.block_max_polygons:,}"
                )
                self._set_blocks_status(f"Error: {message}", True)
                raise ValueError(message)

            if blocks is not None:
                cpp_mesh = blocks.mesh_tree(tree)
                stage.record(
                    density=self.block_density,
                    scale=self.block_scale,
                    blocks=blocks.block_count,
                    polygons=blocks.polygon_count,
                )
                self._set_blocks_status(
                    f"{blocks.block_count:,} blocks, {blocks.polygon_count:,} polygons"
                )

            if cards is not None:
                cards.append(tree, cpp_mesh)
                stage.record(cards=cards.card_count)
                leaf_cards.set_status(f"{cards.card_count:,} cards")

            if clumps is not None:
                try:
                    clumps.append(tree, cpp_mesh)
                except RuntimeError as error:
                    leaf_clumps.set_status(f"Error: {error}", True)
                    raise ValueError(str(error)) from error
                stage.record(
                    clumps=clumps.clump_count,
                    voxels=clumps.voxel_count,
                    clump_polygons=clumps.polygon_count,
                )
                leaf_clumps.set_status(
                    f"{clumps.clump_count:,} clumps, {clumps.voxel_count:,} voxels, "
                    f"{clumps.polygon_count:,} polygons"
                )

        return cpp_mesh

    def _bark_mesh(self, tree, seed_offset):
        if self.mesh_mode == BLOCKS:
            return self._block_fill(seed_offset).mesh_tree(tree)
        return self._mesh_tree(tree)

    def _leaf_lods(self, tree, leaf_cards, seed_offset):
        from ...pivot_painter.leaf_lods import source_vertices

        lods = []
        for level, fraction in leaf_cards.lod_fractions():
            with diagnostics.stage("leaf_lod", level=level, fraction=fraction) as stage:
                cpp_mesh = self._bark_mesh(tree, seed_offset)
                bark_vertices = int(len(cpp_mesh.get_vertices()) / 3)
                cards = leaf_cards.native(seed_offset)
                kept = cards.lod_cards(tree, fraction)
                cards.append(tree, cpp_mesh, fraction)
                vertices = int(len(cpp_mesh.get_vertices()) / 3)
                per_card = int((vertices - bark_vertices) / len(kept)) if kept else 0
                stage.record(cards=len(kept), verts=vertices)
            lods.append((level, cpp_mesh, source_vertices(bark_vertices, kept, per_card)))
        return lods

    def _clump_lods(self, tree, leaf_clumps, seed_offset):
        lods = []
        counts = []
        for level, scale in leaf_clumps.lod_voxel_scales():
            with diagnostics.stage("leaf_lod", level=level, voxel_scale=scale) as stage:
                cpp_mesh = self._bark_mesh(tree, seed_offset)
                bark_vertices = int(len(cpp_mesh.get_vertices()) / 3)
                clumps = leaf_clumps.native(seed_offset)
                lod0 = np.asarray(clumps.append_lod(tree, cpp_mesh, scale), dtype=np.int32)
                sources = np.concatenate(
                    [
                        np.arange(bark_vertices, dtype=np.int32),
                        np.where(lod0 < 0, 0, lod0 + bark_vertices).astype(np.int32),
                    ]
                )
                stage.record(
                    voxel_scale=scale,
                    voxels=clumps.voxel_count,
                    clump_polygons=clumps.polygon_count,
                    verts=int(len(cpp_mesh.get_vertices()) / 3),
                )
                counts.append(f"LOD{level} {clumps.polygon_count:,}")
            lods.append((level, cpp_mesh, sources))
        if counts:
            leaf_clumps.set_status(f"{leaf_clumps.status_message}, {' / '.join(counts)} polygons")
        return lods

    def _write_leaf_lods(self, obj, lods, leaf_stage):
        from ...pivot_painter.leaf_lods import (
            find_lod_objects,
            remove_lod_objects,
            show_lod,
            write_lod_object,
        )

        if not lods:
            if find_lod_objects(obj):
                remove_lod_objects(obj)
                show_lod(obj, 0)
            return

        for level, cpp_mesh, sources in lods:
            write_lod_object(obj, level, cpp_mesh, sources, len(obj.data.materials))
        show_lod(obj, leaf_stage.preview_level())

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
        return resolve_trunk(self)[0] is not None
