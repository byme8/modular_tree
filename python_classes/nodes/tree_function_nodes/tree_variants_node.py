"""SEED_STRIDE is prime so variant N and N+1 cannot alias onto nodes whose
authored seeds differ by one, which would give neighbours shared substructure.
"""

from __future__ import annotations

import bpy

from ...pivot_painter.leaf_lods import copy_foliage_streams, is_lod_object
from ...pivot_painter.tree_asset import link_exclusively, resolve_mesh_object
from ..base_types.node import MtreeNode

SEED_STRIDE = 7919


def mesh_summary(mesher) -> str:
    if mesher.mesh_mode != "BLOCKS":
        summary = f"{mesher.radial_resolution} sides, smooth {mesher.smoothness}"
    else:
        summary = f"Blocks, {mesher.block_density:g}/m, scale {mesher.block_scale:.2f}"
    if mesher.leaf_cards_node() is not None:
        summary += ", cards"
    elif mesher.leaf_clumps_node() is not None:
        summary += ", clumps"
    return summary


class MTreeVariants(bpy.types.Node, MtreeNode):
    """Generate several seed variants of the connected tree configuration"""

    bl_idname = "mt_VariantsNode"
    bl_label = "Tree Variants"

    variant_count: bpy.props.IntProperty(
        name="Variants",
        description="How many trees to generate from this configuration",
        default=20,
        min=1,
        max=200,
    )
    base_seed: bpy.props.IntProperty(
        name="Base Seed",
        description=(
            "Starting offset applied to every seed in the graph. Change it to get "
            "a different set of variants from the same configuration"
        ),
        default=0,
        min=0,
    )
    spacing: bpy.props.FloatProperty(
        name="Spacing",
        description="Distance between generated trees, in metres. 0 stacks them at the origin",
        default=10.0,
        min=0.0,
    )
    grid_columns: bpy.props.IntProperty(
        name="Columns",
        description="Trees per row when laying the variants out",
        default=5,
        min=1,
    )
    collection_name: bpy.props.StringProperty(
        name="Collection",
        description="Collection the generated variants are placed in",
        default="tree_variants",
    )

    bake_foliage_streams: bpy.props.BoolProperty(
        name="Bake Foliage Streams",
        description=(
            "Write the wind and shading data into UV and colour layers on every "
            "variant. Without this the exported meshes carry geometry only, "
            "because FBX cannot store the mesher's raw attributes"
        ),
        default=True,
    )
    bake_morphs: bpy.props.BoolProperty(
        name="Bake Wind Morphs",
        description=(
            "Bake wind poses as shape keys on every variant, so the exported "
            "meshes carry morph targets the engine blends by wind direction. No "
            "armature is created or exported; the rig is only used to compute "
            "the poses"
        ),
        default=False,
    )
    morph_directions: bpy.props.IntProperty(
        name="Wind Directions",
        description=(
            "Compass directions to bake a wind pose for. The engine blends the "
            "two nearest, so more directions bend more accurately at the cost of "
            "one vertex buffer each"
        ),
        default=8,
        min=2,
        max=16,
    )

    status_message: bpy.props.StringProperty(default="")
    status_is_error: bpy.props.BoolProperty(default=False)

    chain_input = ("mt_MeshSocket", "Mesh")
    chain_output = ("mt_VariantsSocket", "Variants")

    def init(self, context):
        self.add_chain_sockets()

    def draw(self, context, layout):
        if self.status_message:
            status_row = layout.row()
            status_row.alert = self.status_is_error
            icon = "ERROR" if self.status_is_error else "CHECKMARK"
            status_row.label(text=self.status_message, icon=icon)

        layout.prop(self, "variant_count")
        layout.prop(self, "base_seed")

        layout_box = layout.box()
        layout_box.label(text="Placement")
        layout_box.prop(self, "spacing")
        layout_box.prop(self, "grid_columns")
        layout_box.prop(self, "collection_name")

        mesher = self.get_mesher()

        mesh_box = layout.box()
        mesh_box.label(text="Mesh")
        if mesher is None:
            mesh_box.label(text="Connect a Tree Mesher", icon="ERROR")
        else:
            mesh_box.label(text=mesh_summary(mesher), icon="INFO")
            if mesher.build_bone_rig:
                from ..export_nodes.foliage_wind_node import (
                    describe_rig_layout,
                    foliage_wind_settings_for,
                )

                rig_layout = describe_rig_layout(foliage_wind_settings_for(mesher))
                mesh_box.label(text=f"Bone rig: {rig_layout}", icon="INFO")
        mesh_box.prop(self, "bake_foliage_streams")
        if not self.bake_foliage_streams:
            warn = mesh_box.row()
            warn.alert = True
            warn.label(text="Exports will carry geometry only", icon="ERROR")

        wind_box = layout.box()
        wind_box.label(text="Wind Clips", icon="ACTION")
        wind_box.prop(self, "bake_morphs")
        if self.bake_morphs:
            wind_box.prop(self, "morph_directions")

        generate_row = layout.row()
        generate_row.enabled = mesher is not None
        props = generate_row.operator("mtree.node_function", text="Generate Variants")
        props.node_tree_name = self.get_node_tree().name
        props.node_name = self.name
        props.function_name = "generate_variants"

    def output_collection_name(self) -> str:
        return self.collection_name or "tree_variants"

    def variant_objects(self) -> list:
        collection = bpy.data.collections.get(self.output_collection_name())
        if collection is None:
            return []
        return [obj for obj in collection.objects if obj.type == "MESH" and not is_lod_object(obj)]

    def _get_collection(self):
        name = self.output_collection_name()
        collection = bpy.data.collections.get(name)
        if collection is None:
            collection = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(collection)
        return collection

    def generate_variants(self):
        self.status_message = ""
        self.status_is_error = False

        mesher = self.get_mesher()
        if mesher is None:
            self.status_message = "Connect this to a Tree Mesher to generate"
            self.status_is_error = True
            return

        collection = self._get_collection()

        try:
            for variant in range(self.variant_count):
                offset = (self.base_seed + variant) * SEED_STRIDE
                self._build_variant(mesher, variant, collection, offset)

            self.status_message = f"Generated {self.variant_count} variants"
            self.status_is_error = False

        except Exception as e:
            self.status_message = f"Error: {str(e)}"
            self.status_is_error = True

    def _build_variant(self, mesher, variant_index, collection, seed_offset):
        name = f"{collection.name}_{variant_index:03d}"

        obj = resolve_mesh_object(name)
        if obj is None:
            obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))

        link_exclusively(obj, collection)
        self._place(obj, variant_index)

        mesher.build_into(obj, seed_offset)

        if self.bake_foliage_streams:
            self._bake_foliage(obj.data)
            copy_foliage_streams(obj)

        if self.bake_morphs:
            count, message = self._bake_morphs(obj)
            if count == 0:
                raise ValueError(f"{name}: {message}")

    def _place(self, obj, variant_index):
        columns = max(1, self.grid_columns)
        column = variant_index % columns
        row = int(variant_index / columns)
        obj.location = (column * self.spacing, row * self.spacing, 0.0)

    def _bake_foliage(self, mesh):
        """Bake the foliage streams into UV and colour layers.

        The mesher's attributes (``stem_id``, ``branch_extent`` and the rest) are
        Blender-only: FBX and glTF cannot carry arbitrary named float attributes,
        so a variant exported without this step arrives in the engine with one UV
        layer, no colours, and none of the wind data — geometrically correct and
        functionally empty.

        Baking here rather than leaving it as a separate manual step is the point:
        twenty variants means twenty bakes, and a forgotten one produces a file
        that looks fine until the shader reads zeros from it.
        """
        from ...nodes.export_nodes.foliage_wind_node import foliage_wind_settings_for
        from ...pivot_painter.formats.foliage import FoliageExporter

        settings = foliage_wind_settings_for(self.get_mesher())
        for key in (
            "build_bone_rig",
            "segments_per_chain",
            "branch_segments_per_chain",
            "rig_depth",
        ):
            settings.pop(key, None)

        FoliageExporter(mesh=mesh, **settings).export()

    def _bake_morphs(self, obj):
        from ...nodes.export_nodes.foliage_wind_node import (
            foliage_wind_settings_for,
            rig_settings,
        )
        from ...pivot_painter.wind_animation import bake_wind_clips, build_rig_for

        settings = foliage_wind_settings_for(self.get_mesher())

        rig = build_rig_for(obj.data, settings["segments_per_chain"], rig_settings(settings))

        return bake_wind_clips(obj, rig, directions=self.morph_directions)
