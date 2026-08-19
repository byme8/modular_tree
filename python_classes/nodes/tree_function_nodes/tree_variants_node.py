"""Node for generating multiple seed variants of one tree configuration.

A tree's randomness is not driven by a single seed. Every function node in the
graph — trunk, each branch level, growth — carries its own ``seed`` socket, so
changing one node's seed only re-rolls that node's contribution and leaves the
rest of the tree identical. Producing a genuinely different tree means offsetting
every seed in the graph at once.

This node does that without duplicating the graph: for each variant it walks the
connected function nodes, offsets each seed by a per-variant amount, builds a
mesh into its own object, then restores the original seeds. The authored
configuration is left exactly as it was.
"""

from __future__ import annotations

import bpy

from ...m_tree_wrapper import lazy_m_tree as m_tree
from ...mesh_utils import create_mesh_from_cpp
from ..base_types.node import MtreeNode

# Socket property name that carries randomness on every function node.
SEED_PROPERTY = "seed"

# Seeds are offset by variant_index * this stride rather than by 1, so that
# variant N and variant N+1 do not share a seed on any node whose own seed
# happens to differ by one. A prime keeps the offsets from aligning with the
# small integer seeds nodes are usually authored with.
SEED_STRIDE = 7919


def collect_seed_sockets(root_node, seen=None):
    """Collect every seed-carrying input socket reachable from a node.

    Walks the same child chain that ``construct_function`` follows, so the set of
    sockets found is exactly the set that contributes to the built tree.

    Args:
        root_node: Node to start from (the mesher's connected trunk).
        seen: Internal recursion guard against graph loops.

    Returns:
        List of input sockets whose ``property_name`` is ``seed``.
    """
    if seen is None:
        seen = set()

    if root_node is None or root_node.name in seen:
        return []
    seen.add(root_node.name)

    sockets = [
        socket
        for socket in root_node.inputs
        if getattr(socket, "property_name", "") == SEED_PROPERTY
    ]

    for child in root_node.get_child_nodes():
        sockets.extend(collect_seed_sockets(child, seen))

    return sockets


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

    radial_resolution: bpy.props.IntProperty(name="Radial Resolution", default=32, min=3)
    smoothness: bpy.props.IntProperty(name="Smoothness", default=4, min=0)
    bake_foliage_streams: bpy.props.BoolProperty(
        name="Bake Foliage Streams",
        description=(
            "Write the wind and shading data into UV and colour layers on every "
            "variant. Without this the exported meshes carry geometry only, "
            "because FBX cannot store the mesher's raw attributes"
        ),
        default=True,
    )

    status_message: bpy.props.StringProperty(default="")
    status_is_error: bpy.props.BoolProperty(default=False)

    def init(self, context):
        self.add_output("mt_TreeSocket", "Tree", is_property=False)

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

        mesh_box = layout.box()
        mesh_box.label(text="Mesh")
        mesh_box.prop(self, "radial_resolution")
        mesh_box.prop(self, "smoothness")
        mesh_box.prop(self, "bake_foliage_streams")
        if not self.bake_foliage_streams:
            warn = mesh_box.row()
            warn.alert = True
            warn.label(text="Exports will carry geometry only", icon="ERROR")

        generate_row = layout.row()
        generate_row.enabled = self._get_trunk_node() is not None
        props = generate_row.operator("mtree.node_function", text="Generate Variants")
        props.node_tree_name = self.get_node_tree().name
        props.node_name = self.name
        props.function_name = "generate_variants"

    def _get_trunk_node(self):
        """Return the function node connected to this node's output."""
        if not self.outputs or not self.outputs[0].links:
            return None
        return self.outputs[0].links[0].to_node

    def _get_collection(self):
        """Get or create the collection the variants are placed in."""
        name = self.collection_name or "tree_variants"
        collection = bpy.data.collections.get(name)
        if collection is None:
            collection = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(collection)
        return collection

    def generate_variants(self):
        """Build one mesh per variant, each with every seed in the graph offset."""
        self.status_message = ""
        self.status_is_error = False

        trunk_node = self._get_trunk_node()
        if trunk_node is None:
            self.status_message = "Connect a Trunk node to generate"
            self.status_is_error = True
            return

        seed_sockets = collect_seed_sockets(trunk_node)
        if not seed_sockets:
            self.status_message = "No seed inputs found in the connected graph"
            self.status_is_error = True
            return

        # Restore the authored seeds even if a build raises partway through, so a
        # failed run cannot silently rewrite the user's configuration.
        original_seeds = [socket.property_value for socket in seed_sockets]
        collection = self._get_collection()

        try:
            for variant in range(self.variant_count):
                offset = (self.base_seed + variant) * SEED_STRIDE
                for socket, original in zip(seed_sockets, original_seeds, strict=True):
                    socket.property_value = original + offset

                self._build_variant(trunk_node, variant, collection)

            self.status_message = f"Generated {self.variant_count} variants"
            self.status_is_error = False

        except Exception as e:
            self.status_message = f"Error: {str(e)}"
            self.status_is_error = True

        finally:
            for socket, original in zip(seed_sockets, original_seeds, strict=True):
                socket.property_value = original

    def _build_variant(self, trunk_node, variant_index, collection):
        """Build a single variant into its own object."""
        tree = m_tree.Tree()
        tree.set_trunk_function(trunk_node.construct_function())
        tree.execute_functions()

        mesher = m_tree.ManifoldMesher()
        mesher.radial_n_points = self.radial_resolution
        mesher.smooth_iterations = self.smoothness
        cpp_mesh = mesher.mesh_tree(tree)

        name = f"{self.collection_name}_{variant_index:03d}"

        # Reuse the object across runs so regenerating does not leave orphans or
        # break references from other parts of the scene.
        obj = bpy.data.objects.get(name)
        if obj is None or not isinstance(obj.data, bpy.types.Mesh):
            mesh = bpy.data.meshes.new(name)
            obj = bpy.data.objects.new(name, mesh)
        else:
            obj.data.clear_geometry()

        if obj.name not in collection.objects:
            for existing in list(obj.users_collection):
                existing.objects.unlink(obj)
            collection.objects.link(obj)

        create_mesh_from_cpp(obj.data, cpp_mesh)

        if self.bake_foliage_streams:
            self._bake_foliage(obj.data)

        column = variant_index % self.grid_columns
        row = variant_index // self.grid_columns
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
        from ...nodes.export_nodes.foliage_wind_node import find_foliage_wind_settings
        from ...pivot_painter.formats.foliage import FoliageExporter

        settings = find_foliage_wind_settings(self.get_node_tree())
        FoliageExporter(mesh=mesh, **settings).export()
