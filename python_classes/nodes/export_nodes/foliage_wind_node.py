"""Node for authoring foliage wind stiffness.

Stiffness decides which parts of the tree move under wind, so it is an authoring
decision rather than an export setting. This node owns those values; the export
node reads them from whichever instance of this node is upstream, and the
viewport debug view reads them too, so what is previewed is what is exported.
"""

from __future__ import annotations

import bpy

from ..base_types.node import MtreeNode


class MTreeFoliageWind(bpy.types.Node, MtreeNode):
    """Control how stiffly each part of the tree responds to wind"""

    bl_idname = "mt_FoliageWindNode"
    bl_label = "Foliage Wind"

    trunk_base_stiffness: bpy.props.FloatProperty(
        name="Trunk Base",
        description=(
            "Trunk stiffness at the root. 1 holds the base rigid; lower values "
            "let the whole trunk sway. The rooted ring is always pinned rigid "
            "regardless, so the tree cannot drift out of the ground"
        ),
        default=1.0,
        min=0.0,
        max=1.0,
    )
    trunk_tip_stiffness: bpy.props.FloatProperty(
        name="Trunk Tip",
        description=(
            "Trunk stiffness at the top. Set below the base value to make the "
            "trunk bend progressively toward the crown"
        ),
        default=0.6,
        min=0.0,
        max=1.0,
    )
    branch_stiffness: bpy.props.FloatProperty(
        name="Branch Scale",
        description=(
            "Scale on the side-branch stiffness curve, which is driven by branch "
            "radius. Lower values make every side branch more compliant"
        ),
        default=1.0,
        min=0.0,
        max=1.0,
    )
    junction_falloff: bpy.props.FloatProperty(
        name="Junction Falloff",
        description=(
            "How much softer a branch may be than the limb carrying it. Values "
            "near 1 keep neighbouring wood moving together; low values let each "
            "level move much more freely than its parent, which makes branches "
            "whip around independently in wind"
        ),
        default=0.6,
        min=0.1,
        max=1.0,
    )

    # How the wind is delivered: displaced vertices, or a bone rig. These are
    # alternatives rather than layers — the consuming shader turns vertex wind
    # off entirely when it finds bones — so they belong with the stiffness
    # curves they share, not with the export settings.
    use_bone_rig: bpy.props.BoolProperty(
        name="Use Bone Rig",
        description=(
            "Drive wind from a bone rig instead of from per-vertex displacement. "
            "Vertex wind translates each vertex along one global direction with "
            "no rotation, so a branch cannot follow the trunk's lean and shears "
            "against it at every junction. Bones compose rotations down the "
            "hierarchy, which is the only way branch motion stays attached to "
            "the limb carrying it. The stiffness curves above drive either path"
        ),
        default=False,
    )
    segments_per_chain: bpy.props.IntProperty(
        name="Trunk Bones",
        description=(
            "How many bones the trunk bends through. One bone can only move it "
            "rigidly, which reproduces the very problem the rig solves; more "
            "bones bend more smoothly at proportional cost"
        ),
        default=4,
        min=1,
        max=16,
    )
    branch_segments_per_chain: bpy.props.IntProperty(
        name="Branch Bones",
        description="Bones per rigged branch chain; one swings the branch rigidly about its base",
        default=2,
        min=1,
        max=16,
    )
    rig_depth: bpy.props.IntProperty(
        name="Rig Depth",
        description=(
            "Deepest branch level that gets bones of its own. Chains below it "
            "are skinned to the bone of the limb they grow from, so twigs and "
            "their leaves move as part of that limb"
        ),
        default=1,
        min=0,
        max=16,
    )

    # Shading scalars packed into vertex colour, authored alongside the wind
    # settings because they share the same along-branch parameter.
    occlusion_base: bpy.props.FloatProperty(
        name="Occlusion Base",
        description="Ambient occlusion at the branch base; the tip is always 1",
        default=0.25,
        min=0.0,
        max=1.0,
    )
    translucency_tip: bpy.props.FloatProperty(
        name="Translucency Tip",
        description=(
            "Back-lit transmission at the branch tip. Wood is opaque, so this "
            "belongs at 0 for trunks and branches; raise it only for twigs thin "
            "enough to genuinely glow against the sky. Leaves are shaded "
            "separately and transmit light by default"
        ),
        default=0.0,
        min=0.0,
        max=1.0,
    )
    normal_blend: bpy.props.FloatProperty(
        name="Normal Blend",
        description="Blend from geometric normals (0) to flat up (1) for the whole plant",
        default=0.0,
        min=0.0,
        max=1.0,
    )

    def init(self, context):
        self.add_input("mt_TreeSocket", "Tree", is_property=False)
        self.add_output("mt_TreeSocket", "Tree")

    def draw(self, context, layout):
        wind_box = layout.box()
        wind_box.label(text="Stiffness")
        wind_box.prop(self, "trunk_base_stiffness")
        wind_box.prop(self, "trunk_tip_stiffness")
        wind_box.prop(self, "branch_stiffness")
        wind_box.prop(self, "junction_falloff")

        # A rigid trunk is a legitimate choice, but it is also the state people
        # land in by accident, so say so rather than letting it look broken.
        if self.trunk_base_stiffness >= 1.0 and self.trunk_tip_stiffness >= 1.0:
            info = wind_box.row()
            info.label(text="Trunk fully rigid; only branches move", icon="INFO")

        delivery_box = layout.box()
        delivery_box.label(text="Wind Delivery")
        delivery_box.prop(self, "use_bone_rig")
        if self.use_bone_rig:
            delivery_box.prop(self, "segments_per_chain")
            delivery_box.prop(self, "branch_segments_per_chain")
            delivery_box.prop(self, "rig_depth")
            if self.segments_per_chain == 1:
                warning = delivery_box.row()
                warning.label(
                    text="1 trunk bone moves the trunk rigidly",
                    icon="ERROR",
                )
            # The engine picks the path by keyword, so a rigged mesh silently
            # ignores the vertex wind it also carries. Saying which one is live
            # avoids tuning a curve through the path that is switched off.
            note = delivery_box.row()
            note.label(text="Vertex wind is replaced, not layered", icon="INFO")

        shading_box = layout.box()
        shading_box.label(text="Shading")
        shading_box.prop(self, "occlusion_base")
        shading_box.prop(self, "translucency_tip")
        shading_box.prop(self, "normal_blend")

        debug_box = layout.box()
        debug_box.label(text="Debug View")
        debug_box.prop(context.scene, "mtree_foliage_debug_channel", text="")

    def get_settings(self) -> dict:
        """Return the authored values as exporter keyword arguments."""
        return {
            "trunk_base_stiffness": self.trunk_base_stiffness,
            "trunk_tip_stiffness": self.trunk_tip_stiffness,
            "branch_stiffness": self.branch_stiffness,
            "junction_falloff": self.junction_falloff,
            "occlusion_base": self.occlusion_base,
            "translucency_tip": self.translucency_tip,
            "normal_blend": self.normal_blend,
            "build_bone_rig": self.use_bone_rig,
            "segments_per_chain": self.segments_per_chain,
            "branch_segments_per_chain": self.branch_segments_per_chain,
            "rig_depth": self.rig_depth,
        }


DEFAULT_FOLIAGE_SETTINGS = {
    "trunk_base_stiffness": 1.0,
    "trunk_tip_stiffness": 1.0,
    "branch_stiffness": 1.0,
    "junction_falloff": 0.6,
    "occlusion_base": 0.25,
    "translucency_tip": 0.0,
    "normal_blend": 0.0,
    "build_bone_rig": False,
    "segments_per_chain": 4,
    "branch_segments_per_chain": 2,
    "rig_depth": 1,
}


RIG_SETTING_KEYS = (
    "trunk_base_stiffness",
    "trunk_tip_stiffness",
    "branch_stiffness",
    "junction_falloff",
    "branch_segments_per_chain",
    "rig_depth",
)


def rig_settings(settings: dict) -> dict:
    return {key: settings[key] for key in RIG_SETTING_KEYS}


def describe_rig_layout(settings: dict) -> str:
    return (
        f"trunk {settings['segments_per_chain']}, branches "
        f"{settings['branch_segments_per_chain']} to depth {settings['rig_depth']}"
    )


def find_foliage_wind_settings(node_tree) -> dict:
    """Find the foliage wind settings in a node tree.

    Falls back to defaults when no such node is present, so the foliage export
    stays usable without one.
    """
    if node_tree is None:
        return dict(DEFAULT_FOLIAGE_SETTINGS)

    for node in node_tree.nodes:
        if node.bl_idname == MTreeFoliageWind.bl_idname:
            return node.get_settings()

    return dict(DEFAULT_FOLIAGE_SETTINGS)
