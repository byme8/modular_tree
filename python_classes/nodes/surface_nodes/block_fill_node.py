from __future__ import annotations

import bpy

from ...m_tree_wrapper import lazy_m_tree as m_tree
from .linking import BLOCK_FILL_ID
from .surface_node import SurfaceNode, on_update_setting


class MTreeBlockFill(bpy.types.Node, SurfaceNode):
    bl_idname = BLOCK_FILL_ID
    bl_label = "Block Fill"

    density: bpy.props.FloatProperty(
        name="Density",
        description="Blocks per metre along every branch chain",
        default=10.0,
        min=0.1,
        max=1000.0,
        update=on_update_setting,
    )
    scale: bpy.props.FloatProperty(
        name="Scale",
        description="Grows or shrinks every block; above 1 they overlap in every direction",
        default=1.0,
        min=0.1,
        max=5.0,
        update=on_update_setting,
    )
    roll_variation: bpy.props.FloatProperty(
        name="Roll Variation",
        description="Each block twists about its branch by a random angle up to this, in degrees",
        default=15.0,
        min=0.0,
        max=180.0,
        update=on_update_setting,
    )
    seed: bpy.props.IntProperty(
        name="Seed",
        description="Seed for the block rolls; the variants offset is added to it",
        default=0,
        update=on_update_setting,
    )
    max_polygons: bpy.props.IntProperty(
        name="Max Polygons",
        description="A build that would exceed this fails and leaves the tree as it was",
        default=200000,
        min=1000,
        max=10000000,
        update=on_update_setting,
    )

    status_message: bpy.props.StringProperty(default="")
    status_is_error: bpy.props.BoolProperty(default=False)

    settings_layout = ("density", "scale", "roll_variation", "seed", "max_polygons")

    def native(self, seed_offset: int = 0):
        fill = m_tree.BlockFill()
        fill.density = self.density
        fill.scale = self.scale
        fill.roll_variation_degrees = self.roll_variation
        fill.seed = self.seed + seed_offset
        return fill
