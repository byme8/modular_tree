from __future__ import annotations

import bpy

from ..base_types.node import MtreeNode
from ..pipeline import BLOCK_FILL_ID


class MTreeBlockFill(bpy.types.Node, MtreeNode):
    bl_idname = BLOCK_FILL_ID
    bl_label = "Block Fill (legacy)"

    density: bpy.props.FloatProperty(name="Density", default=10.0, min=0.1, max=1000.0)
    scale: bpy.props.FloatProperty(name="Scale", default=1.0, min=0.1, max=5.0)
    roll_variation: bpy.props.FloatProperty(name="Roll Variation", default=15.0, min=0.0, max=180.0)
    seed: bpy.props.IntProperty(name="Seed", default=0)
    max_polygons: bpy.props.IntProperty(name="Max Polygons", default=200000, min=1000, max=10000000)

    def init(self, context):
        self.add_input("mt_TreeSocket", "Tree", is_property=False)

    def draw(self, context, layout):
        layout.label(text="Moved to the Tree Mesher's Blocks mode", icon="INFO")
