from __future__ import annotations

import bpy

from ...m_tree_wrapper import lazy_m_tree as m_tree
from ..pipeline import LEAF_CARDS_ID
from .leaf_lod_stage import LeafLodStage, on_update_preview_lod
from .surface_node import SurfaceNode, on_update_setting


class MTreeLeafCards(bpy.types.Node, SurfaceNode, LeafLodStage):
    bl_idname = LEAF_CARDS_ID
    bl_label = "Leaf Cards"

    card_size: bpy.props.FloatProperty(
        name="Card Size",
        description="Side of each card square in metres; 0 disables cards",
        default=0.2,
        min=0.0,
        max=10.0,
        unit="LENGTH",
        update=on_update_setting,
    )
    card_density: bpy.props.FloatProperty(
        name="Card Density",
        description="Cards per metre of leaf-level twig; 0 disables cards",
        default=20.0,
        min=0.0,
        max=1000.0,
        update=on_update_setting,
    )
    spread: bpy.props.FloatProperty(
        name="Spread",
        description=(
            "How far out from the twig a card may sit, in metres; cards fill "
            "the volume out to this distance, so a large value builds a canopy"
        ),
        default=0.3,
        min=0.0,
        max=10.0,
        unit="LENGTH",
        update=on_update_setting,
    )
    tilt_variation: bpy.props.FloatProperty(
        name="Tilt Variation",
        description=(
            "Random tilt of each card pair away from its twig, in degrees; 90 "
            "is fully random orientation, 0 keeps cards aligned with the twig"
        ),
        default=90.0,
        min=0.0,
        max=90.0,
        update=on_update_setting,
    )
    leaf_seed: bpy.props.IntProperty(
        name="Leaf Seed",
        description="Seed for card placement; the variants offset is added to it",
        default=0,
        update=on_update_setting,
    )

    build_lods: bpy.props.BoolProperty(
        name="Build LODs",
        description=(
            "Also build LOD1 and LOD2 objects for every tree: identical bark, "
            "fewer leaf cards, the most hidden cards removed first"
        ),
        default=False,
        update=on_update_setting,
    )
    lod1_fraction: bpy.props.FloatProperty(
        name="LOD1 Cards",
        description="Share of LOD0 cards LOD1 keeps; survivors grow to keep the leaf area",
        default=0.5,
        min=0.01,
        max=1.0,
        subtype="FACTOR",
        update=on_update_setting,
    )
    lod2_fraction: bpy.props.FloatProperty(
        name="LOD2 Cards",
        description="Share of LOD0 cards LOD2 keeps; never more than LOD1 keeps",
        default=0.2,
        min=0.01,
        max=1.0,
        subtype="FACTOR",
        update=on_update_setting,
    )
    preview_lod: bpy.props.EnumProperty(
        name="Preview LOD",
        description="Which level of detail the viewport shows for every tree of this pipeline",
        items=[
            ("0", "LOD0", "The full tree"),
            ("1", "LOD1", "Show LOD1 and hide the others"),
            ("2", "LOD2", "Show LOD2 and hide the others"),
        ],
        default="0",
        update=on_update_preview_lod,
    )

    status_message: bpy.props.StringProperty(default="")
    status_is_error: bpy.props.BoolProperty(default=False)

    settings_layout = ("card_size", "card_density", "spread", "tilt_variation", "leaf_seed")

    def draw(self, context, layout):
        super().draw(context, layout)
        lod_box = layout.box()
        lod_box.prop(self, "build_lods")
        if self.build_lods:
            lod_box.prop(self, "lod1_fraction", slider=True)
            lod_box.prop(self, "lod2_fraction", slider=True)
            lod_box.row().prop(self, "preview_lod", expand=True)

    def lod_fractions(self) -> tuple:
        if not self.build_lods:
            return ()
        return ((1, self.lod1_fraction), (2, min(self.lod2_fraction, self.lod1_fraction)))

    def native(self, seed_offset: int = 0):
        cards = m_tree.LeafCards()
        cards.card_size = self.card_size
        cards.density = self.card_density
        cards.spread = self.spread
        cards.tilt_variation_degrees = self.tilt_variation
        cards.seed = self.leaf_seed + seed_offset
        return cards
