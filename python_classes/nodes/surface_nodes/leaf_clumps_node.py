from __future__ import annotations

import bpy

from ...m_tree_wrapper import lazy_m_tree as m_tree
from ..pipeline import LEAF_CLUMPS_ID
from .leaf_lod_stage import LeafLodStage, on_update_preview_lod
from .surface_node import SurfaceNode, on_update_setting

BUSH = "BUSH"
CONE = "CONE"
PINE_BRANCH = "PINE_BRANCH"

NATIVE_SHAPES = {BUSH: "Bush", CONE: "Cone", PINE_BRANCH: "PineBranch"}

COMMON_SETTINGS = (
    "shape",
    "voxel_size",
    "clump_size",
    "width",
    "depth",
    "height",
    "size_variation",
    "clump_density",
    "spread",
    "erosion",
    "leaf_seed",
    "max_polygons",
)
PINE_SETTINGS = ("tiers", "tilt_variation")


class MTreeLeafClumps(bpy.types.Node, SurfaceNode, LeafLodStage):
    bl_idname = LEAF_CLUMPS_ID
    bl_label = "Leaf Clumps"

    shape: bpy.props.EnumProperty(
        name="Shape",
        items=[
            (BUSH, "Bush", "Ellipsoid clumps on the object axes"),
            (CONE, "Cone", "Upright cones, base below and apex above"),
            (PINE_BRANCH, "Pine Branch", "Tiered discs along the twig, shrinking to the tip"),
        ],
        default=BUSH,
        update=on_update_setting,
    )
    voxel_size: bpy.props.FloatProperty(
        name="Voxel Size",
        description="Edge of one voxel cube in metres",
        default=0.05,
        min=0.005,
        max=1.0,
        unit="LENGTH",
        update=on_update_setting,
    )
    clump_size: bpy.props.FloatProperty(
        name="Clump Size",
        description="Base diameter of one clump in metres; 0 disables clumps",
        default=0.4,
        min=0.0,
        max=10.0,
        unit="LENGTH",
        update=on_update_setting,
    )
    width: bpy.props.FloatProperty(
        name="Width", default=1.0, min=0.1, max=10.0, update=on_update_setting
    )
    depth: bpy.props.FloatProperty(
        name="Depth", default=1.0, min=0.1, max=10.0, update=on_update_setting
    )
    height: bpy.props.FloatProperty(
        name="Height", default=1.0, min=0.1, max=10.0, update=on_update_setting
    )
    size_variation: bpy.props.FloatProperty(
        name="Size Variation",
        description="Each clump's size is scaled by a random factor in 1 ± this",
        default=0.3,
        min=0.0,
        max=0.9,
        subtype="FACTOR",
        update=on_update_setting,
    )
    clump_density: bpy.props.FloatProperty(
        name="Density",
        description="Clumps per metre of leaf-level twig; 0 disables clumps",
        default=4.0,
        min=0.0,
        max=1000.0,
        update=on_update_setting,
    )
    spread: bpy.props.FloatProperty(
        name="Spread",
        description="How far out from the twig a clump centre may sit, in metres",
        default=0.2,
        min=0.0,
        max=10.0,
        unit="LENGTH",
        update=on_update_setting,
    )
    tilt_variation: bpy.props.FloatProperty(
        name="Tilt Variation",
        description="Random tilt of a Pine Branch axis away from its twig, in degrees",
        default=30.0,
        min=0.0,
        max=90.0,
        update=on_update_setting,
    )
    tiers: bpy.props.IntProperty(
        name="Tiers",
        description="Number of discs in a Pine Branch",
        default=4,
        min=1,
        max=16,
        update=on_update_setting,
    )
    erosion: bpy.props.FloatProperty(
        name="Erosion",
        description="Share of surface voxels randomly removed",
        default=0.2,
        min=0.0,
        max=0.9,
        subtype="FACTOR",
        update=on_update_setting,
    )
    leaf_seed: bpy.props.IntProperty(
        name="Leaf Seed",
        description="Seed for placement, size variation and erosion; the variants offset is added to it",
        default=0,
        update=on_update_setting,
    )
    max_polygons: bpy.props.IntProperty(
        name="Max Polygons",
        description="Over this many clump polygons the build fails and the previous tree stays",
        default=200000,
        min=1000,
        max=10000000,
        update=on_update_setting,
    )

    build_lods: bpy.props.BoolProperty(
        name="Build LODs",
        description=(
            "Also build LOD1 and LOD2 objects for every tree: identical bark, "
            "the same clumps at coarser voxels"
        ),
        default=False,
        update=on_update_setting,
    )
    lod1_voxel_scale: bpy.props.FloatProperty(
        name="LOD1 Voxel Scale",
        description="LOD1 voxel size is Voxel Size times this",
        default=2.0,
        min=1.0,
        max=16.0,
        update=on_update_setting,
    )
    lod2_voxel_scale: bpy.props.FloatProperty(
        name="LOD2 Voxel Scale",
        description="LOD2 voxel size is Voxel Size times this, never less than LOD1's",
        default=4.0,
        min=1.0,
        max=16.0,
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

    @property
    def settings_layout(self) -> tuple:
        if self.shape == PINE_BRANCH:
            return COMMON_SETTINGS + PINE_SETTINGS
        return COMMON_SETTINGS

    def draw(self, context, layout):
        super().draw(context, layout)
        lod_box = layout.box()
        lod_box.prop(self, "build_lods")
        if self.build_lods:
            lod_box.prop(self, "lod1_voxel_scale")
            lod_box.prop(self, "lod2_voxel_scale")
            lod_box.row().prop(self, "preview_lod", expand=True)

    def lod_voxel_scales(self) -> tuple:
        if not self.build_lods:
            return ()
        return (
            (1, self.lod1_voxel_scale),
            (2, max(self.lod2_voxel_scale, self.lod1_voxel_scale)),
        )

    def native(self, seed_offset: int = 0):
        clumps = m_tree.LeafClumps()
        clumps.shape = getattr(m_tree.ClumpShape, NATIVE_SHAPES[self.shape])
        clumps.voxel_size = self.voxel_size
        clumps.clump_size = self.clump_size
        clumps.width = self.width
        clumps.depth = self.depth
        clumps.height = self.height
        clumps.size_variation = self.size_variation
        clumps.density = self.clump_density
        clumps.spread = self.spread
        clumps.tilt_variation_degrees = self.tilt_variation
        clumps.tiers = self.tiers
        clumps.erosion = self.erosion
        clumps.seed = self.leaf_seed + seed_offset
        clumps.max_polygons = self.max_polygons
        return clumps
