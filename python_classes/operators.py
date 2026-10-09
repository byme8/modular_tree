"""Blender operators for MTree addon."""

from __future__ import annotations

import time
from random import randint

import bpy
from bpy.utils import register_class, unregister_class

from .m_tree_wrapper import lazy_m_tree as m_tree
from .mesh_utils import create_mesh_from_cpp
from .pivot_painter.formats.foliage import FoliageExporter
from .pivot_painter.formats.foliage_rig import FoliageRigExporter
from .pivot_painter.tree_asset import strip_rig
from .presets import (
    TREE_PRESETS,
    apply_preset,
    apply_preset_to_generator,
    apply_sub_branch_preset,
    apply_trunk_preset,
    get_growth_preset_items,
    get_preset_items,
)
from .presets.leaf_presets import LEAF_PRESETS, get_leaf_preset_items
from .resources.node_groups import distribute_leaves


class ExecuteNodeFunction(bpy.types.Operator):
    """Execute a function on a node by name."""

    bl_idname = "mtree.node_function"
    bl_label = "Node Function callback"
    bl_options = {"REGISTER", "UNDO"}

    node_tree_name: bpy.props.StringProperty()
    node_name: bpy.props.StringProperty()
    function_name: bpy.props.StringProperty()

    def execute(self, context):
        node = bpy.data.node_groups[self.node_tree_name].nodes[self.node_name]
        getattr(node, self.function_name)()
        return {"FINISHED"}


class AddLeavesModifier(bpy.types.Operator):
    """Add leaves distribution modifier to tree."""

    bl_idname = "mtree.add_leaves"
    bl_label = "Add leaves distribution modifier to tree"
    bl_options = {"REGISTER", "UNDO"}

    object_id: bpy.props.StringProperty()
    distribution_mode: bpy.props.IntProperty(
        name="Distribution Mode",
        description="0=Random, 1=Phyllotactic",
        default=0,
        min=0,
        max=1,
    )
    phyllotaxis_angle: bpy.props.FloatProperty(
        name="Phyllotaxis Angle",
        description="Divergence angle in degrees (137.5 = golden angle)",
        default=137.5,
        min=0.0,
        max=360.0,
    )
    billboard_mode: bpy.props.EnumProperty(
        name="Billboard Mode",
        items=[
            ("OFF", "Off", "No billboard rotation"),
            ("AXIAL", "Axial", "Rotate around branch axis toward camera"),
            ("CAMERA", "Camera-Facing", "Fully face the camera"),
        ],
        default="OFF",
    )
    lod_1_distance: bpy.props.FloatProperty(
        name="LOD 1 Distance",
        description="Distance threshold for LOD 1 switching",
        default=20.0,
        min=0.0,
    )
    cull_distance: bpy.props.FloatProperty(
        name="Cull Distance",
        description="Distance beyond which leaves are removed",
        default=100.0,
        min=0.0,
    )
    enable_normal_transfer: bpy.props.BoolProperty(
        name="Enable Normal Transfer",
        description="Transfer proxy volume normals to leaves for smooth canopy shading",
        default=True,
    )

    def execute(self, context):
        ob = bpy.data.objects.get(self.object_id)
        if ob is not None:
            distribute_leaves(
                ob,
                distribution_mode=self.distribution_mode,
                phyllotaxis_angle=self.phyllotaxis_angle,
                billboard_mode=self.billboard_mode,
                lod_1_distance=self.lod_1_distance,
                cull_distance=self.cull_distance,
                camera=bpy.context.scene.camera,
                enable_normal_transfer=self.enable_normal_transfer,
            )
        return {"FINISHED"}


class QuickGenerateTree(bpy.types.Operator):
    """Generate a random tree with preset settings."""

    bl_idname = "mtree.quick_generate"
    bl_label = "Generate Random Tree"
    bl_options = {"REGISTER", "UNDO"}

    seed: bpy.props.IntProperty(
        name="Seed", default=0, min=0, description="Random seed (0 = random)"
    )
    preset: bpy.props.EnumProperty(name="Preset", items=get_preset_items(), default="RANDOM")
    add_leaves: bpy.props.BoolProperty(
        name="Add Leaves", default=True, description="Automatically add leaf distribution"
    )

    # Maps tree presets to matching leaf presets
    _TREE_TO_LEAF_PRESET = {
        "OAK": "OAK",
        "PINE": "PINE",
        "WILLOW": "WILLOW",
    }

    def execute(self, context):
        try:
            seed = self.seed if self.seed != 0 else randint(0, 10000)
            start_time = time.time()

            cpp_mesh = self._generate_tree(seed)
            self._create_blender_object(context, cpp_mesh, f"Tree_{seed}")

            if self.add_leaves:
                active_obj = context.view_layer.objects.active
                if active_obj is not None:
                    leaf_obj = self._create_leaf_for_preset(seed)
                    leaf_kwargs = {}
                    preset = TREE_PRESETS.get(self.preset)
                    if preset and preset.leaf_params:
                        leaf_kwargs = preset.leaf_params
                    distribute_leaves(active_obj, leaf_object=leaf_obj, **leaf_kwargs)

            elapsed = time.time() - start_time
            self.report({"INFO"}, f"Generated tree (seed={seed}) in {elapsed:.2f}s")
            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, f"Failed to generate tree: {str(e)}")
            return {"CANCELLED"}

    def _generate_tree(self, seed: int):
        """Generate tree using C++ library."""
        tree = m_tree.Tree()

        trunk = m_tree.TrunkFunction()
        trunk.seed = seed
        apply_trunk_preset(trunk, self.preset)

        branches = m_tree.BranchFunction()
        branches.seed = seed + 1
        apply_preset(branches, self.preset)

        preset = TREE_PRESETS.get(self.preset)
        if preset and preset.sub_branches:
            sub_branches = m_tree.BranchFunction()
            sub_branches.seed = seed + 2
            apply_sub_branch_preset(sub_branches, self.preset)
            branches.add_child(sub_branches)

        trunk.add_child(branches)
        tree.set_trunk_function(trunk)
        tree.execute_functions()

        mesher = m_tree.ManifoldMesher()
        mesher.radial_n_points = 32
        mesher.smooth_iterations = 4
        return mesher.mesh_tree(tree)

    def _create_blender_object(self, context, cpp_mesh, name: str) -> None:
        """Create Blender mesh object from C++ mesh data."""
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        context.collection.objects.link(obj)
        context.view_layer.objects.active = obj
        obj.select_set(True)

        create_mesh_from_cpp(mesh, cpp_mesh)

    def _create_leaf_for_preset(self, seed: int):
        """Create a unique procedural leaf matching the tree preset."""
        import random

        from .mesh_utils import create_leaf_mesh_from_cpp
        from .resources.node_groups import _get_or_create_resource_collection

        # Pick leaf preset based on tree preset
        if self.preset == "RANDOM":
            leaf_preset_name = random.choice(list(LEAF_PRESETS.keys()))
        else:
            leaf_preset_name = self._TREE_TO_LEAF_PRESET.get(
                self.preset, random.choice(list(LEAF_PRESETS.keys()))
            )

        gen = m_tree.LeafShapeGenerator()
        apply_preset_to_generator(gen, leaf_preset_name)
        gen.seed = seed
        gen.asymmetry_seed = seed

        cpp_mesh = gen.generate()

        obj_name = f"Leaf_{seed}"
        mesh = bpy.data.meshes.new(obj_name)
        obj = bpy.data.objects.new(obj_name, mesh)
        create_leaf_mesh_from_cpp(mesh, cpp_mesh)

        collection = _get_or_create_resource_collection()
        collection.objects.link(obj)
        return obj

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "preset")
        layout.prop(self, "seed")
        layout.prop(self, "add_leaves")


class ExportFoliage(bpy.types.Operator):
    """Bake foliage wind and shading streams, and optionally a bone rig."""

    bl_idname = "mtree.export_foliage"
    bl_label = "Bake Foliage"
    bl_options = {"REGISTER", "UNDO"}

    object_name: bpy.props.StringProperty(name="Object")
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
            "belongs at 0 for trunks and branches"
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
    trunk_base_stiffness: bpy.props.FloatProperty(
        name="Trunk Base",
        description="Trunk stiffness at the root; the rooted ring is always pinned rigid",
        default=1.0,
        min=0.0,
        max=1.0,
    )
    trunk_tip_stiffness: bpy.props.FloatProperty(
        name="Trunk Tip",
        description="Trunk stiffness at the top; below the base value the trunk bends",
        default=1.0,
        min=0.0,
        max=1.0,
    )
    branch_stiffness: bpy.props.FloatProperty(
        name="Branch Scale",
        description="Scale on the radius-driven side-branch stiffness curve",
        default=1.0,
        min=0.0,
        max=1.0,
    )
    junction_falloff: bpy.props.FloatProperty(
        name="Junction Falloff",
        description=(
            "How much softer a branch may be than the limb carrying it; low "
            "values let branches whip independently of their parent"
        ),
        default=0.6,
        min=0.1,
        max=1.0,
    )
    build_bone_rig: bpy.props.BoolProperty(
        name="Build Bone Rig",
        description=(
            "Also generate a bone rig and its binding stream. Bones compose "
            "rotations down the hierarchy, so a branch follows the trunk's lean "
            "instead of shearing against it"
        ),
        default=False,
    )
    segments_per_chain: bpy.props.IntProperty(
        name="Trunk Bones",
        description="Bones the trunk is split into; one moves it rigidly",
        default=4,
        min=1,
        max=16,
    )
    branch_segments_per_chain: bpy.props.IntProperty(
        name="Branch Bones",
        description="Bones per rigged branch chain; one swings it rigidly about its base",
        default=2,
        min=1,
        max=16,
    )
    rig_depth: bpy.props.IntProperty(
        name="Rig Depth",
        description="Deepest branch level with bones of its own; deeper chains ride their limb's bone",
        default=1,
        min=0,
        max=16,
    )

    def execute(self, context):
        obj = bpy.data.objects.get(self.object_name)
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Invalid mesh object")
            return {"CANCELLED"}

        mesh_data = obj.data
        if not isinstance(mesh_data, bpy.types.Mesh):
            self.report({"ERROR"}, "Object data is not a mesh")
            return {"CANCELLED"}

        result = FoliageExporter(
            mesh=mesh_data,
            occlusion_base=self.occlusion_base,
            translucency_tip=self.translucency_tip,
            normal_blend=self.normal_blend,
            trunk_base_stiffness=self.trunk_base_stiffness,
            trunk_tip_stiffness=self.trunk_tip_stiffness,
            branch_stiffness=self.branch_stiffness,
            junction_falloff=self.junction_falloff,
        ).export()

        if not result.success:
            self.report({"ERROR"}, result.message)
            return {"CANCELLED"}

        message = result.message

        if self.build_bone_rig:
            strip_rig(obj)

            rig_result = FoliageRigExporter(
                mesh=mesh_data,
                mesh_object=obj,
                segments_per_chain=self.segments_per_chain,
                trunk_base_stiffness=self.trunk_base_stiffness,
                trunk_tip_stiffness=self.trunk_tip_stiffness,
                branch_stiffness=self.branch_stiffness,
                junction_falloff=self.junction_falloff,
                branch_segments_per_chain=self.branch_segments_per_chain,
                rig_depth=self.rig_depth,
            ).export()

            if not rig_result.success:
                self.report({"ERROR"}, f"Streams written, rig failed: {rig_result.message}")
                return {"CANCELLED"}

            message = f"{message} {rig_result.message}"

        self.report({"INFO"}, message)
        return {"FINISHED"}

    def invoke(self, context, event):
        if context.active_object and context.active_object.type == "MESH":
            self.object_name = context.active_object.name
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "object_name")
        layout.prop(self, "trunk_base_stiffness")
        layout.prop(self, "trunk_tip_stiffness")
        layout.prop(self, "branch_stiffness")
        layout.prop(self, "junction_falloff")
        layout.prop(self, "occlusion_base")
        layout.prop(self, "translucency_tip")
        layout.prop(self, "normal_blend")

        rig_box = layout.box()
        rig_box.prop(self, "build_bone_rig")
        if self.build_bone_rig:
            rig_box.prop(self, "segments_per_chain")
            rig_box.prop(self, "branch_segments_per_chain")
            rig_box.prop(self, "rig_depth")


class GenerateLeaf(bpy.types.Operator):
    """Generate a procedural leaf mesh from LeafShapeNode parameters."""

    bl_idname = "mtree.generate_leaf"
    bl_label = "Generate Leaf"
    bl_options = {"REGISTER", "UNDO"}

    node_tree_name: bpy.props.StringProperty()
    node_name: bpy.props.StringProperty()

    def execute(self, context):
        node_tree = bpy.data.node_groups.get(self.node_tree_name)
        if not node_tree:
            self.report({"ERROR"}, "Node tree not found")
            return {"CANCELLED"}
        node = node_tree.nodes.get(self.node_name)
        if node and hasattr(node, "generate_leaf"):
            node.generate_leaf()
            return {"FINISHED"}
        self.report({"ERROR"}, "Leaf shape node not found")
        return {"CANCELLED"}


class ApplyLeafPreset(bpy.types.Operator):
    """Apply a species preset to a LeafShapeNode."""

    bl_idname = "mtree.apply_leaf_preset"
    bl_label = "Apply Leaf Preset"
    bl_options = {"REGISTER", "UNDO"}

    preset: bpy.props.EnumProperty(
        name="Preset",
        items=get_leaf_preset_items(),
    )
    node_tree_name: bpy.props.StringProperty()
    node_name: bpy.props.StringProperty()

    def execute(self, context):
        node_tree = bpy.data.node_groups.get(self.node_tree_name)
        if not node_tree:
            return {"CANCELLED"}
        node = node_tree.nodes.get(self.node_name)
        if node and hasattr(node, "apply_preset"):
            node.apply_preset(self.preset)
        return {"FINISHED"}


class ApplyBranchNodePreset(bpy.types.Operator):
    """Apply a preset to a branch node."""

    bl_idname = "mtree.apply_branch_node_preset"
    bl_label = "Apply Preset"
    bl_options = {"REGISTER", "UNDO"}

    preset: bpy.props.EnumProperty(name="Preset", items=get_preset_items())
    node_tree_name: bpy.props.StringProperty()
    node_name: bpy.props.StringProperty()

    def execute(self, context):
        node_tree = bpy.data.node_groups.get(self.node_tree_name)
        if not node_tree:
            return {"CANCELLED"}
        node = node_tree.nodes.get(self.node_name)
        if node and hasattr(node, "apply_preset"):
            node.apply_preset(self.preset)
        return {"FINISHED"}


class ApplyTrunkNodePreset(bpy.types.Operator):
    """Apply a preset to a trunk node."""

    bl_idname = "mtree.apply_trunk_node_preset"
    bl_label = "Apply Preset"
    bl_options = {"REGISTER", "UNDO"}

    preset: bpy.props.EnumProperty(name="Preset", items=get_preset_items())
    node_tree_name: bpy.props.StringProperty()
    node_name: bpy.props.StringProperty()

    def execute(self, context):
        node_tree = bpy.data.node_groups.get(self.node_tree_name)
        if not node_tree:
            return {"CANCELLED"}
        node = node_tree.nodes.get(self.node_name)
        if node and hasattr(node, "apply_preset"):
            node.apply_preset(self.preset)
        return {"FINISHED"}


class ApplyGrowthNodePreset(bpy.types.Operator):
    """Apply a preset to a growth node."""

    bl_idname = "mtree.apply_growth_node_preset"
    bl_label = "Apply Growth Preset"
    bl_options = {"REGISTER", "UNDO"}

    preset: bpy.props.EnumProperty(name="Preset", items=get_growth_preset_items())
    node_tree_name: bpy.props.StringProperty()
    node_name: bpy.props.StringProperty()

    def execute(self, context):
        node_tree = bpy.data.node_groups.get(self.node_tree_name)
        if not node_tree:
            return {"CANCELLED"}
        node = node_tree.nodes.get(self.node_name)
        if node and hasattr(node, "apply_preset"):
            node.apply_preset(self.preset)
        return {"FINISHED"}


# Registration

_classes = [
    ExecuteNodeFunction,
    AddLeavesModifier,
    QuickGenerateTree,
    ExportFoliage,
    GenerateLeaf,
    ApplyLeafPreset,
    ApplyBranchNodePreset,
    ApplyTrunkNodePreset,
    ApplyGrowthNodePreset,
]


def register():
    for cls in _classes:
        register_class(cls)


def unregister():
    for cls in _classes:
        unregister_class(cls)
