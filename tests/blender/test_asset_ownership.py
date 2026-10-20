"""Object lifecycle tests. These require a real bpy and cannot run in tier 1.

Every generated tree owns its mesh, armature, wind clips and vertex groups
exclusively. The failures guarded here are silent: two assets sharing an
armature still both look built, and an accumulated rig still renders — it just
renders and exports wrong.
"""

from pathlib import Path

import bpy
import pytest

from python_classes.nodes.base_types.node import SEED_PROPERTY
from python_classes.nodes.pipeline import CHAIN_SOCKETS
from python_classes.pivot_painter.tree_asset import (
    armature_name_for,
    find_armature,
    link_exclusively,
    owning_collection,
    resolve_mesh_object,
    strip_rig,
)

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "blender"

pytestmark = pytest.mark.blender


def make_mesh_object(name, collection=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(0, 0, 0), (0, 0, 1), (1, 0, 1)], [], [(0, 1, 2)])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def attach_rig(obj, collection=None):
    """Give obj an armature, a bind and a vertex group, as a real rig build does."""
    name = armature_name_for(obj)
    armature = bpy.data.objects.new(name, bpy.data.armatures.new(name))
    (collection or owning_collection(obj)).objects.link(armature)
    obj.parent = armature
    obj.modifiers.new("Armature", "ARMATURE").object = armature
    obj.vertex_groups.new(name="stem0_seg0")
    return armature


def rig_state(obj):
    return {
        "armature_modifiers": len([m for m in obj.modifiers if m.type == "ARMATURE"]),
        "vertex_groups": len(obj.vertex_groups),
        "parent": obj.parent.name if obj.parent else None,
        "armature": find_armature(obj) is not None,
        "shape_keys": 0 if obj.data.shape_keys is None else len(obj.data.shape_keys.key_blocks),
    }


class TestAssetIsolation:
    """M6/M9: an operation reaches only the asset it was handed."""

    def test_asset_isolation(self):
        col = bpy.data.collections.new("iso_variants")
        bpy.context.scene.collection.children.link(col)

        a = make_mesh_object("iso_a")
        b = make_mesh_object("iso_b")
        link_exclusively(b, col)

        assert armature_name_for(a) != armature_name_for(b)
        assert owning_collection(b).name == col.name
        assert b.name not in bpy.context.scene.collection.objects

        arm_a = attach_rig(a)
        arm_b = attach_rig(b)

        assert find_armature(a).name == arm_a.name
        assert find_armature(b).name == arm_b.name
        assert arm_b.name in col.objects, "armature must follow its mesh's collection"

        before_b = rig_state(b)
        strip_rig(a)

        assert find_armature(a) is None
        assert rig_state(b) == before_b, "stripping a disturbed b"


class TestRebuild:
    """M7: repeating an operation must not accumulate state."""

    @pytest.mark.parametrize("repeats", [1, 2, 5])
    def test_rebuild_idempotent(self, repeats):
        obj = make_mesh_object(f"rebuild_{repeats}")

        for _ in range(repeats):
            strip_rig(obj)
            attach_rig(obj)

        state = rig_state(obj)
        assert state["armature_modifiers"] == 1
        assert state["vertex_groups"] == 1

        strip_rig(obj)
        assert rig_state(obj) == {
            "armature_modifiers": 0,
            "vertex_groups": 0,
            "parent": None,
            "armature": False,
            "shape_keys": 0,
        }

    def test_strip_survives_missing_rig(self):
        obj = make_mesh_object("never_rigged")
        strip_rig(obj)
        strip_rig(obj)
        assert rig_state(obj)["armature_modifiers"] == 0


class TestNodeGraph:
    """M8/M10: graph-level behaviour that needs registered Blender types."""

    def test_variant_inherits_mesher_settings(self, node_tree):
        mesher, variants = node_tree.mesher, node_tree.variants

        assert "radial_resolution" not in variants.bl_rna.properties, (
            "Variants must not carry its own mesh settings; they would drift"
        )
        found = variants.get_mesher()
        assert found is not None and found.name == mesher.name
        assert [s.bl_idname for s in variants.inputs] == ["mt_MeshSocket"]
        assert [s.bl_idname for s in variants.outputs] == ["mt_VariantsSocket"]

        a = make_mesh_object("inherit_hi")
        b = make_mesh_object("inherit_lo")

        mesher.radial_resolution = 32
        mesher.build_into(a, 0)
        high = len(a.data.vertices)

        mesher.radial_resolution = 8
        mesher.build_into(b, 0)
        low = len(b.data.vertices)

        assert high > low > 0, f"resolution ignored: {high} vs {low}"
        assert len(a.data.vertices) == high, "building b disturbed a"

    def test_seed_offset_does_not_mutate_graph(self, node_tree):
        mesher = node_tree.mesher
        trunk = node_tree.trunk

        seeds = [s for s in trunk.inputs if s.property_name == SEED_PROPERTY]
        assert seeds, "trunk carries no seed socket"
        before = [s.property_value for s in seeds]

        obj = make_mesh_object("seed_probe")
        for variant in range(4):
            mesher.build_into(obj, variant * 7919)

        assert [s.property_value for s in seeds] == before

    def test_socket_edit_schedules_rebuild(self, node_tree):
        """M11: with the 10Hz poll gone, the setter is the only notification."""
        from python_classes.nodes import debounce

        mesher = node_tree.mesher
        trunk = node_tree.trunk

        socket = next(s for s in trunk.inputs if s.bl_idname == "mt_IntSocket")

        debounce.cancel_pending()
        assert not debounce._pending_timers

        socket.property_value = socket.property_value + 1
        assert debounce._pending_timers, "socket edit scheduled no rebuild"

        target = next(iter(debounce._pending_timers))
        assert target[1] == mesher.name, f"rebuild aimed at {target[1]}"

        debounce.cancel_pending()
        assert not debounce._pending_timers, "cancel_pending leaked a timer"

    def test_growth_clamp_does_not_write_sockets(self, node_tree):
        """M12: correcting an invalid threshold pair must not touch the graph."""
        from python_classes.nodes import debounce

        tree = node_tree.tree
        trunk = node_tree.trunk

        growth = tree.nodes.new("mt_GrowthNode")
        tree.links.new(trunk.outputs[0], growth.inputs[0])

        def socket(name):
            return next(s for s in growth.inputs if s.property_name == name)

        socket("enable_flowering").property_value = True
        socket("cut_threshold").property_value = 0.5
        socket("flower_threshold").property_value = 0.2

        authored = socket("flower_threshold").property_value
        debounce.cancel_pending()

        built = growth.construct_function()

        assert socket("flower_threshold").property_value == authored, (
            "construct_function rewrote the authored threshold"
        )
        assert not debounce._pending_timers, "building scheduled a rebuild"
        assert built.flower_threshold > 0.5, built.flower_threshold

    def test_socket_clamp(self, node_tree):
        trunk = node_tree.trunk

        socket = next((s for s in trunk.inputs if s.bl_idname == "mt_IntSocket"), None)
        assert socket is not None, "trunk carries no int socket"

        low, high = socket.min_value, socket.max_value

        socket.property_value = high + 1000
        assert socket.property_value == high

        socket.property_value = low - 1000
        assert socket.property_value == low


class TestGenerate:
    """M15: a failed rig build leaves global mode dirty and poisons later builds."""

    def test_generate_with_and_without_rig(self, node_tree):
        mesher = node_tree.mesher
        mesher.tree_object = ""

        mesher.build_bone_rig = True
        mesher.build_tree()
        assert not mesher.status_is_error, mesher.status_message
        assert bpy.context.mode == "OBJECT", f"left in {bpy.context.mode}"

        tree_obj = resolve_mesh_object(mesher.tree_object)
        assert tree_obj is not None and len(tree_obj.data.vertices) > 0
        assert find_armature(tree_obj) is not None, "rig on, but no armature built"

        mesher.build_bone_rig = False
        mesher.build_tree()
        assert not mesher.status_is_error, mesher.status_message
        assert find_armature(tree_obj) is None, "dead rig survived a rig-off rebuild"

        mesher.build_tree()
        assert not mesher.status_is_error, mesher.status_message
        assert bpy.context.mode == "OBJECT"

    def test_rigged_variants_keep_their_grid_slot(self, node_tree):
        """TEST_POINT: the clip bake re-read a stale matrix_world and pulled every
        variant back to the origin, with its rig left behind at the old slot."""
        node_tree.add_branch()
        node_tree.mesher.build_bone_rig = True
        variants = node_tree.variants
        variants.variant_count = 3
        variants.spacing = 10.0
        variants.collection_name = "layout_variants"
        variants.bake_morphs = True

        lines = ["# world x of each variant and its rig, per generate", ""]
        for run in range(2):
            variants.generate_variants()
            assert not variants.status_is_error, variants.status_message
            bpy.context.view_layer.update()

            for index in range(3):
                obj = bpy.data.objects[f"layout_variants_{index:03d}"]
                armature = find_armature(obj)
                mesh_x = obj.matrix_world.translation.x
                rig_x = armature.matrix_world.translation.x
                lines.append(f"run {run} variant {index}: mesh {mesh_x:.3f} rig {rig_x:.3f}")
                assert mesh_x == pytest.approx(index * 10.0)
                assert rig_x == pytest.approx(index * 10.0)

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "variant_layout.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestEmptyBuild:
    """M16: an empty build reported success, so the rig blamed branch chains."""

    @pytest.mark.parametrize(
        ("length", "resolution", "expect_empty"),
        [
            (0.0, 3.0, True),
            (14.0, 0.0, True),
            # TEST_POINT: neither value is zero here — the product is what matters,
            # so a check for zero would pass these straight through.
            (14.0, 0.05, True),
            (0.3, 3.0, True),
            (0.5, 3.0, False),
            (14.0, 0.1, False),
        ],
    )
    def test_empty_mesh_is_reported_at_the_mesher(
        self, node_tree, length, resolution, expect_empty
    ):
        mesher, trunk = node_tree.mesher, node_tree.trunk

        for name, value in (("length", length), ("resolution", resolution)):
            next(s for s in trunk.inputs if s.property_name == name).property_value = value

        mesher.build_bone_rig = False
        mesher.tree_object = ""
        mesher.build_tree()

        if not expect_empty:
            assert not mesher.status_is_error, mesher.status_message
            return

        assert mesher.status_is_error
        assert "no geometry" in mesher.status_message, mesher.status_message
        assert "segments" in mesher.status_message, (
            "the message must name the product, not a single zeroed value"
        )
        assert "branch chains" not in mesher.status_message


class TestNodeDraw:
    """M18: an exception mid-draw aborts the node, hiding every panel below."""

    class _Layout:
        """Records emitted elements and reads every property drawn."""

        def __init__(self, sink):
            self.sink = sink
            self.enabled = True
            self.alert = False

        def box(self):
            self.sink.append("box")
            return self

        def row(self):
            return self

        def column(self):
            return self

        def label(self, **kwargs):
            self.sink.append("label:" + kwargs.get("text", ""))

        def prop(self, target, name, **kwargs):
            self.sink.append("prop:" + name)
            getattr(target, name)

        def prop_search(self, *args, **kwargs):
            self.sink.append("prop_search")

        def operator(self, idname, **kwargs):
            self.sink.append("op:" + kwargs.get("text", ""))

            class Props:
                def __setattr__(self, key, value):
                    pass

            return Props()

    def _draw(self, node):
        emitted = []
        node.draw_buttons(bpy.context, self._Layout(emitted))
        return emitted

    @pytest.mark.parametrize("built", [False, True])
    def test_draw_emits_wind_controls(self, node_tree, built):
        mesher = node_tree.mesher
        mesher.build_bone_rig = True

        if built:
            mesher.tree_object = ""
            mesher.build_tree()
            assert not mesher.status_is_error, mesher.status_message

        emitted = self._draw(mesher)

        assert not any(e.startswith("label:UI error") for e in emitted), emitted
        assert "prop:preview_wind_angle" in emitted, emitted
        assert "prop:preview_wind_strength" in emitted, emitted
        assert "op:Bake Wind Clips" in emitted, emitted

    def test_draw_never_writes_id_data(self, node_tree):
        """Blender forbids writing ID data while drawing; doing so aborts the
        draw and silently truncates the node.
        """
        mesher = node_tree.mesher
        mesher.build_bone_rig = True

        # TEST_POINT: an object that exists but is unlinked is what makes the
        # old code create-and-link during draw; a fully absent or fully linked
        # object takes branches that happen not to write.
        orphan = make_mesh_object("draw_orphan")
        for collection in list(orphan.users_collection):
            collection.objects.unlink(orphan)
        mesher.tree_object = orphan.name

        before_objects = set(bpy.data.objects.keys())
        before_meshes = set(bpy.data.meshes.keys())

        self._draw(mesher)

        assert set(bpy.data.objects.keys()) == before_objects, "draw created an object"
        assert set(bpy.data.meshes.keys()) == before_meshes, "draw created a mesh"
        assert not orphan.users_collection, "draw linked the object into a collection"
        assert bpy.context.view_layer.objects.active is not orphan, "draw changed the active object"


class TestWindClipOwnership:
    @pytest.mark.parametrize("directions", [4, 8])
    def test_clips_do_not_accumulate(self, node_tree, directions):
        from python_classes.pivot_painter.formats.foliage import FoliageExporter
        from python_classes.pivot_painter.wind_animation import (
            action_prefix_for,
            bake_wind_clips,
            build_rig_for,
        )

        mesher = node_tree.mesher
        node_tree.add_branch()

        obj = make_mesh_object(f"clips_{directions}")
        mesher.build_into(obj, 0)
        FoliageExporter(mesh=obj.data).export()

        rig = build_rig_for(obj.data, mesher.segments_per_chain, mesher._stiffness_settings())

        for _ in range(2):
            count, message = bake_wind_clips(obj, rig, directions=directions)
            assert count == directions, message

        armature = find_armature(obj)
        assert armature is not None, "the bake must leave the rig bound"

        prefix = action_prefix_for(armature)
        clips = [a for a in bpy.data.actions if a.name.startswith(prefix)]
        assert len(clips) == directions, "rebaking accumulated clips"

    def test_two_trees_keep_separate_clips(self, node_tree):
        from python_classes.pivot_painter.formats.foliage import FoliageExporter
        from python_classes.pivot_painter.wind_animation import (
            action_prefix_for,
            bake_wind_clips,
            build_rig_for,
        )

        mesher = node_tree.mesher
        node_tree.add_branch()

        prefixes = []
        for name in ("clip_owner_a", "clip_owner_b"):
            obj = make_mesh_object(name)
            mesher.build_into(obj, 0)
            FoliageExporter(mesh=obj.data).export()

            rig = build_rig_for(obj.data, mesher.segments_per_chain, mesher._stiffness_settings())
            bake_wind_clips(obj, rig, directions=4)

            armature = find_armature(obj)
            assert armature is not None
            prefixes.append(action_prefix_for(armature))

        # TEST_POINT: a shared action prefix is the same defect a shared
        # armature name once was — the second bake silently overwrites the
        # first tree's clips and both trees then animate identically.
        assert prefixes[0] != prefixes[1]
        for prefix in prefixes:
            assert len([a for a in bpy.data.actions if a.name.startswith(prefix)]) == 4


class TestDiagnostics:
    """M17: a build failure must name the stage that failed."""

    def test_stages_are_logged(self, node_tree):
        """The logger does not propagate, so caplog cannot see it: attach a
        handler directly, the same way a user reading the console does.
        """
        import logging

        from python_classes import diagnostics

        mesher, trunk = node_tree.mesher, node_tree.trunk

        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(self.format(record))

        handler = Capture()
        logger = logging.getLogger(diagnostics.LOGGER_NAME)

        diagnostics.set_enabled(True)
        logger.addHandler(handler)
        try:
            obj = make_mesh_object("logged_tree")
            mesher.build_into(obj, 0)

            text = "\n".join(records)
            assert "build_into" in text
            assert "mesh_tree" in text
            assert "verts=" in text

            records.clear()

            socket = next(s for s in trunk.inputs if s.property_name == "length")
            socket.property_value = 0
            with pytest.raises(ValueError):
                mesher.build_into(obj, 0)

            failure = "\n".join(records)
            assert "FAILED" in failure
            assert "verts=0" in failure, failure
        finally:
            logger.removeHandler(handler)
            diagnostics.set_enabled(False)

    def test_disabled_by_default(self):
        from python_classes import diagnostics

        diagnostics.set_enabled(False)
        assert not diagnostics.is_enabled()


class TestRigCache:
    """M14: a stale rig poses bones against topology that no longer exists."""

    def test_cache_invalidates(self, node_tree):
        from python_classes.pivot_painter import wind_animation
        from python_classes.pivot_painter.formats.foliage import FoliageExporter

        mesher = node_tree.mesher
        obj = make_mesh_object("cached_tree")
        mesher.build_into(obj, 0)
        FoliageExporter(mesh=obj.data).export()

        stiffness = mesher._stiffness_settings()
        wind_animation.clear_rig_cache()

        first = wind_animation.build_rig_for(obj.data, 4, stiffness, use_cache=True)
        assert wind_animation.build_rig_for(obj.data, 4, stiffness, use_cache=True) is first

        softer = dict(stiffness, branch_stiffness=0.25)
        assert wind_animation.build_rig_for(obj.data, 4, softer, use_cache=True) is not first

        assert wind_animation.build_rig_for(obj.data, 4, stiffness, use_cache=False) is not first, (
            "bake paths must not read the cache"
        )

        strip_rig(obj)
        assert not wind_animation._rig_cache, "strip_rig left a stale rig cached"


GRAPH_ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "node_graph"

BLOCK_FILL_VALUES = {
    "density": 12.5,
    "scale": 1.25,
    "roll_variation": 20.0,
    "seed": 3,
    "max_polygons": 150000,
}


def reshape_to_legacy(node, inputs=(), outputs=()):
    for sockets, legacy in ((node.inputs, inputs), (node.outputs, outputs)):
        for socket in [s for s in sockets if s.bl_idname in CHAIN_SOCKETS]:
            sockets.remove(socket)
        for name in legacy:
            socket = sockets.new("mt_TreeSocket", name)
            socket.is_property = False
            sockets.move(len(sockets) - 1, 0)


def pin_graph_seeds(tree, seed=7):
    for node in tree.nodes:
        for socket in node.inputs:
            if getattr(socket, "property_name", "") == SEED_PROPERTY:
                socket.property_value = seed


def build_legacy_graph():
    from conftest import Graph

    tree = bpy.data.node_groups.new("mtree_legacy", "mt_MtreeNodeTree")
    nodes = {
        idname: tree.nodes.new(idname)
        for idname in (
            "mt_MesherNode",
            "mt_TrunkNode",
            "mt_BlockFillNode",
            "mt_LeafCardsNode",
            "mt_VariantsNode",
            "mt_MeshExportNode",
            "mt_FoliageWindNode",
        )
    }
    mesher, trunk = nodes["mt_MesherNode"], nodes["mt_TrunkNode"]

    reshape_to_legacy(mesher, outputs=["Tree"])
    reshape_to_legacy(trunk, inputs=["Tree"], outputs=["Tree"])
    reshape_to_legacy(nodes["mt_LeafCardsNode"], inputs=["Tree"])
    reshape_to_legacy(nodes["mt_VariantsNode"], inputs=["Tree"])
    reshape_to_legacy(nodes["mt_MeshExportNode"], inputs=["Tree"])
    reshape_to_legacy(nodes["mt_FoliageWindNode"], inputs=["Tree"], outputs=["Tree"])

    for idname in (
        "mt_TrunkNode",
        "mt_BlockFillNode",
        "mt_LeafCardsNode",
        "mt_VariantsNode",
        "mt_MeshExportNode",
    ):
        tree.links.new(mesher.outputs[0], nodes[idname].inputs[0])

    for name, value in BLOCK_FILL_VALUES.items():
        setattr(nodes["mt_BlockFillNode"], name, value)
    nodes["mt_FoliageWindNode"].trunk_tip_stiffness = 0.3

    branch = Graph(tree, mesher, trunk, nodes["mt_VariantsNode"]).add_branch()
    pin_graph_seeds(tree)
    return tree, {**nodes, "mt_BranchNode": branch}


class TestPipeline:
    def test_legacy_graph_migrates_to_the_same_tree(self, node_tree):
        from python_classes.nodes import migration, pipeline
        from python_classes.nodes.base_types.node_tree import GRAPH_VERSION
        from python_classes.nodes.export_nodes.foliage_wind_node import (
            foliage_wind_settings_for,
        )

        node_tree.add_branch()
        pin_graph_seeds(node_tree.tree)
        reference = node_tree.mesher
        reference.mesh_mode = "BLOCKS"
        for name, value in BLOCK_FILL_VALUES.items():
            setattr(reference, f"block_{name}", value)
        cards = node_tree.tree.nodes.new("mt_LeafCardsNode")
        node_tree.tree.links.new(reference.outputs[0], cards.inputs[0])
        expected = make_mesh_object("migration_expected")
        reference.build_into(expected, 0)

        tree, nodes = build_legacy_graph()
        names = {idname: node.name for idname, node in nodes.items()}
        assert pipeline.needs_migration(tree)

        assert migration.migrate_tree(tree)
        links_after = sorted((link.from_node.name, link.to_node.name) for link in tree.links)
        assert not migration.migrate_tree(tree), "a second migration changed the graph"
        assert sorted((link.from_node.name, link.to_node.name) for link in tree.links) == (
            links_after
        )

        mesher = tree.nodes[names["mt_MesherNode"]]
        chain = [node.name for node in pipeline.single_tree_chain(mesher)]
        variants = tree.nodes[names["mt_VariantsNode"]]
        export = tree.nodes[names["mt_MeshExportNode"]]

        assert tree.graph_version == GRAPH_VERSION
        assert names["mt_BlockFillNode"] not in tree.nodes
        assert pipeline.resolve_trunk(mesher)[0].name == names["mt_TrunkNode"]
        assert pipeline.upstream_node(mesher).name == names["mt_BranchNode"]
        assert chain == [
            names["mt_MesherNode"],
            names["mt_LeafCardsNode"],
            names["mt_FoliageWindNode"],
        ]
        assert pipeline.upstream_node(variants).name == names["mt_FoliageWindNode"]
        assert pipeline.upstream_node(export).name == names["mt_VariantsNode"]
        assert not [
            s for s in tree.nodes[names["mt_TrunkNode"]].inputs if s.bl_idname == "mt_TreeSocket"
        ]
        assert mesher.mesh_mode == "BLOCKS"
        for name, value in BLOCK_FILL_VALUES.items():
            assert getattr(mesher, f"block_{name}") == pytest.approx(value), name
        assert foliage_wind_settings_for(mesher)["trunk_tip_stiffness"] == pytest.approx(0.3)
        assert not pipeline.rejected_links(tree)

        migrated = make_mesh_object("migration_migrated")
        mesher.build_into(migrated, 0)

        assert len(migrated.data.vertices) == len(expected.data.vertices) > 0
        assert len(migrated.data.polygons) == len(expected.data.polygons)

        lines = [
            "# legacy graph: mesher root -> trunk -> branch, Block Fill + Leaf Cards + "
            "Variants + Mesh Export on the mesher, Foliage Wind floating",
            "",
            "links after migration:",
            *(f"  {source} -> {target}" for source, target in links_after),
            "",
            f"single-tree chain: {' -> '.join(chain)}",
            f"mesh_mode: {mesher.mesh_mode}",
            *(f"block_{name}: {getattr(mesher, f'block_{name}'):g}" for name in BLOCK_FILL_VALUES),
            f"vertices expected {len(expected.data.vertices)} "
            f"migrated {len(migrated.data.vertices)}",
            f"polygons expected {len(expected.data.polygons)} "
            f"migrated {len(migrated.data.polygons)}",
        ]
        GRAPH_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (GRAPH_ARTIFACT_DIR / "migration.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        bpy.data.node_groups.remove(tree)

    def test_incompatible_links_are_rejected(self, node_tree):
        from python_classes.nodes import pipeline

        tree, mesher, variants = node_tree.tree, node_tree.mesher, node_tree.variants
        node_tree.add_branch()

        cards = tree.nodes.new("mt_LeafCardsNode")
        stray_branch = tree.nodes.new("mt_BranchNode")
        cards_after_variants = tree.nodes.new("mt_LeafCardsNode")
        second_cards = tree.nodes.new("mt_LeafCardsNode")

        pipeline.cancel_rejections()
        cases = [
            ("mesher -> leaf cards", mesher, cards, True),
            ("mesh -> branch tree input", mesher, stray_branch, False),
            ("variants -> leaf cards", variants, cards_after_variants, False),
            ("second leaf cards on one mesh output", mesher, second_cards, False),
        ]
        for _, source, target, _ in cases:
            tree.links.new(source.outputs[0], target.inputs[0])

        assert bpy.app.timers.is_registered(pipeline.flush_rejections), (
            "NodeTree.update scheduled no removal"
        )
        assert pipeline.upstream_node(cards_after_variants) is None
        assert pipeline.chain_stage(mesher, "mt_LeafCardsNode").name == cards.name

        pipeline.flush_rejections()
        pipeline.cancel_rejections()

        survivors = {(link.from_node.name, link.to_node.name) for link in tree.links}
        lines = ["# link -> kept after the deferred rejection", ""]
        for label, source, target, allowed in cases:
            kept = (source.name, target.name) in survivors
            lines.append(f"{label}: {'kept' if kept else 'rejected'}")
            assert kept == allowed, label

        GRAPH_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (GRAPH_ARTIFACT_DIR / "link_rejection.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


class TestExportPayload:
    def test_rig_detection_ignores_selection(self, node_tree):
        from python_classes.pivot_painter.formats.foliage import FoliageExporter
        from python_classes.pivot_painter.wind_animation import (
            bake_wind_clips,
            build_rig_for,
        )

        tree = node_tree.tree
        export = tree.nodes.new("mt_MeshExportNode")

        mesher = node_tree.mesher
        node_tree.add_branch()

        plain = make_mesh_object("payload_plain")

        rigged = make_mesh_object("payload_rigged")
        mesher.build_into(rigged, 0)
        FoliageExporter(mesh=rigged.data).export()
        rig = build_rig_for(rigged.data, mesher.segments_per_chain, mesher._stiffness_settings())
        bake_wind_clips(rigged, rig, directions=4)

        for obj in bpy.context.scene.objects:
            obj.select_set(False)
        plain.select_set(True)

        assert export._rigged_armatures([rigged]), "rigged tree reported no armature"
        assert export._rigged_armatures([plain]) == []

        # TEST_POINT: reading global selection instead of the passed objects
        # would drop the rig from every file but one in a per-file batch.
        assert export._rigged_armatures([plain, rigged])

        tree.nodes.remove(export)


def test_snapshot(node_tree):
    mesher, variants = node_tree.mesher, node_tree.variants

    col = bpy.data.collections.new("snapshot_variants")
    bpy.context.scene.collection.children.link(col)

    lines = ["# asset ownership after two builds and one rebuild", ""]
    lines.append(f"{'object':>20} {'collection':>20} {'armature':>24} {'verts':>7}")

    assets = []
    for index in range(2):
        name = f"{col.name}_{index:03d}"
        obj = resolve_mesh_object(name) or bpy.data.objects.new(name, bpy.data.meshes.new(name))
        link_exclusively(obj, col)
        mesher.build_into(obj, index * 7919)
        assets.append(obj)

    mesher.build_into(assets[0], 0)

    for obj in assets:
        armature = find_armature(obj)
        lines.append(
            f"{obj.name:>20} {owning_collection(obj).name:>20} "
            f"{(armature.name if armature else '-'):>24} {len(obj.data.vertices):>7}"
        )

    names = {obj.name for obj in assets}
    lines += [
        "",
        "# invariants",
        f"distinct armature names: {len({armature_name_for(o) for o in assets})}/{len(assets)}",
        f"each asset in exactly one collection: {all(len(o.users_collection) == 1 for o in assets)}",
        f"no name collisions: {len(names) == len(assets)}",
        f"variants node carries own mesh settings: "
        f"{'radial_resolution' in variants.bl_rna.properties}",
    ]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "asset_ownership.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert len({armature_name_for(o) for o in assets}) == len(assets)
