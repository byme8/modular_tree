import hashlib
from pathlib import Path

import bpy
import numpy as np
import pytest
from test_asset_ownership import make_mesh_object, rig_state

from python_classes.mesh_utils import FLOAT_ATTRIBUTES, VECTOR3_ATTRIBUTES
from python_classes.nodes.base_types.node import SEED_PROPERTY
from python_classes.resources.node_groups import LEAVES_MODIFIER_NAME

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "block_fill"
BASELINE_DIR = Path(__file__).parent.parent / "artifacts" / "blender"

pytestmark = pytest.mark.blender


def pin_seeds(graph, seed=7):
    for node in graph.tree.nodes:
        for socket in node.inputs:
            if getattr(socket, "property_name", "") == SEED_PROPERTY:
                socket.property_value = seed


def enable_surfaces(graph, blocks=True, leaf_cards=True):
    if blocks:
        graph.mesher.mesh_mode = "BLOCKS"
    if leaf_cards:
        node = graph.tree.nodes.new("mt_LeafCardsNode")
        graph.tree.links.new(graph.mesher.outputs[0], node.inputs[0])
        return node
    return None


def mesh_arrays(mesh):
    arrays = {}

    positions = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", positions)
    arrays["vertices"] = positions.reshape(-1, 3)

    loops = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loops)
    arrays["loops"] = loops

    materials = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("material_index", materials)
    arrays["material_index"] = materials

    if mesh.uv_layers:
        uvs = np.empty(len(mesh.loops) * 2, dtype=np.float32)
        mesh.uv_layers[0].data.foreach_get("uv", uvs)
        arrays["uv"] = uvs.reshape(-1, 2)

    for name in FLOAT_ATTRIBUTES:
        if name in mesh.attributes:
            values = np.empty(len(mesh.vertices), dtype=np.float32)
            mesh.attributes[name].data.foreach_get("value", values)
            arrays[name] = values

    for name in VECTOR3_ATTRIBUTES:
        if name in mesh.attributes:
            values = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.attributes[name].data.foreach_get("vector", values)
            arrays[name] = values.reshape(-1, 3)

    return arrays


def leaf_vertex_mask(mesh):
    mask = np.zeros(len(mesh.vertices), dtype=bool)
    for polygon in mesh.polygons:
        if polygon.material_index == 1:
            mask[list(polygon.vertices)] = True
    return mask


def digest(array):
    rounded = np.round(np.asarray(array, dtype=np.float64), 5)
    rounded[rounded == 0.0] = 0.0
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def describe_arrays(arrays):
    lines = []
    for name in sorted(arrays):
        array = arrays[name]
        flat = np.asarray(array, dtype=np.float64).ravel()
        lines.append(
            f"{name:>16} n={len(array):<7d} min={flat.min():.5f} max={flat.max():.5f} "
            f"sha={digest(array)}"
        )
    return lines


def write_artifact(directory, name, lines):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_manifold_output_is_unchanged(node_tree):
    mesher = node_tree.mesher
    node_tree.add_branch()
    pin_seeds(node_tree)

    obj = make_mesh_object("manifold_baseline")
    mesher.build_into(obj, 0)

    arrays = mesh_arrays(obj.data)
    lines = [
        "# manifold build of the fixture graph (trunk + sparse branch), seed offset 0",
        f"polygons {len(obj.data.polygons)} smooth {all(p.use_smooth for p in obj.data.polygons)}",
        f"material_slots {[m.name for m in obj.data.materials]}",
        "",
        *describe_arrays(arrays),
    ]
    write_artifact(BASELINE_DIR, "manifold_baseline.txt", lines)

    assert len(arrays["vertices"]) > 0
    assert set(arrays["material_index"].tolist()) == {0}
    assert [m.name for m in obj.data.materials] == ["Bark"]
    assert "leaf_attachment" not in arrays


class TestBlockBuild:
    def test_cards_take_leaf_and_blocks_take_bark(self, node_tree):
        node_tree.add_branch()
        pin_seeds(node_tree)
        enable_surfaces(node_tree)

        obj = make_mesh_object("materials_probe")
        node_tree.mesher.build_into(obj, 0)

        slots = [m.name for m in obj.data.materials]
        indices = np.empty(len(obj.data.polygons), dtype=np.int32)
        obj.data.polygons.foreach_get("material_index", indices)
        used = sorted({slots[i] for i in indices.tolist()})

        authored = bpy.data.materials.new("UserBark")
        obj.data.materials[0] = authored
        node_tree.mesher.build_into(obj, 0)
        after_rebuild = [m.name for m in obj.data.materials]

        write_artifact(
            ARTIFACT_DIR,
            "materials.txt",
            [
                f"slots {slots}",
                f"used by polygons {used}",
                f"after user assigns slot 0 and rebuilds {after_rebuild}",
            ],
        )

        assert slots == ["Bark", "Leaf"]
        assert used == ["Bark", "Leaf"]
        assert after_rebuild == ["UserBark", "Leaf"]

    def test_same_inputs_yield_identical_arrays_and_seeds_are_isolated(self, node_tree):
        mesher = node_tree.mesher
        node_tree.add_branch()
        pin_seeds(node_tree)
        leaf_cards = enable_surfaces(node_tree)

        def build(name, block_seed=0, leaf_seed=0):
            mesher.block_seed = block_seed
            leaf_cards.leaf_seed = leaf_seed
            obj = make_mesh_object(name)
            mesher.build_into(obj, 0)
            return obj, mesh_arrays(obj.data), leaf_vertex_mask(obj.data)

        first, base, leaves = build("blocks_a")
        _, again, _ = build("blocks_b")
        _, rerolled, _ = build("blocks_reroll", block_seed=1)
        _, releafed, _ = build("blocks_releaf", leaf_seed=1)

        assert set(base) == set(again) == set(rerolled) == set(releafed)
        for name in base:
            assert np.array_equal(base[name], again[name]), (
                f"{name} differs between identical builds"
            )

        vertex_count = len(leaves)
        for name in base:
            if len(base[name]) == vertex_count:
                assert np.array_equal(base[name][leaves], rerolled[name][leaves]), (
                    f"block seed changed card {name}"
                )
                assert np.array_equal(base[name][~leaves], releafed[name][~leaves]), (
                    f"leaf seed changed block {name}"
                )
            else:
                assert np.array_equal(base[name], rerolled[name]), f"block seed changed {name}"
                assert np.array_equal(base[name], releafed[name]), f"leaf seed changed {name}"

        block_moved = np.any(base["vertices"][~leaves] != rerolled["vertices"][~leaves])
        assert block_moved, "block seed moved no block vertex"

        card_moved = np.any(base["vertices"][leaves] != releafed["vertices"][leaves])
        assert card_moved, "leaf seed moved no card vertex"

        assert not any(p.use_smooth for p in first.data.polygons)
        assert len(first.data.materials) == 2
        assert set(base["material_index"].tolist()) == {0, 1}

        lines = [
            "# block fill + leaf cards build of the fixture graph (trunk + sparse branch)",
            f"polygons {len(first.data.polygons)} flat {not any(p.use_smooth for p in first.data.polygons)}",
            f"material_slots {len(first.data.materials)} leaf_vertices {int(leaves.sum())}/{len(leaves)}",
            f"block_fill status: {mesher.blocks_status_message}",
            f"leaf_cards status: {leaf_cards.status_message}",
            "",
            *describe_arrays(base),
            "",
            "# seed isolation",
            f"block seed moves only block vertices: {block_moved}",
            f"leaf seed moves only card vertices: {card_moved}",
        ]
        write_artifact(ARTIFACT_DIR, "build_snapshot.txt", lines)

    def test_over_cap_leaves_the_object_untouched(self, node_tree):
        from python_classes.pivot_painter.tree_asset import resolve_mesh_object

        mesher = node_tree.mesher
        node_tree.add_branch()
        pin_seeds(node_tree)
        enable_surfaces(node_tree, leaf_cards=False)

        mesher.tree_object = ""
        mesher.build_bone_rig = True
        mesher.build_tree()
        assert not mesher.status_is_error, mesher.status_message

        obj = resolve_mesh_object(mesher.tree_object)
        bpy.ops.mtree.add_leaves(object_id=obj.name)
        before = mesh_arrays(obj.data)
        rig_before = rig_state(obj)
        modifiers_before = [m.name for m in obj.modifiers]

        mesher.block_density = 1000.0
        mesher.block_max_polygons = 1000
        mesher.build_tree()

        assert mesher.status_is_error
        assert "1,000" in mesher.status_message and "polygons" in mesher.status_message
        assert mesher.blocks_status_is_error

        after = mesh_arrays(obj.data)
        for name in before:
            assert np.array_equal(before[name], after[name]), f"{name} changed by a failed build"
        assert rig_state(obj) == rig_before
        assert [m.name for m in obj.modifiers] == modifiers_before
        assert obj.modifiers.get(LEAVES_MODIFIER_NAME) is not None

        lines = [
            "# a build over the polygon cap must not touch the object",
            f"status: {mesher.status_message}",
            f"vertices unchanged: {np.array_equal(before['vertices'], after['vertices'])}",
            f"rig unchanged: {rig_state(obj) == rig_before}",
            f"modifiers unchanged: {[m.name for m in obj.modifiers] == modifiers_before}",
        ]
        write_artifact(ARTIFACT_DIR, "over_cap.txt", lines)

        mesher.build_bone_rig = False
