"""If the seed offset fails to reach every function node, every variant still
generates and still looks like a tree, but they share branch structure and only
the trunk differs — a bug that reads as "the variants are a bit samey".
"""

from pathlib import Path

import pytest
from conftest import load_node_base

node_base = load_node_base()
SEED_PROPERTY = node_base.SEED_PROPERTY
MtreeFunctionNode = node_base.MtreeFunctionNode

SEED_STRIDE = 7919

NODE_NAMES = ("trunk", "branch_a", "sub_branch", "branch_b")

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "tree_variants"


class FakeSocket:
    def __init__(self, property_name, property_value=0, bl_idname="mt_IntSocket"):
        self.property_name = property_name
        self.property_value = property_value
        self.bl_idname = bl_idname
        self.is_property = True


class FakeFunction:
    def __init__(self):
        object.__setattr__(self, "children", [])
        object.__setattr__(self, "assigned", {})

    def __setattr__(self, name, value):
        self.assigned[name] = value

    def add_child(self, child):
        self.children.append(child)


class FakeNode(MtreeFunctionNode):
    """Must subclass the real node: construct_function isinstance-checks its
    children, so a duck-typed stand-in silently builds a childless tree.
    """

    exposed_parameters = []

    def __init__(self, name, sockets=None, children=None):
        self.name = name
        self.inputs = sockets or []
        self._children = children or []
        self.tree_function = FakeFunction

    def get_child_nodes(self):
        return self._children


def make_graph():
    sub = FakeNode(
        "sub_branch",
        [FakeSocket("seed", 5), FakeSocket("length", 1.0)],
    )
    branch_a = FakeNode(
        "branch_a",
        [FakeSocket("seed", 3), FakeSocket("length", 4.0)],
        [sub],
    )
    branch_b = FakeNode(
        "branch_b",
        [FakeSocket("seed", 4), FakeSocket("resolution", 8)],
    )
    trunk = FakeNode(
        "trunk",
        [FakeSocket("seed", 1), FakeSocket("length", 14.0), FakeSocket("start_radius", 0.3)],
        [branch_a, branch_b],
    )
    return trunk


def build_depth_first(trunk, seed_offset=0):
    order = []

    def descend(func):
        order.append(func)
        for child in func.children:
            descend(child)

    descend(trunk.construct_function(seed_offset))
    return {NODE_NAMES[i]: f.assigned for i, f in enumerate(order)}


def seed_state(trunk, seed_offset):
    built = build_depth_first(trunk, seed_offset)
    return tuple(built[n]["seed"] for n in NODE_NAMES)


class TestSeedPropagation:
    """M1/M2/M5: the offset must reach every node and nothing but the seeds."""

    @pytest.mark.parametrize("offset", [0, 1, SEED_STRIDE, 7 * SEED_STRIDE])
    def test_offset_propagates(self, offset):
        built = build_depth_first(make_graph(), offset)

        authored = {"trunk": 1, "branch_a": 3, "sub_branch": 5, "branch_b": 4}
        for node_name, seed in authored.items():
            assert built[node_name]["seed"] == seed + offset, (
                f"{node_name} seed did not receive the offset"
            )

        # TEST_POINT: an offset leaking onto resolution or length corrupts
        # geometry while still producing varied trees, so it reads as working.
        assert built["trunk"]["length"] == 14.0
        assert built["trunk"]["start_radius"] == 0.3
        assert built["branch_a"]["length"] == 4.0
        assert built["sub_branch"]["length"] == 1.0
        assert built["branch_b"]["resolution"] == 8

    def test_graph_not_mutated(self):
        """M3: building must not write back to any socket."""
        trunk = make_graph()

        def snapshot(node, seen=None):
            if seen is None:
                seen = set()
            if node.name in seen:
                return {}
            seen.add(node.name)
            values = {(node.name, s.property_name): s.property_value for s in node.inputs}
            for child in node.get_child_nodes():
                values.update(snapshot(child, seen))
            return values

        before = snapshot(trunk)

        for variant in range(10):
            trunk.construct_function(variant * SEED_STRIDE)

        assert snapshot(trunk) == before


class TestVariantOffsets:
    """M4: distinct variants must not collide on the same seed set."""

    def test_variant_sweep(self):
        trunk = make_graph()

        seen = {}
        for variant in range(20):
            state = seed_state(trunk, variant * SEED_STRIDE)
            assert state not in seen, f"variant {variant} collides with {seen[state]}"
            seen[state] = variant

        assert len(seen) == 20

    def test_stride_avoids_neighbour_collisions(self):
        """With a stride of 1, variant N on a node seeded 2 produces the same
        seed as variant N+1 on a node seeded 1 — they would share substructure.
        """
        trunk = make_graph()

        collisions = 0
        for variant in range(20):
            current = set(seed_state(trunk, variant * SEED_STRIDE))
            following = set(seed_state(trunk, (variant + 1) * SEED_STRIDE))
            collisions += len(current & following)

        assert collisions == 0

    def test_base_seed_shifts_the_whole_set(self):
        trunk = make_graph()

        def family(base):
            return {seed_state(trunk, (base + v) * SEED_STRIDE) for v in range(20)}

        assert not (family(0) & family(100))


def test_snapshot():
    trunk = make_graph()
    authored = seed_state(trunk, 0)

    lines = [
        "# seed assignment per variant",
        f"# stride={SEED_STRIDE}  authored seeds={list(authored)}",
        "",
        f"{'variant':>8} " + " ".join(f"{n:>12}" for n in NODE_NAMES),
    ]
    for variant in range(8):
        row = " ".join(f"{s:>12}" for s in seed_state(trunk, variant * SEED_STRIDE))
        lines.append(f"{variant:>8} {row}")

    after = seed_state(trunk, 0)
    distinct = len({seed_state(trunk, v * SEED_STRIDE) for v in range(8)})

    lines += [
        "",
        "# invariants",
        f"authored seeds unchanged after run: {after == authored}",
        f"distinct variant states: {distinct}/8",
    ]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "seed_assignment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert after == authored
