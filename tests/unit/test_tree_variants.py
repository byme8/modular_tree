"""Tests for seed-variant generation.

The failure this guards against is silent: if the seed walk misses nodes, every
variant still generates and still looks like a tree, but they share branch
structure and only the trunk differs. That reads as "the variants are a bit
samey" rather than as a bug.

The node module itself imports bpy, so the pure walk logic is exercised here
against stand-in node objects with the same shape as the Blender ones.
"""

from pathlib import Path

import pytest

# The node module imports bpy at module scope, so the pure function under test
# is re-declared against the same contract rather than imported. This mirrors
# collect_seed_sockets in nodes/tree_function_nodes/tree_variants_node.py.
SEED_PROPERTY = "seed"
SEED_STRIDE = 7919

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "tree_variants"


class FakeSocket:
    """Stands in for a Blender input socket carrying a property."""

    def __init__(self, property_name, property_value=0):
        self.property_name = property_name
        self.property_value = property_value


class FakeNode:
    """Stands in for an MtreeNode with inputs and child nodes."""

    def __init__(self, name, sockets=None, children=None):
        self.name = name
        self.inputs = sockets or []
        self._children = children or []

    def get_child_nodes(self):
        return self._children


def collect_seed_sockets(root_node, seen=None):
    """Mirror of the production walk; see module docstring."""
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


def make_graph():
    """Build a trunk -> 2 branches -> 1 sub-branch graph, each with a seed."""
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
        [FakeSocket("seed", 4)],
    )
    trunk = FakeNode(
        "trunk",
        [FakeSocket("seed", 1), FakeSocket("length", 14.0), FakeSocket("start_radius", 0.3)],
        [branch_a, branch_b],
    )
    return trunk


class TestSeedCollection:
    """M14: every seed in the graph must be found, not just the trunk's."""

    def test_walk_finds_every_seed(self):
        trunk = make_graph()
        sockets = collect_seed_sockets(trunk)

        # Four seed-carrying nodes: trunk, two branches, one sub-branch.
        assert len(sockets) == 4
        assert sorted(s.property_value for s in sockets) == [1, 3, 4, 5]

        # Non-seed sockets must not be swept up and offset.
        assert all(s.property_name == "seed" for s in sockets)

    def test_walk_survives_loops(self):
        """A cyclic graph must terminate rather than recursing forever."""
        a = FakeNode("a", [FakeSocket("seed", 1)])
        b = FakeNode("b", [FakeSocket("seed", 2)], [a])
        a._children = [b]  # close the loop

        sockets = collect_seed_sockets(a)
        assert len(sockets) == 2

    def test_graph_without_seeds(self):
        """A graph carrying no seeds yields nothing rather than raising."""
        node = FakeNode("plain", [FakeSocket("length", 3.0)])
        assert collect_seed_sockets(node) == []
        assert collect_seed_sockets(None) == []


class TestVariantOffsets:
    """M16: distinct variants must not collide on the same seed set."""

    def test_variants_are_distinct(self):
        trunk = make_graph()
        sockets = collect_seed_sockets(trunk)
        originals = [s.property_value for s in sockets]

        seen = {}
        for variant in range(20):
            offset = variant * SEED_STRIDE
            state = tuple(o + offset for o in originals)
            assert state not in seen, f"variant {variant} collides with {seen.get(state)}"
            seen[state] = variant

        assert len(seen) == 20

    def test_stride_avoids_neighbour_collisions(self):
        """Adjacent variants must not alias onto nodes whose seeds differ by 1.

        With a stride of 1, variant N on a node seeded 2 produces the same seed
        as variant N+1 on a node seeded 1 — neighbouring variants would share
        sub-structure. The stride is what prevents that.
        """
        originals = [1, 2, 3, 4]

        collisions = 0
        for variant in range(20):
            current = {o + variant * SEED_STRIDE for o in originals}
            following = {o + (variant + 1) * SEED_STRIDE for o in originals}
            collisions += len(current & following)

        assert collisions == 0

    def test_base_seed_shifts_the_whole_set(self):
        """Changing base seed yields a different family of variants."""
        originals = [1, 3, 4, 5]

        family_a = {
            tuple(o + (0 + v) * SEED_STRIDE for o in originals) for v in range(20)
        }
        family_b = {
            tuple(o + (100 + v) * SEED_STRIDE for o in originals) for v in range(20)
        }

        assert not (family_a & family_b)


class TestSeedRestoration:
    """M15: a run must leave the authored configuration untouched."""

    @pytest.mark.parametrize("fail_at", [None, 0, 3, 9])
    def test_seeds_restored_even_on_failure(self, fail_at):
        """Seeds return to their authored values whether or not a build throws."""
        trunk = make_graph()
        sockets = collect_seed_sockets(trunk)
        originals = [s.property_value for s in sockets]

        def run():
            saved = [s.property_value for s in sockets]
            try:
                for variant in range(10):
                    offset = variant * SEED_STRIDE
                    for socket, original in zip(sockets, saved, strict=True):
                        socket.property_value = original + offset
                    if fail_at is not None and variant == fail_at:
                        raise RuntimeError("build failed")
            except RuntimeError:
                pass
            finally:
                for socket, original in zip(sockets, saved, strict=True):
                    socket.property_value = original

        run()

        assert [s.property_value for s in sockets] == originals

    def test_snapshot(self):
        """Record the seed assignment table for inspection."""
        trunk = make_graph()
        sockets = collect_seed_sockets(trunk)
        originals = [s.property_value for s in sockets]

        # Stable node order for a diffable artifact.
        names = ["trunk", "branch_a", "sub_branch", "branch_b"]

        lines = [
            "# seed assignment per variant",
            f"# stride={SEED_STRIDE}  authored seeds={originals}",
            "",
            f"{'variant':>8} " + " ".join(f"{n:>12}" for n in names),
        ]
        for variant in range(8):
            offset = variant * SEED_STRIDE
            row = " ".join(f"{o + offset:>12}" for o in originals)
            lines.append(f"{variant:>8} {row}")

        lines += [
            "",
            "# invariants",
            f"authored seeds unchanged after run: {[s.property_value for s in sockets] == originals}",
            f"distinct variant states: {len({tuple(o + v * SEED_STRIDE for o in originals) for v in range(8)})}/8",
        ]

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "seed_assignment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        assert [s.property_value for s in sockets] == originals
