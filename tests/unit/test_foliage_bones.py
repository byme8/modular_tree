"""Tests for foliage bone rig construction.

A rig fails the same way the vertex streams do: wrong values produce perfectly
plausible geometry, and the defect only appears once wind runs in the consuming
engine. Worse, the two worst failures — a parent index pointing forward, a bone
index out of range — are read as raw buffer offsets by the shader, so they
sample another branch's pose rather than erroring.

Each test therefore writes a deterministic artifact to
``tests/artifacts/foliage_bones/`` for inspection rather than relying on a
pass/fail tick alone.
"""

from pathlib import Path

import numpy as np
import pytest
from foliage_bones import (
    BLEND_BAND,
    DEFAULT_SEGMENTS_PER_CHAIN,
    build_rig,
    measure_rest_pose_error,
    pack_bone_uv,
    resolve_chain_parents,
    validate_rig,
    _chain_table,
)

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "foliage_bones"

PRECISION = 6


def _fmt(value: float) -> str:
    """Format a float at fixed precision, normalising negative zero."""
    formatted = f"{value:.{PRECISION}f}"
    if formatted.startswith("-0.0") and float(formatted) == 0:
        return "0." + "0" * PRECISION
    return formatted


def write_artifact(name: str, lines: list[str]) -> None:
    """Write a deterministic snapshot artifact."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_synthetic_tree():
    """Build a known four-chain tree without the native mesher.

    Chain 0 is a vertical trunk. Chains 1 and 2 are side branches rooted partway
    up it, running along +X and +Y. Chain 3 is a twig rooted on chain 1, which is
    what makes parent recovery non-trivial: a depth-2 chain must attach to the
    correct depth-1 branch rather than to the trunk or to its sibling.

    Deliberately mirrors the fixture in ``test_foliage_streams`` so the two
    modules are exercised against the same topology, with the extra depth-2
    chain the rig needs and the streams do not.
    """
    branches = [
        # (stem_id, depth, base, direction, extent, radius, n_sections)
        (0, 0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 4.0, 0.5, 9),
        (1, 1, np.array([0.0, 0.0, 2.0]), np.array([1.0, 0.0, 0.0]), 2.0, 0.2, 7),
        (2, 1, np.array([0.0, 0.0, 3.0]), np.array([0.0, 1.0, 0.0]), 1.0, 0.1, 5),
        (3, 2, np.array([1.5, 0.0, 2.0]), np.array([0.0, 0.0, 1.0]), 0.8, 0.05, 5),
    ]

    positions, stem_ids, extents, radii, pivots, directions, depths = [], [], [], [], [], [], []

    for stem_id, depth, base, direction, extent, radius, n_sections in branches:
        for section in range(n_sections):
            t = section / (n_sections - 1)
            perpendicular = (
                np.array([0.0, 1.0, 0.0]) if abs(direction[2]) > 0.5 else np.array([0.0, 0.0, 1.0])
            )
            for side in (-1.0, 1.0):
                positions.append(base + direction * extent * t + perpendicular * radius * side)
                stem_ids.append(float(stem_id))
                extents.append(extent)
                radii.append(radius)
                pivots.append(base)
                directions.append(direction)
                depths.append(float(depth))

    return {
        "vertex_positions": np.array(positions),
        "stem_ids": np.array(stem_ids),
        "branch_extents": np.array(extents),
        "radii": np.array(radii),
        "pivot_positions": np.array(pivots),
        "directions": np.array(directions),
        "hierarchy_depths": np.array(depths),
    }


class TestSegmentationSweep:
    """T1: sweep segment count and blend band over a single straight chain.

    Covers M6 (a chain must be segmented to bend at all) and M7 (the blend at a
    segment boundary must be continuous) across the parameter space rather than
    pinning one configuration.
    """

    def test_segmentation_and_blending_across_parameters(self):
        extent = 4.0
        base = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])

        # Dense sampling along the chain so segment boundaries are crossed.
        samples = 41
        t_values = np.linspace(0.0, 1.0, samples)
        positions = np.array([base + direction * extent * t for t in t_values])
        n = len(positions)

        lines = [
            "# bone segmentation sweep: one straight chain, extent 4.0m",
            "# columns: segments band | bone_count | max_weight_a min_weight_a "
            "| distinct_bones | max_bind_distance",
        ]

        for segments in (1, 2, 4, 8):
            for band in (0.0, 0.25, 0.5):
                rig = build_rig(
                    vertex_positions=positions,
                    stem_ids=np.zeros(n),
                    branch_extents=np.full(n, extent),
                    radii=np.full(n, 0.5),
                    pivot_positions=np.tile(base, (n, 1)),
                    directions=np.tile(direction, (n, 1)),
                    hierarchy_depths=np.zeros(n),
                    segments_per_chain=segments,
                    blend_band=band,
                )

                # M6: the chain is split into as many bones as requested, so it
                # can bend rather than move rigidly.
                assert rig.bone_count == segments

                # Bones run root to tip with each segment parented to the one
                # before it, so a forward sweep is valid.
                assert not validate_rig(rig)

                distinct = len(set(rig.bone_indices_a.tolist()))
                error = measure_rest_pose_error(rig, positions)

                # M5: every vertex binds to a bone whose rest segment spans it.
                # These vertices sit exactly on the chain axis, so the distance
                # is zero up to floating point.
                assert error["max_distance"] == pytest.approx(0.0, abs=1e-9)

                # M4: weights stay in range whatever the band.
                assert np.all(rig.weights_a >= 0.0)
                assert np.all(rig.weights_a <= 1.0)

                # M7: with no band nothing blends; with a band, vertices near a
                # boundary do. A single-bone chain never blends regardless.
                if band == 0.0 or segments == 1:
                    assert np.all(rig.weights_a == 1.0)
                else:
                    assert np.any(rig.weights_a < 1.0)
                    # The blend never crosses to a bone more than one away, and
                    # never inverts: weightA stays the dominant weight.
                    assert np.all(rig.weights_a >= 0.5)
                    step = np.abs(rig.bone_indices_a - rig.bone_indices_b)
                    assert np.all(step <= 1)

                lines.append(
                    f"segments={segments} band={_fmt(band)} | bones={rig.bone_count} "
                    f"| weight_a max={_fmt(float(rig.weights_a.max()))} "
                    f"min={_fmt(float(rig.weights_a.min()))} "
                    f"| distinct={distinct} "
                    f"| max_bind_dist={_fmt(error['max_distance'])}"
                )

        # Per-vertex detail at the default configuration, so the blend profile
        # itself is reviewable rather than only its extremes.
        rig = build_rig(
            vertex_positions=positions,
            stem_ids=np.zeros(n),
            branch_extents=np.full(n, extent),
            radii=np.full(n, 0.5),
            pivot_positions=np.tile(base, (n, 1)),
            directions=np.tile(direction, (n, 1)),
            hierarchy_depths=np.zeros(n),
        )
        lines.append("")
        lines.append(
            f"# blend profile at segments={DEFAULT_SEGMENTS_PER_CHAIN} band={_fmt(BLEND_BAND)}"
        )
        lines.append("# t | z | indexA indexB weightA")
        for i, t in enumerate(t_values):
            lines.append(
                f"{_fmt(t)} | {_fmt(positions[i][2])} | "
                f"{int(rig.bone_indices_a[i])} {int(rig.bone_indices_b[i])} "
                f"{_fmt(float(rig.weights_a[i]))}"
            )

        write_artifact("segmentation_sweep.txt", lines)


class TestTopology:
    """T2: parent recovery and forward-sweep ordering on a branching tree.

    Covers M1 (parent index below own index) and M2 (correct parent chain), the
    two failures the consuming engine cannot detect at all.
    """

    def test_parents_are_recovered_and_ordered(self):
        tree = make_synthetic_tree()

        chains = _chain_table(
            tree["stem_ids"],
            tree["branch_extents"],
            tree["pivot_positions"],
            tree["directions"],
            tree["radii"],
            tree["hierarchy_depths"],
        )
        parents = resolve_chain_parents(chains)

        # M2: the trunk is a root; both depth-1 branches hang off the trunk; the
        # depth-2 twig hangs off chain 1, whose chain its pivot actually touches,
        # not off chain 2 and not off the trunk.
        assert parents[0] == -1
        assert parents[1] == 0
        assert parents[2] == 0
        assert parents[3] == 1

        rig = build_rig(**tree)

        # M1: every bone's parent precedes it, so the forward pose sweep reads
        # only already-computed parents.
        assert not validate_rig(rig)
        for index, bone in enumerate(rig.bones):
            assert bone.parent < index

        # A branch attaches to the parent segment that geometrically contains
        # the junction, so it swings on an arc that actually passes through the
        # point where it joins. Attaching to either end of the parent chain puts
        # it on the wrong arc, which is the failure the whole rig exists to fix.
        #
        # Checked against the junction position rather than against an expected
        # bone index, so the assertion cannot encode the same off-by-one the
        # implementation might have.
        for bone in rig.bones:
            if bone.segment != 0 or bone.parent < 0:
                continue
            parent = rig.bones[bone.parent]
            junction_offset = float(
                (bone.rest_position - parent.rest_position) @ parent.rest_direction
            )
            # The junction lies within the parent segment's own span.
            assert -1e-9 <= junction_offset <= parent.length + 1e-9, (
                f"stem {bone.stem_id} joins {junction_offset:.4f}m along a parent "
                f"segment only {parent.length:.4f}m long"
            )

        # Chain 1 roots at z=2.0 on a trunk whose segments are 1m long, so it
        # must hang off the segment spanning z=2..3, not off the trunk tip.
        chain_one_root = next(b for b in rig.bones if b.stem_id == 1 and b.segment == 0)
        assert rig.bones[chain_one_root.parent].rest_position[2] == pytest.approx(2.0)

        lines = [
            "# bone rig topology: four-chain synthetic tree",
            f"# bones={rig.bone_count} vertices={len(tree['vertex_positions'])}",
            "",
            "# chain parents (stem_id -> parent stem_id, -1 = root)",
        ]
        for stem_id in sorted(parents):
            lines.append(
                f"stem {stem_id} depth={chains[stem_id]['depth']} "
                f"extent={_fmt(chains[stem_id]['extent'])} "
                f"radius={_fmt(chains[stem_id]['radius'])} -> parent {parents[stem_id]}"
            )

        lines.append("")
        lines.append("# bones in emission order")
        lines.append("# idx | stem seg depth | parent | rest_position | length stiffness swing")
        for index, bone in enumerate(rig.bones):
            position = " ".join(_fmt(v) for v in bone.rest_position)
            lines.append(
                f"{index:3d} | {bone.stem_id} {bone.segment} {bone.depth} "
                f"| {bone.parent:3d} | {position} "
                f"| {_fmt(bone.length)} {_fmt(bone.stiffness)} {_fmt(bone.swing_weight)}"
            )

        lines.append("")
        lines.append("# invariants")
        lines.append(f"forward_sweep_valid: {all(b.parent < i for i, b in enumerate(rig.bones))}")
        lines.append(f"roots: {[i for i, b in enumerate(rig.bones) if b.parent < 0]}")

        write_artifact("topology.txt", lines)


class TestBinding:
    """T3: per-vertex binding on the branching tree.

    Covers M3 (index range), M4 (weight validity) and M5 (bind distance) — the
    three properties that must hold for every vertex, checked exhaustively and
    written into the artifact with any violation.
    """

    def test_every_vertex_binds_soundly(self):
        tree = make_synthetic_tree()
        rig = build_rig(**tree)

        positions = tree["vertex_positions"]
        ids = tree["stem_ids"].astype(int)

        # M3: indices are raw buffer offsets in the shader, so out of range
        # silently samples another branch's pose.
        assert np.all(rig.bone_indices_a >= 0)
        assert np.all(rig.bone_indices_a < rig.bone_count)
        assert np.all(rig.bone_indices_b >= 0)
        assert np.all(rig.bone_indices_b < rig.bone_count)

        # M4: weightB is derived as 1 - weightA and nothing renormalises it.
        assert np.all(rig.weights_a >= 0.0)
        assert np.all(rig.weights_a <= 1.0)

        # A vertex only ever binds to bones of its own chain: binding across
        # chains would make a branch follow another branch's pose.
        for vertex in range(len(positions)):
            assert rig.bones[int(rig.bone_indices_a[vertex])].stem_id == ids[vertex]
            assert rig.bones[int(rig.bone_indices_b[vertex])].stem_id == ids[vertex]

        # M5: the bound bone's rest segment spans the vertex. The tolerance is
        # the branch radius, since vertices sit on a ring around the axis.
        error = measure_rest_pose_error(rig, positions)
        assert error["max_distance"] <= float(tree["radii"].max()) + 1e-6

        packed = pack_bone_uv(rig.bone_indices_a, rig.bone_indices_b, rig.weights_a)
        assert len(packed) == len(positions) * 4
        # The fourth component is unused by the contract and must be zero.
        assert np.all(packed[3::4] == 0.0)
        assert packed[0::4] == pytest.approx(rig.bone_indices_a.astype(float))
        assert packed[1::4] == pytest.approx(rig.bone_indices_b.astype(float))
        assert packed[2::4] == pytest.approx(rig.weights_a)

        problems = validate_rig(rig)
        assert not problems

        lines = [
            "# per-vertex bone binding: four-chain synthetic tree",
            f"# bones={rig.bone_count} vertices={len(positions)}",
            f"# max_bind_distance={_fmt(error['max_distance'])} "
            f"mean={_fmt(error['mean_distance'])} worst_vertex={error['worst_vertex']}",
            "",
            "# violations (empty means every invariant held)",
        ]
        lines.extend(problems if problems else ["none"])

        lines.append("")
        lines.append("# vtx | stem | position | indexA indexB weightA | bind_distance")

        starts = np.array([b.rest_position for b in rig.bones])
        ends = np.array([b.rest_position + b.rest_direction * b.length for b in rig.bones])

        for vertex in range(len(positions)):
            bone = int(rig.bone_indices_a[vertex])
            axis = ends[bone] - starts[bone]
            axis_sq = float(axis @ axis)
            if axis_sq <= 1e-12:
                distance = float(np.linalg.norm(positions[vertex] - starts[bone]))
            else:
                t = min(1.0, max(0.0, float((positions[vertex] - starts[bone]) @ axis) / axis_sq))
                distance = float(np.linalg.norm(positions[vertex] - (starts[bone] + axis * t)))

            position = " ".join(_fmt(v) for v in positions[vertex])
            lines.append(
                f"{vertex:3d} | {ids[vertex]} | {position} "
                f"| {int(rig.bone_indices_a[vertex])} {int(rig.bone_indices_b[vertex])} "
                f"{_fmt(float(rig.weights_a[vertex]))} | {_fmt(distance)}"
            )

        write_artifact("binding.txt", lines)

    def test_empty_and_degenerate_input(self):
        """A rig built from nothing is empty rather than malformed.

        The export path can be pointed at a mesh with no branch attributes, and
        an empty rig with a non-zero bone count would index into nothing.
        """
        rig = build_rig(
            vertex_positions=np.zeros((0, 3)),
            stem_ids=np.zeros(0),
            branch_extents=np.zeros(0),
            radii=np.zeros(0),
            pivot_positions=np.zeros((0, 3)),
            directions=np.zeros((0, 3)),
            hierarchy_depths=np.zeros(0),
        )
        assert rig.bone_count == 0
        assert not validate_rig(rig)
        assert len(pack_bone_uv(rig.bone_indices_a, rig.bone_indices_b, rig.weights_a)) == 0

        # A chain too short to segment gets one bone rather than a division by
        # its own near-zero length.
        n = 4
        rig = build_rig(
            vertex_positions=np.zeros((n, 3)),
            stem_ids=np.zeros(n),
            branch_extents=np.full(n, 1e-9),
            radii=np.full(n, 0.01),
            pivot_positions=np.zeros((n, 3)),
            directions=np.tile(np.array([0.0, 0.0, 1.0]), (n, 1)),
            hierarchy_depths=np.zeros(n),
        )
        assert rig.bone_count == 1
        assert not validate_rig(rig)
        assert np.all(rig.weights_a == 1.0)
