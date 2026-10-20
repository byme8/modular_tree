from pathlib import Path

import numpy as np
import pytest
from foliage_bones import (
    ANTIPARALLEL_BAND,
    BLEND_BAND,
    DEFAULT_SEGMENTS_PER_CHAIN,
    _chain_table,
    build_rig,
    deform_vertices,
    measure_rest_pose_error,
    pack_bone_uv,
    pose_rig,
    resolve_chain_parents,
    validate_rig,
)

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "foliage_bones"

PRECISION = 6


def _fmt(value: float) -> str:
    formatted = f"{value:.{PRECISION}f}"
    if formatted.startswith("-0.0") and float(formatted) == 0:
        return "0." + "0" * PRECISION
    return formatted


def write_artifact(name: str, lines: list[str]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_synthetic_tree():
    branches = [
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
    def test_segmentation_and_blending_across_parameters(self):
        extent = 4.0
        base = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])

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

                assert rig.bone_count == segments

                assert not validate_rig(rig)

                distinct = len(set(rig.bone_indices_a.tolist()))
                error = measure_rest_pose_error(rig, positions)

                assert error["max_distance"] == pytest.approx(0.0, abs=1e-9)

                assert np.all(rig.weights_a >= 0.0)
                assert np.all(rig.weights_a <= 1.0)

                if band == 0.0 or segments == 1:
                    assert np.all(rig.weights_a == 1.0)
                else:
                    assert np.any(rig.weights_a < 1.0)
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

        assert parents[0] == -1
        assert parents[1] == 0
        assert parents[2] == 0
        assert parents[3] == 1

        rig = build_rig(**tree)

        assert not validate_rig(rig)
        for index, bone in enumerate(rig.bones):
            assert bone.parent < index

        for bone in rig.bones:
            if bone.segment != 0 or bone.parent < 0:
                continue
            parent = rig.bones[bone.parent]
            junction_offset = float(
                (bone.rest_position - parent.rest_position) @ parent.rest_direction
            )
            assert -1e-9 <= junction_offset <= parent.length + 1e-9, (
                f"stem {bone.stem_id} joins {junction_offset:.4f}m along a parent "
                f"segment only {parent.length:.4f}m long"
            )

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


class TestRigDepth:
    def test_chains_below_rig_depth_ride_their_ancestor_bone(self):
        tree = make_synthetic_tree()
        ids = tree["stem_ids"].astype(int)
        positions = tree["vertex_positions"]

        lines = [
            "# rig depth sweep on the synthetic tree: stems 0 (depth 0), 1 and 2 (depth 1), 3 (depth 2 on stem 1)",
            "# columns: rig_depth | bones | stem: own_bones attach_bone",
        ]

        # TEST_POINT: (rig_depth, expected bone count, unrigged stem -> bone it rides);
        # stem 3 hangs 1.5m along stem 1, so it must ride stem 1's second bone
        # at depth 1 and the trunk bone at z=2 at depth 0.
        cases = [
            (None, 10, {}),
            (0, 4, {1: 2, 2: 3, 3: 2}),
            (1, 8, {3: 5}),
            (2, 10, {}),
        ]

        for rig_depth, expected_bones, riders in cases:
            rig = build_rig(**tree, branch_segments_per_chain=2, rig_depth=rig_depth)

            assert rig.bone_count == expected_bones, (rig_depth, rig.bone_count)
            assert not validate_rig(rig)

            per_stem = []
            for stem in sorted(set(ids.tolist())):
                mask = ids == stem
                own = sum(1 for bone in rig.bones if bone.stem_id == stem)
                bound = sorted(set(rig.bone_indices_a[mask].tolist()))
                per_stem.append(f"{stem}: {own} {bound if own == 0 else '-'}")

                if stem in riders:
                    assert own == 0
                    assert bound == [riders[stem]]
                    assert np.all(rig.bone_indices_b[mask] == riders[stem])
                    assert np.all(rig.weights_a[mask] == 1.0)
                else:
                    assert own > 0

            lines.append(f"{rig_depth} | {rig.bone_count} | " + " | ".join(per_stem))

            matrices = pose_rig(rig, 0.0, 1.0)
            posed = deform_vertices(rig, positions, matrices)
            for stem, bone in riders.items():
                mask = ids == stem
                homogeneous = np.column_stack([positions[mask], np.ones(mask.sum())])
                rigid = (matrices[bone] @ homogeneous.T).T[:, :3]
                assert np.abs(posed[mask] - rigid).max() < 1e-9
                assert np.abs(posed[mask] - positions[mask]).max() > 1e-3

        write_artifact("rig_depth.txt", lines)


class TestBinding:
    def test_every_vertex_binds_soundly(self):
        tree = make_synthetic_tree()
        rig = build_rig(**tree)

        positions = tree["vertex_positions"]
        ids = tree["stem_ids"].astype(int)

        assert np.all(rig.bone_indices_a >= 0)
        assert np.all(rig.bone_indices_a < rig.bone_count)
        assert np.all(rig.bone_indices_b >= 0)
        assert np.all(rig.bone_indices_b < rig.bone_count)

        assert np.all(rig.weights_a >= 0.0)
        assert np.all(rig.weights_a <= 1.0)

        for vertex in range(len(positions)):
            assert rig.bones[int(rig.bone_indices_a[vertex])].stem_id == ids[vertex]
            assert rig.bones[int(rig.bone_indices_b[vertex])].stem_id == ids[vertex]

        error = measure_rest_pose_error(rig, positions)
        assert error["max_distance"] <= float(tree["radii"].max()) + 1e-6

        packed = pack_bone_uv(rig.bone_indices_a, rig.bone_indices_b, rig.weights_a)
        assert len(packed) == len(positions) * 4
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

    def test_skin_weights_sum_to_one(self):
        tree = make_synthetic_tree()
        rig = build_rig(**tree)

        for vertex in range(len(tree["vertex_positions"])):
            index_a = int(rig.bone_indices_a[vertex])
            index_b = int(rig.bone_indices_b[vertex])
            weight_a = float(rig.weights_a[vertex])

            accumulated = {index_a: weight_a}
            if weight_a < 1.0:
                extra = 1.0 if index_b == index_a else 1.0 - weight_a
                accumulated[index_b] = accumulated.get(index_b, 0.0) + extra

            assert sum(accumulated.values()) == pytest.approx(1.0), (
                f"vertex {vertex} weights sum to {sum(accumulated.values())}"
            )

    def test_empty_and_degenerate_input(self):
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


def _tip_direction(rig, matrices, stem_id):
    indices = [i for i, bone in enumerate(rig.bones) if bone.stem_id == stem_id]
    last = indices[-1]
    direction = matrices[last][:3, :3] @ rig.bones[last].rest_direction
    return direction / np.linalg.norm(direction)


class TestWindPose:
    @pytest.mark.parametrize(
        "label,trunk_base,trunk_tip",
        [("soft_trunk", 0.1, 0.0), ("default_trunk", 1.0, 0.6)],
    )
    def test_wind_moves_the_tree_like_wind(self, label, trunk_base, trunk_tip):
        tree = make_synthetic_tree()
        rig = build_rig(
            **tree,
            trunk_base_stiffness=trunk_base,
            trunk_tip_stiffness=trunk_tip,
        )
        positions = tree["vertex_positions"]

        rest = deform_vertices(rig, positions, pose_rig(rig, 0.0, 0.0))
        rest_drift = float(np.abs(rest - positions).max())

        downwind = deform_vertices(rig, positions, pose_rig(rig, 0.0, 1.0))
        offset = downwind - positions
        most_upwind = float(offset[:, 0].min())
        largest_drop = float(offset[:, 2].min())

        step = 0.5
        samples = {}
        for degrees in np.arange(0.0, 360.0 + step, step):
            samples[float(degrees)] = deform_vertices(
                rig, positions, pose_rig(rig, float(degrees), 1.0)
            )

        keys = sorted(samples)
        jumps = [
            float(np.linalg.norm(samples[b] - samples[a], axis=1).max())
            for a, b in zip(keys, keys[1:], strict=False)
        ]
        worst_jump = max(jumps)
        worst_at = keys[jumps.index(worst_jump) + 1]
        peak = max(float(np.linalg.norm(samples[d] - positions, axis=1).max()) for d in keys)

        def interpolation_error(directions):
            spacing = 360.0 / directions
            baked = [
                deform_vertices(rig, positions, pose_rig(rig, i * spacing, 1.0))
                for i in range(directions)
            ]
            worst = 0.0
            for i in range(directions):
                for fraction in np.linspace(0.0, 1.0, 9):
                    degrees = i * spacing + fraction * spacing
                    truth = deform_vertices(rig, positions, pose_rig(rig, degrees, 1.0))
                    blend = (1.0 - fraction) * baked[i] + fraction * baked[(i + 1) % directions]
                    worst = max(worst, float(np.linalg.norm(truth - blend, axis=1).max()))
            return worst

        counts = (8, 16, 32)
        errors = {n: interpolation_error(n) for n in counts}

        matrices = pose_rig(rig, 0.0, 1.0)
        tips = {stem: _tip_direction(rig, matrices, stem) for stem in (0, 1, 2, 3)}

        violations = []
        if rest_drift > 1e-9:
            violations.append(f"rest pose drifts by {rest_drift:.9f}")
        if most_upwind < -1e-3:
            violations.append(f"a vertex travels {abs(most_upwind):.4f} upwind")
        if largest_drop < -0.5 * peak:
            violations.append(
                f"a vertex drops {abs(largest_drop):.4f}, over half the "
                f"{peak:.4f} peak displacement"
            )
        if tips[0][2] < -0.5:
            violations.append(
                f"trunk tip points {tips[0][2]:+.2f} in Z; it has folded back on itself"
            )
        for stem in (1, 3):
            if tips[stem][2] < -0.35:
                violations.append(
                    f"stem {stem} tip points {tips[stem][2]:+.2f} in Z; wind drove it into the ground"
                )
        if worst_jump > 0.15 * peak:
            violations.append(
                f"pose jumps {worst_jump:.6f} at {worst_at:g} deg, "
                f"{100 * worst_jump / peak:.1f}% of peak displacement"
            )
        coarse, fine = counts[0], counts[1]
        if errors[fine] > 0.6 * errors[coarse]:
            violations.append(
                f"doubling {coarse} to {fine} directions cut error only "
                f"{errors[coarse] / errors[fine]:.2f}x; expected about 4x"
            )

        finest = errors[counts[-1]]
        if finest > 0.05 * peak:
            violations.append(
                f"{counts[-1]} directions still leave {finest:.6f} error, "
                f"{100 * finest / peak:.1f}% of peak displacement"
            )

        lines = [
            f"# wind pose: {label} (trunk base {trunk_base:g}, tip {trunk_tip:g})",
            f"# {rig.bone_count} bones, {len(positions)} vertices, "
            f"antiparallel band {ANTIPARALLEL_BAND:g}",
            "",
            f"rest_drift_at_zero_strength {_fmt(rest_drift)}",
            f"peak_displacement {_fmt(peak)}",
            "",
            "# wind blowing along +X at full strength",
            f"most_upwind_travel {_fmt(most_upwind)}",
            f"largest_drop {_fmt(largest_drop)}",
            "",
            "# posed direction of each chain tip; Z below zero points at the ground",
            *(
                f"stem{stem}_tip {_fmt(tips[stem][0])} {_fmt(tips[stem][1])} {_fmt(tips[stem][2])}"
                for stem in sorted(tips)
            ),
            "",
            f"# largest vertex move between adjacent {step:g} deg samples",
            f"worst_jump {_fmt(worst_jump)} at_degrees {_fmt(worst_at)}",
            "",
            "# interpolating N baked directions against the true pose between them",
            "# columns: directions | max_error | ratio_to_previous",
        ]

        previous = None
        for n in counts:
            ratio = "-" if previous is None else _fmt(errors[n] / previous)
            lines.append(f"{n:3d} {_fmt(errors[n])} {ratio}")
            previous = errors[n]

        lines.extend(["", "# violations"] + ([f"  {v}" for v in violations] or ["  none"]))
        write_artifact(f"wind_pose_{label}.txt", lines)

        assert not violations, "; ".join(violations)
