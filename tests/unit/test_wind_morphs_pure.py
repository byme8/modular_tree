from pathlib import Path

import numpy as np
import pytest
from foliage_bones import (
    ANTIPARALLEL_BAND,
    Bone,
    BoneRig,
    deform_vertices,
    pose_rig,
    validate_rig,
)

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "wind_morphs"

PRECISION = 6


def _fmt(value: float) -> str:
    formatted = f"{value:.{PRECISION}f}"
    if formatted.startswith("-0.0") and float(formatted) == 0:
        return "0." + "0" * PRECISION
    return formatted


def write_artifact(name: str, lines: list[str]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def chain_bones_with_stiffness(
    stiffness_per_bone,
    length=1.0,
    direction=(0.0, 0.0, 1.0),
    origin=(0.0, 0.0, 0.0),
):
    axis = np.asarray(direction, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    base = np.asarray(origin, dtype=np.float64)

    return [
        Bone(
            rest_position=base + axis * (length * segment),
            rest_direction=axis.copy(),
            parent=segment - 1,
            stiffness=float(value),
            length=length,
            swing_weight=1.0,
            stem_id=0,
            segment=segment,
            depth=0,
        )
        for segment, value in enumerate(stiffness_per_bone)
    ]


def tree_bones(trunk_stiffness, branches):
    axis = np.array([0.0, 0.0, 1.0])
    bones = [
        Bone(
            rest_position=axis * index,
            rest_direction=axis.copy(),
            parent=index - 1,
            stiffness=float(value),
            length=1.0,
            swing_weight=1.0,
            stem_id=0,
            segment=index,
            depth=0,
        )
        for index, value in enumerate(trunk_stiffness)
    ]

    for stem_id, (attach, tilt_degrees, stiffness_per_bone) in enumerate(branches, start=1):
        tilt = np.radians(tilt_degrees)
        direction = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
        origin = axis * float(attach)
        parent = attach - 1 if attach > 0 else -1

        for segment, value in enumerate(stiffness_per_bone):
            bones.append(
                Bone(
                    rest_position=origin + direction * segment,
                    rest_direction=direction.copy(),
                    parent=parent if segment == 0 else len(bones) - 1,
                    stiffness=float(value),
                    length=1.0,
                    swing_weight=1.0,
                    stem_id=stem_id,
                    segment=segment,
                    depth=1,
                )
            )

    return bones


def chains_of(bones):
    chains = {}
    for index, bone in enumerate(bones):
        chains.setdefault(bone.stem_id, []).append(index)
    return chains


def chain_polylines(bones, matrices):
    lines = []
    for indices in chains_of(bones).values():
        segment_bones = [bones[i] for i in indices]
        segment_matrices = [matrices[i] for i in indices]
        lines.append(joint_polyline(segment_bones, segment_matrices))
    return lines


def rig_from_bones(bones, vertices_per_bone=1):
    count = len(bones)
    indices = np.repeat(np.arange(count, dtype=np.int32), vertices_per_bone)

    return BoneRig(
        bones=bones,
        bone_indices_a=indices,
        bone_indices_b=indices.copy(),
        weights_a=np.ones(len(indices), dtype=np.float64),
    )


def joint_polyline(bones, matrices):
    points = [
        np.asarray(matrix)[:3, :3] @ bone.rest_position + np.asarray(matrix)[:3, 3]
        for bone, matrix in zip(bones, matrices)
    ]

    last, tip_matrix = bones[-1], np.asarray(matrices[-1])
    tip = last.rest_position + last.rest_direction * last.length
    points.append(tip_matrix[:3, :3] @ tip + tip_matrix[:3, 3])

    return np.array(points)


def _svg_polyline(points, stroke, width, opacity=1.0):
    coords = " ".join(f"{x:.4f},{y:.4f}" for x, y in points)
    return (
        f'  <polyline points="{coords}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width:.3f}" stroke-linecap="round" '
        f'stroke-linejoin="round" opacity="{opacity:.3f}"/>'
    )


def _svg_joints(points, fill, radius, opacity=1.0):
    return [
        f'  <circle cx="{x:.4f}" cy="{y:.4f}" r="{radius:.3f}" fill="{fill}" '
        f'stroke="#ffffff" stroke-width="1" opacity="{opacity:.3f}"/>'
        for x, y in points
    ]


def write_svg(name, curves, caption, extent=6.0, size=420, joint_radius=0.0, labels=()):
    scale = size / (2.0 * extent)

    def project_xz(point):
        return (size * 0.5 + point[0] * scale, size - 20.0 - point[2] * scale)

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">',
        f'  <rect width="{size}" height="{size}" fill="#ffffff"/>',
        f'  <line x1="0" y1="{size - 20}" x2="{size}" y2="{size - 20}" '
        f'stroke="#cccccc" stroke-width="1"/>',
        f'  <line x1="{size * 0.5}" y1="0" x2="{size * 0.5}" y2="{size}" '
        f'stroke="#eeeeee" stroke-width="1"/>',
    ]

    for points, stroke, width, opacity in curves:
        projected = [project_xz(p) for p in points]
        body.append(_svg_polyline(projected, stroke, width, opacity))
        if joint_radius > 0.0:
            body.extend(_svg_joints(projected, stroke, joint_radius, opacity))

    body.append(
        f'  <text x="8" y="16" font-family="monospace" font-size="11" '
        f'fill="#333333">{caption}</text>'
    )
    for offset, (text, colour) in enumerate(labels):
        body.append(
            f'  <text x="8" y="{32 + offset * 14}" font-family="monospace" '
            f'font-size="10" fill="{colour}">{text}</text>'
        )
    body.append("</svg>")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / name).write_text("\n".join(body) + "\n", encoding="utf-8")


def bake_morph_targets(rig, rest, directions):
    step = 360.0 / directions
    return [
        (index * step, deform_vertices(rig, rest, pose_rig(rig, index * step, 1.0)))
        for index in range(directions)
    ]


def cumulative_turn_degrees(bone, matrix):
    turned = np.asarray(matrix)[:3, :3] @ bone.rest_direction
    cosine = float(np.clip(turned @ bone.rest_direction, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


class TestStiffnessResponse:

    @pytest.mark.parametrize(
        "label,stiffness",
        [
            ("uniform_soft", [0.0, 0.0, 0.0, 0.0]),
            ("uniform_mid", [0.5, 0.5, 0.5, 0.5]),
            ("rigid_base_soft_tip", [1.0, 0.7, 0.4, 0.0]),
            ("soft_base_rigid_tip", [0.0, 0.4, 0.7, 1.0]),
            ("fully_rigid", [1.0, 1.0, 1.0, 1.0]),
        ],
    )
    def test_stiffness_governs_deflection(self, label, stiffness):
        bones = chain_bones_with_stiffness(stiffness)
        rig = rig_from_bones(bones)

        assert not validate_rig(rig)

        matrices = pose_rig(rig, 0.0, 1.0)
        posed = joint_polyline(bones, matrices)
        rest_joints = joint_polyline(bones, pose_rig(rig, 0.0, 0.0))

        tip_travel = float(posed[-1][0] - rest_joints[-1][0])
        out_of_plane = float(np.abs(posed[:, 1]).max())
        turns = [cumulative_turn_degrees(bone, matrices[i]) for i, bone in enumerate(bones)]

        rigid = all(value >= 1.0 for value in stiffness)

        violations = []
        if out_of_plane > 1e-12:
            violations.append(f"XZ wind left the plane by {out_of_plane:.3e} in Y")
        if rigid and abs(tip_travel) > 1e-12:
            violations.append(f"fully rigid chain still moved {tip_travel:.3e}")
        if not rigid and tip_travel <= 0.0:
            violations.append(f"compliant chain did not travel downwind ({tip_travel:.6f})")

        for index, bone in enumerate(bones):
            if bone.stiffness >= 1.0 and index == 0 and turns[index] > 1e-9:
                violations.append(
                    f"root bone at stiffness 1.0 turned {turns[index]:.6f} deg"
                )

        write_svg(
            f"stiffness_{label}.svg",
            [(rest_joints, "#bbbbbb", 3.0, 1.0), (posed, "#1f77b4", 4.0, 1.0)],
            f"{label}  stiffness={stiffness}  tip_travel={tip_travel:.3f}",
        )

        write_artifact(
            f"stiffness_{label}.txt",
            [
                f"# stiffness response: {label}",
                "# one 4-bone chain along +Z, wind along +X at full strength",
                "",
                f"tip_travel_downwind {_fmt(tip_travel)}",
                f"out_of_plane_y {_fmt(out_of_plane)}",
                "",
                "# columns: bone | stiffness | sway=1-s | cumulative_turn_deg",
                *(
                    f"{index:4d} {_fmt(bone.stiffness)} "
                    f"{_fmt(max(0.0, 1.0 - bone.stiffness))} {_fmt(turns[index])}"
                    for index, bone in enumerate(bones)
                ),
                "",
                "# violations",
                *([f"  {v}" for v in violations] or ["  none"]),
            ],
        )

        assert not violations, "; ".join(violations)

    def test_softer_bones_deflect_further(self):
        levels = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

        travels = []
        curves = []
        for shade, value in enumerate(levels):
            bones = chain_bones_with_stiffness([value] * 4)
            rig = rig_from_bones(bones)
            posed = joint_polyline(bones, pose_rig(rig, 0.0, 1.0))
            travels.append(float(posed[-1][0]))
            grey = int(30 + 150 * shade / max(1, len(levels) - 1))
            curves.append((posed, f"rgb({grey},{grey},200)", 3.0, 1.0))

        violations = [
            f"stiffness {levels[i]:g} -> {levels[i + 1]:g} did not increase travel "
            f"({travels[i]:.6f} -> {travels[i + 1]:.6f})"
            for i in range(len(levels) - 1)
            if travels[i + 1] <= travels[i]
        ]

        write_svg(
            "stiffness_monotonic.svg",
            curves,
            "uniform stiffness 1.0 (dark) to 0.0 (light), wind +X",
        )

        write_artifact(
            "stiffness_monotonic.txt",
            [
                "# uniform stiffness sweep, 4-bone chain, wind +X at full strength",
                "# columns: stiffness | tip_x",
                *(f"{_fmt(v)} {_fmt(t)}" for v, t in zip(levels, travels)),
                "",
                "# violations",
                *([f"  {v}" for v in violations] or ["  none"]),
            ],
        )

        assert not violations, "; ".join(violations)


class TestAmplitudeLaw:

    @pytest.mark.parametrize("segments", [1, 2, 4, 8])
    @pytest.mark.parametrize("stiffness", [0.0, 0.25, 0.5, 0.75])
    def test_turn_matches_sway_squared_over_segment_count(self, segments, stiffness):
        bones = chain_bones_with_stiffness([stiffness] * segments)
        rig = rig_from_bones(bones)
        matrices = pose_rig(rig, 0.0, 1.0)

        sway = max(0.0, 1.0 - stiffness)
        expected_per_bone = 180.0 * sway * sway / segments

        root_turn = cumulative_turn_degrees(bones[0], matrices[0])
        tip_turn = cumulative_turn_degrees(bones[-1], matrices[-1])

        violations = []
        if abs(root_turn - expected_per_bone) > 1e-9:
            violations.append(
                f"root turned {root_turn:.9f} deg, expected sway^2/segments = "
                f"{expected_per_bone:.9f}"
            )

        if tip_turn < root_turn - 1e-9:
            violations.append(
                f"tip turned {tip_turn:.9f} deg, less than the root's {root_turn:.9f}"
            )

        tip_height = float(joint_polyline(bones, matrices)[-1][2])
        single_bone = segments == 1

        if not single_bone and tip_height < -1e-9:
            violations.append(
                f"tip sank to Z={tip_height:.6f}; a chain of {segments} bones drove "
                f"itself underground"
            )
        if single_bone and expected_per_bone > 90.0 and tip_height >= 0.0:
            violations.append(
                f"a lone bone turning {expected_per_bone:.6f} deg stayed above ground "
                f"at Z={tip_height:.6f}; the unparented overbend has been fixed"
            )

        write_artifact(
            f"amplitude_s{stiffness:g}_n{segments}.txt",
            [
                f"# amplitude law: stiffness {stiffness:g}, {segments} segments",
                "# per-bone turn should equal 180 * (1-stiffness)^2 / segments",
                "# the chain saturates at 90 deg: leverage falls as a bone nears the",
                "# wind axis, so turns do not accumulate linearly toward the tip",
                "",
                f"sway {_fmt(sway)}",
                f"expected_per_bone_deg {_fmt(expected_per_bone)}",
                f"measured_root_deg {_fmt(root_turn)}",
                f"measured_tip_deg {_fmt(tip_turn)}",
                "",
                "# violations",
                *([f"  {v}" for v in violations] or ["  none"]),
            ],
        )

        assert not violations, "; ".join(violations)

    def test_deflection_scales_with_strength(self):
        bones = chain_bones_with_stiffness([0.0, 0.0, 0.0, 0.0])
        rig = rig_from_bones(bones)

        rows = []
        for strength in (0.0, 0.25, 0.5, 0.75, 1.0):
            matrices = pose_rig(rig, 0.0, strength)
            turn = cumulative_turn_degrees(bones[0], matrices[0])
            rows.append((strength, turn, 180.0 * strength / len(bones)))

        violations = [
            f"strength {s:g} turned {turn:.9f} deg, expected {expected:.9f}"
            for s, turn, expected in rows
            if abs(turn - expected) > 1e-9
        ]

        write_artifact(
            "amplitude_strength_scaling.txt",
            [
                "# free chain (stiffness 0), per-bone turn is linear in strength",
                "# columns: strength | measured_deg | expected_deg",
                *(f"{_fmt(s)} {_fmt(t)} {_fmt(e)}" for s, t, e in rows),
                "",
                "# violations",
                *([f"  {v}" for v in violations] or ["  none"]),
            ],
        )

        assert not violations, "; ".join(violations)


class TestBranchedTreeAtFullWind:

    TREE = (
        [1.0, 0.85, 0.6, 0.35],
        [
            (1, 60.0, [0.5, 0.25]),
            (2, -60.0, [0.5, 0.25]),
            (3, 25.0, [0.4, 0.15]),
            (3, -25.0, [0.4, 0.15]),
        ],
    )

    def _posed(self, bones, rig, angle, strength):
        matrices = pose_rig(rig, angle, strength)
        return matrices, np.array(
            [
                np.asarray(matrices[i])[:3, :3] @ bone.rest_position
                + np.asarray(matrices[i])[:3, 3]
                for i, bone in enumerate(bones)
            ]
        )

    def test_every_chain_responds_at_full_wind(self):
        trunk_stiffness, branches = self.TREE
        bones = tree_bones(trunk_stiffness, branches)
        rig = rig_from_bones(bones)

        assert not validate_rig(rig)

        rest_matrices, rest = self._posed(bones, rig, 0.0, 0.0)
        matrices, posed = self._posed(bones, rig, 0.0, 1.0)

        chains = chains_of(bones)
        per_chain = {}
        for stem_id, indices in chains.items():
            drift = float(np.abs(posed[indices] - rest[indices]).max())
            downwind = float((posed[indices] - rest[indices])[:, 0].min())
            per_chain[stem_id] = (drift, downwind)

        violations = []

        for stem_id, (drift, downwind) in sorted(per_chain.items()):
            softest = min(bones[i].stiffness for i in chains[stem_id])
            if softest < 1.0 and drift <= 1e-6:
                violations.append(
                    f"stem {stem_id} has a bone at stiffness {softest:g} but did not move"
                )
            if downwind < -1e-6:
                violations.append(
                    f"stem {stem_id} moved {abs(downwind):.6f} upwind of its rest pose"
                )

        out_of_plane = float(np.abs(posed[:, 1]).max())
        if out_of_plane > 1e-12:
            violations.append(f"planar tree left the XZ plane by {out_of_plane:.3e}")

        if float(posed[:, 2].min()) < -1e-6:
            violations.append(f"a joint sank to Z={float(posed[:, 2].min()):.6f}")

        curves = [
            (line, "#bbbbbb", 3.0, 1.0) for line in chain_polylines(bones, rest_matrices)
        ] + [(line, "#08306b", 3.5, 1.0) for line in chain_polylines(bones, matrices)]

        write_svg(
            "branched_full_wind.svg",
            curves,
            "branched tree at full wind (rest grey, posed dark)",
            joint_radius=3.5,
            labels=[
                (f"trunk {trunk_stiffness}", "#333333"),
                (f"{len(chains)} chains, {len(bones)} bones, wind +X", "#777777"),
            ],
        )

        write_artifact(
            "branched_full_wind.txt",
            [
                "# branched tree, wind +X at full strength",
                f"# trunk stiffness {trunk_stiffness}",
                f"# branches (attach, tilt_deg, stiffness) {branches}",
                "",
                f"out_of_plane_y {_fmt(out_of_plane)}",
                f"lowest_joint_z {_fmt(float(posed[:, 2].min()))}",
                "",
                "# columns: stem | bones | softest_stiffness | max_drift | min_dx",
                *(
                    f"{stem_id:5d} {len(chains[stem_id]):5d} "
                    f"{_fmt(min(bones[i].stiffness for i in chains[stem_id]))} "
                    f"{_fmt(per_chain[stem_id][0])} {_fmt(per_chain[stem_id][1])}"
                    for stem_id in sorted(chains)
                ),
                "",
                "# violations",
                *([f"  {v}" for v in violations] or ["  none"]),
            ],
        )

        assert not violations, "; ".join(violations)

    def test_branch_response_depends_on_angle_to_wind(self):
        trunk = [1.0, 1.0, 1.0]
        tilts = (0.0, 25.0, 45.0, 70.0, 90.0, 110.0, 135.0, 180.0)
        branches = [(3, tilt, [0.3, 0.1]) for tilt in tilts]

        bones = tree_bones(trunk, branches)
        rig = rig_from_bones(bones)

        rest_matrices, rest = self._posed(bones, rig, 0.0, 0.0)
        matrices, posed = self._posed(bones, rig, 0.0, 1.0)

        chains = chains_of(bones)
        rows = []
        for stem_id in sorted(chains):
            if stem_id == 0:
                continue
            indices = chains[stem_id]
            direction = bones[indices[0]].rest_direction
            leverage = float(np.linalg.norm(np.cross(direction, [1.0, 0.0, 0.0])))
            drift = float(np.abs(posed[indices] - rest[indices]).max())
            rows.append((tilts[stem_id - 1], leverage, drift))

        violations = []

        trunk_drift = float(np.abs(posed[chains[0]] - rest[chains[0]]).max())
        if trunk_drift > 1e-12:
            violations.append(f"rigid trunk moved {trunk_drift:.3e}")

        for tilt, leverage, drift in rows:
            if leverage < 1e-9 and drift > 1e-6:
                violations.append(
                    f"branch at {tilt:g} deg is parallel to the wind but moved {drift:.6f}"
                )

        curves = [
            (line, "#bbbbbb", 2.5, 1.0) for line in chain_polylines(bones, rest_matrices)
        ] + [(line, "#d62728", 3.0, 1.0) for line in chain_polylines(bones, matrices)]

        write_svg(
            "branch_angle_response.svg",
            curves,
            "branches fanned 0-180 deg, rigid trunk, full wind +X",
            joint_radius=3.0,
            labels=[("rest grey, posed red", "#777777")],
        )

        write_artifact(
            "branch_angle_response.txt",
            [
                "# branch response versus angle to the wind, full strength +X",
                "# rigid trunk; every branch has identical stiffness [0.3, 0.1]",
                f"# antiparallel easing band {ANTIPARALLEL_BAND:g}",
                "",
                "# columns: tilt_deg | leverage=|dir x wind| | max_drift",
                *(f"{_fmt(t)} {_fmt(l)} {_fmt(d)}" for t, l, d in rows),
                "",
                "# violations",
                *([f"  {v}" for v in violations] or ["  none"]),
            ],
        )

        assert not violations, "; ".join(violations)

    def test_mirrored_branches_respond_asymmetrically(self):
        bones = tree_bones([1.0, 0.8, 0.5], [(2, 70.0, [0.4, 0.15]), (2, -70.0, [0.4, 0.15])])
        rig = rig_from_bones(bones)

        rest_matrices, rest = self._posed(bones, rig, 0.0, 0.0)
        posed_matrices, posed = self._posed(bones, rig, 0.0, 1.0)

        write_svg(
            "mirror_rest.svg",
            [(line, "#666666", 4.0, 1.0) for line in chain_polylines(bones, rest_matrices)],
            "before: trunk with one branch right, one left",
            joint_radius=4.5,
            labels=[("no wind", "#777777")],
        )

        write_svg(
            "mirror_full_wind.svg",
            [(line, "#bbbbbb", 3.0, 1.0) for line in chain_polylines(bones, rest_matrices)]
            + [(line, "#08306b", 4.0, 1.0) for line in chain_polylines(bones, posed_matrices)],
            "after: full wind blowing right (+X)",
            joint_radius=4.5,
            labels=[
                ("grey = rest, blue = full wind", "#777777"),
                ("the upwind (left) branch folds much further", "#08306b"),
            ],
        )

        chains = chains_of(bones)
        downwind_tip = posed[chains[1]][-1]
        upwind_tip = posed[chains[2]][-1]
        downwind_drift = float(np.abs(posed[chains[1]] - rest[chains[1]]).max())
        upwind_drift = float(np.abs(posed[chains[2]] - rest[chains[2]]).max())

        violations = []
        if downwind_drift <= 1e-6 or upwind_drift <= 1e-6:
            violations.append("a mirrored branch did not respond at all")
        if (posed[chains[1]] - rest[chains[1]])[:, 0].min() < -1e-6:
            violations.append("the downwind branch travelled upwind")
        if (posed[chains[2]] - rest[chains[2]])[:, 0].min() < -1e-6:
            violations.append("the upwind branch travelled upwind")

        downwind_travel = float(downwind_tip[0]) - float(rest[chains[1]][-1][0])
        upwind_travel = float(upwind_tip[0]) - float(rest[chains[2]][-1][0])

        if upwind_travel <= downwind_travel:
            violations.append(
                f"the upwind tip travelled {upwind_travel:.6f} along the wind, no more "
                f"than the downwind tip's {downwind_travel:.6f}; a branch already lying "
                f"near the wind axis should have the least leverage left"
            )

        if upwind_travel <= 0.0:
            violations.append("the upwind branch tip was not carried toward the wind")

        write_artifact(
            "branch_mirror_asymmetry.txt",
            [
                "# mirrored branches at +/-70 deg under the same wind",
                "# both are pushed downwind, but not by equal amounts:",
                "# leverage and parent composition differ once they tilt",
                "",
                f"downwind_branch_drift {_fmt(downwind_drift)}",
                f"upwind_branch_drift {_fmt(upwind_drift)}",
                "",
                "# columns: branch | rest_tip_x | posed_tip_x | travel",
                f"downwind {_fmt(float(rest[chains[1]][-1][0]))} "
                f"{_fmt(float(downwind_tip[0]))} "
                f"{_fmt(float(downwind_tip[0]) - float(rest[chains[1]][-1][0]))}",
                f"  upwind {_fmt(float(rest[chains[2]][-1][0]))} "
                f"{_fmt(float(upwind_tip[0]))} "
                f"{_fmt(float(upwind_tip[0]) - float(rest[chains[2]][-1][0]))}",
                "",
                "# violations",
                *([f"  {v}" for v in violations] or ["  none"]),
            ],
        )

        assert not violations, "; ".join(violations)


class TestWindPowerRamp:

    STRENGTHS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    RAMP_COLOURS = (
        "#bbbbbb",
        "#c6dbef",
        "#9ecae1",
        "#4292c6",
        "#2171b5",
        "#08306b",
    )

    TREES = {
        "sapling": ([1.0, 0.8, 0.5, 0.2], [(2, 55.0, [0.5, 0.2]), (3, -50.0, [0.4, 0.1])]),
        "stiff_oak": ([1.0, 0.95, 0.9, 0.85], [(2, 60.0, [0.9, 0.8]), (3, -60.0, [0.9, 0.8])]),
        "willow": ([0.9, 0.6, 0.3, 0.05], [(2, 70.0, [0.2, 0.05]), (3, -65.0, [0.15, 0.0])]),
        "bare_trunk": ([1.0, 0.7, 0.4, 0.1], []),
    }

    @pytest.mark.parametrize("label", sorted(TREES))
    def test_power_ramp_renders_node_configuration(self, label):
        trunk_stiffness, branches = self.TREES[label]
        bones = tree_bones(trunk_stiffness, branches)
        rig = rig_from_bones(bones)

        assert not validate_rig(rig)

        curves = []
        rows = []
        for strength, colour in zip(self.STRENGTHS, self.RAMP_COLOURS):
            matrices = pose_rig(rig, 0.0, strength)
            width = 3.5 if strength in (0.0, 1.0) else 2.5
            for polyline in chain_polylines(bones, matrices):
                curves.append((polyline, colour, width, 1.0))

            positions = np.array(
                [
                    np.asarray(matrices[i])[:3, :3] @ bone.rest_position
                    + np.asarray(matrices[i])[:3, 3]
                    for i, bone in enumerate(bones)
                ]
            )
            rows.append((strength, positions))

        stems = len(chains_of(bones))
        write_svg(
            f"power_ramp_{label}.svg",
            curves,
            f"{label}  wind +X, power 0.0 (grey) to 1.0 (dark blue)",
            joint_radius=3.5,
            labels=[
                (f"trunk stiffness {trunk_stiffness}", "#333333"),
                (f"{len(bones)} bones in {stems} chains; circles are joints", "#777777"),
            ],
        )

        rest = rows[0][1]
        violations = []

        if float(np.abs(rows[0][1] - np.array([b.rest_position for b in bones])).max()) > 1e-12:
            violations.append("zero power does not reproduce the rest configuration")

        drifts = [float(np.abs(positions - rest).max()) for _, positions in rows]
        for index in range(len(rows) - 1):
            if drifts[index + 1] < drifts[index] - 1e-9:
                violations.append(
                    f"power {self.STRENGTHS[index]:g} -> {self.STRENGTHS[index + 1]:g} "
                    f"reduced total displacement ({drifts[index]:.6f} -> {drifts[index + 1]:.6f})"
                )

        for strength, positions in rows:
            downwind = float((positions - rest)[:, 0].min())
            if downwind < -1e-6:
                violations.append(
                    f"power {strength:g} pushed a joint {abs(downwind):.6f} upwind"
                )
            sunk = float(positions[:, 2].min())
            if sunk < -1e-6:
                violations.append(f"power {strength:g} drove a joint to Z={sunk:.6f}")

        write_artifact(
            f"power_ramp_{label}.txt",
            [
                f"# wind power ramp: {label}",
                f"# trunk stiffness {trunk_stiffness}",
                f"# branches (attach, tilt_deg, stiffness) {branches}",
                f"# {len(bones)} bones in {stems} chains, wind along +X",
                "",
                "# columns: power | max_drift | trunk_tip_x | trunk_tip_z | lowest_z",
                *(
                    f"{_fmt(strength)} {_fmt(drifts[index])} "
                    f"{_fmt(positions[len(trunk_stiffness) - 1][0])} "
                    f"{_fmt(positions[len(trunk_stiffness) - 1][2])} "
                    f"{_fmt(float(positions[:, 2].min()))}"
                    for index, (strength, positions) in enumerate(rows)
                ),
                "",
                "# violations",
                *([f"  {v}" for v in violations] or ["  none"]),
            ],
        )

        assert not violations, "; ".join(violations)


class TestMorphTargets:

    @pytest.mark.parametrize("directions", [4, 8, 16])
    def test_morph_set_spans_every_direction(self, directions):
        bones = chain_bones_with_stiffness([1.0, 0.6, 0.3, 0.0])
        rig = rig_from_bones(bones)
        rest = np.array([bone.rest_position for bone in bones])

        morphs = bake_morph_targets(rig, rest, directions)

        assert len(morphs) == directions

        deltas = [posed - rest for _, posed in morphs]
        magnitudes = [float(np.linalg.norm(delta, axis=1).max()) for delta in deltas]

        azimuths = [
            float(np.degrees(np.arctan2(delta[-1][1], delta[-1][0])) % 360.0)
            for delta in deltas
        ]

        spacing = 360.0 / directions
        violations = []

        for index, magnitude in enumerate(magnitudes):
            if magnitude <= 1e-6:
                violations.append(f"morph {index} is identical to rest")

        spread = max(magnitudes) - min(magnitudes)
        if spread > 1e-6 * max(magnitudes):
            violations.append(
                f"morph magnitudes vary by {spread:.6f}; wind is not isotropic"
            )

        for index, azimuth in enumerate(azimuths):
            expected = (index * spacing) % 360.0
            error = abs((azimuth - expected + 180.0) % 360.0 - 180.0)
            if error > 1e-6:
                violations.append(
                    f"morph {index} pushes toward {azimuth:.4f} deg, expected {expected:.4f}"
                )

        pairs = [
            float(np.abs(deltas[i] - deltas[j]).max())
            for i in range(directions)
            for j in range(i + 1, directions)
        ]
        if min(pairs) <= 1e-9:
            violations.append("two morph targets are duplicates")

        write_artifact(
            f"morph_set_{directions:02d}.txt",
            [
                f"# morph set: {directions} directions at full strength",
                f"# {rig.bone_count} bones, spacing {spacing:g} deg",
                "",
                f"uniform_magnitude {_fmt(magnitudes[0])}",
                f"closest_pair_distance {_fmt(min(pairs))}",
                "",
                "# columns: morph | wind_deg | max_delta | tip_azimuth_deg",
                *(
                    f"{index:5d} {_fmt(angle)} {_fmt(magnitudes[index])} {_fmt(azimuths[index])}"
                    for index, (angle, _) in enumerate(morphs)
                ),
                "",
                "# violations",
                *([f"  {v}" for v in violations] or ["  none"]),
            ],
        )

        assert not violations, "; ".join(violations)

    def test_morphs_are_deltas_against_a_stable_rest(self):
        bones = chain_bones_with_stiffness([1.0, 0.6, 0.3, 0.0])
        rig = rig_from_bones(bones)
        rest = np.array([bone.rest_position for bone in bones])

        zero = deform_vertices(rig, rest, pose_rig(rig, 0.0, 0.0))
        drift = float(np.abs(zero - rest).max())

        first = bake_morph_targets(rig, rest, 8)
        second = bake_morph_targets(rig, rest, 8)
        rebake_difference = max(
            float(np.abs(a[1] - b[1]).max()) for a, b in zip(first, second)
        )

        violations = []
        if drift > 1e-12:
            violations.append(f"zero-strength pose drifts {drift:.3e} from rest")
        if rebake_difference > 0.0:
            violations.append(f"baking twice differed by {rebake_difference:.3e}")

        write_artifact(
            "morph_rest_stability.txt",
            [
                "# morph targets are deltas against an unmoved rest pose",
                "",
                f"zero_strength_drift {_fmt(drift)}",
                f"rebake_difference {_fmt(rebake_difference)}",
                "",
                "# violations",
                *([f"  {v}" for v in violations] or ["  none"]),
            ],
        )

        assert not violations, "; ".join(violations)

    def test_morph_fan_is_visually_even(self):
        bones = chain_bones_with_stiffness([1.0, 0.6, 0.3, 0.0])
        rig = rig_from_bones(bones)

        curves = [(joint_polyline(bones, pose_rig(rig, 0.0, 0.0)), "#bbbbbb", 3.0, 1.0)]

        for angle, colour in ((0.0, "#1f77b4"), (180.0, "#d62728")):
            curves.append((joint_polyline(bones, pose_rig(rig, angle, 1.0)), colour, 3.5, 1.0))

        for strength in (0.25, 0.5, 0.75):
            curves.append(
                (joint_polyline(bones, pose_rig(rig, 0.0, strength)), "#1f77b4", 2.0, 0.45)
            )

        write_svg(
            "morph_fan.svg",
            curves,
            "wind 0 deg (blue) vs 180 deg (red), with 0.25/0.5/0.75 strength",
        )


class TestDirectionSymmetry:

    def test_opposing_and_wrapped_directions(self):
        bones = chain_bones_with_stiffness([1.0, 0.6, 0.3, 0.0])
        rig = rig_from_bones(bones)
        rest = np.array([bone.rest_position for bone in bones])

        violations = []
        rows = []

        for angle in (0.0, 30.0, 90.0, 210.0):
            forward = deform_vertices(rig, rest, pose_rig(rig, angle, 1.0))
            opposed = deform_vertices(rig, rest, pose_rig(rig, angle + 180.0, 1.0))
            wrapped = deform_vertices(rig, rest, pose_rig(rig, angle + 360.0, 1.0))

            wrap_error = float(np.abs(wrapped - forward).max())

            mirrored = (forward - rest) + (opposed - rest)
            mirror_error = float(np.abs(mirrored[:, :2]).max())
            height_error = float(
                np.abs((forward - rest)[:, 2] - (opposed - rest)[:, 2]).max()
            )

            rows.append((angle, wrap_error, mirror_error, height_error))

            if wrap_error > 1e-9:
                violations.append(
                    f"{angle:g} and {angle + 360:g} deg differ by {wrap_error:.3e}"
                )
            if mirror_error > 1e-9:
                violations.append(
                    f"{angle:g} and {angle + 180:g} deg are not horizontal mirrors "
                    f"({mirror_error:.3e})"
                )
            if height_error > 1e-9:
                violations.append(
                    f"{angle:g} and {angle + 180:g} deg drop by different amounts "
                    f"({height_error:.3e})"
                )

        write_artifact(
            "direction_symmetry.txt",
            [
                "# wind direction symmetry on a 4-bone chain at full strength",
                "",
                "# columns: angle_deg | wrap_360_error | mirror_180_xy_error | mirror_180_z_error",
                *(f"{_fmt(a)} {_fmt(w)} {_fmt(m)} {_fmt(h)}" for a, w, m, h in rows),
                "",
                "# violations",
                *([f"  {v}" for v in violations] or ["  none"]),
            ],
        )

        assert not violations, "; ".join(violations)
