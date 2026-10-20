"""Tests for foliage vertex stream computation.

The streams these functions produce are as invisible as a bone rig: wrong values
still yield plausible geometry, and the defect only appears once wind runs in the
consuming engine. So each test writes a deterministic artifact to
``tests/artifacts/foliage_streams/`` for inspection rather than relying on a
pass/fail tick alone.
"""

from pathlib import Path

import numpy as np
import pytest
from foliage_streams import (
    ROOTED_EPSILON,
    WIND_REACH_FACTOR,
    compute_all_streams,
    compute_along_branch_position,
    compute_blade_height,
    compute_foliage_colors,
    compute_phase,
    compute_stiffness,
    compute_wind_expanded_bounds,
    pack_height_uv,
    pack_wind_uv,
)

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "foliage_streams"

# Fixed decimal precision so last-bit noise cannot produce a spurious diff.
PRECISION = 6


def _fmt(value: float) -> str:
    """Format a float at fixed precision, normalising negative zero."""
    formatted = f"{value:.{PRECISION}f}"
    return (
        "0." + "0" * PRECISION
        if formatted.startswith("-0.0") and float(formatted) == 0
        else formatted
    )


def write_artifact(name: str, lines: list[str]) -> None:
    """Write a deterministic snapshot artifact."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_synthetic_tree():
    """Build a known three-branch tree without the native mesher.

    Branch 0 is a vertical trunk of length 4 and radius 0.5. Branches 1 and 2 are
    thinner side branches of length 2 and 1, rooted partway up the trunk and
    running along +X and +Y. Each branch contributes vertices sampled evenly
    along its own chain, mimicking the mesher's ring-per-section output.

    Returns a dict of the arrays the stream functions consume.
    """
    branches = [
        # (stem_id, depth, base, direction, extent, radius, n_sections)
        (0, 0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 4.0, 0.5, 5),
        (1, 1, np.array([0.0, 0.0, 2.0]), np.array([1.0, 0.0, 0.0]), 2.0, 0.2, 4),
        (2, 2, np.array([0.0, 0.0, 3.0]), np.array([0.0, 1.0, 0.0]), 1.0, 0.05, 3),
    ]

    positions, stem_ids, extents, radii, pivots, directions, depths = [], [], [], [], [], [], []

    for stem_id, depth, base, direction, extent, radius, n_sections in branches:
        for section in range(n_sections):
            t = section / (n_sections - 1)
            # Two vertices per section on opposite sides of the branch, so the
            # radial offset that a real ring has is represented.
            perpendicular = np.array([0.0, 1.0, 0.0]) if direction[2] else np.array([0.0, 0.0, 1.0])
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


class TestStreamSweep:
    """T1: sweep along-branch position and check every derived stream."""

    def test_streams_across_position_sweep(self):
        """All position-derived streams behave correctly across the full range.

        Covers M3 (tint ramp linearity), M4 (blade height scale) and M5
        (position clamping) in one sweep rather than one test per channel.
        """
        # Sweep t beyond [0, 1] on both ends to exercise clamping.
        raw_t = np.linspace(-0.5, 1.5, 21)
        extent = 3.0
        base = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])

        positions = np.array([base + direction * extent * t for t in raw_t])
        pivots = np.tile(base, (len(raw_t), 1))
        dirs = np.tile(direction, (len(raw_t), 1))
        extents = np.full(len(raw_t), extent)

        along = compute_along_branch_position(positions, pivots, dirs, extents)

        # M5: never escapes [0, 1] regardless of input.
        assert np.all(along >= 0.0)
        assert np.all(along <= 1.0)

        # Within range the mapping is exactly the parametric position.
        in_range = (raw_t >= 0.0) & (raw_t <= 1.0)
        assert along[in_range] == pytest.approx(raw_t[in_range])

        colors = compute_foliage_colors(
            along, occlusion_base=0.25, translucency_tip=0.8, normal_blend=0.3
        )
        alpha = colors[3::4]
        red = colors[0::4]
        green = colors[1::4]
        blue = colors[2::4]

        # M3: alpha is the raw position, not a reshaped curve.
        assert alpha == pytest.approx(along)
        # Channels are independent scalars with the documented endpoints.
        assert red == pytest.approx(0.25 + 0.75 * along)
        assert green == pytest.approx(0.8 * along)
        assert blue == pytest.approx(np.full(len(along), 0.3))

        # M4: blade height is metres, passed through unscaled.
        heights = compute_blade_height(extents)
        assert heights == pytest.approx(extents)

        lines = [
            "# foliage stream sweep along a single branch",
            f"# extent={_fmt(extent)}m occlusion_base=0.250000 translucency_tip=0.800000 normal_blend=0.300000",
            "",
            f"{'raw_t':>10} {'along':>10} {'color.r':>10} {'color.g':>10} {'color.b':>10} {'color.a':>10} {'height_m':>10}",
        ]
        for i, t in enumerate(raw_t):
            lines.append(
                f"{_fmt(t):>10} {_fmt(along[i]):>10} {_fmt(red[i]):>10} "
                f"{_fmt(green[i]):>10} {_fmt(blue[i]):>10} {_fmt(alpha[i]):>10} {_fmt(heights[i]):>10}"
            )

        lines += [
            "",
            "# invariants",
            f"along_min={_fmt(along.min())} along_max={_fmt(along.max())}",
            f"clamped_low={int(np.sum(raw_t < 0.0))} clamped_high={int(np.sum(raw_t > 1.0))}",
        ]
        write_artifact("sweep.txt", lines)


class TestPerBladeInvariants:
    """T2: invariants that hold per branch chain on a whole tree."""

    def test_tree_invariants(self):
        """Phase is constant per blade, roots are rigid, bounds cover wind reach.

        Covers M1 (phase constancy), M2 (rooted stiffness) and M6 (bounds).
        """
        tree = make_synthetic_tree()
        streams = compute_all_streams(**tree)

        stem_ids = tree["stem_ids"]
        phase = streams["phase"]
        stiffness = streams["stiffness"]
        along = streams["along_branch"]

        lines = [
            "# per-blade invariants on the synthetic tree",
            f"# vertices={len(stem_ids)}",
            "",
            f"{'stem_id':>8} {'verts':>6} {'phase':>10} {'extent_m':>10} {'radius':>8} {'stiff_min':>10} {'stiff_max':>10} {'along_min':>10} {'along_max':>10}",
        ]

        # Canonical sort order so the artifact diffs cleanly.
        for stem_id in sorted(set(stem_ids.tolist())):
            mask = stem_ids == stem_id
            blade_phase = phase[mask]

            # M1: a gradient within one blade tears it apart under wind.
            assert blade_phase.min() == pytest.approx(blade_phase.max()), (
                f"phase varies within blade {stem_id}"
            )

            # M2: only the ground-rooted trunk base is pinned rigid. A side
            # branch's base is a joint, not a root, and pinning it there made
            # thin high branches stiffer than the trunk carrying them.
            chain_base = mask & (along <= ROOTED_EPSILON)
            assert np.any(chain_base), f"blade {stem_id} has no base vertex"

            is_trunk = tree["hierarchy_depths"][mask][0] == 0
            if is_trunk:
                assert np.all(stiffness[chain_base] == 1.0), (
                    "trunk root must be exactly rigid or the tree drifts out of the ground"
                )
            else:
                assert np.all(stiffness[chain_base] < 1.0), (
                    f"branch {stem_id} base is pinned rigid; it is a joint, not a root"
                )

            lines.append(
                f"{int(stem_id):>8} {int(mask.sum()):>6} {_fmt(blade_phase[0]):>10} "
                f"{_fmt(tree['branch_extents'][mask][0]):>10} {_fmt(tree['radii'][mask][0]):>8} "
                f"{_fmt(stiffness[mask].min()):>10} {_fmt(stiffness[mask].max()):>10} "
                f"{_fmt(along[mask].min()):>10} {_fmt(along[mask].max()):>10}"
            )

        # Distinct blades must not collide onto one phase, or they sway in lockstep.
        blade_phases = [phase[stem_ids == s][0] for s in sorted(set(stem_ids.tolist()))]
        assert len(set(np.round(blade_phases, 6))) == len(blade_phases)

        # Thicker wood is never less rigid than thinner wood.
        trunk = stem_ids == 0
        twig = stem_ids == 2
        assert stiffness[trunk].min() >= stiffness[twig].max()

        # M6: bounds expand horizontally by the wind reach of the tallest blade.
        min_corner, max_corner = streams["bounds"]
        verts = tree["vertex_positions"]
        reach = streams["blade_height"].max() * WIND_REACH_FACTOR
        assert min_corner[:2] == pytest.approx(verts.min(axis=0)[:2] - reach)
        assert max_corner[:2] == pytest.approx(verts.max(axis=0)[:2] + reach)
        # Vertical bounds are untouched; wind reach is horizontal.
        assert min_corner[2] == pytest.approx(verts.min(axis=0)[2])
        assert max_corner[2] == pytest.approx(verts.max(axis=0)[2])

        lines += [
            "",
            "# bounds",
            f"rest_min=({_fmt(verts.min(axis=0)[0])}, {_fmt(verts.min(axis=0)[1])}, {_fmt(verts.min(axis=0)[2])})",
            f"rest_max=({_fmt(verts.max(axis=0)[0])}, {_fmt(verts.max(axis=0)[1])}, {_fmt(verts.max(axis=0)[2])})",
            f"wind_reach={_fmt(reach)}",
            f"expanded_min=({_fmt(min_corner[0])}, {_fmt(min_corner[1])}, {_fmt(min_corner[2])})",
            f"expanded_max=({_fmt(max_corner[0])}, {_fmt(max_corner[1])}, {_fmt(max_corner[2])})",
        ]
        write_artifact("tree_invariants.txt", lines)

    def test_packed_streams_interleave_correctly(self):
        """Packed UV streams keep channel order and length."""
        tree = make_synthetic_tree()
        streams = compute_all_streams(**tree)

        wind_uv = streams["wind_uv"]
        height_uv = streams["height_uv"]
        count = len(tree["stem_ids"])

        assert len(wind_uv) == count * 2
        assert len(height_uv) == count * 2
        assert wind_uv[0::2] == pytest.approx(streams["stiffness"])
        assert wind_uv[1::2] == pytest.approx(streams["phase"])
        assert height_uv[0::2] == pytest.approx(streams["blade_height"])
        # The contract leaves this channel unused; it must be zero, not garbage.
        assert np.all(height_uv[1::2] == 0.0)

        assert pack_wind_uv(np.array([]), np.array([])).size == 0
        assert pack_height_uv(np.array([])).size == 0


class TestDegenerateInput:
    """T3: degenerate geometry must not produce NaN or escape range."""

    @pytest.mark.parametrize(
        "label,extent,direction",
        [
            ("zero_extent", 0.0, np.array([0.0, 0.0, 1.0])),
            ("sub_millimetre_extent", 1e-6, np.array([0.0, 0.0, 1.0])),
            ("zero_direction", 2.0, np.array([0.0, 0.0, 0.0])),
            ("negative_extent", -1.0, np.array([0.0, 0.0, 1.0])),
        ],
    )
    def test_degenerate_branches_stay_finite(self, label, extent, direction):
        """M7: no NaN, no out-of-range values, from degenerate branches."""
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        pivots = np.zeros((3, 3))
        dirs = np.tile(direction, (3, 1))
        extents = np.full(3, extent)

        along = compute_along_branch_position(positions, pivots, dirs, extents)
        assert np.all(np.isfinite(along))
        assert np.all((along >= 0.0) & (along <= 1.0))

        stiffness = compute_stiffness(np.full(3, 0.1), along)
        assert np.all(np.isfinite(stiffness))
        assert np.all((stiffness >= 0.0) & (stiffness <= 1.0))

        # Height is a displacement bound; a negative one would invert the clamp.
        heights = compute_blade_height(extents)
        assert np.all(heights >= 0.0)

    def test_empty_input(self):
        """Empty meshes produce empty streams, not exceptions."""
        empty3 = np.zeros((0, 3))
        empty1 = np.zeros(0)

        assert compute_along_branch_position(empty3, empty3, empty3, empty1).size == 0
        assert compute_stiffness(empty1, empty1).size == 0
        assert compute_phase(empty1).size == 0
        assert compute_blade_height(empty1).size == 0
        assert compute_foliage_colors(empty1).size == 0

        min_corner, max_corner = compute_wind_expanded_bounds(empty3, empty1)
        assert np.all(min_corner == 0.0)
        assert np.all(max_corner == 0.0)

    def test_uniform_radius_tree_still_pins_roots(self):
        """A tree of constant radius is fully rigid, and roots stay exactly 1.

        Guards the max_radius normalisation: dividing by the max makes every
        vertex 1.0 here, which is correct, but the rooted pin must still hold if
        the mapping is later changed.
        """
        along = np.array([0.0, 0.5, 1.0])
        stiffness = compute_stiffness(np.array([0.3, 0.3, 0.3]), along)
        assert stiffness == pytest.approx(np.array([1.0, 1.0, 1.0]))
        assert stiffness[0] == 1.0


class TestTrunkStiffnessControl:
    """T6: the trunk stiffness controls must actually reach the trunk."""

    def test_trunk_controls_bend_the_trunk(self):
        """M10: a lowered trunk tip must produce a gradient down the trunk.

        The default radius curve normalises against the tree's largest radius,
        which is the trunk, so an uncontrolled trunk sits at 1.0 everywhere and
        never bends. This is the regression guard for that.
        """
        tree = make_synthetic_tree()
        trunk = tree["hierarchy_depths"] == 0

        rigid = compute_all_streams(**tree)["stiffness"]
        # Without controls the trunk is uniformly rigid; that is the behaviour
        # the controls exist to override.
        assert rigid[trunk].min() == pytest.approx(1.0)

        bending = compute_all_streams(**tree, trunk_base_stiffness=1.0, trunk_tip_stiffness=0.2)[
            "stiffness"
        ]

        along = compute_all_streams(**tree)["along_branch"]
        trunk_tip = trunk & (along >= 1.0 - 1e-9)

        # The tip must actually be compliant now.
        assert bending[trunk_tip] == pytest.approx(0.2)
        # And the trunk must vary rather than sitting flat.
        assert bending[trunk].max() - bending[trunk].min() > 0.5

        # M11: the rooted ring stays exactly rigid even with a soft trunk.
        soft = compute_all_streams(**tree, trunk_base_stiffness=0.0, trunk_tip_stiffness=0.0)[
            "stiffness"
        ]
        rooted = trunk & (along <= ROOTED_EPSILON)
        assert np.all(soft[rooted] == 1.0)
        # Everything above the root is free to move.
        assert soft[trunk & ~rooted].max() == pytest.approx(0.0)

    def test_branch_scale_leaves_trunk_alone(self):
        """Branch scale must affect side branches without touching the trunk."""
        tree = make_synthetic_tree()
        trunk = tree["hierarchy_depths"] == 0
        side = ~trunk

        base = compute_all_streams(**tree, trunk_tip_stiffness=0.5)["stiffness"]
        scaled = compute_all_streams(**tree, trunk_tip_stiffness=0.5, branch_stiffness=0.25)[
            "stiffness"
        ]

        assert scaled[trunk] == pytest.approx(base[trunk])

        along = compute_all_streams(**tree)["along_branch"]
        movable_side = side & (along > ROOTED_EPSILON)
        assert np.all(scaled[movable_side] < base[movable_side])

    def test_stiffness_control_sweep(self):
        """Sweep the trunk controls and record the resulting profiles."""
        tree = make_synthetic_tree()
        trunk = tree["hierarchy_depths"] == 0
        along = compute_all_streams(**tree)["along_branch"]

        settings = [
            (1.0, 1.0, "rigid trunk, branches only"),
            (1.0, 0.6, "default bend toward crown"),
            (1.0, 0.2, "strong lean"),
            (0.5, 0.0, "whole trunk sways"),
        ]

        lines = [
            "# trunk stiffness control sweep",
            "# rooted vertices are pinned to exactly 1.0 in every configuration",
            "",
        ]

        for base_stiff, tip_stiff, label in settings:
            stiffness = compute_all_streams(
                **tree, trunk_base_stiffness=base_stiff, trunk_tip_stiffness=tip_stiff
            )["stiffness"]

            rooted = trunk & (along <= ROOTED_EPSILON)
            assert np.all(stiffness[rooted] == 1.0), f"{label}: root not pinned"
            assert np.all((stiffness >= 0.0) & (stiffness <= 1.0)), f"{label}: out of range"

            lines.append(f"## base={_fmt(base_stiff)} tip={_fmt(tip_stiff)}  # {label}")
            lines.append(f"{'along':>10} {'stiffness':>10}")

            # One row per distinct trunk height, in canonical order.
            for t in sorted(set(np.round(along[trunk], 6).tolist())):
                at_height = trunk & (np.round(along, 6) == t)
                lines.append(f"{_fmt(t):>10} {_fmt(stiffness[at_height][0]):>10}")
            lines.append("")

        write_artifact("stiffness_controls.txt", lines)


class TestJunctionContinuity:
    """T9: stiffness must not spike upward where a branch meets its parent."""

    def test_branch_base_is_not_stiffer_than_its_parent(self):
        """A thin branch high on the trunk must stay softer than the trunk.

        The rooted pin used to fire at the base of every chain, not just at the
        ground, so each side branch became fully rigid exactly where it joined
        its parent. The result was a hard spot at every junction and thin twigs
        reading as stiffer than the trunk carrying them.
        """
        tree = make_synthetic_tree()
        streams = compute_all_streams(**tree, trunk_tip_stiffness=0.5)

        stiffness = streams["stiffness"]
        along = streams["along_branch"]
        depths = tree["hierarchy_depths"]
        radii = tree["radii"]

        trunk_top = stiffness[(depths == 0) & (along >= 1.0 - 1e-9)]
        assert len(trunk_top) > 0

        for depth in (1, 2):
            branch = depths == depth
            if not np.any(branch):
                continue

            base = branch & (along <= ROOTED_EPSILON)
            assert np.any(base)

            # The branch is thinner than the trunk, so it must be less rigid.
            assert radii[branch][0] < radii[depths == 0][0]
            assert np.all(stiffness[base] < 1.0), f"depth {depth} branch base pinned rigid"
            assert stiffness[base].max() <= trunk_top.max() + 1e-9, (
                f"depth {depth} branch base is stiffer than the trunk top"
            )

    def test_junction_drop_is_gradual(self):
        """Neighbouring levels must move at comparable rates.

        The shader bends by ``sway**2``, so a large stiffness gap across a
        junction makes parent and child swing very differently and the branch
        whips around independently of the limb carrying it. Clamping each level
        to its parent's minimum satisfied the ordering rule but compounded the
        drops, collapsing trunk-to-twig within two junctions.
        """
        from foliage_streams import JUNCTION_FALLOFF

        tree = make_synthetic_tree()
        streams = compute_all_streams(**tree, trunk_tip_stiffness=0.4)
        stiffness = streams["stiffness"]
        depths = tree["hierarchy_depths"]

        levels = sorted(set(depths.tolist()))
        for depth in levels[1:]:
            parent = depths == (depth - 1)
            child = depths == depth
            if not np.any(parent) or not np.any(child):
                continue

            parent_min = stiffness[parent].min()
            child_min = stiffness[child].min()

            # Never stiffer than the parent: motion damps outward.
            assert stiffness[child].max() <= parent_min + 1e-9

            # Never drastically softer either: the ratio is what governs how
            # differently the two pieces of wood move.
            assert child_min >= parent_min * JUNCTION_FALLOFF - 1e-9, (
                f"depth {depth} drops to {child_min:.4f} from {parent_min:.4f}, "
                f"a {parent_min / max(child_min, 1e-9):.1f}x gap"
            )

    def test_no_upward_spike_along_any_chain(self):
        """Within a branch, stiffness must not rise toward the tip."""
        tree = make_synthetic_tree()
        streams = compute_all_streams(**tree, trunk_tip_stiffness=0.4)
        stiffness = streams["stiffness"]
        along = streams["along_branch"]
        stem_ids = tree["stem_ids"]

        lines = ["# stiffness along each chain, base to tip", ""]

        for stem_id in sorted(set(stem_ids.tolist())):
            mask = stem_ids == stem_id
            order = np.argsort(along[mask])
            profile = stiffness[mask][order]

            # Wood does not get more rigid toward its tip.
            assert np.all(np.diff(profile) <= 1e-9), (
                f"chain {stem_id} stiffness rises toward the tip: {profile.tolist()}"
            )

            lines.append(
                f"chain {int(stem_id)} depth {int(tree['hierarchy_depths'][mask][0])}: "
                + " ".join(_fmt(v) for v in profile)
            )

        write_artifact("junction_continuity.txt", lines)


class TestStiffnessDistribution:
    """T8: stiffness must span the usable range, not pile up at one end."""

    def test_radius_curve_spreads_across_range(self):
        """M18: a realistic radius spread must not collapse to near-zero.

        Real trees run a half-metre trunk against sub-millimetre twigs. Under a
        linear radius ratio almost every vertex lands in the bottom tenth, which
        makes the whole tree uniformly floppy and leaves no visible gradient
        between a thick limb and a thin twig.
        """
        # Log-spaced radii spanning the three orders of magnitude a real tree has.
        radii = np.geomspace(0.0002, 0.5, 400)
        along = np.full(len(radii), 0.5)  # nothing rooted, so no pinning

        stiffness = compute_stiffness(radii, along)

        hist, _ = np.histogram(stiffness, bins=10, range=(0.0, 1.0))
        occupied = int(np.count_nonzero(hist))

        # The distribution must reach across most of the range rather than
        # collapsing into one or two buckets.
        assert occupied >= 7, f"stiffness occupies only {occupied}/10 buckets: {hist.tolist()}"

        # No single bucket may swallow the mesh.
        assert hist.max() < 0.5 * len(radii), f"one bucket holds {hist.max()}/{len(radii)}"

        # Ordering must survive the curve: thicker wood is always stiffer.
        assert np.all(np.diff(stiffness) >= 0)
        assert stiffness[0] < stiffness[-1]
        assert stiffness[-1] == pytest.approx(1.0)

        lines = [
            "# stiffness across a log-spaced radius range",
            f"# radii {radii[0]:.6f} .. {radii[-1]:.6f}, {len(radii)} samples",
            "",
            f"{'radius':>12} {'stiffness':>12}",
        ]
        for i in range(0, len(radii), 40):
            lines.append(f"{_fmt(radii[i]):>12} {_fmt(stiffness[i]):>12}")
        lines += [
            "",
            f"histogram(10 bins): {hist.tolist()}",
            f"buckets occupied: {occupied}/10",
            f"median stiffness: {_fmt(float(np.median(stiffness)))}",
        ]
        write_artifact("stiffness_distribution.txt", lines)

    def test_branch_scale_still_applies(self):
        """The curve change must not break the branch scale control."""
        radii = np.geomspace(0.001, 0.5, 50)
        along = np.full(len(radii), 0.5)

        full = compute_stiffness(radii, along)
        scaled = compute_stiffness(radii, along, branch_stiffness=0.25)

        assert np.all(scaled <= full)
        assert scaled.max() == pytest.approx(0.25 * full.max())


class TestTranslucency:
    """T7: wood must not transmit light; leaves must."""

    def test_wood_defaults_to_opaque(self):
        """M12: a default branch export writes zero translucency everywhere.

        Trunks and branches are opaque. A non-zero green channel makes twig tips
        glow when back-lit, which only shows up under specific lighting and so
        survives a casual look at the export.
        """
        tree = make_synthetic_tree()
        streams = compute_all_streams(**tree)
        translucency = streams["colors"][1::4]

        assert np.all(translucency == 0.0)

        # The other channels must still carry their gradients — this is a
        # translucency fix, not a flattening of vertex colour.
        occlusion = streams["colors"][0::4]
        tint = streams["colors"][3::4]
        assert occlusion.max() > occlusion.min()
        assert tint.max() > tint.min()

        # Raising it stays possible for genuinely thin twigs.
        lit = compute_all_streams(**tree, translucency_tip=0.8)["colors"][1::4]
        assert lit.max() == pytest.approx(0.8 * streams["along_branch"].max())

    def test_leaf_colors_transmit_by_default(self):
        """Leaves invert the wood defaults: translucent and normal-flattened."""
        from foliage_streams import compute_leaf_colors

        position = np.linspace(0.0, 1.0, 9)
        colors = compute_leaf_colors(position)

        translucency = colors[1::4]
        normal_blend = colors[2::4]

        # Transmission rises to the thin tip rather than sitting at zero.
        assert translucency[0] == pytest.approx(0.0)
        assert translucency[-1] == pytest.approx(1.0)
        assert np.all(np.diff(translucency) > 0)

        # Canopy leaves flatten their normals so the canopy lights as a volume.
        assert np.all(normal_blend == 1.0)

        # Tint ramp stays the honest petiole-to-tip position.
        assert colors[3::4] == pytest.approx(position)

    def test_leaf_blade_position(self):
        """M13: petiole-to-tip parameter is derived from leaf geometry alone."""
        from foliage_streams import compute_leaf_blade_position

        # A leaf lying along +Y, petiole at the origin.
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.2, 0.5, 0.0],
            ]
        )
        position = compute_leaf_blade_position(verts)

        assert position.min() == pytest.approx(0.0)
        assert position.max() == pytest.approx(1.0)
        # The petiole vertex is the zero point, the far tip the one point.
        assert position[0] == pytest.approx(0.0)
        assert position[2] == pytest.approx(1.0)
        assert np.all((position >= 0.0) & (position <= 1.0))

        # An explicit attachment point overrides the inferred one.
        explicit = compute_leaf_blade_position(verts, attachment_points=np.array([0.0, 1.0, 0.0]))
        assert explicit[2] == pytest.approx(0.0)

        # Degenerate leaves stay finite instead of dividing by zero.
        single = compute_leaf_blade_position(np.array([[1.0, 1.0, 1.0]]))
        assert single.shape == (1,)
        assert np.all(np.isfinite(single))
        assert compute_leaf_blade_position(np.zeros((0, 3))).size == 0

    def test_cards_in_the_tree_mesh_take_leaf_colors(self):
        from foliage_streams import compute_leaf_vertex_mask

        tree = make_synthetic_tree()
        wood_count = len(tree["vertex_positions"])

        twig_base = np.array([1.5, 0.0, 2.0])
        twig_direction = np.array([0.0, 0.0, 1.0])
        attachments = [twig_base + twig_direction * 0.2, twig_base + twig_direction * 0.6]
        sizes = (0.2, 0.05)

        card_positions, card_attachments = [], []
        for attachment, size in zip(attachments, sizes, strict=True):
            half = size / 2.0
            centre = attachment + np.array([0.05, 0.0, 0.0])
            for sx, sz in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
                card_positions.append(centre + np.array([0.0, sx * half, sz * half]))
                card_attachments.append(attachment)
            for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                card_positions.append(centre + np.array([sx * half, 0.0, sz * half]))
                card_attachments.append(attachment)

        card_count = len(card_positions)
        arrays = {
            "vertex_positions": np.vstack([tree["vertex_positions"], np.array(card_positions)]),
            "stem_ids": np.concatenate([tree["stem_ids"], np.full(card_count, 3.0)]),
            "branch_extents": np.concatenate([tree["branch_extents"], np.full(card_count, 0.8)]),
            "radii": np.concatenate([tree["radii"], np.full(card_count, 0.05)]),
            "pivot_positions": np.vstack(
                [tree["pivot_positions"], np.tile(twig_base, (card_count, 1))]
            ),
            "directions": np.vstack([tree["directions"], np.tile(twig_direction, (card_count, 1))]),
            "hierarchy_depths": np.concatenate(
                [tree["hierarchy_depths"], np.full(card_count, 2.0)]
            ),
        }
        total = wood_count + card_count

        quads = int(card_count / 4)
        loop_vertices = np.concatenate([np.arange(wood_count), np.arange(wood_count, total)])
        loop_starts = np.concatenate([[0], wood_count + np.arange(quads) * 4])
        loop_totals = np.concatenate([[wood_count], np.full(quads, 4)])
        materials = np.concatenate([[0], np.ones(quads, dtype=int)])

        mask = compute_leaf_vertex_mask(total, loop_vertices, loop_starts, loop_totals, materials)
        assert mask.sum() == card_count and not mask[:wood_count].any()

        leaf_attachments = np.vstack(
            [arrays["vertex_positions"][:wood_count], np.array(card_attachments)]
        )
        plain = compute_all_streams(**arrays)
        streams = compute_all_streams(**arrays, leaf_mask=mask, leaf_attachments=leaf_attachments)

        wood_slice = slice(0, wood_count * 4)
        assert np.array_equal(streams["colors"][wood_slice], plain["colors"][wood_slice])
        assert np.array_equal(streams["stiffness"], plain["stiffness"])
        assert np.array_equal(streams["phase"], plain["phase"])
        assert np.array_equal(streams["wind_uv"], plain["wind_uv"])

        colors = streams["colors"].reshape(-1, 4)[wood_count:]
        blade = streams["leaf_blade"][wood_count:]
        assert np.all(colors[:, 2] == 1.0)
        assert colors[:, 1] == pytest.approx(blade)
        assert colors[:, 3] == pytest.approx(blade)

        # TEST_POINT: the second card is a quarter the size of the first; a
        # mesh-wide normalisation would cap its blade position at 0.25.
        for card in range(2):
            per_card = blade[card * 8 : (card + 1) * 8]
            assert per_card.max() == pytest.approx(1.0)
            assert per_card.min() > 0.0

        lines = [
            "# leaf cards baked into the tree mesh: material-1 vertices take leaf colours",
            f"# wood vertices {wood_count}, card vertices {card_count} on stem 3",
            "",
            f"{'vertex':>6} {'card':>4} {'blade':>10} {'r':>10} {'g':>10} {'b':>10} {'a':>10} "
            f"{'stiffness':>10}",
        ]
        for i in range(card_count):
            v = wood_count + i
            lines.append(
                f"{v:>6} {int(i / 8):>4} {_fmt(blade[i]):>10} {_fmt(colors[i, 0]):>10} "
                f"{_fmt(colors[i, 1]):>10} {_fmt(colors[i, 2]):>10} {_fmt(colors[i, 3]):>10} "
                f"{_fmt(streams['stiffness'][v]):>10}"
            )
        wood_unchanged = np.array_equal(streams["colors"][wood_slice], plain["colors"][wood_slice])
        twig_stiffness = np.allclose(
            streams["stiffness"][wood_count:], plain["stiffness"][wood_count:]
        )
        lines += [
            "",
            "# invariants",
            f"wood colours unchanged by the mask: {wood_unchanged}",
            f"card stiffness equals its twig's: {twig_stiffness}",
        ]
        write_artifact("card_streams.txt", lines)

    def test_wood_and_leaf_snapshot(self):
        """Record wood and leaf channel values side by side."""
        from foliage_streams import compute_leaf_blade_position, compute_leaf_colors

        tree = make_synthetic_tree()
        wood = compute_all_streams(**tree)["colors"]
        wood_along = compute_all_streams(**tree)["along_branch"]

        leaf_verts = np.array([[0.0, t, 0.0] for t in np.linspace(0.0, 1.0, 9)])
        leaf_pos = compute_leaf_blade_position(leaf_verts)
        leaf = compute_leaf_colors(leaf_pos)

        lines = [
            "# vertex colour: wood is opaque, leaves transmit",
            "# channels: r=occlusion g=translucency b=normal_blend a=tint",
            "",
            "## wood (branch mesh, default settings)",
            f"{'along':>10} {'r':>10} {'g':>10} {'b':>10} {'a':>10}",
        ]
        for t in sorted(set(np.round(wood_along, 6).tolist()))[:10]:
            i = int(np.where(np.round(wood_along, 6) == t)[0][0])
            lines.append(
                f"{_fmt(t):>10} {_fmt(wood[i * 4]):>10} {_fmt(wood[i * 4 + 1]):>10} "
                f"{_fmt(wood[i * 4 + 2]):>10} {_fmt(wood[i * 4 + 3]):>10}"
            )

        lines += [
            "",
            "## leaf (leaf mesh, default settings)",
            f"{'blade_t':>10} {'r':>10} {'g':>10} {'b':>10} {'a':>10}",
        ]
        for i, t in enumerate(leaf_pos):
            lines.append(
                f"{_fmt(t):>10} {_fmt(leaf[i * 4]):>10} {_fmt(leaf[i * 4 + 1]):>10} "
                f"{_fmt(leaf[i * 4 + 2]):>10} {_fmt(leaf[i * 4 + 3]):>10}"
            )

        lines += [
            "",
            "# invariants",
            f"wood_translucency_max={_fmt(wood[1::4].max())}  (must be 0)",
            f"leaf_translucency_max={_fmt(leaf[1::4].max())}",
        ]
        write_artifact("translucency.txt", lines)

        assert wood[1::4].max() == 0.0


class TestDebugView:
    """T5: the debug view must show the channel it claims to show."""

    def test_channels_map_to_their_streams(self):
        """M8: each channel reads the stream its label names.

        A channel wired to the wrong stream produces plausible colours and a
        wrong diagnosis, which is worse than no debug view at all.
        """
        from foliage_debug_drawer import (
            DEBUG_CHANNELS,
            compute_channel_colors,
            stem_id_to_color,
            value_to_viridis,
        )

        tree = make_synthetic_tree()
        streams = compute_all_streams(**tree)
        stem_ids = tree["stem_ids"]
        colors = streams["colors"]

        expected = {
            "STIFFNESS": streams["stiffness"],
            "PHASE": streams["phase"],
            "OCCLUSION": colors[0::4],
            "TRANSLUCENCY": colors[1::4],
            "NORMAL_BLEND": colors[2::4],
            "TINT": colors[3::4],
        }

        for channel, source in expected.items():
            actual = compute_channel_colors(streams, stem_ids, channel)
            assert actual is not None, f"channel {channel} produced nothing"
            assert actual == pytest.approx(value_to_viridis(source)), (
                f"channel {channel} does not read its own stream"
            )

        # Blade height is normalised for display; raw metres would clip.
        heights = streams["blade_height"]
        assert compute_channel_colors(streams, stem_ids, "BLADE_HEIGHT") == pytest.approx(
            value_to_viridis(heights / heights.max())
        )
        assert compute_channel_colors(streams, stem_ids, "STRUCTURE") == pytest.approx(
            stem_id_to_color(stem_ids)
        )

        # Off and unknown channels draw nothing rather than raising.
        assert compute_channel_colors(streams, stem_ids, "NONE") is None
        assert compute_channel_colors(streams, stem_ids, "NOT_A_CHANNEL") is None

        # Channel ids must stay aligned with the consuming shader's numbering.
        numbering = {ident: number for ident, _, _, number in DEBUG_CHANNELS}
        assert numbering["STIFFNESS"] == 1
        assert numbering["PHASE"] == 2
        assert numbering["OCCLUSION"] == 3
        assert numbering["TRANSLUCENCY"] == 4
        assert numbering["NORMAL_BLEND"] == 5
        assert numbering["TINT"] == 6
        assert numbering["STRUCTURE"] == 8
        assert numbering["SHADED_OCCLUSION"] == 9

        grey = compute_channel_colors(streams, stem_ids, "SHADED_OCCLUSION")
        assert grey[:, 0] == pytest.approx(colors[0::4])
        assert grey[:, 0] == pytest.approx(grey[:, 2])

    def test_canopy_exposure_darkens_only_occlusion(self):
        tree = make_synthetic_tree()
        plain = compute_all_streams(**tree)["colors"].reshape(-1, 4)
        exposure = np.linspace(0.0, 1.0, len(plain))

        shaded = compute_all_streams(**tree, sky_exposure=exposure, sky_occlusion=0.5)
        shaded = shaded["colors"].reshape(-1, 4)

        assert shaded[:, 0] == pytest.approx(plain[:, 0] * (1.0 - 0.5 * (1.0 - exposure)))
        assert shaded[:, 1:] == pytest.approx(plain[:, 1:])
        off = compute_all_streams(**tree, sky_exposure=exposure, sky_occlusion=0.0)
        assert off["colors"] == pytest.approx(plain.ravel())

    def test_ramp_separates_values(self):
        """M9: the ramp must not collapse distinct values to one colour."""
        from foliage_debug_drawer import stem_id_to_color, value_to_viridis

        sweep = np.linspace(0.0, 1.0, 32)
        ramp = value_to_viridis(sweep)

        assert ramp.shape == (32, 4)
        assert np.all((ramp >= 0.0) & (ramp <= 1.0))
        assert np.all(ramp[:, 3] == 1.0)

        # Adjacent samples must differ, or a gradient reads as a flat patch.
        deltas = np.linalg.norm(np.diff(ramp[:, :3], axis=0), axis=1)
        assert np.all(deltas > 1e-3)

        # Out-of-range input clamps rather than wrapping to a wrong colour.
        assert value_to_viridis(np.array([-1.0])) == pytest.approx(
            value_to_viridis(np.array([0.0]))
        )
        assert value_to_viridis(np.array([2.0])) == pytest.approx(value_to_viridis(np.array([1.0])))

        # Structure view: neighbouring blade ids must not share a colour.
        blade_colors = stem_id_to_color(np.arange(8, dtype=float))
        for i in range(len(blade_colors) - 1):
            assert np.linalg.norm(blade_colors[i, :3] - blade_colors[i + 1, :3]) > 0.2

        lines = [
            "# debug view ramp",
            "",
            f"{'value':>10} {'r':>10} {'g':>10} {'b':>10}",
        ]
        for i, value in enumerate(sweep):
            lines.append(
                f"{_fmt(value):>10} {_fmt(ramp[i][0]):>10} "
                f"{_fmt(ramp[i][1]):>10} {_fmt(ramp[i][2]):>10}"
            )

        lines += [
            "",
            "# structure view, blade id to colour",
            "",
            f"{'stem_id':>10} {'r':>10} {'g':>10} {'b':>10}",
        ]
        for i in range(len(blade_colors)):
            lines.append(
                f"{i:>10} {_fmt(blade_colors[i][0]):>10} "
                f"{_fmt(blade_colors[i][1]):>10} {_fmt(blade_colors[i][2]):>10}"
            )
        write_artifact("debug_ramp.txt", lines)


class TestFullStreamSnapshot:
    """T4: whole-tree snapshot of every stream, for eyeball and model review."""

    def test_full_tree_snapshot(self):
        """Emit every per-vertex stream value for the synthetic tree."""
        tree = make_synthetic_tree()
        streams = compute_all_streams(**tree)

        lines = [
            "# full foliage stream dump for the synthetic tree",
            "# uv1 = (stiffness, phase)  uv2 = (blade_height_m, 0)  color = (occl, transl, nblend, tint)",
            "",
            f"{'v':>4} {'stem':>5} {'pos_x':>10} {'pos_y':>10} {'pos_z':>10} "
            f"{'uv1.x':>10} {'uv1.y':>10} {'uv2.x':>10} "
            f"{'col.r':>10} {'col.g':>10} {'col.b':>10} {'col.a':>10}",
        ]

        positions = tree["vertex_positions"]
        colors = streams["colors"]
        for i in range(len(tree["stem_ids"])):
            lines.append(
                f"{i:>4} {int(tree['stem_ids'][i]):>5} "
                f"{_fmt(positions[i][0]):>10} {_fmt(positions[i][1]):>10} {_fmt(positions[i][2]):>10} "
                f"{_fmt(streams['stiffness'][i]):>10} {_fmt(streams['phase'][i]):>10} "
                f"{_fmt(streams['blade_height'][i]):>10} "
                f"{_fmt(colors[i * 4]):>10} {_fmt(colors[i * 4 + 1]):>10} "
                f"{_fmt(colors[i * 4 + 2]):>10} {_fmt(colors[i * 4 + 3]):>10}"
            )

        # Any violated invariant is written into the artifact with its location,
        # so a reviewer reading the file alone sees the defect.
        violations = []
        for i in range(len(tree["stem_ids"])):
            if not 0.0 <= streams["stiffness"][i] <= 1.0:
                violations.append(f"vertex {i}: stiffness out of range")
            if not 0.0 <= streams["along_branch"][i] <= 1.0:
                violations.append(f"vertex {i}: along_branch out of range")
            if streams["blade_height"][i] < 0.0:
                violations.append(f"vertex {i}: negative blade height")

        lines += ["", "# violations", *(violations or ["none"])]
        write_artifact("full_streams.txt", lines)

        assert not violations
