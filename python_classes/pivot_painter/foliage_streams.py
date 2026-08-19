"""Pure computation of foliage vertex streams.

Maps Mtree's per-vertex branch attributes onto the vertex-stream contract that
real-time foliage shaders consume (stiffness, phase, blade height, and the four
independent scalars packed into vertex colour).

The contract these functions target treats a *blade* as the unit of wind motion.
In Mtree a blade is a branch chain: the mesher assigns one ``stem_id`` per chain
and keeps it constant from the chain's base to its tip, so ``stem_id`` is the
blade key and ``branch_extent`` is that blade's total length.

Everything here is free of Blender and of the native module, so it is testable
in isolation. Callers supply plain arrays read off the mesh attributes that
``ManifoldMesher`` already emits: ``stem_id``, ``branch_extent``, ``radius``,
``pivot_position`` and ``direction``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:  # package import (addon runtime)
    from .core import compute_stem_id_hash
except ImportError:  # flat import (test harness puts pivot_painter/ on sys.path)
    from core import compute_stem_id_hash

# A vertex whose along-chain position is within this fraction of the chain base
# is treated as rooted, and its stiffness is pinned to exactly 1.0.
ROOTED_EPSILON = 1e-4

# Wind displacement is clamped by the consuming shader to blade_height * this
# factor. Mesh bounds must be expanded by the same amount or instances bending
# near the screen edge get culled while still visible.
WIND_REACH_FACTOR = 0.25

# Lower bound on a branch's stiffness as a fraction of its parent's. The shader
# bends by sway**2, so a large gap across a junction makes the child swing at a
# very different rate from the parent and the branch whips independently. Wood
# still softens outward, just never abruptly: 0.6 means a branch is at most
# moderately looser than the limb carrying it.
JUNCTION_FALLOFF = 0.6


def compute_along_branch_position(
    vertex_positions: NDArray[np.floating],
    pivot_positions: NDArray[np.floating],
    directions: NDArray[np.floating],
    branch_extents: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute normalised position along each vertex's branch chain.

    The chain base is the vertex's ``pivot_position`` and the chain runs along
    ``direction`` for ``branch_extent`` metres. Projecting the vertex offset onto
    the branch direction gives distance travelled along the chain, which
    normalises to ``0`` at the base and ``1`` at the tip.

    Projection is used rather than raw distance so that a vertex on the far side
    of a thick trunk is not pushed artificially far along the chain by the
    trunk's radius.

    Args:
        vertex_positions: Nx3 array of vertex world positions.
        pivot_positions: Nx3 array of branch-chain base positions.
        directions: Nx3 array of branch direction vectors.
        branch_extents: Length of each vertex's branch chain, in metres.

    Returns:
        Array of normalised positions clamped to [0, 1].
    """
    verts = np.asarray(vertex_positions, dtype=np.float64).reshape(-1, 3)
    pivots = np.asarray(pivot_positions, dtype=np.float64).reshape(-1, 3)
    dirs = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    extents = np.asarray(branch_extents, dtype=np.float64).ravel()

    if len(verts) == 0:
        return np.zeros(0, dtype=np.float64)

    # Normalise directions, falling back to +Z for degenerate vectors so the
    # projection stays finite.
    lengths = np.linalg.norm(dirs, axis=1, keepdims=True)
    degenerate = lengths[:, 0] <= 1e-6
    safe_dirs = np.where(lengths > 1e-6, dirs / np.maximum(lengths, 1e-12), 0.0)
    safe_dirs[degenerate] = np.array([0.0, 0.0, 1.0])

    travelled = np.einsum("ij,ij->i", verts - pivots, safe_dirs)

    # Extents below one millimetre carry no usable gradient; report those
    # vertices as rooted rather than dividing by a near-zero length.
    usable = extents > 1e-3
    position = np.zeros(len(verts), dtype=np.float64)
    position[usable] = travelled[usable] / extents[usable]

    return np.clip(position, 0.0, 1.0)


def compute_stiffness(
    radii: NDArray[np.floating],
    along_branch: NDArray[np.floating],
    hierarchy_depths: NDArray[np.floating] | None = None,
    max_radius: float | None = None,
    trunk_base_stiffness: float = 1.0,
    trunk_tip_stiffness: float = 1.0,
    branch_stiffness: float = 1.0,
    junction_falloff: float = JUNCTION_FALLOFF,
) -> NDArray[np.floating]:
    """Compute per-vertex wind stiffness.

    Stiffness is the wind authority: the consuming shader derives
    ``sway = 1 - stiffness`` and bends by ``sway**2``, so this decides which
    parts of the tree move at all, and the response is strongly non-linear —
    ``0.5`` sits much closer to rigid than to free.

    The trunk and the side branches are controlled separately, because a single
    curve cannot serve both. Radius is the honest signal for side branches — a
    thick limb resists wind and a thin twig does not — but the trunk is the
    thickest wood on the tree, so a radius-derived curve normalised against the
    tree maximum pins the whole trunk at ``1`` and it never bends at all.

    Trunk vertices (``hierarchy_depth == 0``) therefore interpolate from
    ``trunk_base_stiffness`` at the root to ``trunk_tip_stiffness`` at the top,
    which is what produces whole-tree lean. Side branches keep the radius curve,
    scaled by ``branch_stiffness``.

    Rooted vertices are pinned to exactly ``1.0`` regardless of the settings:
    radius alone would leave them slightly below ``1`` and let the trunk base
    drift out of the ground.

    Args:
        radii: Per-vertex branch radius.
        along_branch: Per-vertex normalised position along the branch chain.
        hierarchy_depths: Per-vertex depth, ``0`` for trunk. When omitted every
            vertex is treated as a side branch.
        max_radius: Radius mapping to full rigidity for side branches. Defaults
            to the largest radius present.
        trunk_base_stiffness: Trunk stiffness at the root. ``1`` holds the base
            rigid; lower values let the whole trunk sway.
        trunk_tip_stiffness: Trunk stiffness at the top. Lower than the base
            value makes the trunk bend progressively toward the crown.
        branch_stiffness: Scale applied to the side-branch radius curve. Lower
            values make every side branch more compliant.

    Returns:
        Stiffness values in [0, 1]; ``1`` is rigid, ``0`` is a free tip.
    """
    radius_values = np.asarray(radii, dtype=np.float64).ravel()
    position = np.asarray(along_branch, dtype=np.float64).ravel()

    if len(radius_values) == 0:
        return np.zeros(0, dtype=np.float64)

    if max_radius is None:
        max_radius = float(radius_values.max())
    max_radius = max(max_radius, 1e-6)

    normalised = np.clip(radius_values / max_radius, 0.0, 1.0)

    # Branch radius spans orders of magnitude — a half-metre trunk against
    # sub-millimetre twigs — so a linear ratio drives almost every vertex to
    # nearly zero and compresses the whole limb-to-twig gradient into a range
    # too narrow to see or to tune. The square root spreads that distribution
    # across the usable range while preserving the ordering, so thicker wood is
    # still always stiffer than thinner wood.
    stiffness = np.sqrt(normalised) * branch_stiffness

    if hierarchy_depths is not None:
        depths = np.asarray(hierarchy_depths, dtype=np.float64).ravel()
        trunk = depths <= 0.0
        # Interpolate along the trunk rather than by radius, so the trunk can
        # bend even though it is the thickest wood on the tree.
        stiffness[trunk] = (
            trunk_base_stiffness
            + (trunk_tip_stiffness - trunk_base_stiffness) * position[trunk]
        )

        # Step each level down from its parent by a bounded amount.
        #
        # Two things have to hold at a junction. A branch must not be stiffer
        # than the wood carrying it, or the joint reads as a hard spot and motion
        # grows outward instead of damping. But the drop must also be *gradual*:
        # the shader bends by ``sway**2``, so a large stiffness gap makes parent
        # and child swing at very different rates and the branch whips around
        # independently of the limb it hangs on.
        #
        # Clamping to the parent minimum satisfied the first and violated the
        # second — each level inherited the softest parent value and then fell
        # again by its own radius, so the ratios compounded and the trunk-to-twig
        # range collapsed within two junctions. Instead each level is held inside
        # a band relative to its parent: never stiffer than it, never less than
        # ``JUNCTION_FALLOFF`` of it. Motion still damps outward, but neighbouring
        # wood always moves at comparable rates.
        for depth in sorted(set(depths[depths > 0].tolist())):
            parent = depths == (depth - 1)
            if not np.any(parent):
                continue
            level = depths == depth
            parent_stiffness = float(stiffness[parent].min())
            stiffness[level] = np.clip(
                stiffness[level],
                parent_stiffness * junction_falloff,
                parent_stiffness,
            )

    stiffness = np.clip(stiffness, 0.0, 1.0)

    # Pin only genuinely ground-rooted geometry.
    #
    # ``position`` is normalised along each *chain*, so it is 0 at the base of
    # every branch, not just at the ground. Pinning on that alone made each side
    # branch rigid where it joins its parent — a thin twig high in the canopy
    # ended up stiffer than the trunk carrying it, producing a visible hard spot
    # at every junction. Only the trunk (depth 0) actually touches the ground, so
    # only the trunk's own base is pinned; when no depth information is supplied
    # nothing is pinned, because there is no way to tell a root from a joint.
    if hierarchy_depths is not None:
        depths = np.asarray(hierarchy_depths, dtype=np.float64).ravel()
        rooted = (depths <= 0.0) & (position <= ROOTED_EPSILON)
        stiffness[rooted] = 1.0

    return stiffness


def compute_phase(stem_ids: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute the per-blade wind phase.

    Phase does two unrelated jobs in the consuming shader: it offsets the sway
    wave so neighbouring branches do not move in lockstep, and it acts as the
    per-blade id that density thresholding tests against. Both require the value
    to be *constant across a blade* and to differ between blades — a gradient
    within one branch tears that branch apart under wind.

    Deriving it from ``stem_id`` alone guarantees constancy, because the mesher
    stamps one ``stem_id`` across an entire branch chain. The golden-ratio hash
    spreads sequential ids across the range instead of letting neighbouring
    branches land on neighbouring phases.

    Args:
        stem_ids: Per-vertex branch chain id.

    Returns:
        Phase values in [0, 1).
    """
    return compute_stem_id_hash(np.asarray(stem_ids, dtype=np.float64).ravel())


def compute_blade_height(branch_extents: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute per-vertex blade height in metres.

    This is the wind displacement bound, not a normalised value: the consuming
    shader clamps wind offset to ``height * WIND_REACH_FACTOR``. Zero pins the
    geometry still, so it is passed through in world units exactly as the mesher
    measured the branch chain.

    Args:
        branch_extents: Per-vertex branch chain length, in metres.

    Returns:
        Blade heights in metres, clamped to be non-negative.
    """
    extents = np.asarray(branch_extents, dtype=np.float64).ravel()
    return np.maximum(extents, 0.0)


def pack_wind_uv(
    stiffness: NDArray[np.floating],
    phase: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Pack stiffness and phase into an interleaved UV stream.

    Args:
        stiffness: Per-vertex stiffness.
        phase: Per-vertex phase.

    Returns:
        Flat array of UV pairs (length = vertex_count * 2).
    """
    return np.column_stack(
        [
            np.asarray(stiffness, dtype=np.float64).ravel(),
            np.asarray(phase, dtype=np.float64).ravel(),
        ]
    ).ravel()


def pack_height_uv(blade_height: NDArray[np.floating]) -> NDArray[np.floating]:
    """Pack blade height into an interleaved UV stream.

    The second channel is unused by the contract and is written as zero rather
    than left undefined.

    Args:
        blade_height: Per-vertex blade height in metres.

    Returns:
        Flat array of UV pairs (length = vertex_count * 2).
    """
    heights = np.asarray(blade_height, dtype=np.float64).ravel()
    return np.column_stack([heights, np.zeros(len(heights))]).ravel()


def compute_foliage_colors(
    along_branch: NDArray[np.floating],
    occlusion_base: float = 0.25,
    translucency_tip: float = 0.0,
    normal_blend: float = 0.0,
) -> NDArray[np.floating]:
    """Compute the four independent scalars packed into vertex colour.

    Nothing samples this as a colour. Each channel is a separate lighting or
    tinting input, and all four are functions of normalised position along the
    branch:

        r = lerp(occlusion_base, 1, t)   ambient occlusion, darker at the root
        g = lerp(0, translucency_tip, t) back-lit transmission, rising to the tip
        b = normal_blend                 geometric-to-flat normal blend, constant
        a = t                            tint ramp coordinate, raw root to tip

    ``a`` stays the honest normalised position because a material's root/mid/tip
    ramp positions are authored against it; reshaping it silently rescales the
    artist's ramp.

    **Translucency defaults to zero here because this function shades wood.**
    Back-lit transmission is light passing through a thin surface; trunks and
    branches are opaque, and a non-zero value makes twig tips glow when lit from
    behind. Leaves are the geometry that legitimately transmits light, and they
    are a separate mesh — see ``compute_leaf_colors``.

    Args:
        along_branch: Per-vertex normalised position along the branch chain.
        occlusion_base: Occlusion value at the branch base.
        translucency_tip: Transmission at the branch tip. Leave at ``0`` for
            wood; raise it only for thin twigs that genuinely read as
            translucent against the sky.
        normal_blend: Constant normal-flattening amount for the whole plant.

    Returns:
        Flat array of RGBA values (length = vertex_count * 4).
    """
    position = np.asarray(along_branch, dtype=np.float64).ravel()
    count = len(position)

    colors = np.zeros(count * 4, dtype=np.float64)
    colors[0::4] = occlusion_base + (1.0 - occlusion_base) * position
    colors[1::4] = translucency_tip * position
    colors[2::4] = normal_blend
    colors[3::4] = position

    return colors


def compute_leaf_blade_position(
    vertex_positions: NDArray[np.floating],
    attachment_points: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Compute normalised petiole-to-tip position across a leaf mesh.

    Leaves carry none of the branch attributes — they are generated as a
    separate mesh — so the along-blade parameter cannot come from
    ``branch_extent`` and ``pivot_position``. It is derived from the leaf's own
    geometry instead: the attachment point is where the petiole meets the
    branch, and distance from it, normalised by the farthest vertex, runs
    ``0`` at the stalk to ``1`` at the leaf tip.

    Args:
        vertex_positions: Nx3 array of leaf vertex positions.
        attachment_points: Nx3 array of per-vertex attachment positions. When
            omitted the mesh minimum along its longest axis is used, which is
            correct for a single leaf authored about its own origin.

    Returns:
        Array of normalised positions in [0, 1].
    """
    verts = np.asarray(vertex_positions, dtype=np.float64).reshape(-1, 3)

    if len(verts) == 0:
        return np.zeros(0, dtype=np.float64)

    if attachment_points is None:
        # Fall back to the extreme along the leaf's longest axis, which is the
        # petiole for a leaf modelled about its own origin.
        extent = verts.max(axis=0) - verts.min(axis=0)
        axis = int(np.argmax(extent))
        origin = verts[np.argmin(verts[:, axis])]
    else:
        origin = np.asarray(attachment_points, dtype=np.float64).reshape(-1, 3)

    distance = np.linalg.norm(verts - origin, axis=1)
    peak = float(distance.max())

    if peak <= 1e-9:
        return np.zeros(len(verts), dtype=np.float64)

    return np.clip(distance / peak, 0.0, 1.0)


def compute_leaf_colors(
    blade_position: NDArray[np.floating],
    occlusion_base: float = 0.5,
    translucency_tip: float = 1.0,
    normal_blend: float = 1.0,
) -> NDArray[np.floating]:
    """Compute the vertex colour scalars for leaf geometry.

    Same four channels as ``compute_foliage_colors``, but the defaults invert
    because leaves are the opposite material to wood:

    - **Translucency defaults to ``1``.** A leaf is a thin surface that
      genuinely transmits light, and back-lit transmission is most of what makes
      foliage read as foliage rather than as green cardboard.
    - **Normal blend defaults to ``1``.** Flattening leaf normals toward the
      canopy up-vector is what stops individual leaves reading as faceted shells
      and makes a canopy light as one volume.
    - **Occlusion base is higher.** A leaf sits in the canopy rather than at the
      dark base of a trunk, so its shaded end is not as dark.

    Args:
        blade_position: Per-vertex normalised petiole-to-tip position.
        occlusion_base: Occlusion at the petiole; the tip is always ``1``.
        translucency_tip: Transmission at the leaf tip, where the blade is
            thinnest.
        normal_blend: Blend from geometric normals to flat up.

    Returns:
        Flat array of RGBA values (length = vertex_count * 4).
    """
    return compute_foliage_colors(
        blade_position,
        occlusion_base=occlusion_base,
        translucency_tip=translucency_tip,
        normal_blend=normal_blend,
    )


def compute_wind_expanded_bounds(
    vertex_positions: NDArray[np.floating],
    blade_height: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute mesh bounds expanded by the wind reach.

    Bounds fitted to the rest pose pop as instances bend out of them near the
    screen edge, so the rest geometry is expanded horizontally by the largest
    blade's wind reach.

    Args:
        vertex_positions: Nx3 array of vertex world positions.
        blade_height: Per-vertex blade height in metres.

    Returns:
        Tuple of (min_corner, max_corner) as 3-element arrays.
    """
    verts = np.asarray(vertex_positions, dtype=np.float64).reshape(-1, 3)

    if len(verts) == 0:
        return (np.zeros(3), np.zeros(3))

    heights = np.asarray(blade_height, dtype=np.float64).ravel()
    reach = float(heights.max()) * WIND_REACH_FACTOR if len(heights) else 0.0

    horizontal = np.array([reach, reach, 0.0])
    return (verts.min(axis=0) - horizontal, verts.max(axis=0) + horizontal)


def compute_all_streams(
    vertex_positions: NDArray[np.floating],
    stem_ids: NDArray[np.floating],
    branch_extents: NDArray[np.floating],
    radii: NDArray[np.floating],
    pivot_positions: NDArray[np.floating],
    directions: NDArray[np.floating],
    hierarchy_depths: NDArray[np.floating] | None = None,
    occlusion_base: float = 0.25,
    translucency_tip: float = 0.0,
    normal_blend: float = 0.0,
    trunk_base_stiffness: float = 1.0,
    trunk_tip_stiffness: float = 1.0,
    branch_stiffness: float = 1.0,
    junction_falloff: float = JUNCTION_FALLOFF,
) -> dict:
    """Compute every foliage vertex stream from mesher attributes.

    This shades **wood**. Translucency defaults to zero because trunks and
    branches are opaque; leaves are a separate mesh and use
    ``compute_leaf_colors``.

    Args:
        vertex_positions: Nx3 array of vertex world positions.
        stem_ids: Per-vertex branch chain id.
        branch_extents: Per-vertex branch chain length, in metres.
        radii: Per-vertex branch radius.
        pivot_positions: Nx3 array of branch chain base positions.
        directions: Nx3 array of branch direction vectors.
        hierarchy_depths: Per-vertex depth, ``0`` for trunk. Required for the
            trunk stiffness controls to have any effect.
        occlusion_base: Occlusion value at the branch base.
        translucency_tip: Translucency value at the branch tip.
        normal_blend: Constant normal-flattening amount for the whole plant.
        trunk_base_stiffness: Trunk stiffness at the root.
        trunk_tip_stiffness: Trunk stiffness at the top.
        branch_stiffness: Scale applied to the side-branch radius curve.

    Returns:
        Dict with keys ``along_branch``, ``stiffness``, ``phase``,
        ``blade_height``, ``wind_uv``, ``height_uv``, ``colors`` and ``bounds``.
    """
    along_branch = compute_along_branch_position(
        vertex_positions, pivot_positions, directions, branch_extents
    )
    stiffness = compute_stiffness(
        radii,
        along_branch,
        hierarchy_depths=hierarchy_depths,
        trunk_base_stiffness=trunk_base_stiffness,
        trunk_tip_stiffness=trunk_tip_stiffness,
        branch_stiffness=branch_stiffness,
        junction_falloff=junction_falloff,
    )
    phase = compute_phase(stem_ids)
    blade_height = compute_blade_height(branch_extents)

    return {
        "along_branch": along_branch,
        "stiffness": stiffness,
        "phase": phase,
        "blade_height": blade_height,
        "wind_uv": pack_wind_uv(stiffness, phase),
        "height_uv": pack_height_uv(blade_height),
        "colors": compute_foliage_colors(
            along_branch, occlusion_base, translucency_tip, normal_blend
        ),
        "bounds": compute_wind_expanded_bounds(vertex_positions, blade_height),
    }
