"""Pure computation of a foliage bone rig.

The vertex-stream wind path (``foliage_streams``) displaces each vertex by a
translation along one global wind direction, scaled by ``sway**2``. That is a
grass model: it approximates bending only because a blade is a thin strip whose
stiffness gradient does the shearing. Wood is not thin, and the approximation
fails structurally on a tree — when the trunk leans, a branch four metres up
should translate *and rotate* with it, and the vertex path carries no rotation
at all. Branches therefore shear against a trunk that has moved somewhere else,
and the gap scales with ``branch_extent``: an eight-metre trunk chain is allowed
two metres of displacement while the twigs hanging off it move centimetres in
their own unrotated frame.

A rig fixes that by construction. Poses compose down the chain, so a child bone
inherits its parent's rotation before applying its own bend.

Everything here is free of Blender and of the native module. Callers supply the
same per-vertex arrays ``ManifoldMesher`` already emits that ``foliage_streams``
consumes: ``stem_id``, ``branch_extent``, ``radius``, ``pivot_position``,
``direction`` and ``hierarchy_depth``.

Two structural facts about that input shape the whole module:

**Parent linkage is not in the mesh.** ``PivotPainterContext`` carries no parent
id, so the mesher discards the linkage it knows at recursion time. It is
recovered here geometrically: a chain's ``pivot_position`` lies on its parent's
chain by construction, so the parent is the chain passing nearest that point
among those that could carry it.

**One bone per chain is not enough.** A single bone can only move its chain
rigidly, which reproduces the very failure the rig exists to fix. Chains are
subdivided into bone segments along their length so the trunk can actually bend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

try:  # package import (addon runtime)
    from .foliage_streams import JUNCTION_FALLOFF, compute_stiffness
except ImportError:  # flat import (test harness puts pivot_painter/ on sys.path)
    from foliage_streams import JUNCTION_FALLOFF, compute_stiffness

# Target length of one bone segment, as a fraction of the chain it belongs to.
# Four segments per chain is the point where a trunk reads as bending rather
# than as hinging, without paying for bones that never visibly separate.
DEFAULT_SEGMENTS_PER_CHAIN = 4

# A chain shorter than this carries no usable bend and gets a single bone.
MIN_SEGMENTED_LENGTH = 1e-3

# Vertices within this distance of a segment boundary blend between the two
# bones either side of it, as a fraction of segment length. A hard assignment
# creases the mesh at every boundary; the shader only has two weights per
# vertex, so the blend band must stay inside one segment on each side.
BLEND_BAND = 0.5

# Weight below which the second bone is not worth binding. The consuming shader
# skips the second fetch entirely above 0.999 on the first weight, so anything
# under this threshold is encoded as rigid.
MIN_BLEND_WEIGHT = 1e-3


@dataclass
class Bone:
    """One bone record in the rig.

    Mirrors the ``FoliageBoneData`` layout the consuming engine expects. The
    fields are deliberately flat scalars and vectors rather than a transform
    matrix, because the engine composes the rest transform itself and any
    disagreement between this record and the authored vertex positions deforms
    the mesh while it stands still.
    """

    rest_position: NDArray[np.floating]
    rest_direction: NDArray[np.floating]
    parent: int
    stiffness: float
    length: float
    swing_weight: float
    stem_id: int
    segment: int
    depth: int


@dataclass
class BoneRig:
    """A complete rig: the bone records plus the per-vertex binding."""

    bones: list[Bone] = field(default_factory=list)
    bone_indices_a: NDArray[np.integer] = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    bone_indices_b: NDArray[np.integer] = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    weights_a: NDArray[np.floating] = field(default_factory=lambda: np.zeros(0))

    @property
    def bone_count(self) -> int:
        return len(self.bones)


def _chain_table(
    stem_ids: NDArray[np.floating],
    branch_extents: NDArray[np.floating],
    pivot_positions: NDArray[np.floating],
    directions: NDArray[np.floating],
    radii: NDArray[np.floating],
    hierarchy_depths: NDArray[np.floating],
) -> dict[int, dict]:
    """Collapse per-vertex arrays into one record per branch chain.

    ``stem_id`` is constant across a chain and every other attribute the rig
    needs is constant with it, so the first vertex carrying an id defines the
    chain. Radius is taken as the maximum over the chain rather than the first
    vertex, because the first vertex encountered is not necessarily the chain
    base and radius is the stiffness signal.
    """
    ids = np.asarray(stem_ids, dtype=np.float64).ravel().astype(np.int64)
    extents = np.asarray(branch_extents, dtype=np.float64).ravel()
    pivots = np.asarray(pivot_positions, dtype=np.float64).reshape(-1, 3)
    dirs = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    radius_values = np.asarray(radii, dtype=np.float64).ravel()
    depths = np.asarray(hierarchy_depths, dtype=np.float64).ravel().astype(np.int64)

    chains: dict[int, dict] = {}

    for stem_id in np.unique(ids):
        mask = ids == stem_id
        chains[int(stem_id)] = {
            "stem_id": int(stem_id),
            "extent": float(extents[mask][0]),
            "pivot": pivots[mask][0].copy(),
            # Direction is per-node in the mesher, so it curves along the chain.
            # The chain's overall axis is the mean of its vertices' directions,
            # which is the honest single axis for a curved chain and matches
            # what the rest pose has to describe.
            "direction": _safe_normalise(dirs[mask].mean(axis=0)),
            "radius": float(radius_values[mask].max()),
            "depth": int(depths[mask][0]),
        }

    return chains


def _safe_normalise(vector: NDArray[np.floating]) -> NDArray[np.floating]:
    """Normalise a vector, falling back to +Z when it is degenerate."""
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return np.asarray(vector, dtype=np.float64) / length


def resolve_chain_parents(chains: dict[int, dict]) -> dict[int, int]:
    """Recover each chain's parent chain from geometry.

    The mesher knows the linkage at recursion time and discards it, so it is
    rebuilt here. A side branch's ``pivot_position`` is placed *on* its parent's
    chain, so the parent is the chain whose axis passes nearest that point.

    ``hierarchy_depth`` constrains the search rather than the geometry alone:
    a candidate must sit exactly one level above the child, which rules out a
    twig that happens to pass close to another twig, and guarantees the
    resulting parent index is topologically ordered.

    Returns:
        Mapping of stem_id to parent stem_id, with ``-1`` for roots.
    """
    parents: dict[int, int] = {}

    by_depth: dict[int, list[dict]] = {}
    for chain in chains.values():
        by_depth.setdefault(chain["depth"], []).append(chain)

    for stem_id, chain in chains.items():
        depth = chain["depth"]
        if depth <= 0:
            parents[stem_id] = -1
            continue

        candidates = by_depth.get(depth - 1, [])
        if not candidates:
            # A chain whose parent level is absent cannot be attached without
            # inventing a linkage, so it is treated as its own root. Reporting
            # it as rooted is safe; guessing a parent silently deforms it.
            parents[stem_id] = -1
            continue

        best_id = -1
        best_distance = float("inf")
        for candidate in candidates:
            distance = _distance_to_segment(
                chain["pivot"],
                candidate["pivot"],
                candidate["pivot"] + candidate["direction"] * candidate["extent"],
            )
            if distance < best_distance:
                best_distance = distance
                best_id = candidate["stem_id"]

        parents[stem_id] = best_id

    return parents


def _distance_to_segment(
    point: NDArray[np.floating],
    start: NDArray[np.floating],
    end: NDArray[np.floating],
) -> float:
    """Shortest distance from a point to a finite segment.

    A finite segment rather than an infinite line: a branch attaches somewhere
    along its parent's actual length, and an infinite line would let a distant
    chain win by lying on the parent's extension beyond its tip.
    """
    axis = end - start
    axis_length_sq = float(axis @ axis)

    if axis_length_sq <= 1e-12:
        return float(np.linalg.norm(point - start))

    t = float((point - start) @ axis) / axis_length_sq
    t = min(1.0, max(0.0, t))

    return float(np.linalg.norm(point - (start + axis * t)))


def _segment_count(extent: float, segments_per_chain: int) -> int:
    """Number of bone segments to split a chain of this length into."""
    if extent <= MIN_SEGMENTED_LENGTH:
        return 1
    return max(1, int(segments_per_chain))


def build_bones(
    chains: dict[int, dict],
    parents: dict[int, int],
    segments_per_chain: int = DEFAULT_SEGMENTS_PER_CHAIN,
    trunk_base_stiffness: float = 1.0,
    trunk_tip_stiffness: float = 0.6,
    branch_stiffness: float = 1.0,
    junction_falloff: float = JUNCTION_FALLOFF,
) -> tuple[list[Bone], dict[int, list[int]]]:
    """Build the ordered bone list and the chain-to-bone-index map.

    Bones are emitted in ascending depth, then ascending stem_id, then ascending
    segment. **That order is the contract, not a convenience:** the consuming
    pose pass is a single forward sweep that reads its parent's already-computed
    pose, so a parent appearing after its child produces garbage silently rather
    than erroring. Emitting by depth guarantees every parent precedes every
    child, and within a chain each segment's parent is the segment before it.

    Stiffness is **delegated to** ``compute_stiffness``, not reimplemented here.
    Both wind paths are driven by the same authored curves, so a second
    implementation would let the rig and the vertex streams disagree about how
    stiff the same branch is — and since the engine runs only one path at a
    time, that divergence would surface as "the tree moves differently after
    enabling bones" rather than as a visible bug.

    The sample points are the bone segment starts rather than the vertices, so
    each bone gets the stiffness of the position it actually pivots from.

    Returns:
        Tuple of (bones, map of stem_id to its bone indices in segment order).
    """
    ordered = sorted(chains.values(), key=lambda chain: (chain["depth"], chain["stem_id"]))

    # Build the sample set first, so the whole rig's stiffness comes from one
    # call and shares the tree-wide radius normalisation the vertex path uses.
    sample_radii: list[float] = []
    sample_positions: list[float] = []
    sample_depths: list[float] = []

    for chain in ordered:
        count = _segment_count(chain["extent"], segments_per_chain)
        for segment in range(count):
            sample_radii.append(chain["radius"])
            sample_positions.append(segment / count if count else 0.0)
            sample_depths.append(float(chain["depth"]))

    stiffness_values = compute_stiffness(
        np.array(sample_radii),
        np.array(sample_positions),
        hierarchy_depths=np.array(sample_depths),
        trunk_base_stiffness=trunk_base_stiffness,
        trunk_tip_stiffness=trunk_tip_stiffness,
        branch_stiffness=branch_stiffness,
        junction_falloff=junction_falloff,
    )

    bones: list[Bone] = []
    chain_bones: dict[int, list[int]] = {}

    # Walks the stiffness samples in the same order they were built, so a bone
    # takes the value computed for its own segment.
    sample_cursor = 0

    for chain in ordered:
        stem_id = chain["stem_id"]
        extent = chain["extent"]
        count = _segment_count(extent, segments_per_chain)
        segment_length = extent / count if count else extent

        parent_stem = parents.get(stem_id, -1)
        # Attach to the parent *segment that actually contains the junction*,
        # found by projecting this chain's pivot onto the parent chain.
        #
        # Neither end of the parent will do. Its last bone makes a branch rooted
        # low on a limb swing from the limb's tip, on an arc that does not pass
        # through the junction at all; its first bone makes a branch ignore every
        # bend between the limb's base and the junction. Both are the wrong-arc
        # failure the rig exists to remove, just at opposite ends.
        parent_index = -1
        if parent_stem >= 0 and parent_stem in chain_bones:
            parent_chain = chains[parent_stem]
            parent_indices = chain_bones[parent_stem]
            parent_segments = len(parent_indices)

            travelled = float((chain["pivot"] - parent_chain["pivot"]) @ parent_chain["direction"])
            parent_segment_length = parent_chain["extent"] / parent_segments
            parent_segment_length = max(parent_segment_length, 1e-9)

            segment_at_junction = int(travelled / parent_segment_length)
            segment_at_junction = min(parent_segments - 1, max(0, segment_at_junction))

            parent_index = parent_indices[segment_at_junction]

        indices: list[int] = []

        for segment in range(count):
            # Position along the chain where this segment starts.
            offset = chain["direction"] * (segment_length * segment)

            bones.append(
                Bone(
                    rest_position=chain["pivot"] + offset,
                    rest_direction=chain["direction"].copy(),
                    parent=parent_index if segment == 0 else len(bones) - 1,
                    stiffness=float(stiffness_values[sample_cursor]),
                    length=float(segment_length),
                    swing_weight=1.0,
                    stem_id=stem_id,
                    segment=segment,
                    depth=chain["depth"],
                )
            )
            indices.append(len(bones) - 1)
            sample_cursor += 1

        chain_bones[stem_id] = indices

    return bones, chain_bones


def bind_vertices(
    vertex_positions: NDArray[np.floating],
    stem_ids: NDArray[np.floating],
    chains: dict[int, dict],
    chain_bones: dict[int, list[int]],
    blend_band: float = BLEND_BAND,
) -> tuple[NDArray[np.integer], NDArray[np.integer], NDArray[np.floating]]:
    """Bind each vertex to two bones of its own chain.

    A vertex belongs to the chain its ``stem_id`` names, and to the segment its
    projected position along that chain falls in. Near a segment boundary it
    blends between the two adjacent bones, because a hard assignment creases the
    mesh at every boundary.

    The consuming shader stores exactly two bones and one weight, deriving the
    second as ``1 - weightA``, so the blend must involve at most two bones and
    the weights are normalised by construction. Blending is therefore restricted
    to immediate neighbours: the band cannot exceed half a segment on each side.

    Returns:
        Tuple of (index_a, index_b, weight_a) arrays, one entry per vertex.
    """
    verts = np.asarray(vertex_positions, dtype=np.float64).reshape(-1, 3)
    ids = np.asarray(stem_ids, dtype=np.float64).ravel().astype(np.int64)

    count = len(verts)
    index_a = np.zeros(count, dtype=np.int32)
    index_b = np.zeros(count, dtype=np.int32)
    weight_a = np.ones(count, dtype=np.float64)

    band = min(0.5, max(0.0, blend_band))

    for stem_id, bone_indices in chain_bones.items():
        mask = ids == stem_id
        if not np.any(mask):
            continue

        chain = chains[stem_id]
        segments = len(bone_indices)

        # Distance along the chain, in segments. Projection rather than raw
        # distance so a vertex on the far side of a thick trunk is not pushed
        # artificially far along by the trunk's own radius.
        travelled = (verts[mask] - chain["pivot"]) @ chain["direction"]
        segment_length = chain["extent"] / segments if segments else chain["extent"]
        segment_length = max(segment_length, 1e-9)

        position = np.clip(travelled / segment_length, 0.0, float(segments) - 1e-9)

        primary = np.floor(position).astype(np.int64)
        fraction = position - primary

        # Blend toward the previous bone in the first band of a segment and
        # toward the next one in the last band; hold rigid in between.
        neighbour = primary.copy()
        blend = np.zeros(len(primary), dtype=np.float64)

        if band > 0.0 and segments > 1:
            toward_previous = fraction < band
            toward_next = fraction > (1.0 - band)

            neighbour[toward_previous] = primary[toward_previous] - 1
            # Half weight exactly on the boundary, falling to zero a full band in.
            blend[toward_previous] = 0.5 * (1.0 - fraction[toward_previous] / band)

            neighbour[toward_next] = primary[toward_next] + 1
            blend[toward_next] = 0.5 * (1.0 - (1.0 - fraction[toward_next]) / band)

            # A neighbour off either end of the chain does not exist; those
            # vertices stay rigid on their own segment rather than binding to
            # another chain's bone.
            outside = (neighbour < 0) | (neighbour >= segments)
            neighbour[outside] = primary[outside]
            blend[outside] = 0.0

        blend[blend < MIN_BLEND_WEIGHT] = 0.0

        lookup = np.array(bone_indices, dtype=np.int32)
        index_a[mask] = lookup[primary]
        index_b[mask] = lookup[neighbour]
        weight_a[mask] = 1.0 - blend

    return index_a, index_b, weight_a


def pack_bone_uv(
    index_a: NDArray[np.integer],
    index_b: NDArray[np.integer],
    weight_a: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Pack the binding into the four-component stream the shader reads.

    Layout is ``(indexA, indexB, weightA, 0)``. ``weightB`` is deliberately
    absent: the shader derives it as ``1 - weightA``, so emitting it would let
    the pair fail to sum to one, and nothing renormalises them.

    Returns:
        Flat array of 4-component values (length = vertex_count * 4).
    """
    count = len(np.asarray(index_a).ravel())
    packed = np.zeros(count * 4, dtype=np.float64)
    packed[0::4] = np.asarray(index_a, dtype=np.float64).ravel()
    packed[1::4] = np.asarray(index_b, dtype=np.float64).ravel()
    packed[2::4] = np.asarray(weight_a, dtype=np.float64).ravel()
    return packed


def build_rig(
    vertex_positions: NDArray[np.floating],
    stem_ids: NDArray[np.floating],
    branch_extents: NDArray[np.floating],
    radii: NDArray[np.floating],
    pivot_positions: NDArray[np.floating],
    directions: NDArray[np.floating],
    hierarchy_depths: NDArray[np.floating],
    segments_per_chain: int = DEFAULT_SEGMENTS_PER_CHAIN,
    trunk_base_stiffness: float = 1.0,
    trunk_tip_stiffness: float = 0.6,
    branch_stiffness: float = 1.0,
    junction_falloff: float = JUNCTION_FALLOFF,
    blend_band: float = BLEND_BAND,
) -> BoneRig:
    """Build a complete bone rig from the mesher's per-vertex attributes.

    Args:
        vertex_positions: Nx3 array of vertex positions.
        stem_ids: Per-vertex branch chain id.
        branch_extents: Per-vertex chain length in metres.
        radii: Per-vertex branch radius.
        pivot_positions: Nx3 array of chain base positions.
        directions: Nx3 array of branch direction vectors.
        hierarchy_depths: Per-vertex depth, ``0`` for trunk. Required: parent
            recovery and the trunk stiffness split both depend on it.
        segments_per_chain: Bones per chain. One would move each chain rigidly,
            which is the failure the rig exists to fix.
        trunk_base_stiffness: Trunk bone stiffness at the root.
        trunk_tip_stiffness: Trunk bone stiffness at the crown.
        branch_stiffness: Scale on the side-branch radius curve.
        blend_band: Fraction of a segment over which adjacent bones blend.

    Returns:
        A ``BoneRig`` with bones in forward-sweep order and per-vertex binding.
    """
    chains = _chain_table(
        stem_ids, branch_extents, pivot_positions, directions, radii, hierarchy_depths
    )

    if not chains:
        return BoneRig()

    parents = resolve_chain_parents(chains)
    bones, chain_bones = build_bones(
        chains,
        parents,
        segments_per_chain=segments_per_chain,
        trunk_base_stiffness=trunk_base_stiffness,
        trunk_tip_stiffness=trunk_tip_stiffness,
        branch_stiffness=branch_stiffness,
        junction_falloff=junction_falloff,
    )
    index_a, index_b, weight_a = bind_vertices(
        vertex_positions, stem_ids, chains, chain_bones, blend_band=blend_band
    )

    return BoneRig(
        bones=bones,
        bone_indices_a=index_a,
        bone_indices_b=index_b,
        weights_a=weight_a,
    )


def validate_rig(rig: BoneRig) -> list[str]:
    """Check the invariants that fail silently in the consuming engine.

    Every one of these produces plausible geometry rather than an error when
    violated, which is why they are checked here rather than left to the engine:

    - **Parent index below own index.** The pose pass is a single forward sweep
      reading the parent's already-computed pose; a parent after its child reads
      uninitialised data.
    - **Bone indices in range.** They are used as raw buffer offsets with only a
      low-side clamp, so an out-of-range index samples another branch's pose.
    - **Weights in [0, 1].** ``weightB`` is derived as ``1 - weightA``, so a
      weight outside the range produces a negative partner weight.

    Returns:
        List of human-readable violations; empty when the rig is sound.
    """
    problems: list[str] = []

    for index, bone in enumerate(rig.bones):
        if bone.parent >= index:
            problems.append(
                f"bone {index} (stem {bone.stem_id} seg {bone.segment}) has parent "
                f"{bone.parent} at or after itself; forward sweep would read it unposed"
            )
        if bone.parent < -1:
            problems.append(f"bone {index} has parent {bone.parent}; roots must use -1")

    count = rig.bone_count
    for name, indices in (("indexA", rig.bone_indices_a), ("indexB", rig.bone_indices_b)):
        if len(indices) == 0:
            continue
        out_of_range = np.where((indices < 0) | (indices >= count))[0]
        for vertex in out_of_range[:8]:
            problems.append(
                f"vertex {int(vertex)} {name}={int(indices[vertex])} outside [0, {count})"
            )

    if len(rig.weights_a):
        bad = np.where((rig.weights_a < 0.0) | (rig.weights_a > 1.0))[0]
        for vertex in bad[:8]:
            problems.append(
                f"vertex {int(vertex)} weightA={rig.weights_a[vertex]:.6f} outside [0, 1]"
            )

    return problems


def measure_rest_pose_error(
    rig: BoneRig,
    vertex_positions: NDArray[np.floating],
) -> dict:
    """Measure how far the rig's rest pose drifts from the authored vertices.

    This is the single most important property of a rig and the one that fails
    most invisibly. The engine composes the skin transform as the posed
    transform times the inverse rest transform, so at zero wind every bone must
    resolve to identity. If the rest pose disagrees with the vertex positions at
    all, the mesh deforms while standing perfectly still — and because it
    deforms *consistently*, it reads as "the model is slightly wrong" rather
    than as a rigging bug.

    At rest the skin transform is identity by construction here, because bones
    store their rest frame and nothing else. What this measures is the weaker
    but still necessary property that every vertex is bound to a bone whose rest
    segment actually spans it: a vertex bound to a bone metres away is not
    wrong at rest, but it swings on the wrong arc the moment wind starts.

    Returns:
        Dict with ``max_distance``, ``mean_distance`` and ``worst_vertex``.
    """
    verts = np.asarray(vertex_positions, dtype=np.float64).reshape(-1, 3)

    if not rig.bones or len(verts) == 0:
        return {"max_distance": 0.0, "mean_distance": 0.0, "worst_vertex": -1}

    starts = np.array([bone.rest_position for bone in rig.bones])
    ends = np.array(
        [bone.rest_position + bone.rest_direction * bone.length for bone in rig.bones]
    )

    distances = np.zeros(len(verts))
    for i, vertex in enumerate(verts):
        bone = int(rig.bone_indices_a[i])
        distances[i] = _distance_to_segment(vertex, starts[bone], ends[bone])

    return {
        "max_distance": float(distances.max()),
        "mean_distance": float(distances.mean()),
        "worst_vertex": int(np.argmax(distances)),
    }
