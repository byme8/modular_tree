
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

try:
    from .foliage_streams import JUNCTION_FALLOFF, compute_stiffness
except ImportError:
    from foliage_streams import JUNCTION_FALLOFF, compute_stiffness

DEFAULT_SEGMENTS_PER_CHAIN = 4

MIN_SEGMENTED_LENGTH = 1e-3

BLEND_BAND = 0.5

MIN_BLEND_WEIGHT = 1e-3

ANTIPARALLEL_BAND = 0.4


@dataclass
class Bone:

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
            "direction": _safe_normalise(dirs[mask].mean(axis=0)),
            "radius": float(radius_values[mask].max()),
            "depth": int(depths[mask][0]),
        }

    return chains


def _safe_normalise(vector: NDArray[np.floating]) -> NDArray[np.floating]:
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return np.asarray(vector, dtype=np.float64) / length


def resolve_chain_parents(chains: dict[int, dict]) -> dict[int, int]:
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
    axis = end - start
    axis_length_sq = float(axis @ axis)

    if axis_length_sq <= 1e-12:
        return float(np.linalg.norm(point - start))

    t = float((point - start) @ axis) / axis_length_sq
    t = min(1.0, max(0.0, t))

    return float(np.linalg.norm(point - (start + axis * t)))


def _segment_count(extent: float, segments_per_chain: int) -> int:
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
    ordered = sorted(chains.values(), key=lambda chain: (chain["depth"], chain["stem_id"]))

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

    sample_cursor = 0

    for chain in ordered:
        stem_id = chain["stem_id"]
        extent = chain["extent"]
        count = _segment_count(extent, segments_per_chain)
        segment_length = extent / count if count else extent

        parent_stem = parents.get(stem_id, -1)
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

        travelled = (verts[mask] - chain["pivot"]) @ chain["direction"]
        segment_length = chain["extent"] / segments if segments else chain["extent"]
        segment_length = max(segment_length, 1e-9)

        position = np.clip(travelled / segment_length, 0.0, float(segments) - 1e-9)

        primary = np.floor(position).astype(np.int64)
        fraction = position - primary

        neighbour = primary.copy()
        blend = np.zeros(len(primary), dtype=np.float64)

        if band > 0.0 and segments > 1:
            toward_previous = fraction < band
            toward_next = fraction > (1.0 - band)

            neighbour[toward_previous] = primary[toward_previous] - 1
            blend[toward_previous] = 0.5 * (1.0 - fraction[toward_previous] / band)

            neighbour[toward_next] = primary[toward_next] + 1
            blend[toward_next] = 0.5 * (1.0 - (1.0 - fraction[toward_next]) / band)

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


def _axis_rotation(axis: NDArray[np.floating], angle: float) -> NDArray[np.floating]:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)

    unit = axis / norm
    cross = np.array(
        [
            [0.0, -unit[2], unit[1]],
            [unit[2], 0.0, -unit[0]],
            [-unit[1], unit[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def pose_rig(
    rig: BoneRig,
    wind_angle_degrees: float,
    strength: float,
) -> list[NDArray[np.floating]]:
    if not rig.bones:
        return []

    radians = np.radians(wind_angle_degrees)
    wind = np.array([np.cos(radians), np.sin(radians), 0.0])

    segments_in_chain: dict[int, int] = {}
    for bone in rig.bones:
        segments_in_chain[bone.stem_id] = segments_in_chain.get(bone.stem_id, 0) + 1

    matrices: list[NDArray[np.floating]] = []

    for bone in rig.bones:
        parent_rotation = (
            matrices[bone.parent][:3, :3] if bone.parent >= 0 else np.eye(3)
        )

        posed_direction = parent_rotation @ bone.rest_direction
        posed_norm = float(np.linalg.norm(posed_direction))
        if posed_norm > 1e-12:
            posed_direction = posed_direction / posed_norm

        sway = max(0.0, 1.0 - bone.stiffness)
        amount = sway * sway * strength / segments_in_chain[bone.stem_id]

        axis = np.cross(posed_direction, wind)
        leverage = float(np.linalg.norm(axis))

        if leverage < 1e-9:
            local = np.eye(3)
        else:
            easing = min(1.0, leverage / ANTIPARALLEL_BAND)
            world = _axis_rotation(axis / leverage, amount * easing * np.pi)
            local = parent_rotation.T @ world @ parent_rotation

        pivot = bone.rest_position
        matrix = np.eye(4)
        matrix[:3, :3] = local
        matrix[:3, 3] = pivot - local @ pivot

        matrices.append(
            matrices[bone.parent] @ matrix if bone.parent >= 0 else matrix
        )

    return matrices


def deform_vertices(
    rig: BoneRig,
    vertex_positions: NDArray[np.floating],
    matrices: list[NDArray[np.floating]],
) -> NDArray[np.floating]:
    verts = np.asarray(vertex_positions, dtype=np.float64).reshape(-1, 3)

    if not matrices or len(verts) == 0:
        return verts.copy()

    stacked = np.array(matrices)
    homogeneous = np.column_stack([verts, np.ones(len(verts))])

    first = np.einsum("vij,vj->vi", stacked[rig.bone_indices_a], homogeneous)[:, :3]
    second = np.einsum("vij,vj->vi", stacked[rig.bone_indices_b], homogeneous)[:, :3]

    weights = rig.weights_a.reshape(-1, 1)
    return weights * first + (1.0 - weights) * second
