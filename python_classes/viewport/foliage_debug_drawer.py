"""Viewport debug view for foliage vertex streams.

The stream values are invisible in a normal viewport: a wrong stiffness or a
phase that drifts within one branch still produces perfectly plausible geometry,
and the defect only appears once wind runs in the consuming engine. This draws
the mesh shaded by one stream at a time so the values can be checked before
export.

Channel numbering matches the consuming shader's own debug view, so a channel
checked here and a channel checked in-engine mean the same thing.
"""

from __future__ import annotations

import numpy as np

# The colour-mapping logic below is pure and is tested without Blender, so the
# Blender modules are imported lazily inside the functions that draw.
try:
    import bpy
except ImportError:  # running outside Blender, under test
    bpy = None

try:  # package import (addon runtime)
    from ..pivot_painter.foliage_streams import compute_all_streams
except ImportError:  # flat import (test harness puts pivot_painter/ on sys.path)
    from foliage_streams import compute_all_streams  # type: ignore[no-redef]

_draw_handler = None

# Cached GPU batch for the debug overlay. The draw handler runs on every viewport
# redraw, so the batch is rebuilt only when _cached_key changes.
_cached_key = None
_cached_batch = None
_cached_shader = None


def invalidate_debug_cache():
    """Drop the cached overlay batch, forcing a rebuild on the next redraw."""
    global _cached_key, _cached_batch, _cached_shader
    _cached_key = None
    _cached_batch = None
    _cached_shader = None


# Channel ids mirror the consuming shader's debug view numbering.
DEBUG_CHANNELS = [
    ("NONE", "Off", "Disable the debug view", 0),
    ("STIFFNESS", "1 Stiffness", "Wind authority: 1 rigid, 0 free tip", 1),
    ("PHASE", "2 Phase", "Per-blade id; must be flat within a branch", 2),
    ("OCCLUSION", "3 Occlusion", "Vertex colour red channel", 3),
    ("TRANSLUCENCY", "4 Translucency", "Vertex colour green channel", 4),
    ("NORMAL_BLEND", "5 Normal Blend", "Vertex colour blue channel", 5),
    ("TINT", "6 Tint Ramp", "Vertex colour alpha channel", 6),
    ("BLADE_HEIGHT", "7 Blade Height", "Wind displacement bound, in metres", 7),
    ("STRUCTURE", "8 Structure", "Branch chains in distinct colours", 8),
]

SOURCE_FLOAT_ATTRIBUTES = ["stem_id", "branch_extent", "radius", "hierarchy_depth"]
SOURCE_VECTOR_ATTRIBUTES = ["pivot_position", "direction"]


def value_to_viridis(values: np.ndarray) -> np.ndarray:
    """Map scalars in [0, 1] to a perceptually ordered colour ramp.

    A ramp is used rather than greyscale because a flat grey patch and a shallow
    grey gradient are hard to tell apart by eye, and telling them apart is
    exactly what the phase check requires.

    Args:
        values: Array of scalars, expected in [0, 1].

    Returns:
        Nx4 RGBA array.
    """
    v = np.clip(np.asarray(values, dtype=np.float64).ravel(), 0.0, 1.0)

    # Four-stop approximation of viridis: dark blue, teal, green, yellow.
    stops = np.array(
        [
            [0.267, 0.005, 0.329],
            [0.188, 0.408, 0.557],
            [0.208, 0.718, 0.472],
            [0.993, 0.906, 0.144],
        ]
    )
    scaled = v * (len(stops) - 1)
    lower = np.clip(np.floor(scaled).astype(int), 0, len(stops) - 2)
    frac = (scaled - lower)[:, None]

    rgb = stops[lower] * (1.0 - frac) + stops[lower + 1] * frac
    return np.column_stack([rgb, np.ones(len(v))])


def stem_id_to_color(stem_ids: np.ndarray) -> np.ndarray:
    """Map branch chain ids to visually distinct colours.

    Structure view answers "is this one blade or several?", so neighbouring ids
    must not land on neighbouring colours. The golden-ratio hash that generates
    phase is reused as the hue, which also makes this view a direct visual check
    that phase is constant per chain.

    Args:
        stem_ids: Per-vertex branch chain id.

    Returns:
        Nx4 RGBA array.
    """
    hue = np.mod(np.asarray(stem_ids, dtype=np.float64).ravel() * 0.61803398875, 1.0)

    # Cheap HSV->RGB at full saturation and value.
    h6 = hue * 6.0
    sector = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)

    p = np.zeros_like(hue)
    q = 1.0 - f
    t = f
    ones = np.ones_like(hue)

    table = [
        (ones, t, p),
        (q, ones, p),
        (p, ones, t),
        (p, q, ones),
        (t, p, ones),
        (ones, p, q),
    ]
    r = np.select([sector == i for i in range(6)], [table[i][0] for i in range(6)])
    g = np.select([sector == i for i in range(6)], [table[i][1] for i in range(6)])
    b = np.select([sector == i for i in range(6)], [table[i][2] for i in range(6)])

    return np.column_stack([r, g, b, np.ones(len(hue))])


def compute_channel_colors(streams: dict, stem_ids: np.ndarray, channel: str) -> np.ndarray | None:
    """Compute per-vertex debug colours for one channel.

    Blade height is normalised by its own maximum for display only — it is a
    metre value, so showing it raw would clip everything above one metre to the
    same colour.

    Args:
        streams: Output of ``compute_all_streams``.
        stem_ids: Per-vertex branch chain id.
        channel: Channel identifier from ``DEBUG_CHANNELS``.

    Returns:
        Nx4 RGBA array, or None if the channel is unknown or disabled.
    """
    colors = streams["colors"]

    if channel == "STIFFNESS":
        return value_to_viridis(streams["stiffness"])
    if channel == "PHASE":
        return value_to_viridis(streams["phase"])
    if channel == "OCCLUSION":
        return value_to_viridis(colors[0::4])
    if channel == "TRANSLUCENCY":
        return value_to_viridis(colors[1::4])
    if channel == "NORMAL_BLEND":
        return value_to_viridis(colors[2::4])
    if channel == "TINT":
        return value_to_viridis(colors[3::4])
    if channel == "BLADE_HEIGHT":
        heights = streams["blade_height"]
        peak = max(float(heights.max()) if len(heights) else 0.0, 1e-6)
        return value_to_viridis(heights / peak)
    if channel == "STRUCTURE":
        return stem_id_to_color(stem_ids)

    return None


def _find_node_tree_for_object(obj):
    """Find the MTree node tree whose mesher produced this object."""
    for node_tree in bpy.data.node_groups:
        if node_tree.bl_idname != "mt_MtreeNodeTree":
            continue
        for node in node_tree.nodes:
            if node.bl_idname == "mt_MesherNode" and getattr(node, "tree_object", "") == obj.name:
                return node_tree
    return None


def read_mesh_streams(mesh, settings: dict | None = None) -> tuple[dict, np.ndarray] | None:
    """Compute foliage streams from a mesh's mesher attributes.

    The stiffness and shading settings must match the ones the export will use,
    or the preview shows values that are never written.

    Returns None when the mesh lacks the attributes, so the overlay simply does
    not draw rather than raising inside a draw handler.
    """
    required = SOURCE_FLOAT_ATTRIBUTES + SOURCE_VECTOR_ATTRIBUTES
    if any(name not in mesh.attributes for name in required):
        return None

    num_verts = len(mesh.vertices)
    if num_verts == 0:
        return None

    positions = np.zeros(num_verts * 3)
    mesh.vertices.foreach_get("co", positions)

    values = {}
    for name in SOURCE_FLOAT_ATTRIBUTES:
        buffer = np.zeros(num_verts)
        mesh.attributes[name].data.foreach_get("value", buffer)
        values[name] = buffer

    for name in SOURCE_VECTOR_ATTRIBUTES:
        buffer = np.zeros(num_verts * 3)
        mesh.attributes[name].data.foreach_get("vector", buffer)
        values[name] = buffer.reshape(-1, 3)

    streams = compute_all_streams(
        vertex_positions=positions.reshape(-1, 3),
        stem_ids=values["stem_id"],
        branch_extents=values["branch_extent"],
        radii=values["radius"],
        pivot_positions=values["pivot_position"],
        directions=values["direction"],
        hierarchy_depths=values["hierarchy_depth"],
        **(settings or {}),
    )
    return streams, values["stem_id"]


def draw_foliage_debug():
    """Draw the active object's mesh shaded by the selected stream."""
    import gpu
    from gpu_extras.batch import batch_for_shader

    scene = bpy.context.scene
    channel = getattr(scene, "mtree_foliage_debug_channel", "NONE")
    if channel == "NONE":
        return

    obj = bpy.context.active_object
    if obj is None or obj.type != "MESH":
        return

    # Preview with the same settings the export will use, so what is shown is
    # what gets written.
    try:
        from ..nodes.export_nodes.foliage_wind_node import find_foliage_wind_settings

        settings = find_foliage_wind_settings(_find_node_tree_for_object(obj))
    except ImportError:
        settings = None

    # This handler runs on EVERY viewport redraw — every mouse move, every
    # navigation frame. Recomputing the streams and rebuilding the GPU batch each
    # time stalls the whole interface on a tree of any size, so the result is
    # cached and only rebuilt when something that affects it actually changes.
    cache_key = (
        obj.name,
        channel,
        len(obj.data.vertices),
        tuple(sorted((settings or {}).items())),
    )

    global _cached_key, _cached_batch, _cached_shader
    if cache_key == _cached_key and _cached_batch is not None:
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(True)
        _cached_shader.bind()
        _cached_batch.draw(_cached_shader)
        gpu.state.depth_mask_set(False)
        gpu.state.depth_test_set("NONE")
        return

    result = read_mesh_streams(obj.data, settings)
    if result is None:
        return

    streams, stem_ids = result
    colors = compute_channel_colors(streams, stem_ids, channel)
    if colors is None:
        return

    mesh = obj.data
    mesh.calc_loop_triangles()

    num_verts = len(mesh.vertices)
    coords = np.zeros(num_verts * 3)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape(-1, 3)

    tri_count = len(mesh.loop_triangles)
    tri_verts = np.zeros(tri_count * 3, dtype=int)
    mesh.loop_triangles.foreach_get("vertices", tri_verts)
    indices = tri_verts.reshape(-1, 3)

    # Draw in world space so the overlay tracks the object transform.
    matrix = np.array(obj.matrix_world.transposed())
    homogeneous = np.column_stack([coords, np.ones(len(coords))])
    world_coords = (homogeneous @ matrix)[:, :3]

    shader = gpu.shader.from_builtin("SMOOTH_COLOR")
    batch = batch_for_shader(
        shader,
        "TRIS",
        {"pos": world_coords.tolist(), "color": colors.reshape(-1, 4).tolist()},
        indices=indices.tolist(),
    )

    _cached_key = cache_key
    _cached_batch = batch
    _cached_shader = shader

    gpu.state.depth_test_set("LESS_EQUAL")
    gpu.state.depth_mask_set(True)
    shader.bind()
    batch.draw(shader)
    gpu.state.depth_mask_set(False)
    gpu.state.depth_test_set("NONE")


def register():
    """Register the scene property and the draw handler."""
    global _draw_handler

    bpy.types.Scene.mtree_foliage_debug_channel = bpy.props.EnumProperty(
        name="Foliage Debug",
        description="Shade the active tree by one foliage vertex stream",
        items=[(ident, label, desc) for ident, label, desc, _ in DEBUG_CHANNELS],
        default="NONE",
    )

    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_foliage_debug, (), "WINDOW", "POST_VIEW"
        )


def unregister():
    """Unregister the draw handler and the scene property."""
    global _draw_handler
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        _draw_handler = None

    if hasattr(bpy.types.Scene, "mtree_foliage_debug_channel"):
        del bpy.types.Scene.mtree_foliage_debug_channel
