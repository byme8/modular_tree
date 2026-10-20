"""The node graph is one strict left-to-right pipeline:

    structure (Trunk -> Branches ...) -> Tree Mesher -> Leaf Cards | Leaf Clumps -> Foliage Wind
        -> Tree Variants -> Mesh Export / Foliage Export

Each stage finds its neighbours by walking links from itself, never by scanning
the whole tree, so two pipelines in one node tree cannot read each other's
settings.
"""

from __future__ import annotations

TREE_SOCKET = "mt_TreeSocket"
MESH_SOCKET = "mt_MeshSocket"
VARIANTS_SOCKET = "mt_VariantsSocket"
CHAIN_SOCKETS = frozenset({TREE_SOCKET, MESH_SOCKET, VARIANTS_SOCKET})

TRUNK_ID = "mt_TrunkNode"
BRANCH_ID = "mt_BranchNode"
GROWTH_ID = "mt_GrowthNode"
PIPE_RADIUS_ID = "mt_PipeRadiusNode"
MESHER_ID = "mt_MesherNode"
BLOCK_FILL_ID = "mt_BlockFillNode"
LEAF_CARDS_ID = "mt_LeafCardsNode"
LEAF_CLUMPS_ID = "mt_LeafClumpsNode"
FOLIAGE_WIND_ID = "mt_FoliageWindNode"
VARIANTS_ID = "mt_VariantsNode"
MESH_EXPORT_ID = "mt_MeshExportNode"
FOLIAGE_EXPORT_ID = "mt_FoliageExportNode"

STRUCTURE_IDS = frozenset({TRUNK_ID, BRANCH_ID, GROWTH_ID, PIPE_RADIUS_ID})
LEAF_STAGES = frozenset({LEAF_CARDS_ID, LEAF_CLUMPS_ID})
SINGLE_TREE_STAGES = frozenset({MESHER_ID, FOLIAGE_WIND_ID}) | LEAF_STAGES
MESH_SIDE_IDS = (
    frozenset({FOLIAGE_WIND_ID, VARIANTS_ID, MESH_EXPORT_ID, FOLIAGE_EXPORT_ID}) | LEAF_STAGES
)
EXCLUSIVE_STAGES = frozenset({FOLIAGE_WIND_ID}) | LEAF_STAGES

ACCEPTED_UPSTREAM = {
    BRANCH_ID: STRUCTURE_IDS,
    GROWTH_ID: STRUCTURE_IDS,
    PIPE_RADIUS_ID: STRUCTURE_IDS,
    MESHER_ID: STRUCTURE_IDS,
    LEAF_CARDS_ID: frozenset({MESHER_ID}),
    LEAF_CLUMPS_ID: frozenset({MESHER_ID}),
    FOLIAGE_WIND_ID: frozenset({MESHER_ID}) | LEAF_STAGES,
    VARIANTS_ID: SINGLE_TREE_STAGES,
    MESH_EXPORT_ID: SINGLE_TREE_STAGES | {VARIANTS_ID},
    FOLIAGE_EXPORT_ID: SINGLE_TREE_STAGES | {VARIANTS_ID},
}


def link_is_compatible(link) -> bool:
    from_chain = link.from_socket.bl_idname in CHAIN_SOCKETS
    to_chain = link.to_socket.bl_idname in CHAIN_SOCKETS
    if not from_chain and not to_chain:
        return True
    if not (from_chain and to_chain):
        return False
    return link.from_node.bl_idname in ACCEPTED_UPSTREAM.get(link.to_node.bl_idname, ())


def _link_key(link) -> tuple:
    return (
        link.from_node.name,
        link.from_socket.identifier,
        link.to_node.name,
        link.to_socket.identifier,
    )


def allowed_links(tree) -> list:
    allowed = []
    claimed = set()
    for link in tree.links:
        if not link_is_compatible(link):
            continue
        if link.to_node.bl_idname in EXCLUSIVE_STAGES:
            outlet = (link.from_node.name, link.from_socket.identifier)
            if outlet in claimed:
                continue
            claimed.add(outlet)
        allowed.append(link)
    return allowed


def rejected_links(tree) -> list:
    kept = {_link_key(link) for link in allowed_links(tree)}
    return [link for link in tree.links if _link_key(link) not in kept]


def upstream_node(node):
    for link in allowed_links(node.id_data):
        if link.to_node.name == node.name and link.to_socket.bl_idname in CHAIN_SOCKETS:
            return link.from_node
    return None


def downstream_nodes(node) -> list:
    return [
        link.to_node
        for link in allowed_links(node.id_data)
        if link.from_node.name == node.name and link.from_socket.bl_idname in CHAIN_SOCKETS
    ]


def resolve_trunk(mesher) -> tuple:
    node = upstream_node(mesher)
    if node is None:
        return None, "Connect a Trunk's structure to the Skeleton input"

    seen = set()
    while node.bl_idname != TRUNK_ID:
        if node.name in seen:
            return None, "The structure feeding the Skeleton loops back on itself"
        seen.add(node.name)
        parent = upstream_node(node)
        if parent is None:
            return None, f"{node.name} has no Trunk upstream"
        node = parent
    return node, ""


def single_tree_chain(mesher) -> list:
    chain = [mesher]
    names = {mesher.name}
    current = mesher
    while True:
        following = next(
            (
                node
                for node in downstream_nodes(current)
                if node.bl_idname in EXCLUSIVE_STAGES and node.name not in names
            ),
            None,
        )
        if following is None:
            return chain
        chain.append(following)
        names.add(following.name)
        current = following


def chain_stage(mesher, bl_idname: str):
    if mesher is None:
        return None
    return next((node for node in single_tree_chain(mesher) if node.bl_idname == bl_idname), None)


def mesher_upstream(node):
    seen = set()
    while node is not None and node.name not in seen:
        if node.bl_idname == MESHER_ID:
            return node
        if node.bl_idname not in MESH_SIDE_IDS:
            return None
        seen.add(node.name)
        node = upstream_node(node)
    return None


def mesher_downstream(node):
    tree = node.id_data
    edges: dict[str, list[str]] = {}
    for link in allowed_links(tree):
        edges.setdefault(link.from_node.name, []).append(link.to_node.name)

    seen = {node.name}
    queue = [node.name]
    while queue:
        name = queue.pop(0)
        candidate = tree.nodes.get(name)
        if candidate is not None and candidate.bl_idname == MESHER_ID:
            return candidate
        for following in edges.get(name, ()):
            if following not in seen:
                seen.add(following)
                queue.append(following)
    return None


def find_mesher(node):
    if node.bl_idname in MESH_SIDE_IDS:
        return mesher_upstream(node)
    return mesher_downstream(node)


def chain_sockets_conform(node) -> bool:
    return _sockets_match(node.inputs, getattr(node, "chain_input", None)) and _sockets_match(
        node.outputs, getattr(node, "chain_output", None)
    )


def _sockets_match(sockets, expected) -> bool:
    chain = [socket for socket in sockets if socket.bl_idname in CHAIN_SOCKETS]
    if expected is None:
        return not chain
    return (
        len(chain) == 1 and sockets[0].bl_idname == expected[0] and sockets[0].name == expected[1]
    )


def conform_chain_sockets(node) -> bool:
    changed = _conform(node.inputs, getattr(node, "chain_input", None))
    return _conform(node.outputs, getattr(node, "chain_output", None)) or changed


def _conform(sockets, expected) -> bool:
    if _sockets_match(sockets, expected):
        return False

    for socket in [socket for socket in sockets if socket.bl_idname in CHAIN_SOCKETS]:
        sockets.remove(socket)

    if expected is not None:
        socket = sockets.new(expected[0], expected[1])
        socket.is_property = False
        sockets.move(len(sockets) - 1, 0)
    return True


_pending_rejections: set[str] = set()


def schedule_rejection(tree):
    import bpy

    _pending_rejections.add(tree.name)
    if not bpy.app.timers.is_registered(flush_rejections):
        bpy.app.timers.register(flush_rejections, first_interval=0.0)


def cancel_rejections():
    import bpy

    if bpy.app.timers.is_registered(flush_rejections):
        bpy.app.timers.unregister(flush_rejections)
    _pending_rejections.clear()


def flush_rejections():
    import bpy

    from .. import diagnostics

    names = sorted(_pending_rejections)
    _pending_rejections.clear()
    for name in names:
        tree = bpy.data.node_groups.get(name)
        if tree is None or needs_migration(tree):
            continue
        for link in rejected_links(tree):
            diagnostics.log(
                "rejected link %s.%s -> %s.%s",
                link.from_node.name,
                link.from_socket.name,
                link.to_node.name,
                link.to_socket.name,
            )
            tree.links.remove(link)
    return None


def needs_migration(tree) -> bool:
    return any(
        node.bl_idname == BLOCK_FILL_ID
        or (node.bl_idname.startswith("mt_") and not chain_sockets_conform(node))
        for node in tree.nodes
    )
