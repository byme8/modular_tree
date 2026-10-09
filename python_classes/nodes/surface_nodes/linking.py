from __future__ import annotations

BLOCK_FILL_ID = "mt_BlockFillNode"
LEAF_CARDS_ID = "mt_LeafCardsNode"


def linked_surface_nodes(mesher, bl_idname: str) -> list:
    tree = mesher.id_data
    ordered = []
    names = set()

    for link in tree.links:
        node = link.to_node
        if (
            link.from_node.name == mesher.name
            and node.bl_idname == bl_idname
            and node.name not in names
        ):
            ordered.append(node)
            names.add(node.name)

    for node in tree.nodes:
        if node.bl_idname != bl_idname or node.name in names:
            continue
        reached = node.get_mesher()
        if reached is not None and reached.name == mesher.name:
            ordered.append(node)
            names.add(node.name)

    return ordered


def acting_surface_node(mesher, bl_idname: str):
    linked = linked_surface_nodes(mesher, bl_idname)
    return linked[0] if linked else None
