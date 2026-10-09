from bpy.utils import register_class, unregister_class

from .block_fill_node import MTreeBlockFill
from .leaf_cards_node import MTreeLeafCards
from .linking import BLOCK_FILL_ID, LEAF_CARDS_ID, acting_surface_node, linked_surface_nodes

classes = [MTreeBlockFill, MTreeLeafCards]


def register():
    for cls in classes:
        register_class(cls)


def unregister():
    for cls in classes:
        unregister_class(cls)
