from bpy.utils import register_class, unregister_class

from .block_fill_node import MTreeBlockFill
from .leaf_cards_node import MTreeLeafCards
from .leaf_clumps_node import MTreeLeafClumps

classes = [MTreeBlockFill, MTreeLeafCards, MTreeLeafClumps]


def register():
    for cls in classes:
        register_class(cls)


def unregister():
    for cls in classes:
        unregister_class(cls)
