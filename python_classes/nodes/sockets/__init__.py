import bpy
from bpy.utils import register_class, unregister_class

from .bool_socket import MtreeBoolSocket
from .float_socket import MtreeFloatSocket
from .int_socket import MtreeIntSocket
from .mesh_socket import MeshSocket
from .property_socket import MtreePropertySocket
from .tree_socket import TreeSocket
from .variants_socket import VariantsSocket

classes = [
    MtreeBoolSocket,
    MtreeFloatSocket,
    TreeSocket,
    MeshSocket,
    VariantsSocket,
    MtreeIntSocket,
    MtreePropertySocket,
]


def register():
    for cls in classes:
        register_class(cls)


def unregister():
    for cls in reversed(classes):
        unregister_class(cls)
