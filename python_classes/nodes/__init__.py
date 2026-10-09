import bpy
import nodeitems_utils
from bpy.utils import register_class, unregister_class

from . import (
    base_types,
    debounce,
    export_nodes,
    node_categories,
    properties,
    sockets,
    surface_nodes,
    tree_function_nodes,
)


@bpy.app.handlers.persistent
def _migrate_loaded_nodes(_file_path):
    """Bring nodes saved by older versions up to the current socket layout.

    A node's init() runs only when it is created, so sockets restored from a
    .blend keep whatever shape they were saved with.
    """
    for tree in bpy.data.node_groups:
        if tree.bl_idname != "mt_MtreeNodeTree":
            continue
        for node in tree.nodes:
            migrate = getattr(node, "migrate_sockets", None)
            if migrate is not None:
                migrate()


def register():
    base_types.register()
    sockets.register()
    tree_function_nodes.register()
    export_nodes.register()
    surface_nodes.register()
    properties.register()
    node_categories.register()
    debounce.register()

    if _migrate_loaded_nodes not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_migrate_loaded_nodes)


def unregister():
    if _migrate_loaded_nodes in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_migrate_loaded_nodes)

    debounce.unregister()
    node_categories.unregister()
    base_types.unregister()
    sockets.unregister()
    properties.unregister()
    surface_nodes.unregister()
    export_nodes.unregister()
    tree_function_nodes.unregister()
