import bpy
import nodeitems_utils
from bpy.utils import register_class, unregister_class

from . import (
    base_types,
    debounce,
    export_nodes,
    migration,
    node_categories,
    pipeline,
    properties,
    sockets,
    surface_nodes,
    tree_function_nodes,
)


def register():
    base_types.register()
    sockets.register()
    tree_function_nodes.register()
    export_nodes.register()
    surface_nodes.register()
    properties.register()
    node_categories.register()
    debounce.register()

    if migration.migrate_loaded_trees not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(migration.migrate_loaded_trees)
    bpy.app.timers.register(migration.migrate_all_trees, first_interval=0.0)


def unregister():
    if migration.migrate_loaded_trees in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(migration.migrate_loaded_trees)
    if bpy.app.timers.is_registered(migration.migrate_all_trees):
        bpy.app.timers.unregister(migration.migrate_all_trees)
    pipeline.cancel_rejections()

    debounce.unregister()
    node_categories.unregister()
    base_types.unregister()
    sockets.unregister()
    properties.unregister()
    surface_nodes.unregister()
    export_nodes.unregister()
    tree_function_nodes.unregister()
