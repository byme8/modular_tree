"""Debounced tree rebuild.

Sockets store their value through a `get`/`set` pair, which Blender does not
pair with an `update` callback, so `MtreeSocket.write_stored` calls
`notify_socket_changed` here instead. An earlier version polled every node tree
at 10Hz because the callbacks appeared not to fire; they were in fact firing and
raising, which the socket clamp fix resolved.
"""

import logging

import bpy

logger = logging.getLogger(__name__)

_pending_timers = {}

DEBOUNCE_DELAY = 0.3

# PERF: a pose costs ~4.8ms at 112 bones, scales with bone count, and re-skins
# the whole mesh; a dragged slider firing per increment stalls the UI.
PREVIEW_DELAY = 0.05


def schedule_preview(node, method):
    key = (node.id_data.name, node.name, method)

    existing = _pending_timers.get(key)
    if existing is not None:
        try:
            bpy.app.timers.unregister(existing)
        except ValueError:
            logger.debug("Preview timer already expired for node %s", node.name)
        del _pending_timers[key]

    def _do_preview():
        _pending_timers.pop(key, None)
        node_tree = bpy.data.node_groups.get(key[0])
        if node_tree:
            resolved = node_tree.nodes.get(key[1])
            if resolved:
                getattr(resolved, method)()
        return None

    _pending_timers[key] = _do_preview
    bpy.app.timers.register(_do_preview, first_interval=PREVIEW_DELAY)


def schedule_build(node, method="build_tree", delay=DEBOUNCE_DELAY):
    if not getattr(node, "auto_update", True):
        return

    key = (node.id_data.name, node.name)

    # Cancel existing timer for this key
    existing = _pending_timers.get(key)
    if existing is not None:
        try:
            bpy.app.timers.unregister(existing)
        except ValueError:
            logger.debug("Timer already expired for node %s", node.name)
        del _pending_timers[key]

    def _do_build():
        _pending_timers.pop(key, None)
        node_tree = bpy.data.node_groups.get(key[0])
        if node_tree:
            resolved = node_tree.nodes.get(key[1])
            if resolved:
                getattr(resolved, method)()
        return None

    _pending_timers[key] = _do_build
    bpy.app.timers.register(_do_build, first_interval=delay)


def notify_socket_changed(socket):
    """Route a socket edit to whatever should rebuild because of it."""
    node = getattr(socket, "node", None)
    if node is None:
        return

    schedule_node_rebuild(node)


def schedule_node_rebuild(node):
    mesher = node.get_mesher() if hasattr(node, "get_mesher") else None
    if mesher is not None:
        schedule_build(mesher)
        return

    auto_method = getattr(node, "auto_build_method", None)
    if auto_method:
        schedule_build(node, method=auto_method)


def cancel_pending():
    for timer in list(_pending_timers.values()):
        try:
            bpy.app.timers.unregister(timer)
        except ValueError:
            logger.debug("Timer already expired during cancel")
    _pending_timers.clear()


def register():
    pass


def unregister():
    cancel_pending()
