import bpy

GRAPH_VERSION = 1


class MtreeNodeTree(bpy.types.NodeTree):
    bl_idname = "mt_MtreeNodeTree"
    bl_label = "Mtree"
    bl_icon = "ONIONSKIN_ON"

    graph_version: bpy.props.IntProperty(default=0)

    def update(self):
        """Called when links or topology change.

        Deliberately does not trigger a rebuild. Building removes and recreates
        objects and armatures, and doing that from a timer while Blender is
        still settling a link change crashes it outright. Parameter edits still
        auto-rebuild through the socket callbacks; topology changes wait for
        Generate.
        """
        from ..pipeline import needs_migration, rejected_links, schedule_rejection

        if needs_migration(self):
            return
        if rejected_links(self):
            schedule_rejection(self)
