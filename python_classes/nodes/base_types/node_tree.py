import bpy


class MtreeNodeTree(bpy.types.NodeTree):
    bl_idname = "mt_MtreeNodeTree"
    bl_label = "Mtree"
    bl_icon = "ONIONSKIN_ON"

    def update(self):
        """Called when links or topology change.

        Deliberately does not trigger a rebuild. Building removes and recreates
        objects and armatures, and doing that from a timer while Blender is
        still settling a link change crashes it outright. Parameter edits still
        auto-rebuild through the socket callbacks; topology changes wait for
        Generate.
        """
