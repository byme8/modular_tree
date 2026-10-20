import bpy

from ..base_types.socket import MtreeSocket


class MeshSocket(bpy.types.NodeSocket, MtreeSocket):
    bl_idname = "mt_MeshSocket"
    bl_label = "Mesh Socket"

    color = (0.25, 0.55, 0.95, 1)

    def draw(self, context, layout, node, text):
        layout.label(text=text)
