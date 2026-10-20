import bpy

from ..base_types.socket import MtreeSocket


class VariantsSocket(bpy.types.NodeSocket, MtreeSocket):
    bl_idname = "mt_VariantsSocket"
    bl_label = "Variants Socket"

    color = (0.85, 0.45, 0.85, 1)

    def draw(self, context, layout, node, text):
        layout.label(text=text)
