import bpy

from ..base_types.socket import MtreeSocket


class MtreeIntSocket(bpy.types.NodeSocket, MtreeSocket):
    bl_idname = "mt_IntSocket"
    bl_label = "Mtree Int Socket"

    color = (1.0, 0.4, 0.216, 0.5)

    min_value: bpy.props.IntProperty(default=-(10**9))
    max_value: bpy.props.IntProperty(default=10**9)

    property_value: bpy.props.IntProperty(
        default=0,
        update=lambda self, context: self.on_value_changed(),
    )

    def draw(self, context, layout, node, text):
        if self.is_output or self.is_linked:
            layout.label(text=text)
        else:
            layout.prop(self, "property_value", text=text)
