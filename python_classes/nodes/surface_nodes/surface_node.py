from __future__ import annotations

from ..base_types.node import MtreeNode
from ..debounce import schedule_node_rebuild


def on_update_setting(node, context):
    schedule_node_rebuild(node)


class SurfaceNode(MtreeNode):
    chain_input = ("mt_MeshSocket", "Mesh")
    chain_output = ("mt_MeshSocket", "Mesh")
    settings_layout: tuple[str, ...] = ()

    def init(self, context):
        self.add_chain_sockets()

    def draw(self, context, layout):
        if self.status_message:
            status_row = layout.row()
            status_row.alert = self.status_is_error
            icon = "ERROR" if self.status_is_error else "CHECKMARK"
            status_row.label(text=self.status_message, icon=icon)

        for name in self.settings_layout:
            layout.prop(self, name)

        mesher = self.get_mesher()
        if mesher is None:
            layout.label(text="Connect to a Tree Mesher", icon="ERROR")

        generate_row = layout.row()
        generate_row.enabled = mesher is not None
        props = generate_row.operator("mtree.node_function", text="Generate")
        props.node_tree_name = self.get_node_tree().name
        props.node_name = self.name
        props.function_name = "generate"

    def generate(self):
        mesher = self.get_mesher()
        if mesher is None:
            self.status_message = "Connect to a Tree Mesher"
            self.status_is_error = True
            return
        mesher.build_tree()

    def set_status(self, message: str, is_error: bool = False):
        self.status_message = message
        self.status_is_error = is_error
