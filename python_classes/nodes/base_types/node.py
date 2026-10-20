from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bpy.types import NodeSocket, NodeTree

SEED_PROPERTY = "seed"


class MtreeNode:
    chain_input = None
    chain_output = None

    @classmethod
    def poll(cls, nodeTree: NodeTree) -> bool:
        return nodeTree.bl_idname == "mt_MtreeNodeTree"

    def get_node_tree(self) -> NodeTree:
        return self.id_data

    def get_child_nodes(self) -> list:
        child_nodes = []
        for socket in self.outputs:
            for link in socket.links:
                child_nodes.append(link.to_node)
        return child_nodes

    def add_chain_sockets(self) -> None:
        if self.chain_input is not None:
            self.add_input(*self.chain_input, is_property=False)
        if self.chain_output is not None:
            self.add_output(*self.chain_output, is_property=False)

    def add_input(self, socket_type: str, name: str, **kwargs) -> NodeSocket:
        socket = self.inputs.new(socket_type, name)
        for key, value in kwargs.items():
            setattr(socket, key, value)
        return socket

    def add_output(self, socket_type: str, name: str, **kwargs) -> NodeSocket:
        socket = self.outputs.new(socket_type, name)
        for key, value in kwargs.items():
            setattr(socket, key, value)
        return socket

    def get_mesher(self) -> MtreeNode | None:
        from ..pipeline import find_mesher

        return find_mesher(self)

    # Node events, don't override ----------------

    def draw_buttons(self, context, layout) -> None:
        self.draw(context, layout)

    def draw_buttons_ext(self, context, layout) -> None:
        self.draw_inspector(context, layout)

    # Functions that can be overriden -------------

    def draw(self, context, layout) -> None:
        pass

    def draw_inspector(self, context, layout) -> None:
        pass


class MtreeFunctionNode(MtreeNode):
    chain_input = ("mt_TreeSocket", "Tree")
    chain_output = ("mt_TreeSocket", "Tree")
    exposed_parameters = []  # List defined for each sub class
    advanced_parameters = []
    tree_function = None  # tree function type, as defined in m_tree. Should be overriden

    def draw(self, context, layout):
        for parameter in self.exposed_parameters:
            layout.prop(self, parameter)

    def draw_inspector(self, context, layout):
        for parameter in self.exposed_parameters + self.advanced_parameters:
            layout.prop(self, parameter)

    def construct_function(self, seed_offset: int = 0):
        """seed_offset is applied in flight, never written to the sockets: a
        mutated graph would be observed by the change poller mid-build.
        """
        if self.tree_function is None:
            raise ValueError(f"tree_function not defined for {self.__class__.__name__}")
        function_instance = self.tree_function()
        for parameter in self.exposed_parameters:
            setattr(function_instance, parameter, getattr(self, parameter))

        for input_socket in self.inputs:
            if input_socket.is_property:
                if input_socket.bl_idname == "mt_PropertySocket":
                    property = input_socket.get_property()
                    setattr(function_instance, input_socket.property_name, property)
                else:
                    value = input_socket.property_value
                    if seed_offset and input_socket.property_name == SEED_PROPERTY:
                        value = value + seed_offset
                    setattr(function_instance, input_socket.property_name, value)

        for child in self.get_child_nodes():
            if isinstance(child, MtreeFunctionNode):
                child_function = child.construct_function(seed_offset)
                function_instance.add_child(child_function)

        return function_instance


class MtreePropertyNode(MtreeNode):
    property_type = None  # tree Property type, as defined in m_tree. Should be overriden

    def get_property(self):
        if self.property_type is None:
            raise ValueError(f"property_type not defined for {self.__class__.__name__}")
        property = self.property_type()
        for input_socket in self.inputs:
            if input_socket.is_property:
                setattr(property, input_socket.property_name, input_socket.property_value)
        return property
