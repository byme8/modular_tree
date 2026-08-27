import bpy


class MtreeSocket:
    is_property: bpy.props.BoolProperty(default=True)
    property_name: bpy.props.StringProperty()
    property_value: None
    color: (0.5, 0.5, 0.5, 1)
    description: bpy.props.StringProperty(name="Description", default="")

    def draw_color(self, context, node):
        return self.color

    def request_rebuild(self):
        from ..debounce import notify_socket_changed

        notify_socket_changed(self)

    def on_value_changed(self):
        """Clamp to bounds, then schedule a rebuild.

        property_value must stay a plain, persisted property: it is the name
        Blender wrote into existing .blend files, and backing it with a get/set
        pair makes every saved graph load as zero. So bounds are enforced after
        the fact rather than in a setter — `self["property_value"] = ...` is not
        available, because sockets do not support id properties.

        The reassignment re-enters this callback once; the second pass is
        already within bounds and stops.
        """
        clamped = max(self.min_value, min(self.max_value, self.property_value))
        if clamped != self.property_value:
            self.property_value = clamped
            return

        self.request_rebuild()
