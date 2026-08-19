from . import crown_shape_drawer, foliage_debug_drawer


def register():
    crown_shape_drawer.register()
    foliage_debug_drawer.register()


def unregister():
    foliage_debug_drawer.unregister()
    crown_shape_drawer.unregister()
