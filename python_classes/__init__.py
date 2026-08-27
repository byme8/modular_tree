import os

from . import diagnostics, nodes, operators, panels, viewport


def register():
    if os.environ.get(diagnostics.ENV_VAR):
        diagnostics.set_enabled(True)

    operators.register()
    nodes.register()
    panels.register()
    viewport.register()


def unregister():
    viewport.unregister()
    operators.unregister()
    nodes.unregister()
    panels.unregister()
