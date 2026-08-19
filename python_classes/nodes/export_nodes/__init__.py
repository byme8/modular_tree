from bpy.utils import register_class, unregister_class

from .foliage_wind_node import MTreeFoliageWind
from .mesh_export_node import MTreeMeshExport
from .pivot_painter_node import MTreePivotPainterExport

classes = [MTreePivotPainterExport, MTreeFoliageWind, MTreeMeshExport]


def register():
    for cls in classes:
        register_class(cls)


def unregister():
    for cls in classes:
        unregister_class(cls)
