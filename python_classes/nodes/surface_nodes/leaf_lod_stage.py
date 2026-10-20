from __future__ import annotations

from ...pivot_painter.tree_asset import resolve_mesh_object
from ..pipeline import VARIANTS_ID, downstream_nodes, single_tree_chain


def on_update_preview_lod(node, context):
    node.show_preview_lod()


class LeafLodStage:
    def preview_level(self) -> int:
        return int(self.preview_lod) if self.build_lods else 0

    def tree_objects(self) -> list:
        mesher = self.get_mesher()
        if mesher is None:
            return []

        objects = []
        tree_obj = resolve_mesh_object(mesher.tree_object)
        if tree_obj is not None:
            objects.append(tree_obj)

        for stage in single_tree_chain(mesher):
            for node in downstream_nodes(stage):
                if node.bl_idname == VARIANTS_ID:
                    objects.extend(node.variant_objects())
        return objects

    def show_preview_lod(self):
        from ...pivot_painter.leaf_lods import find_lod_objects, show_lod

        for obj in self.tree_objects():
            if find_lod_objects(obj):
                show_lod(obj, self.preview_level())
