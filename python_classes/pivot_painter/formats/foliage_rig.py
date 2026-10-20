from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import bpy

from .. import ExportResult
from ..foliage_bones import (
    BLEND_BAND,
    DEFAULT_SEGMENTS_PER_CHAIN,
    build_rig,
    measure_rest_pose_error,
    validate_rig,
)

SOURCE_FLOAT_ATTRIBUTES = ["stem_id", "branch_extent", "radius", "hierarchy_depth"]
SOURCE_VECTOR_ATTRIBUTES = ["pivot_position", "direction"]


class FoliageRigExporter:
    def __init__(
        self,
        mesh: bpy.types.Mesh,
        segments_per_chain: int = DEFAULT_SEGMENTS_PER_CHAIN,
        trunk_base_stiffness: float = 1.0,
        trunk_tip_stiffness: float = 0.6,
        branch_stiffness: float = 1.0,
        junction_falloff: float = 0.6,
        blend_band: float = BLEND_BAND,
        mesh_object: bpy.types.Object | None = None,
        branch_segments_per_chain: int | None = None,
        rig_depth: int | None = None,
    ):
        self.mesh = mesh
        self.mesh_object = mesh_object
        self.segments_per_chain = segments_per_chain
        self.branch_segments_per_chain = branch_segments_per_chain
        self.rig_depth = rig_depth
        self.trunk_base_stiffness = trunk_base_stiffness
        self.trunk_tip_stiffness = trunk_tip_stiffness
        self.branch_stiffness = branch_stiffness
        self.junction_falloff = junction_falloff
        self.blend_band = blend_band
        self.rig = None

    def export(self) -> ExportResult:
        validation_error = self.validate()
        if validation_error:
            return validation_error

        source = self._read_source_attributes()

        self.rig = build_rig(
            vertex_positions=source["positions"],
            stem_ids=source["stem_id"],
            branch_extents=source["branch_extent"],
            radii=source["radius"],
            pivot_positions=source["pivot_position"],
            directions=source["direction"],
            hierarchy_depths=source["hierarchy_depth"],
            segments_per_chain=self.segments_per_chain,
            trunk_base_stiffness=self.trunk_base_stiffness,
            trunk_tip_stiffness=self.trunk_tip_stiffness,
            branch_stiffness=self.branch_stiffness,
            junction_falloff=self.junction_falloff,
            blend_band=self.blend_band,
            branch_segments_per_chain=self.branch_segments_per_chain,
            rig_depth=self.rig_depth,
        )

        if self.rig.bone_count == 0:
            from ... import diagnostics

            diagnostics.warn("  rig produced 0 bones from %s", diagnostics.describe_mesh(self.mesh))
            return ExportResult(
                success=False,
                message="No branch chains found; nothing to rig.",
            )

        problems = validate_rig(self.rig)
        if problems:
            return ExportResult(
                success=False,
                message="Rig failed validation: " + "; ".join(problems[:3]),
            )

        error = measure_rest_pose_error(self.rig, source["positions"])

        message = (
            f"Built rig: {self.rig.bone_count} bones, "
            f"max bind distance {error['max_distance']:.4f}m."
        )

        return ExportResult(success=True, message=message)

    def validate(self) -> ExportResult | None:
        from ... import diagnostics

        diagnostics.log("  validate %s", diagnostics.describe_mesh(self.mesh))

        if len(self.mesh.vertices) == 0:
            return ExportResult(
                success=False,
                message=(
                    "Tree mesh is empty. The trunk is divided into "
                    "length x resolution segments; if that product rounds below "
                    "one, nothing is generated."
                ),
            )

        for attr in SOURCE_FLOAT_ATTRIBUTES + SOURCE_VECTOR_ATTRIBUTES:
            if attr not in self.mesh.attributes:
                return ExportResult(
                    success=False,
                    message=f"Missing attribute '{attr}'. Generate tree with mesher first.",
                )
        return None

    def _read_source_attributes(self) -> dict:
        num_verts = len(self.mesh.vertices)

        positions = np.zeros(num_verts * 3)
        self.mesh.vertices.foreach_get("co", positions)

        source: dict[str, np.ndarray] = {"positions": positions.reshape(-1, 3)}

        for name in SOURCE_FLOAT_ATTRIBUTES:
            scalars = np.zeros(num_verts)
            self._read_attribute(name, scalars, num_verts, is_vector=False)
            source[name] = scalars

        for name in SOURCE_VECTOR_ATTRIBUTES:
            vectors = np.zeros((num_verts, 3))
            self._read_attribute(name, vectors, num_verts, is_vector=True)
            source[name] = vectors

        return source

    def _read_attribute(
        self,
        attr_name: str,
        target: np.ndarray,
        num_verts: int,
        is_vector: bool,
    ) -> None:
        attr = self.mesh.attributes[attr_name]

        if attr.domain == "POINT":
            if is_vector:
                flat = np.zeros(num_verts * 3)
                attr.data.foreach_get("vector", flat)
                target[:] = flat.reshape(-1, 3)
            else:
                attr.data.foreach_get("value", target)

        elif attr.domain == "CORNER":
            num_loops = len(self.mesh.loops)
            if is_vector:
                flat = np.zeros(num_loops * 3)
                attr.data.foreach_get("vector", flat)
                loop_data = flat.reshape(-1, 3)
            else:
                loop_data = np.zeros(num_loops)
                attr.data.foreach_get("value", loop_data)

            loop_to_vert = np.zeros(num_loops, dtype=int)
            self.mesh.loops.foreach_get("vertex_index", loop_to_vert)

            has_value = np.zeros(num_verts, dtype=bool)
            for loop_idx, vert_idx in enumerate(loop_to_vert):
                if not has_value[vert_idx]:
                    target[vert_idx] = loop_data[loop_idx]
                    has_value[vert_idx] = True
