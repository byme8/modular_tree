"""Foliage vertex-stream export.

Writes the UV and vertex-colour streams that real-time foliage shaders read, in
addition to position and normal. A mesh carrying only position and normal is not
drawn with default shading by such a shader — every stream it fails to declare
is fed zero, which silently deforms the result.

The computation lives in ``foliage_streams``; this module only moves data on and
off the Blender mesh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import bpy

from ..exporter import ExportResult
from ..foliage_streams import compute_all_streams

# Attribute names produced by ManifoldMesher that the streams are derived from.
SOURCE_FLOAT_ATTRIBUTES = ["stem_id", "branch_extent", "radius", "hierarchy_depth"]
SOURCE_VECTOR_ATTRIBUTES = ["pivot_position", "direction"]


class FoliageExporter:
    """Writes foliage wind and shading streams onto a tree mesh.

    Streams written:
        uv1.x  stiffness       1 = rigid, 0 = free tip
        uv1.y  phase           constant per branch chain, [0, 1)
        uv2.x  blade height    branch chain length in metres
        uv2.y  unused          0
        color.r occlusion      dark root to lit tip
        color.g translucency   root to tip transmission
        color.b normal blend   geometric to flat-up, constant
        color.a tint ramp      raw normalised root to tip position

    uv0 is left untouched: the mesher's bark parameterisation already occupies
    it and is what the base map is authored against.
    """

    WIND_UV_NAME = "FoliageWind"
    HEIGHT_UV_NAME = "FoliageHeight"
    COLOR_NAME = "FoliageShading"

    def __init__(
        self,
        mesh: bpy.types.Mesh,
        occlusion_base: float = 0.25,
        translucency_tip: float = 0.0,
        normal_blend: float = 0.0,
        trunk_base_stiffness: float = 1.0,
        trunk_tip_stiffness: float = 1.0,
        branch_stiffness: float = 1.0,
        junction_falloff: float = 0.6,
    ):
        self.mesh = mesh
        self.occlusion_base = occlusion_base
        self.translucency_tip = translucency_tip
        self.normal_blend = normal_blend
        self.trunk_base_stiffness = trunk_base_stiffness
        self.trunk_tip_stiffness = trunk_tip_stiffness
        self.branch_stiffness = branch_stiffness
        self.junction_falloff = junction_falloff

    def export(self) -> ExportResult:
        """Compute and write all foliage streams."""
        validation_error = self.validate()
        if validation_error:
            return validation_error

        source = self._read_source_attributes()
        streams = compute_all_streams(
            vertex_positions=source["positions"],
            stem_ids=source["stem_id"],
            branch_extents=source["branch_extent"],
            radii=source["radius"],
            pivot_positions=source["pivot_position"],
            directions=source["direction"],
            hierarchy_depths=source["hierarchy_depth"],
            occlusion_base=self.occlusion_base,
            translucency_tip=self.translucency_tip,
            normal_blend=self.normal_blend,
            trunk_base_stiffness=self.trunk_base_stiffness,
            trunk_tip_stiffness=self.trunk_tip_stiffness,
            branch_stiffness=self.branch_stiffness,
            junction_falloff=self.junction_falloff,
        )

        # Slot 0 is the mesher's bark parameterisation, which the base map is
        # authored against. Wind must be uv1 and height uv2 for the engine to
        # find them.
        self._write_uv_layer(self.WIND_UV_NAME, streams["wind_uv"], slot=1)
        self._write_uv_layer(self.HEIGHT_UV_NAME, streams["height_uv"], slot=2)
        self._write_color_layer(streams["colors"])

        return ExportResult(
            success=True,
            message=(
                f"Wrote foliage streams: {self.WIND_UV_NAME}, "
                f"{self.HEIGHT_UV_NAME}, {self.COLOR_NAME}. "
                "Expand mesh bounds by the wind reach before instancing."
            ),
        )

    def validate(self) -> ExportResult | None:
        """Check that the mesh carries the attributes the streams derive from."""
        required = SOURCE_FLOAT_ATTRIBUTES + SOURCE_VECTOR_ATTRIBUTES
        for attr in required:
            if attr not in self.mesh.attributes:
                return ExportResult(
                    success=False,
                    message=f"Missing attribute '{attr}'. Generate tree with mesher first.",
                )
        return None

    def _read_source_attributes(self) -> dict:
        """Read vertex positions and the mesher attributes into arrays."""
        num_verts = len(self.mesh.vertices)

        positions = np.zeros(num_verts * 3)
        self.mesh.vertices.foreach_get("co", positions)

        # Holds both Nx3 vectors and length-N scalars, so it is annotated rather
        # than inferred from the first assignment.
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
        """Read a mesh attribute into the target array, handling domain types."""
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

    def _write_uv_layer(self, name: str, per_vertex_uv: np.ndarray, slot: int) -> None:
        """Write a per-vertex UV stream to a named UV layer at a known slot.

        UV layers are corner-domain in Blender, so per-vertex values are
        broadcast to every loop referencing that vertex. The streams are
        constant per vertex by construction, so no seam is introduced.

        **Slot order is part of the contract, not a detail.** Engines address UV
        layers by index — ``uv1``, ``uv2`` — not by name, so a layer that lands
        in the wrong position is read as a different stream entirely. Appending
        by name alone puts the layer wherever the list happens to end, which
        depends on whether the pivot painter export ran first. Padding layers are
        created so the requested slot is reached deterministically.
        """
        while len(self.mesh.uv_layers) < slot:
            self.mesh.uv_layers.new(name=f"UVPad{len(self.mesh.uv_layers)}")

        if name not in self.mesh.uv_layers:
            self.mesh.uv_layers.new(name=name)

        actual = self.mesh.uv_layers.find(name)
        if actual != slot:
            raise ValueError(
                f"UV layer '{name}' landed in slot {actual}, expected {slot}. "
                "Remove extra UV layers, or run this export before Pivot Painter."
            )

        uv_layer = self.mesh.uv_layers[name]

        num_loops = len(self.mesh.loops)
        loop_to_vert = np.zeros(num_loops, dtype=int)
        self.mesh.loops.foreach_get("vertex_index", loop_to_vert)

        per_vertex = per_vertex_uv.reshape(-1, 2)
        uv_layer.data.foreach_set("uv", per_vertex[loop_to_vert].ravel())

    def _write_color_layer(self, colors: np.ndarray) -> None:
        """Write the packed RGBA scalars to a point-domain colour attribute."""
        if self.COLOR_NAME not in self.mesh.color_attributes:
            self.mesh.color_attributes.new(
                name=self.COLOR_NAME,
                type="FLOAT_COLOR",
                domain="POINT",
            )
        self.mesh.color_attributes[self.COLOR_NAME].data.foreach_set("color", colors)
