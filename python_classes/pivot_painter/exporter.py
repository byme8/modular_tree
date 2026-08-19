"""Main Pivot Painter exporter class."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bpy


class ExportFormat(Enum):
    """Supported export formats for Pivot Painter data."""

    UE5 = "UE5"
    UE4 = "UE4"
    UNITY = "UNITY"
    FOLIAGE = "FOLIAGE"


@dataclass
class ExportResult:
    """Result of an export operation."""

    success: bool
    message: str
    files_created: list[str] | None = None


class PivotPainterExporter:
    """Exports Pivot Painter 2.0 data for game engines.

    This class handles the extraction of pivot painter attributes from
    Blender meshes and exports them in engine-specific formats.
    """

    REQUIRED_ATTRIBUTES = [
        "stem_id",
        "hierarchy_depth",
        "pivot_position",
        "branch_extent",
        "direction",  # X-Vector for Pivot Painter 2.0
    ]

    LEAF_ATTRIBUTES = [
        "leaf_attachment_point",
        "leaf_facing_direction",
    ]

    def __init__(
        self,
        mesh: bpy.types.Mesh,
        export_format: ExportFormat,
        texture_size: int = 1024,
        export_path: str = "",
        occlusion_base: float = 0.25,
        translucency_tip: float = 0.0,
        normal_blend: float = 0.0,
        trunk_base_stiffness: float = 1.0,
        trunk_tip_stiffness: float = 1.0,
        branch_stiffness: float = 1.0,
        junction_falloff: float = 0.6,
        build_bone_rig: bool = False,
        segments_per_chain: int = 4,
        create_armature: bool = True,
        mesh_object: bpy.types.Object | None = None,
    ):
        self.mesh = mesh
        # Only the rig needs this: the armature modifier and vertex groups live
        # on the object, not on the mesh data every other format writes to.
        self.mesh_object = mesh_object
        self.export_format = export_format
        self.texture_size = texture_size
        self.export_path = export_path
        self.occlusion_base = occlusion_base
        self.translucency_tip = translucency_tip
        self.normal_blend = normal_blend
        self.trunk_base_stiffness = trunk_base_stiffness
        self.trunk_tip_stiffness = trunk_tip_stiffness
        self.branch_stiffness = branch_stiffness
        self.junction_falloff = junction_falloff
        self.build_bone_rig = build_bone_rig
        self.segments_per_chain = segments_per_chain
        self.create_armature = create_armature

    @property
    def has_leaf_data(self) -> bool:
        """Check if the mesh has leaf-specific attributes."""
        return all(attr in self.mesh.attributes for attr in self.LEAF_ATTRIBUTES)

    def validate(self) -> ExportResult | None:
        """Validate that the mesh has required attributes.

        Returns:
            ExportResult with error if validation fails, None if valid.
        """
        for attr in self.REQUIRED_ATTRIBUTES:
            if attr not in self.mesh.attributes:
                return ExportResult(
                    success=False,
                    message=f"Missing attribute '{attr}'. Generate tree with mesher first.",
                )
        return None

    def export(self, object_name: str) -> ExportResult:
        """Export pivot painter data in the specified format.

        Args:
            object_name: Name of the object (used for file naming).

        Returns:
            ExportResult indicating success or failure.
        """
        # The foliage format derives its streams from a different attribute set
        # (it needs radius, and does not need the pivot textures), so it runs its
        # own validation rather than the Pivot Painter one.
        if self.export_format == ExportFormat.FOLIAGE:
            return self._export_foliage()

        validation_error = self.validate()
        if validation_error:
            return validation_error

        if self.export_format == ExportFormat.UNITY:
            return self._export_unity()
        else:
            return self._export_unreal(object_name)

    def _export_foliage(self) -> ExportResult:
        """Write foliage wind and shading vertex streams onto the mesh.

        When a rig is requested the bone binding is written too. The two are not
        alternatives at export time even though they are at *render* time: a
        skinned mesh still reads phase for density fade and still reads every
        shading channel, so the streams are written either way. What the rig
        replaces is the vertex-displacement wind, which the engine turns off by
        keyword when it finds bones.
        """
        from .formats.foliage import FoliageExporter

        exporter = FoliageExporter(
            mesh=self.mesh,
            occlusion_base=self.occlusion_base,
            translucency_tip=self.translucency_tip,
            normal_blend=self.normal_blend,
            trunk_base_stiffness=self.trunk_base_stiffness,
            trunk_tip_stiffness=self.trunk_tip_stiffness,
            branch_stiffness=self.branch_stiffness,
            junction_falloff=self.junction_falloff,
        )
        result = exporter.export()

        if not result.success or not self.build_bone_rig:
            return result

        rig_result = self._export_foliage_rig()

        # A failed rig is reported as a failure even though the streams landed:
        # a mesh carrying stream wind and a broken binding is worse than one
        # carrying stream wind alone, because the engine will trust the binding.
        if not rig_result.success:
            return ExportResult(
                success=False,
                message=f"Streams written, but rig failed: {rig_result.message}",
            )

        return ExportResult(
            success=True,
            message=f"{result.message} {rig_result.message}",
        )

    def _export_foliage_rig(self) -> ExportResult:
        """Build the bone rig, write its binding, and bind it for preview."""
        from .formats.foliage_rig import FoliageRigExporter

        exporter = FoliageRigExporter(
            mesh=self.mesh,
            mesh_object=self.mesh_object,
            segments_per_chain=self.segments_per_chain,
            trunk_base_stiffness=self.trunk_base_stiffness,
            trunk_tip_stiffness=self.trunk_tip_stiffness,
            branch_stiffness=self.branch_stiffness,
            junction_falloff=self.junction_falloff,
            create_armature=self.create_armature,
        )
        return exporter.export()

    def _export_unity(self) -> ExportResult:
        """Export vertex colors for Unity wind shader."""
        from .formats.unity import UnityExporter

        exporter = UnityExporter(self.mesh)
        return exporter.export()

    def _export_unreal(self, object_name: str) -> ExportResult:
        """Export textures and UV2 for Unreal Engine."""
        from .formats.unreal import UnrealExporter

        exporter = UnrealExporter(
            mesh=self.mesh,
            object_name=object_name,
            texture_size=self.texture_size,
            export_path=self.export_path,
            is_ue5=(self.export_format == ExportFormat.UE5),
            include_leaf_data=self.has_leaf_data,
        )
        return exporter.export()
