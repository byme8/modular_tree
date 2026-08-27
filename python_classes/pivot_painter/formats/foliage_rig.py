
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

MIN_DISPLAY_BONE_LENGTH = 1e-3


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
        create_armature: bool = True,
        armature_name: str = "MtreeFoliageRig",
        mesh_object: bpy.types.Object | None = None,
        collection: bpy.types.Collection | None = None,
    ):
        self.mesh = mesh
        self.mesh_object = mesh_object
        self.collection = collection
        self.segments_per_chain = segments_per_chain
        self.trunk_base_stiffness = trunk_base_stiffness
        self.trunk_tip_stiffness = trunk_tip_stiffness
        self.branch_stiffness = branch_stiffness
        self.junction_falloff = junction_falloff
        self.blend_band = blend_band
        self.create_armature = create_armature
        self.armature_name = armature_name
        self.rig = None
        self.armature_object = None

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
        )

        if self.rig.bone_count == 0:
            from ... import diagnostics

            diagnostics.warn(
                "  rig produced 0 bones from %s", diagnostics.describe_mesh(self.mesh)
            )
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

        if self.create_armature:
            armature_object = self.build_armature()
            self.armature_object = armature_object
            message += f" Armature '{armature_object.name}' created."

            if self.mesh_object is not None:
                self.bind_to_armature(self.mesh_object, armature_object)
                message += " Pose it in Pose Mode to preview the deformation."
            else:
                message += " No mesh object given, so it is not bound: posing it will do nothing."

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

    def build_armature(self) -> bpy.types.Object:
        import bpy
        from mathutils import Vector

        armature_data = bpy.data.armatures.new(self.armature_name)
        armature_object = bpy.data.objects.new(self.armature_name, armature_data)

        collection = self.collection
        if collection is None:
            collection = bpy.context.scene.collection
        collection.objects.link(armature_object)

        previous_active = bpy.context.view_layer.objects.active
        previous_mode = bpy.context.mode

        if previous_active is not None and previous_active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        bpy.context.view_layer.objects.active = armature_object

        try:
            bpy.ops.object.mode_set(mode="EDIT")

            edit_bones = []
            for bone in self.rig.bones:
                edit_bone = armature_data.edit_bones.new(
                    f"stem{bone.stem_id}_seg{bone.segment}"
                )

                head = Vector(bone.rest_position.tolist())
                length = max(bone.length, MIN_DISPLAY_BONE_LENGTH)
                tail = head + Vector(bone.rest_direction.tolist()) * length

                edit_bone.head = head
                edit_bone.tail = tail
                edit_bone.use_connect = False

                edit_bones.append(edit_bone)

            for index, bone in enumerate(self.rig.bones):
                if bone.parent >= 0:
                    edit_bones[index].parent = edit_bones[bone.parent]

        finally:
            if bpy.context.view_layer.objects.active is armature_object:
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.context.view_layer.objects.active = previous_active
            if previous_active is not None and previous_mode.startswith("EDIT"):
                bpy.ops.object.mode_set(mode="EDIT")

        return armature_object

    def bind_to_armature(
        self,
        mesh_object: bpy.types.Object,
        armature_object: bpy.types.Object,
    ) -> None:
        self.write_vertex_groups(mesh_object)

        modifier = next(
            (m for m in mesh_object.modifiers if m.type == "ARMATURE"),
            None,
        )
        if modifier is None:
            modifier = mesh_object.modifiers.new(name="FoliageRig", type="ARMATURE")

        modifier.object = armature_object
        modifier.use_vertex_groups = True

        armature_object.matrix_world = mesh_object.matrix_world.copy()
        mesh_object.parent = armature_object
        mesh_object.matrix_parent_inverse = armature_object.matrix_world.inverted()

    def write_vertex_groups(self, mesh_object: bpy.types.Object) -> None:
        for bone in self.rig.bones:
            name = f"stem{bone.stem_id}_seg{bone.segment}"
            if name not in mesh_object.vertex_groups:
                mesh_object.vertex_groups.new(name=name)

        for vertex in range(len(self.mesh.vertices)):
            index_a = int(self.rig.bone_indices_a[vertex])
            index_b = int(self.rig.bone_indices_b[vertex])
            weight_a = float(self.rig.weights_a[vertex])

            bone_a = self.rig.bones[index_a]
            mesh_object.vertex_groups[f"stem{bone_a.stem_id}_seg{bone_a.segment}"].add(
                [vertex], weight_a, "REPLACE"
            )

            if weight_a >= 1.0:
                continue

            bone_b = self.rig.bones[index_b]
            group_b = f"stem{bone_b.stem_id}_seg{bone_b.segment}"

            if index_b == index_a:
                mesh_object.vertex_groups[group_b].add([vertex], 1.0, "REPLACE")
            else:
                mesh_object.vertex_groups[group_b].add([vertex], 1.0 - weight_a, "REPLACE")
