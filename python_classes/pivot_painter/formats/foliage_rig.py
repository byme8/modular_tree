"""Foliage bone rig export.

Turns the rig computed in ``foliage_bones`` into two things:

- a Blender armature, so the rig is visible and inspectable in the viewport
  rather than existing only as numbers in a stream;
- the ``uv3`` bone-binding stream on the mesh, which is what the consuming
  shader actually reads.

The computation lives in ``foliage_bones``; this module only moves data on and
off Blender datablocks.

**The armature is a preview, not the export product.** The consuming engine
reads the bone records and the binding stream, not a Blender armature — it has
its own rig asset format. The armature exists so a human can see whether the
bones follow the wood before trusting anything downstream, which is the same
reason the stream debug view exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import bpy

from ..exporter import ExportResult
from ..foliage_bones import (
    BLEND_BAND,
    DEFAULT_SEGMENTS_PER_CHAIN,
    build_rig,
    measure_rest_pose_error,
    pack_bone_uv,
    validate_rig,
)

SOURCE_FLOAT_ATTRIBUTES = ["stem_id", "branch_extent", "radius", "hierarchy_depth"]
SOURCE_VECTOR_ATTRIBUTES = ["pivot_position", "direction"]

# A bone shorter than this is below Blender's own minimum and gets dropped from
# the armature silently, which would desynchronise armature bone indices from
# the rig's. Short bones are lengthened for display only; the exported records
# keep their true length.
MIN_DISPLAY_BONE_LENGTH = 1e-3


class FoliageRigExporter:
    """Builds a bone rig from a tree mesh and writes the binding stream.

    Streams written:
        uv3.x  indexA    first bone, raw buffer offset
        uv3.y  indexB    second bone, raw buffer offset
        uv3.z  weightA   weight of indexA; weightB is derived as 1 - weightA
        uv3.w  unused    0

    Only emit this on a mesh that will actually be drawn skinned. On an unskinned
    mesh the shader keyword is off, nothing reads the stream, and it costs 16
    bytes per vertex.
    """

    BONE_UV_NAME = "FoliageBones"
    BONE_UV_SLOT = 3

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
    ):
        self.mesh = mesh
        # The object owning the mesh. Needed to attach the armature modifier and
        # vertex groups, which live on the object rather than on the mesh data.
        self.mesh_object = mesh_object
        self.segments_per_chain = segments_per_chain
        self.trunk_base_stiffness = trunk_base_stiffness
        self.trunk_tip_stiffness = trunk_tip_stiffness
        self.branch_stiffness = branch_stiffness
        self.junction_falloff = junction_falloff
        self.blend_band = blend_band
        self.create_armature = create_armature
        self.armature_name = armature_name
        self.rig = None

    def export(self) -> ExportResult:
        """Build the rig, write the binding stream, optionally build an armature."""
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
            return ExportResult(
                success=False,
                message="No branch chains found; nothing to rig.",
            )

        # Every one of these violations produces plausible geometry in the
        # engine rather than an error, so the export refuses rather than
        # shipping a rig that deforms silently.
        problems = validate_rig(self.rig)
        if problems:
            return ExportResult(
                success=False,
                message="Rig failed validation: " + "; ".join(problems[:3]),
            )

        packed = pack_bone_uv(
            self.rig.bone_indices_a, self.rig.bone_indices_b, self.rig.weights_a
        )
        self._write_bone_stream(packed)

        error = measure_rest_pose_error(self.rig, source["positions"])

        message = (
            f"Built rig: {self.rig.bone_count} bones, "
            f"max bind distance {error['max_distance']:.4f}m. "
            f"Wrote {self.BONE_UV_NAME}."
        )

        if self.create_armature:
            armature_object = self.build_armature()
            message += f" Armature '{armature_object.name}' created."

            # Binding is what makes the rig *testable*. An armature on its own
            # can be posed and nothing happens, which looks exactly like a
            # broken rig; with the modifier and groups in place, dragging a bone
            # in Pose Mode deforms the mesh the same way the engine will.
            if self.mesh_object is not None:
                self.bind_to_armature(self.mesh_object, armature_object)
                message += " Pose it in Pose Mode to preview the deformation."
            else:
                message += " No mesh object given, so it is not bound: posing it will do nothing."

        return ExportResult(success=True, message=message)

    def validate(self) -> ExportResult | None:
        """Check that the mesh carries the attributes the rig derives from."""
        for attr in SOURCE_FLOAT_ATTRIBUTES + SOURCE_VECTOR_ATTRIBUTES:
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

    def _write_bone_stream(self, packed: np.ndarray) -> None:
        """Write the four-component bone binding as a corner-domain attribute.

        A UV layer holds only two components, so the four-component binding goes
        to a generic ``FLOAT_COLOR`` attribute rather than to ``uv3`` proper.
        Exporters map it onto the engine's fourth UV channel at write time; the
        name is what identifies it.

        Indices are stored as raw floats, not normalised. The shader reads them
        as buffer offsets, so any rescaling here silently points every vertex at
        the wrong bone.

        **Unverified downstream risk.** ``FLOAT_COLOR`` is float32 and does not
        clamp on write, so a bone index of 12 survives inside Blender. What is
        *not* established is whether it survives FBX/glTF export: the FBX writer
        is configured with ``colors_type="LINEAR"`` to stop gamma-encoding these
        channels, and vertex colour is conventionally 0-1, so an exporter or
        importer that normalises would turn every index into 1.0 and bind the
        whole mesh to one bone. Check the exported file before trusting it, and
        if it does clamp, this has to move to a real UV pair (two layers of two
        components) rather than a colour attribute.
        """
        if self.BONE_UV_NAME in self.mesh.color_attributes:
            self.mesh.color_attributes.remove(
                self.mesh.color_attributes[self.BONE_UV_NAME]
            )

        self.mesh.color_attributes.new(
            name=self.BONE_UV_NAME,
            type="FLOAT_COLOR",
            domain="POINT",
        )
        self.mesh.color_attributes[self.BONE_UV_NAME].data.foreach_set("color", packed)

    def build_armature(self) -> bpy.types.Object:
        """Create a Blender armature mirroring the rig, for viewport inspection.

        Bone names encode their identity as ``stem<id>_seg<n>``, so a bone picked
        in the viewport can be traced back to a row in the exported records and
        to the vertices bound to it.

        Parenting uses ``use_connect = False`` even where a bone starts exactly
        at its parent's tip. Connecting them makes Blender move the parent's tail
        whenever the child's head moves, which would silently edit the rest pose
        the export just measured — and a rest pose that disagrees with the vertex
        positions deforms the mesh while it stands still.
        """
        import bpy
        from mathutils import Vector

        armature_data = bpy.data.armatures.new(self.armature_name)
        armature_object = bpy.data.objects.new(self.armature_name, armature_data)
        bpy.context.collection.objects.link(armature_object)

        previous_active = bpy.context.view_layer.objects.active
        previous_mode = previous_active.mode if previous_active else "OBJECT"

        # Entering Edit Mode on the new armature fails outright if the current
        # active object is itself in Edit or Pose Mode, which is an ordinary
        # state to run an export from. Drop to Object Mode first.
        if previous_active is not None and previous_mode != "OBJECT":
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

            # Parent in a second pass: a bone cannot be parented to one that
            # does not exist yet, and the rig's forward ordering guarantees the
            # parent exists by index but not that Blender has created it.
            for index, bone in enumerate(self.rig.bones):
                if bone.parent >= 0:
                    edit_bones[index].parent = edit_bones[bone.parent]

        finally:
            # Leaving Blender stuck in Edit Mode on a half-built armature is far
            # worse than the error that put it there, because every subsequent
            # operator then fails for an unrelated-looking reason.
            if bpy.context.object and bpy.context.object.mode == "EDIT":
                bpy.ops.object.mode_set(mode="OBJECT")

        if previous_active is not None:
            bpy.context.view_layer.objects.active = previous_active
            if previous_mode != "OBJECT":
                bpy.ops.object.mode_set(mode=previous_mode)

        return armature_object

    def bind_to_armature(
        self,
        mesh_object: bpy.types.Object,
        armature_object: bpy.types.Object,
    ) -> None:
        """Bind the mesh to the armature so posing a bone deforms it.

        Writes the vertex groups, adds an Armature modifier, and parents the
        mesh to the armature. After this, Pose Mode on the armature previews
        exactly the deformation the rig encodes.

        **This is a preview of the binding, not of the engine's wind.** Blender
        blends with linear skinning while the shader uses dual quaternions, so a
        heavily twisted pose will not match pixel for pixel. What it does verify
        is the part that actually goes wrong: which bone owns which vertices,
        and whether branches follow the limb carrying them.

        The modifier is reused if one already pointing at an armature exists, so
        re-running the export does not stack a second one.
        """
        self.write_vertex_groups(mesh_object)

        modifier = next(
            (m for m in mesh_object.modifiers if m.type == "ARMATURE"),
            None,
        )
        if modifier is None:
            modifier = mesh_object.modifiers.new(name="FoliageRig", type="ARMATURE")

        modifier.object = armature_object
        modifier.use_vertex_groups = True

        # Parent without moving the mesh. Setting `parent` alone makes Blender
        # apply the parent's transform on top of the child's, which teleports
        # the tree; the inverse matrix cancels that. The armature is built in the
        # mesh's own space, so the correct inverse is identity — but reading it
        # from the armature keeps this right if the armature is ever placed.
        mesh_object.parent = armature_object
        mesh_object.matrix_parent_inverse = armature_object.matrix_world.inverted()

    def write_vertex_groups(self, mesh_object: bpy.types.Object) -> None:
        """Write the binding as Blender vertex groups, so it can be weight-painted.

        This duplicates the ``uv3`` stream in a form Blender's own tools read.
        It is for inspection only — the engine reads the stream, not the groups —
        but a wrong binding is far easier to see in weight paint than in a column
        of float indices.
        """
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

            if index_b != index_a and weight_a < 1.0:
                bone_b = self.rig.bones[index_b]
                mesh_object.vertex_groups[
                    f"stem{bone_b.stem_id}_seg{bone_b.segment}"
                ].add([vertex], 1.0 - weight_a, "REPLACE")
