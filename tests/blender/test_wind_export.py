import re
import tempfile
from pathlib import Path

import bpy
import numpy as np
import pytest
from test_asset_ownership import make_mesh_object
from test_block_fill import pin_seeds

from python_classes.pivot_painter.formats.foliage import FoliageExporter
from python_classes.pivot_painter.tree_asset import find_armature
from python_classes.pivot_painter.wind_animation import (
    STRENGTH_STOPS,
    action_fcurves,
    action_name,
    action_prefix_for,
    bake_wind_clips,
    build_rig_for,
    clear_pose,
    stop_frame,
)

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "wind_animation"
EXPORT_ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "wind_export"

VARIANT_COUNT = 3
DIRECTIONS = 4
ROUND_TRIP_TOLERANCE = 1e-4

pytestmark = pytest.mark.blender


def read_fbx_text(path):
    return path.read_bytes().decode("latin-1")


class TestRiggedExport:
    @pytest.fixture
    def exported(self, node_tree):
        mesher = node_tree.mesher
        node_tree.add_branch()

        obj = make_mesh_object("export_tree")
        mesher.build_into(obj, 0)
        FoliageExporter(mesh=obj.data).export()

        rig = build_rig_for(obj.data, mesher.segments_per_chain, mesher._stiffness_settings())
        bake_wind_clips(obj, rig, directions=4)

        export = node_tree.tree.nodes.new("mt_MeshExportNode")
        export.export_format = "FBX"
        export.target_engine = "UNITY"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export_tree.fbx"
            export._write_file(str(path), [obj])

            assert path.exists(), "no file written"
            yield obj, path, read_fbx_text(path)

        node_tree.tree.nodes.remove(export)

    def test_file_carries_skeleton_and_clips(self, exported):
        obj, path, text = exported

        root_bone = f"{obj.parent.name}_stem0_seg0"
        assert root_bone in text, "no bones in the exported file"

        clips = [
            name
            for name in (f"export_tree_rig_Wind_{d:03d}" for d in (0, 90, 180, 270))
            if name in text
        ]

        rows = [
            f"bones present {root_bone in text}",
            f"clips {len(clips)}/4",
            "",
            *sorted(clips),
        ]
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "export_payload.txt").write_text("\n".join(rows) + "\n")

        # TEST_POINT: the exporters deliberately stripped the skeleton while
        # wind shipped as shape keys; wind now rides on the rig, so a file
        # without bones and clips animates as a static tree and looks fine.
        assert len(clips) == 4, f"expected 4 wind clips, found {clips}"

    def test_unrigged_mesh_still_exports(self, node_tree):
        export = node_tree.tree.nodes.new("mt_MeshExportNode")
        export.export_format = "FBX"

        plain = make_mesh_object("export_plain")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export_plain.fbx"
            export._write_file(str(path), [plain])
            assert path.exists()

        node_tree.tree.nodes.remove(export)


def source_name(imported_name):
    return re.sub(r"\.\d{3}$", "", imported_name)


def import_file(path):
    bpy.ops.object.select_all(action="DESELECT")
    actions_before = set(bpy.data.actions.keys())
    if path.suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj.name for obj in bpy.context.selected_objects]
    actions = sorted(set(bpy.data.actions.keys()) - actions_before)
    return imported, actions


def discard_import(object_names, action_names):
    for name in object_names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Armature):
                bpy.data.armatures.remove(data)
    for name in action_names:
        action = bpy.data.actions.get(name)
        if action is not None:
            bpy.data.actions.remove(action)


def takes_on(armature, new_actions):
    fbx_prefix = f"{armature.name}|"
    takes = {name.split("|")[-1] for name in new_actions if name.startswith(fbx_prefix)}
    if armature.animation_data is not None:
        takes |= {
            source_name(strip.action.name)
            for track in armature.animation_data.nla_tracks
            for strip in track.strips
            if strip.action is not None
        }
    return sorted(takes)


def world_vertices(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    local = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", local)
    evaluated.to_mesh_clear()

    local = local.reshape(-1, 3)
    matrix = np.array(evaluated.matrix_world)
    return local @ matrix[:3, :3].T + matrix[:3, 3]


def play(armature, action, frame):
    armature.animation_data.action = action
    if action is None:
        clear_pose(armature)
    elif getattr(action, "slots", None):
        armature.animation_data.action_slot = action.slots[0]
    bpy.context.scene.frame_set(frame)


class TestVariantExport:
    @pytest.fixture
    def variants_export(self, node_tree):
        node_tree.add_branch()
        mesher, variants = node_tree.mesher, node_tree.variants

        mesher.build_bone_rig = True
        mesher.morph_directions = DIRECTIONS
        variants.variant_count = VARIANT_COUNT
        variants.morph_directions = DIRECTIONS
        variants.bake_morphs = True
        variants.spacing = 10.0
        variants.collection_name = "wexp"
        variants.generate_variants()
        assert not variants.status_is_error, variants.status_message

        export = node_tree.tree.nodes.new("mt_MeshExportNode")
        node_tree.tree.links.new(variants.outputs[0], export.inputs[0])
        export.target_engine = "UNREAL"

        objects = sorted(variants.variant_objects(), key=lambda obj: obj.name)
        armatures = [find_armature(obj).name for obj in objects]
        assert len(armatures) == VARIANT_COUNT

        with tempfile.TemporaryDirectory() as directory:
            export.export_path = directory + "/"
            yield export, Path(directory), [obj.name for obj in objects], armatures

        node_tree.tree.nodes.remove(export)

    @pytest.mark.parametrize("export_format", ["FBX", "GLTF"])
    @pytest.mark.parametrize("separate_files", [False, True])
    def test_each_armature_exports_only_its_own_clips(
        self, variants_export, export_format, separate_files
    ):
        export, directory, _, armatures = variants_export
        export.export_format = export_format
        export.separate_files = separate_files

        export.do_export()
        assert not export.status_is_error, export.status_message

        nla_before = {
            name: len(bpy.data.objects[name].animation_data.nla_tracks) for name in armatures
        }

        rows = [f"format {export_format} separate {separate_files}", ""]
        failures = []
        for path in sorted(directory.iterdir()):
            imported, new_actions = import_file(path)
            try:
                print(f"IMPORT {path.name} objects={sorted(imported)} actions={new_actions}")
                rows.append(f"file {path.suffix}")
                for name in sorted(imported):
                    armature = bpy.data.objects[name]
                    if armature.type != "ARMATURE":
                        continue
                    owner = source_name(name)
                    takes = takes_on(armature, new_actions)
                    rows.append(f"  {owner}: {len(takes)} takes")
                    rows.extend(f"    {take}" for take in takes)

                    own_prefix = f"{owner}_Wind_"
                    foreign = [
                        take
                        for take in takes
                        if own_prefix not in take
                        or any(f"{other}_Wind_" in take for other in armatures if other != owner)
                    ]
                    if foreign or len(takes) != DIRECTIONS:
                        failures.append(f"{path.name} {owner}: {takes}")
            finally:
                discard_import(imported, new_actions)

        EXPORT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        mode = "separate" if separate_files else "combined"
        (EXPORT_ARTIFACT_DIR / f"take_ownership_{export_format.lower()}_{mode}.txt").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )

        # TEST_POINT: variants share bone names' shape, so a writer that offers
        # every action whose paths resolve hands each skeleton the others' clips.
        assert not failures, failures
        assert nla_before == {
            name: len(bpy.data.objects[name].animation_data.nla_tracks) for name in armatures
        }, "the export left temporary NLA tracks behind"
        assert all(bpy.data.objects[name].animation_data.action is None for name in armatures)

    def test_full_strength_frame_survives_the_round_trip(self, variants_export):
        export, directory, objects, armatures = variants_export
        export.export_format = "FBX"
        export.separate_files = False

        export.do_export()
        assert not export.status_is_error, export.status_message

        source_object = bpy.data.objects[objects[1]]
        source_armature = bpy.data.objects[armatures[1]]
        angle = 90.0
        frame = stop_frame(STRENGTH_STOPS[-1])

        play(source_armature, bpy.data.actions[action_name(source_armature, angle)], frame)
        expected = world_vertices(source_object)
        play(source_armature, None, frame)
        rest = world_vertices(source_object)

        (path,) = sorted(directory.iterdir())
        imported, new_actions = import_file(path)
        try:
            imported_armature = next(
                bpy.data.objects[name]
                for name in imported
                if bpy.data.objects[name].type == "ARMATURE"
                and source_name(name) == source_armature.name
            )
            imported_mesh = next(
                bpy.data.objects[name]
                for name in imported
                if bpy.data.objects[name].type == "MESH"
                and bpy.data.objects[name].parent is not None
                and bpy.data.objects[name].parent.name == imported_armature.name
            )
            clip = action_name(source_armature, angle)
            take = next(
                name
                for name in new_actions
                if name.startswith(f"{imported_armature.name}|") and name.endswith(clip)
            )

            play(imported_armature, bpy.data.actions[take], frame)
            played = world_vertices(imported_mesh)
            play(imported_armature, None, frame)
            rest_error = float(np.abs(world_vertices(imported_mesh) - rest).max())
        finally:
            discard_import(imported, new_actions)

        assert played.shape == expected.shape, (played.shape, expected.shape)
        error = float(np.abs(played - expected).max())
        swing = float(np.abs(expected - rest).max())
        print(f"ROUND TRIP max error {error:.3e} rest {rest_error:.3e} swing {swing:.3e}")

        rows = [
            f"clip {clip} frame {frame} strength {STRENGTH_STOPS[-1]:.2f}",
            f"vertices match {played.shape == expected.shape}",
            f"bends {swing > 10 * ROUND_TRIP_TOLERANCE}",
            f"rest within {ROUND_TRIP_TOLERANCE:g} {rest_error < ROUND_TRIP_TOLERANCE}",
            f"posed within {ROUND_TRIP_TOLERANCE:g} {error < ROUND_TRIP_TOLERANCE}",
        ]
        EXPORT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (EXPORT_ARTIFACT_DIR / "round_trip.txt").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )

        assert swing > 10 * ROUND_TRIP_TOLERANCE, "the clip does not bend; the check is vacuous"
        assert rest_error < ROUND_TRIP_TOLERANCE, f"rest pose moved by {rest_error:.6f}"
        assert error < ROUND_TRIP_TOLERANCE, f"full-strength pose moved by {error:.6f}"

    @pytest.mark.parametrize("separate_files", [False, True])
    def test_lods_off_keeps_the_file_names(self, variants_export, separate_files):
        export, directory, objects, _ = variants_export
        export.export_format = "FBX"
        export.separate_files = separate_files

        export.do_export()
        assert not export.status_is_error, export.status_message

        expected = [f"{name}.fbx" for name in objects] if separate_files else ["wexp.fbx"]
        assert sorted(path.name for path in directory.iterdir()) == expected

    def test_clip_keys_sit_at_strength_spaced_frames(self, variants_export):
        _, _, _, armatures = variants_export
        expected_frames = sorted(stop_frame(strength) for strength in STRENGTH_STOPS)

        rows = [f"stops {', '.join(f'{s:.2f}' for s in STRENGTH_STOPS)}", ""]
        for armature_name in armatures:
            prefix = action_prefix_for(bpy.data.objects[armature_name])
            for action in sorted(
                (a for a in bpy.data.actions if a.name.startswith(prefix)), key=lambda a: a.name
            ):
                fcurves = action_fcurves(action)
                frames = sorted({round(p.co.x) for fc in fcurves for p in fc.keyframe_points})
                modes = sorted({p.interpolation for fc in fcurves for p in fc.keyframe_points})
                rows.append(f"{action.name} frames {frames} interpolation {modes}")

                assert fcurves, f"{action.name} carries no curves"
                assert frames == expected_frames, (action.name, frames)
                assert modes == ["LINEAR"], (action.name, modes)

        EXPORT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (EXPORT_ARTIFACT_DIR / "key_frames.txt").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )


LOD_VARIANTS = 2
LOD_LEVELS = (0, 1, 2)


def triangles(mesh):
    return sum(len(polygon.vertices) - 2 for polygon in mesh.polygons)


def per_vertex_uv(mesh, name):
    loops = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loops)
    uvs = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    mesh.uv_layers[name].data.foreach_get("uv", uvs)
    values = np.zeros((len(mesh.vertices), 2), dtype=np.float32)
    values[loops] = uvs.reshape(-1, 2)
    return values


class TestLodExport:
    @pytest.fixture
    def lod_export(self, node_tree):
        node_tree.add_branch()
        mesher, variants = node_tree.mesher, node_tree.variants

        cards = node_tree.tree.nodes.new("mt_LeafCardsNode")
        node_tree.tree.links.new(mesher.outputs[0], cards.inputs[0])
        cards.build_lods = True
        pin_seeds(node_tree)

        mesher.build_bone_rig = True
        mesher.morph_directions = DIRECTIONS
        variants.variant_count = LOD_VARIANTS
        variants.morph_directions = DIRECTIONS
        variants.bake_morphs = True
        variants.collection_name = "wlod"
        variants.generate_variants()
        assert not variants.status_is_error, variants.status_message

        export = node_tree.tree.nodes.new("mt_MeshExportNode")
        node_tree.tree.links.new(variants.outputs[0], export.inputs[0])
        export.target_engine = "UNREAL"

        trees = sorted(obj.name for obj in variants.variant_objects())
        with tempfile.TemporaryDirectory() as directory:
            export.export_path = directory + "/"
            yield export, Path(directory), trees

        node_tree.tree.nodes.remove(export)

    def test_lod_meshes_carry_the_foliage_streams(self, lod_export):
        _, _, trees = lod_export
        for tree_name in trees:
            tree = bpy.data.objects[tree_name].data
            wind = per_vertex_uv(tree, FoliageExporter.WIND_UV_NAME)
            for level in LOD_LEVELS[1:]:
                lod = bpy.data.objects[f"{tree_name}_LOD{level}"].data
                assert [layer.name for layer in lod.uv_layers] == [
                    layer.name for layer in tree.uv_layers
                ]
                assert FoliageExporter.COLOR_NAME in lod.color_attributes
                sources = np.empty(len(lod.vertices), dtype=np.int32)
                lod.attributes["lod0_vertex"].data.foreach_get("value", sources)
                lod_wind = per_vertex_uv(lod, FoliageExporter.WIND_UV_NAME)
                assert np.array_equal(lod_wind, wind[sources])

    @pytest.mark.parametrize("export_format", ["FBX", "GLTF"])
    @pytest.mark.parametrize("separate_files", [False, True])
    def test_each_lod_file_shares_the_skeleton_and_only_lod0_has_clips(
        self, lod_export, export_format, separate_files
    ):
        export, directory, trees = lod_export
        export.export_format = export_format
        export.separate_files = separate_files

        export.do_export()
        assert not export.status_is_error, export.status_message

        extension = "fbx" if export_format == "FBX" else "glb"
        stems = trees if separate_files else ["wlod"]
        expected = sorted(
            f"{stem}_LOD{level}.{extension}" for stem in stems for level in LOD_LEVELS
        )
        assert sorted(path.name for path in directory.iterdir()) == expected

        armatures = [find_armature(bpy.data.objects[name]).name for name in trees]
        rig_of = dict(zip(trees, armatures, strict=True))
        source_bones = {
            name: sorted(bone.name for bone in bpy.data.objects[name].data.bones)
            for name in armatures
        }
        source_triangles = {
            f"{tree}_LOD{level}" if level else tree: triangles(
                bpy.data.objects[f"{tree}_LOD{level}" if level else tree].data
            )
            for tree in trees
            for level in LOD_LEVELS
        }
        assert all(obj.hide_viewport for obj in bpy.data.objects if obj.name.endswith("_LOD2"))

        rows = [f"format {export_format} separate {separate_files}", ""]
        failures = []
        for path in sorted(directory.iterdir()):
            level = int(path.stem[-1])
            imported, new_actions = import_file(path)
            try:
                rows.append(path.name)
                for name in sorted(imported):
                    obj = bpy.data.objects[name]
                    owner = source_name(name)
                    if obj.type == "ARMATURE":
                        takes = takes_on(obj, new_actions)
                        bones = sorted(bone.name for bone in obj.data.bones)
                        rows.append(f"  {owner}: {len(bones)} bones, {len(takes)} takes")
                        if bones != source_bones[owner]:
                            failures.append(f"{path.name} {owner}: bones differ")
                        wanted = DIRECTIONS if level == 0 else 0
                        foreign = [take for take in takes if f"{owner}_Wind_" not in take]
                        if foreign or len(takes) != wanted:
                            failures.append(f"{path.name} {owner}: {takes}")
                    elif obj.type == "MESH":
                        count = triangles(obj.data)
                        rows.append(f"  {owner}: {count} triangles")
                        if count != source_triangles.get(owner):
                            failures.append(f"{path.name} {owner}: {count} triangles")
                        parent = None if obj.parent is None else source_name(obj.parent.name)
                        if parent != rig_of[re.sub(r"_LOD\d$", "", owner)]:
                            failures.append(f"{path.name} {owner}: parent {parent}")
            finally:
                discard_import(imported, new_actions)

        EXPORT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        mode = "separate" if separate_files else "combined"
        (EXPORT_ARTIFACT_DIR / f"lod_files_{export_format.lower()}_{mode}.txt").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )

        assert not failures, failures
        assert all(bpy.data.objects[name].animation_data.action is None for name in armatures)
