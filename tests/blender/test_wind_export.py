import tempfile
from pathlib import Path

import bpy
import pytest

from python_classes.pivot_painter.formats.foliage import FoliageExporter
from python_classes.pivot_painter.wind_animation import bake_wind_clips, build_rig_for
from test_asset_ownership import make_mesh_object

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "wind_animation"

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

        rig = build_rig_for(
            obj.data, mesher.segments_per_chain, mesher._stiffness_settings()
        )
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

        assert "stem0_seg0" in text, "no bones in the exported file"

        clips = [
            name
            for name in (f"export_tree_rig_Wind_{d:03d}" for d in (0, 90, 180, 270))
            if name in text
        ]

        rows = [
            f"bones present {'stem0_seg0' in text}",
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
