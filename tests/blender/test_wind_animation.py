from pathlib import Path

import bpy
import numpy as np
import pytest
from test_asset_ownership import make_mesh_object

from python_classes.pivot_painter.foliage_bones import deform_vertices, pose_rig
from python_classes.pivot_painter.formats.foliage import FoliageExporter
from python_classes.pivot_painter.wind_animation import (
    STRENGTH_STOPS,
    action_name,
    bake_wind_clips,
    build_rig_for,
    stop_frame,
)

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "wind_animation"

pytestmark = pytest.mark.blender


def evaluated_vertices(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    skinned = evaluated.to_mesh()

    out = np.empty(len(skinned.vertices) * 3, dtype=np.float32)
    skinned.vertices.foreach_get("co", out)
    evaluated.to_mesh_clear()

    return out.reshape(-1, 3).astype(np.float64)


def rest_vertices(mesh):
    out = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", out)
    return out.reshape(-1, 3).astype(np.float64)


@pytest.fixture(params=["manifold", "blocks_cards"])
def rigged_tree(request, node_tree):
    mesher = node_tree.mesher
    node_tree.add_branch()

    if request.param == "blocks_cards":
        mesher.mesh_mode = "BLOCKS"
        cards = node_tree.tree.nodes.new("mt_LeafCardsNode")
        node_tree.tree.links.new(mesher.outputs[0], cards.inputs[0])

    obj = make_mesh_object(f"wind_clips_{request.param}")
    mesher.build_into(obj, 0)
    FoliageExporter(mesh=obj.data).export()

    rig = build_rig_for(obj.data, mesher.segments_per_chain, mesher._stiffness_settings())
    return obj, rig, request.param


class TestWindClips:
    def test_playback_matches_the_solver(self, rigged_tree):
        obj, rig, surface = rigged_tree
        rest = rest_vertices(obj.data)

        directions = 4
        count, message = bake_wind_clips(obj, rig, directions=directions)
        assert count == directions, message

        armature = obj.parent
        step = 360.0 / directions

        rows = [
            f"surface {surface} vertices {len(rest)} bones {rig.bone_count}",
            f"directions {directions}",
            f"stops {', '.join(f'{s:.2f}' for s in STRENGTH_STOPS)}",
            "",
            "angle  frame  strength  bends  matches_solver",
        ]

        worst = 0.0
        for index in range(directions):
            angle = index * step
            armature.animation_data.action = bpy.data.actions[action_name(armature, angle)]

            for strength in STRENGTH_STOPS:
                frame = stop_frame(strength)
                bpy.context.scene.frame_set(frame)

                played = evaluated_vertices(obj)
                expected = deform_vertices(rig, rest, pose_rig(rig, angle, strength))

                error = float(np.abs(played - expected).max())
                swing = float(np.abs(expected - rest).max())
                worst = max(worst, error)

                rows.append(
                    f"{angle:5.1f}  {frame:5d}  {strength:8.2f}  "
                    f"{str(swing > 1e-4):5s}  {error < 1e-4}"
                )

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / f"clip_playback_{surface}.txt").write_text("\n".join(rows) + "\n")

        # TEST_POINT: the armature and the solver diverging is what shipped a
        # tree that looked one way in Blender and another in the engine; a
        # tolerance loose enough to hide that defeats the whole test.
        assert worst < 1e-4, f"clip playback drifts from the solver by {worst:.6f}"

    def test_rest_frame_is_unposed(self, rigged_tree):
        obj, rig, _ = rigged_tree
        rest = rest_vertices(obj.data)

        bake_wind_clips(obj, rig, directions=4)

        armature = obj.parent
        armature.animation_data.action = bpy.data.actions[action_name(armature, 0.0)]
        bpy.context.scene.frame_set(stop_frame(0.0))

        assert float(np.abs(evaluated_vertices(obj) - rest).max()) < 1e-4

    def test_rebaking_does_not_accumulate_clips(self, rigged_tree):
        from python_classes.pivot_painter.wind_animation import action_prefix_for

        obj, rig, _ = rigged_tree

        def clip_names():
            prefix = action_prefix_for(obj.parent)
            return sorted(a.name for a in bpy.data.actions if a.name.startswith(prefix))

        bake_wind_clips(obj, rig, directions=4)
        first = clip_names()

        bake_wind_clips(obj, rig, directions=4)
        second = clip_names()

        assert first == second, "a rebake left stale clips behind"
        assert len(second) == 4

    def test_stops_are_ordered_and_bracket_the_range(self):
        assert STRENGTH_STOPS[0] == 0.0
        assert STRENGTH_STOPS[-1] == 1.0
        assert list(STRENGTH_STOPS) == sorted(STRENGTH_STOPS)
