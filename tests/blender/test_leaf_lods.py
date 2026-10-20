from pathlib import Path

import bpy
import numpy as np
import pytest
from test_asset_ownership import make_mesh_object
from test_block_fill import enable_surfaces, leaf_vertex_mask, mesh_arrays, pin_seeds
from test_wind_animation import evaluated_vertices

from python_classes.pivot_painter.formats.foliage import FoliageExporter
from python_classes.pivot_painter.leaf_lods import find_lod_objects, read_source_vertices
from python_classes.pivot_painter.tree_asset import find_armature
from python_classes.pivot_painter.wind_animation import apply_pose, build_rig_for

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "leaf_lods"

pytestmark = pytest.mark.blender


def lod_graph(node_tree):
    node_tree.add_branch()
    pin_seeds(node_tree)
    cards = enable_surfaces(node_tree, blocks=False)
    cards.build_lods = True
    return cards


def positions(mesh):
    return mesh_arrays(mesh)["vertices"].astype(np.float64)


def leaf_area(mesh):
    return sum(p.area for p in mesh.polygons if p.material_index == 1)


def leaf_polygons(mesh):
    return sum(1 for p in mesh.polygons if p.material_index == 1)


class CardLayout:
    def __init__(self, tree_mesh):
        mask = leaf_vertex_mask(tree_mesh)
        self.bark_vertices = int(np.argmax(mask))
        assert not mask[: self.bark_vertices].any() and mask[self.bark_vertices :].all()
        self.cards = int(leaf_polygons(tree_mesh) / 2)
        self.per_card = int((len(mask) - self.bark_vertices) / self.cards)

    def card_ids(self, lod):
        sources = read_source_vertices(lod)[self.bark_vertices :: self.per_card]
        return ((sources - self.bark_vertices) / self.per_card).astype(int).tolist()

    def centres(self, mesh):
        cards = positions(mesh)[self.bark_vertices :]
        return cards.reshape(-1, self.per_card, 3).mean(axis=1)


def build_tree(node_tree, name):
    obj = make_mesh_object(name)
    node_tree.mesher.build_into(obj, 0)
    return obj


def occlusion_channel(mesh, sky_occlusion):
    FoliageExporter(mesh=mesh, sky_occlusion=sky_occlusion).export()
    colors = np.zeros(len(mesh.vertices) * 4)
    mesh.color_attributes[FoliageExporter.COLOR_NAME].data.foreach_get("color", colors)
    return colors[0::4].copy()


class TestCanopyOcclusion:
    def test_canopy_darkens_interior_cards_and_the_bark_under_them(self, node_tree):
        lod_graph(node_tree)
        obj = build_tree(node_tree, "ao_tree")
        layout = CardLayout(obj.data)
        bark = layout.bark_vertices

        ramp = occlusion_channel(obj.data, 0.0)
        shaded = occlusion_channel(obj.data, 1.0)
        exposure = shaded / ramp
        assert np.all((exposure > 0.0) & (exposure <= 1.0 + 1e-6))

        card_exposure = exposure[bark:].reshape(-1, layout.per_card).mean(axis=1)
        centres = layout.centres(obj.data)
        distance = np.linalg.norm(centres - centres.mean(axis=0), axis=1)
        inner = distance <= np.median(distance)
        inner_mean = float(card_exposure[inner].mean())
        outer_mean = float(card_exposure[~inner].mean())
        assert inner_mean < outer_mean

        bark_exposure = exposure[:bark]
        assert bark_exposure.min() < 0.9

        lines = [
            "# fixture graph (trunk + sparse branch) + default Leaf Cards, canopy occlusion 1.0",
            "# exposure = baked occlusion / ramp-only occlusion",
            "",
            f"cards {layout.cards} inner mean {inner_mean:.4f} outer mean {outer_mean:.4f}",
            f"bark min {bark_exposure.min():.4f} mean {bark_exposure.mean():.4f} "
            f"max {bark_exposure.max():.4f}",
        ]
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "canopy_occlusion.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


class TestLeafLods:
    def test_lods_nest_cull_the_interior_and_keep_the_area(self, node_tree):
        cards = lod_graph(node_tree)
        obj = build_tree(node_tree, "lod_tree")

        cards.build_lods = False
        reference = build_tree(node_tree, "lod_reference")
        cards.build_lods = True

        tree_arrays = mesh_arrays(obj.data)
        for name, array in mesh_arrays(reference.data).items():
            assert np.array_equal(array, tree_arrays[name]), f"LODs changed LOD0 {name}"

        lods = {lod.name: lod for lod in find_lod_objects(obj)}
        assert sorted(lods) == ["lod_tree_LOD1", "lod_tree_LOD2"]
        lod1, lod2 = lods["lod_tree_LOD1"], lods["lod_tree_LOD2"]

        layout = CardLayout(obj.data)
        bark = layout.bark_vertices
        ids1, ids2 = layout.card_ids(lod1), layout.card_ids(lod2)

        assert len(ids1) == round(0.5 * layout.cards)
        assert len(ids2) == round(0.2 * layout.cards)
        assert set(ids2) <= set(ids1)

        bark_polygons = len(obj.data.polygons) - leaf_polygons(obj.data)
        tree_centres = layout.centres(obj.data)
        for lod, ids in ((lod1, ids1), (lod2, ids2)):
            assert len(lod.data.polygons) - leaf_polygons(lod.data) == bark_polygons
            assert np.array_equal(positions(lod.data)[:bark], positions(obj.data)[:bark])
            assert np.allclose(layout.centres(lod.data), tree_centres[ids], atol=1e-5)

        crown_centre = tree_centres.mean(axis=0)
        distance = np.linalg.norm(tree_centres - crown_centre, axis=1)
        culled = sorted(set(range(layout.cards)) - set(ids2))
        kept_distance = float(distance[ids2].mean())
        culled_distance = float(distance[culled].mean())
        assert culled_distance < kept_distance

        area0 = leaf_area(obj.data)
        area1, area2 = leaf_area(lod1.data), leaf_area(lod2.data)
        assert area2 / area0 == pytest.approx(1.0, abs=0.02)
        assert area1 / area0 == pytest.approx(1.0, abs=0.02)

        lines = ["# fixture graph (trunk + sparse branch) + default Leaf Cards, manifold", ""]
        for level, mesh in enumerate((obj.data, lod1.data, lod2.data)):
            lines.append(
                f"LOD{level} verts {len(mesh.vertices)} polys {len(mesh.polygons)} "
                f"cards {int(leaf_polygons(mesh) / 2)} leaf_area {leaf_area(mesh):.4f}"
            )
        lines += [
            "",
            f"LOD1 area ratio {area1 / area0:.4f}",
            f"LOD2 area ratio {area2 / area0:.4f}",
            f"mean card distance to crown centre: kept in LOD2 {kept_distance:.4f}, "
            f"culled {culled_distance:.4f}",
        ]
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "lod_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_same_build_twice_gives_identical_lods(self, node_tree):
        lod_graph(node_tree)
        first = build_tree(node_tree, "lod_twice_a")
        second = build_tree(node_tree, "lod_twice_b")
        for lod_a, lod_b in zip(find_lod_objects(first), find_lod_objects(second), strict=True):
            arrays_a, arrays_b = mesh_arrays(lod_a.data), mesh_arrays(lod_b.data)
            assert set(arrays_a) == set(arrays_b)
            for name in arrays_a:
                assert np.array_equal(arrays_a[name], arrays_b[name]), f"{lod_a.name} {name}"

    def test_lods_skin_to_the_tree_rig_and_follow_its_lifecycle(self, node_tree):
        cards = lod_graph(node_tree)
        mesher = node_tree.mesher
        mesher.build_bone_rig = True

        obj = build_tree(node_tree, "lod_rigged")
        mesher.build_into(obj, 0)

        named = [o.name for o in bpy.data.objects if o.name.startswith("lod_rigged_LOD")]
        assert sorted(named) == ["lod_rigged_LOD1", "lod_rigged_LOD2"]

        armature = find_armature(obj)
        bones = {bone.name for bone in armature.data.bones}
        for lod in find_lod_objects(obj):
            modifiers = [m for m in lod.modifiers if m.type == "ARMATURE"]
            assert len(modifiers) == 1 and modifiers[0].object.name == armature.name
            assert lod.parent.name == armature.name
            assert np.allclose(np.array(lod.matrix_world), np.array(obj.matrix_world), atol=1e-6)
            assert {group.name for group in lod.vertex_groups} <= bones
            weights = [sum(g.weight for g in v.groups) for v in lod.data.vertices]
            assert np.allclose(weights, 1.0, atol=1e-5)

        rig = build_rig_for(obj.data, mesher.segments_per_chain, mesher._stiffness_settings())
        apply_pose(armature, rig, 45.0, 1.0)
        for lod in find_lod_objects(obj):
            lod.hide_viewport = False
        bpy.context.view_layer.update()

        bark = CardLayout(obj.data).bark_vertices
        posed_tree = evaluated_vertices(obj)
        for lod in find_lod_objects(obj):
            posed = evaluated_vertices(lod)
            assert np.allclose(posed[:bark], posed_tree[:bark], atol=1e-5)
            assert np.abs(posed - positions(lod.data)).max() > 1e-3, f"{lod.name} did not bend"

        cards.build_lods = False
        mesher.build_into(obj, 0)
        assert find_lod_objects(obj) == []
        assert not [m for m in bpy.data.meshes if m.name.startswith("lod_rigged_LOD")]

    def test_each_variant_owns_its_lods_and_preview_switches_them(self, node_tree):
        cards = lod_graph(node_tree)
        variants = node_tree.variants
        variants.variant_count = 2
        variants.collection_name = "lod_variants"
        variants.bake_foliage_streams = False
        variants.generate_variants()
        assert not variants.status_is_error, variants.status_message

        trees = variants.variant_objects()
        assert sorted(obj.name for obj in trees) == ["lod_variants_000", "lod_variants_001"]
        for obj in trees:
            lods = find_lod_objects(obj)
            assert [lod.parent.name for lod in lods] == [obj.name, obj.name]
            assert [lod.users_collection[0].name for lod in lods] == ["lod_variants"] * 2

        cards.preview_lod = "2"
        for obj in trees:
            assert obj.display_type == "BOUNDS"
            shown = [lod.name for lod in find_lod_objects(obj) if not lod.hide_viewport]
            assert shown == [f"{obj.name}_LOD2"]

        cards.preview_lod = "0"
        for obj in trees:
            assert obj.display_type == "TEXTURED"
            assert all(lod.hide_viewport for lod in find_lod_objects(obj))
