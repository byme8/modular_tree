# Unreal handoff: instanced wind from Mtree's skinned trees

What Mtree exports, what the consuming Unreal node has to honour, and which
assumptions in that node would silently produce a wrong-looking forest.

Nothing in this document requires a change to Mtree. The exporter already writes
the Unreal axis convention and the full clip set.

## The target workflow

PCG scatters trees. Every instance needs wind. Skeletal meshes cannot be
instanced the way static meshes can — Unreal has no instanced skeletal mesh
component, and one component per tree is a performance cliff at forest density.

The plan that avoids it: bake the bone poses once per tree, and at runtime
evaluate the wind ring **once per frame for the whole forest** into a shared map.
Each instance samples that map using its own direction and position. The ring is
derived data, not asset data, so it costs one evaluation per frame rather than
one per instance.

## What ships in the FBX

One file per tree, from the `Mesh Export` node with `Target = Unreal`
(Z-up, -Y forward).

| Part | Detail |
|---|---|
| Mesh | triangulated, flat shaded, `UVMap` |
| Material slots | 0 bark, 1 leaf cards |
| Armature | bones named `stem{stem_id}_seg{segment}` |
| Skin weights | every vertex bound; no unweighted geometry |
| Clips | `{armature}_Wind_{angle:03d}`, one per baked direction |
| Keyframes per clip | 3, at wind strengths 0.0, 0.25, 1.0 |

The armature is named from the mesh, and actions are prefixed with the armature
name, so two trees in one scene never share clips.

Bone transforms are the entire wind payload. There is no wind texture, no vertex
animation cache, and no per-direction mesh.

## Direction count is authored, not fixed

`Wind Directions` on the Tree Mesher node: default **8**, range 2–16. The artist
sets it per tree.

**The Unreal node must discover the count from the imported asset, not hardcode
it.** Enumerate the `{armature}_Wind_*` actions and read the angle out of the
suffix; the angles are evenly spaced at `360 / count` degrees starting at 0.

## Two interpolation traps

Both are the same mistake on different axes, and both are already paid for on the
Mtree side. An implementer who "simplifies" either one reintroduces an artifact
that looks like a modelling bug rather than an interpolation bug.

**Strength: three stops, not two.** Bending is rotation. A straight lerp from
rest to full wind cuts the chord across the arc and pulls branch tips inward. The
error grows with the square of the rotation angle, so it is worst on the
compliant twigs and mildest on the stiff trunk. The 0.25 keyframe exists to put a
sample where gentle wind actually sits. Interpolate rest → 0.25 → full, never
rest → full.

**Direction: slerp, not lerp.** Blending two baked directions is a blend across a
rotation, with the same chord error one axis over. Interpolate the bone rotations
as quaternions. Lerping matrices or raw components reproduces exactly the artifact
the strength stops were added to avoid. At 8 directions the gap is 45°, which is
large enough for this to be visible on twigs.

## What the bones mean

Bone count comes from three settings on the Foliage Wind node: **Trunk Bones**
(default 4), **Branch Bones** (default 2) and **Rig Depth** (default 1).

- depth 0 chain → Trunk Bones bones
- depth 1 … Rig Depth → Branch Bones bones
- deeper than Rig Depth → **no bones of its own**

Every vertex of an unrigged chain is weight-1 to the bone its nearest rigged
ancestor carries at the fork, resolved through unrigged intermediates. So twigs
and their leaf cards move as part of the limb they grow from.

Do not expect a bone per twig. A default tree rigs at single-digit bone counts;
the same tree at Rig Depth 16 rigs in the thousands. The whole per-frame-ring
plan depends on the bone count staying small, so Rig Depth is a budget control,
not a quality knob.

Leaf cards are ordinary skinned geometry on material slot 1. They carry their
twig's attributes, so they bend with the twig rather than being animated
separately.

## Do not run both wind paths

Mtree has a second, unrelated wind delivery: per-vertex foliage streams
(stiffness, phase, blade height, shading scalars) consumed by a vertex shader,
documented in `docs/foliage_streams.md`. That path is an **alternative** to
skinning, not a layer on top of it. A mesh that is skinned and also stream-driven
gets bent twice.

Pick one per asset. The skinned path is the one this document describes.

## Rotational symmetry is unmeasured

If the ring is ever cut from N baked directions down to one, by rotating a single
baked pose about world up, that assumes the tree bends symmetrically — that
bending north and bending east are the same deformation, turned. A real skeleton
is asymmetric, so this is an assumption, not a fact, and nobody has measured the
residual.

It does not affect the current plan, which interpolates between real baked
directions and therefore captures asymmetry by construction. It only matters if
someone later tries to shrink the asset.

The measurement is cheap if it is ever wanted: bake at a high direction count and
compare each real clip against the rotated direction-0 pose.
`tests/blender/test_wind_animation.py` already holds the pose path to 1e-4 and is
where that comparison would go.

## Known gaps on the Mtree side

- No automated test covers **exporting** a blocks-and-cards tree. A manual probe
  confirmed 2 material slots, both material indices, bones and clips in the
  written FBX; the manifold tree is the only surface the committed export test
  builds.
- `Mesh Export` defaults to `Objects = Collection`, `Collection = tree_variants`.
  A single mesher-built tree is not in that collection, so Check and Export both
  report "no mesh objects found" until the source is switched to `Selected` or
  the collection name is corrected.

## Open questions for the Unreal side

These decide the shape of the node and are not answerable from this repo.

1. Unreal version, and whether PCG is doing the scattering at runtime or baking
   to instances.
2. Where the per-frame ring lives: a component, a world subsystem, or a material
   parameter collection.
3. Whether instances need individually varying wind (gameplay forces, local
   gusts) or only the shared global ring. Individual response breaks the single
   shared map and changes the design.
4. C++ node or Blueprint.
