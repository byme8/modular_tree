# Foliage vertex streams

Mtree does not simulate wind. It bakes the per-vertex data that a real-time
foliage shader needs in order to do wind itself, and the correctness of that data
is entirely Mtree's responsibility — the consuming engine cannot detect bad
values, it can only animate them.

A foliage shader reads several vertex streams beyond position and normal, and
supplies **zero** for any stream a mesh does not declare. A mesh exported without
them is not drawn "with default shading"; it is drawn with wind, LOD fade,
lighting and tinting all fed zero, which silently deforms it. This is why the
foliage export exists as a separate format rather than as an option on the
Pivot Painter path.

## Authoring

Two nodes are involved, and the split matters:

- **Foliage Wind** owns stiffness and the shading scalars. These are authoring
  decisions about how the tree behaves, not export settings, so they live on
  their own node and survive independently of how the mesh is exported. Both the
  export and the viewport debug view read from it, so what is previewed is what
  gets written.
- **Pivot Painter Export** in `Foliage Streams` mode writes the result onto the
  mesh. Without a Foliage Wind node in the tree it falls back to defaults and
  says so.

## What gets written

`Pivot Painter Export` → format `Foliage Streams` writes three layers onto the
tree mesh, leaving `uv0` alone:

| Layer | Channel | Semantic | Range |
|---|---|---|---|
| `uv0` (existing) | `xy` | bark parameterisation from the mesher | unbounded `v` |
| `FoliageWind` | `x` | **Stiffness** | `1` rigid → `0` free tip |
| `FoliageWind` | `y` | **Phase** | `[0,1)`, constant per branch |
| `FoliageHeight` | `x` | **Blade height** | metres |
| `FoliageHeight` | `y` | unused | `0` |
| `FoliageShading` | `r` | **Occlusion** | dark root → lit tip |
| `FoliageShading` | `g` | **Translucency** | `0` root → tip value |
| `FoliageShading` | `b` | **Normal blend** | `0` geometric → `1` flat up |
| `FoliageShading` | `a` | **Tint ramp position** | `0` root → `1` tip |

Layer names are what the exporter writes; map them onto whichever `uv1`/`uv2`
slots the target engine expects at import time.

## What a "blade" is here

The stream contract is written for grass, where a blade is the obvious unit of
motion. Mtree produces branching wood, so the mapping needs a decision: **a blade
is a branch chain**.

`ManifoldMesher` already assigns one `stem_id` per chain and keeps it constant
from the chain's base to its tip, forking a new id at every side branch
(`ManifoldMesher.cpp:425`). That id is the blade key, and `branch_extent` — the
summed length down the chain — is the blade length. Everything else is derived
from those two plus `radius`, `pivot_position` and `direction`.

No native change was needed for this. Normalised position along a chain is
recovered per-vertex by projecting `vertex - pivot_position` onto `direction` and
dividing by `branch_extent`, so the whole computation is pure Python in
`pivot_painter/foliage_streams.py` and testable without Blender or the compiled
module.

## What the values satisfy

**Stiffness is the wind authority.** The consuming shader derives
`sway = 1 - stiffness` and bends by `sway²`, so this decides which parts of the
tree move at all, and the response is strongly non-linear — `0.5` sits much
closer to rigid than to free.

The trunk and the side branches are controlled **separately**, because one curve
cannot serve both. Radius is the honest signal for side branches: a thick limb
resists wind, a thin twig does not. But the trunk is the thickest wood on the
tree, so a radius curve normalised against the tree maximum pins the entire trunk
at `1.0` and it never bends at all.

Trunk vertices (`hierarchy_depth == 0`) therefore interpolate from
**Trunk Base** at the root to **Trunk Tip** at the top, which is what produces
whole-tree lean. Side branches keep the radius curve, scaled by **Branch Scale**.

Regardless of the settings, **every vertex at a chain base is pinned to exactly
`1.0`**, not merely close to it — radius alone would leave rooted vertices
slightly below `1` and let the trunk drift out of the ground. A fully compliant
trunk (`base = tip = 0`) still has a rigid rooted ring.

Defaults are `Trunk Base 1.0`, `Trunk Tip 0.6`, `Branch Scale 1.0`. Setting both
trunk values to `1.0` restores the rigid-trunk behaviour, where only side
branches move; the node says so in its UI, because that state is easy to reach by
accident and looks like a bug.

**Phase must be constant across a branch and differ between branches.** It does
two unrelated jobs — offsetting the sway wave so neighbouring branches do not
move in lockstep, and acting as the per-blade id that density fade thresholds
against. A gradient *within* one branch tears that branch apart under wind and
makes parts of it fade out at different distances than the rest.

Deriving phase from `stem_id` alone guarantees constancy by construction. The
golden-ratio hash spreads sequential ids across the range so that adjacent
branches do not land on adjacent phases.

**Blade height is metres, not a normalised value.** It is the wind displacement
bound: offset is clamped to `height * 0.25`. Zero pins the geometry still; too
large lets vertices swing outside the mesh bounds and get culled while still on
screen. `branch_extent` is passed through in world units.

**Vertex colour is four independent scalars, not a colour.** Nothing samples it
as RGB. All four are functions of normalised position `t` along the branch:

```
r = lerp(occlusion_base, 1, t)      // darker at root
g = lerp(0, translucency_tip, t)    // more transmission toward tip
b = normal_blend                    // constant per plant
a = t                               // raw root→tip position
```

`a` stays the honest normalised position, because a material's root/mid/tip ramp
positions are authored against it — reshaping `a` silently rescales the artist's
ramp.

**`r` also carries canopy occlusion.** With **Canopy Occlusion** `s` above 0
(Foliage Wind, default 1), the ramp is multiplied by how much sky each vertex
sees through the leaf cards:

```
r *= 1 - s * (1 - sky_exposure)
```

`sky_exposure` is the same estimate that ranks cards for leaf LODs
(`sky_exposure` in `LeafCards.cpp`): leaf-card area splatted into a 32-cell grid,
32 Fibonacci-sphere rays per vertex, mean transmittance. It is evaluated at every
vertex, bark included, so a card buried in the crown and the trunk under it both
darken. Rays from points outside the grid are clipped to it, so a trunk base
below the crown still sees the crown above it.

It is baked for the rest pose and does not follow the wind. It is meant for
distant LODs on low-end hardware, where the shader multiplies albedo by `r` in
place of real-time AO; LOD1/LOD2 copy it from LOD0 through `lod0_vertex`.

## Bounds

Set mesh bounds to the **rest** geometry expanded by the wind reach, not the rest
geometry alone. `compute_wind_expanded_bounds` returns rest bounds expanded
horizontally by `max_blade_height * 0.25`; bounds that fit the rest pose exactly
will pop as instances bend out of them near the screen edge.

The exporter does not apply this to the Blender object — Blender recomputes
bounds from geometry — so it must be applied on the engine side at import.

## Skinning

Skinning is an alternative to this wind path, not a layer on top of it: a mesh
with a rig replaces vertex-stream wind entirely. **Mtree emits no rig and must
not emit a bone-binding stream** — on an unskinned mesh the skinning keyword is
off, nothing reads the stream, and it costs 16 bytes per vertex.

Phase remains required even on skinned meshes, because density fade still reads
it. So the constant-per-branch rule above holds either way.

Mtree's branch hierarchy would map onto a bone rig cleanly — `hierarchy_depth`
gives a topological order satisfying the "parent index below own index" rule that
a forward pose sweep requires — but that is unbuilt. The blocking requirement is
that at zero wind every bone must resolve to identity, which means the rest pose
has to describe the mesh exactly as its vertices are authored; a rest pose that
disagrees even slightly deforms the mesh while it stands still, and reads as "the
model is slightly wrong" rather than as a rigging bug.

## Checking the result

These values are invisible in a normal viewport: wrong ones produce perfectly
plausible geometry, and the defect only appears once wind runs in the engine. The
debug view shades the active tree by one stream at a time.

Set it from the `Pivot Painter Export` node in `Foliage Streams` mode, or from
`Scene.mtree_foliage_debug_channel`. Channel numbering matches the consuming
shader's own debug view, so a channel checked here and in-engine mean the same
thing:

`1` stiffness, `2` phase, `3` occlusion, `4` translucency, `5` normal blend,
`6` tint, `7` blade height, `8` structure, `9` canopy AO.

`9` draws `r` in grey rather than on the viridis ramp: it is the shading a
low-end material applies, so it previews how the distant tree will look. It
appears without a bake, because the debug view computes every channel through
`FoliageExporter.compute_streams`, the same path the bake writes. That path
also gives leaf cards their leaf colours, which the old preview did not.

**Check phase first.** It should render as flat constant patches, one per branch
chain. Any gradient inside a single branch is the defect described above. The
structure view (`8`) is the cross-check: it colours by `stem_id` using the same
hash that generates phase, so the two views should show identical patch
boundaries.

## Test artifacts

`tests/unit/test_foliage_streams.py` writes deterministic snapshots to
`tests/artifacts/foliage_streams/`:

- `sweep.txt` — every position-derived stream across the full `t` range,
  including out-of-range clamping
- `tree_invariants.txt` — per-branch phase constancy, rooted stiffness, and
  wind-expanded bounds on a synthetic three-branch tree
- `full_streams.txt` — every stream value for every vertex, with any violated
  invariant written into the file
- `stiffness_controls.txt` — the trunk stiffness profile under four control
  settings, showing the rooted pin holding in each
- `debug_ramp.txt` — the debug colour ramp and the structure-view blade colours

They are fixed-precision and canonically ordered, so a real change shows up in
`git diff` and float noise does not.
