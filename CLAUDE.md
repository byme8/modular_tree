# modular_tree

Blender extension (`blender_manifest.toml`, `blender_version_min = 4.3.1`) wrapping a
pybind11 C++ core (`m_tree`). Tree generation is a node graph; the mesher node builds
the tree, downstream nodes bake wind streams, rigs and exports onto the result.

## Comments

Code is self-explanatory or it gets rewritten until it is. No comments, and no
docstrings that restate the signature, the types, or what the code does. Rename the
function instead.

A docstring earns its place only when it states something the code cannot: an
invariant a reader would otherwise break, a constraint a downstream API forces on
us, or the reason a non-obvious approach was chosen over the obvious one. One or two
lines. If you are writing it to explain *what* happens, delete it.

Tests may carry `TEST_POINT:` on a case whose rationale is not evident from the
assertion — why that case exists, not what it checks.

Both are enforced by `law.mjs`, which polices docstrings under the same test as
comments. When it rejects something, delete it — moving the same words into a
docstring or a longer name is the same violation, not a fix.

## Asset ownership

Every generated tree — the mesher's preview tree and each variant alike — owns its
mesh, armature, wind clips and vertex groups exclusively. **Identity flows from the
target object, never from a module-level constant.** `pivot_painter/tree_asset.py`
owns this: `armature_name_for(obj)`, `find_armature(obj)`, `strip_rig(obj)`,
`owning_collection(obj)`; `wind_animation.action_prefix_for(armature)` extends the
same rule to actions.

A shared constant like `ARMATURE_NAME = "MtreeRig"` means two assets alive at once
write the same armature and clobber each other. That was the original bug; do not
reintroduce it in any form.

Consequences to preserve:
- Operations take the target object (or its name) as a parameter.
- Every operation is re-runnable in any order, any number of times. Rebuilding one
  asset cannot disturb another; rebuilding the same asset twice yields the same
  result rather than accumulating rigs.
- Never identify "the thing I just created" by diffing `bpy.data.objects.keys()` —
  it adopts anything else that appeared. Use the creating call's return value.

## Blender API

- **`bpy.data` for identity, `bpy.context` for "what is the user pointing at".**
  `bpy.context.scene.objects.get()` returns `None` for an object that exists but is
  not linked to the current scene. Use `bpy.data.objects.get()` to look things up.
- **Prefer the data API over `bpy.ops`.** Operators depend on global context and fail
  from timers, handlers and background mode.
- **`temp_override` does not work for `bpy.ops.object.mode_set`.** It switches the
  real active object's mode, not the overridden one, so `edit_bones.new()` then fails
  with "not in edit mode" *and* leaves global mode dirty — poisoning every later
  build with "Cannot add vertices in edit mode". Set
  `bpy.context.view_layer.objects.active` and restore it in a `finally`. Read
  `bpy.context.mode` (not `active.mode`) before switching, and check the active
  object is still yours before switching back.
- **Never link to `bpy.context.collection`** — the result depends on UI state, so the
  same run produces different scene layouts. Pass the collection explicitly.
- **Store names, not references.** `bpy.types.Object`/`Mesh` are pointers into arrays
  that undo, file load and `.remove()` invalidate. `debounce.py` re-resolves node
  trees by name for this reason; do the same everywhere.
- **Logic in plain functions; operators are thin UI wrappers.** Calling an operator
  from a node button loses the error message to the operator's report.
- **Never assign `self["prop"] = value` on a `NodeSocket`.** The ID-property subscript
  is the usual way to write inside an `update=` callback without recursing, but
  sockets do not support it — it raises `TypeError: id properties not supported for
  this type` on every call. All three socket classes clamped that way, so min/max did
  nothing and the console filled with tracebacks. Clamp with a hidden `stored_value`
  plus `get`/`set` on the public property instead.
- **A `get`/`set` property never fires `update`.** `MtreeSocket.write_stored` therefore
  calls `notify_socket_changed` explicitly; it is the graph's only change signal for
  value sockets. `bool_socket` has no clamp and so uses a plain `update=` callback.
  Any new socket type must do one or the other or edits to it will silently stop
  rebuilding.
- **Never write a socket from inside `construct_function`.** It schedules a rebuild
  from within a build, and it overwrites what the user authored. Correct the value on
  the built function instead — `growth_node` clamps `flower_threshold` that way.
- **`obj.matrix_world` is only refreshed by a depsgraph evaluation.** Inside one batch
  it still holds the placement from before any transform written since, so reading it
  after setting `location` returns the old slot. Use `tree_asset.authored_world_matrix(obj)`,
  which composes `matrix_basis` up the parent chain. Variants re-read the stale matrix
  while binding the rig and were all pulled back to the origin.

## Node graph

One strict left-to-right pipeline; `nodes/pipeline.py` owns it.

```
Trunk -> Branches/Growth/Pipe Radius ... -> Tree Mesher -> Leaf Cards -> Foliage Wind
    -> Tree Variants -> Mesh Export / Foliage Export
```

Leaf Cards, Foliage Wind and Tree Variants are optional. The exports attach after any
single-tree stage (they export the mesher's tree object) or after Tree Variants (they
export its collection).

- **Three chain socket types**, each with its own colour: `mt_TreeSocket` (structure),
  `mt_MeshSocket` (one built tree), `mt_VariantsSocket` (a set). `ACCEPTED_UPSTREAM`
  says which node may feed which; a mesh output feeds at most one Leaf Cards or Foliage
  Wind, so the single-tree chain is a line. Every node declares its chain sockets as
  `chain_input`/`chain_output`, and `init()` builds them from that.
- **Block Fill is a Tree Mesher mesh mode** (`mesh_mode` Manifold | Blocks), not a node:
  it decides how the tree mesh is built, so it cannot sit beside the mesher.
- **Materials `Bark` (slot 0) and `Leaf` (slot 1)** match the native mesher's polygon
  material indices. They are shared by name across every tree and variant so one edit
  reaches all of them; the build fills only empty slots, so a user-assigned material
  survives rebuilds. `Leaf` is added only when Leaf Cards acts.
- **Settings resolve along the chain, never by scanning the node tree.** The mesher
  walks upstream from its Skeleton input to the unique Trunk (`resolve_trunk`) and
  downstream to its own Leaf Cards and Foliage Wind (`chain_stage`,
  `foliage_wind_settings_for(mesher)`). Variants and exports walk upstream to their
  mesher. A tree-wide scan would let two pipelines in one node tree read each other's
  wind settings — the `ARMATURE_NAME` defect in graph form.
- **Incompatible links are removed, deferred.** `NodeTree.update()` finds them and
  `bpy.app.timers` removes them. Writing `link.is_valid = False` there does not stick:
  Blender re-validates every link after the Python callback returns (verified on 5.2).
  Removing links inside the callback risks the link-drag operator still holding the
  link. Until the timer runs, every traversal goes through `allowed_links`, so a
  rejected link is never followed.
- **Old files migrate on load.** `migration.migrate_loaded_trees` (`load_post`) rebuilds
  old-shape trees — mesher at the root, Block Fill/Leaf Cards/Variants/exports fanned
  off its output, Foliage Wind found by scan — into the chain, and stamps
  `graph_version`. It runs only while `needs_migration(tree)` is true, so a tree built
  in the new shape is never re-chained. `MTreeBlockFill` stays registered (hidden from
  the Add menu) only so migration can read it: an unregistered node loads as
  `NodeUndefined`, its `bl_idname` gone, so a Block Fill can no longer be told apart.
- Removing or renaming a chain socket is a graph-shape change: bump `GRAPH_VERSION` and
  teach `migration.py` the old shape.

## Diagnosing a build failure

Blender reduces an exception inside an operator or node callback to a one-line status
string and discards the traceback, so a failure deep in the pipeline surfaces as an
unrelated-sounding message. `python_classes/diagnostics.py` logs every stage.

Turn it on with **Log Pipeline** on the Tree Mesher node, or `MTREE_LOG=1` (or
`MTREE_LOG=C:\path\build.log`) before starting Blender. Output goes to the system
console — on Windows, *Window → Toggle System Console*.

```
> build_into target=tree resolution=4 rig=True
  graph nodes=3 links=[Tree Mesher->Trunk Trunk->Branches]
< mesh_tree 0.000s verts=0 polys=0
  built verts=0 polys=0 attrs=all
! build_into FAILED after 0.002s
```

Read it as a bracket: `>` enters a stage, `<` leaves with results, `!` failed with the
traceback. The stage whose result contradicts the error message is the real culprit —
above, "no branch chains" was a lie and `mesh_tree verts=0` was the truth.

Wrap new pipeline stages in `diagnostics.stage(...)` and log mesh state with
`describe_mesh` — it reports what the rig reads (vertex count, attributes, stem count),
not what the outliner shows.

## Wind clips

Wind ships as a skinned armature plus one animation clip per compass direction, and
Unity blends them. `wind_animation.py` owns all of it. Shape keys were the previous
delivery and are gone: `wind_morphs.py`, its tests and `tools/verify_morphs.py` were
deleted rather than kept alongside, because two wind paths that must agree is exactly
the defect described below.

Each clip carries three keyframes at the strengths in `STRENGTH_STOPS` — 0.0, 0.25,
1.0 — so the engine picks a point along the clip from wind power. Bending is rotation,
so a straight lerp from rest to full cuts the chord across the arc and pulls branch
tips inward; the error grows with the square of the rotation angle, which makes it
worst on the compliant twigs and mildest on the stiff trunk. The 0.25 stop exists to
put a keyframe where gentle wind actually sits. Adding stops is cheap here — three
keyframes per bone — where under shape keys each stop cost a whole vertex buffer.

A stop sits at frame `stop_frame(strength)` = `1 + round(strength * 4)` — frames 1, 2,
5 — with LINEAR interpolation. Engines sample a clip linearly in *time*, so evenly
spaced frames for unevenly spaced strengths would bend the strength ramp; spacing the
keys by strength makes time proportional to strength, and LINEAR makes the frames the
FBX writer bakes between stops (3, 4) the straight blend of the 0.25 and 1.0 poses.

The chain hierarchy is not optional. Stiffness is a scalar per vertex: it says how much
a vertex resists wind, never what it rotates about. Only the chains make a twig inherit
its parent limb's lean instead of shearing off it.

**`apply_pose` must write `matrix_basis`, never `rotation_quaternion`.** `pose_rig`
returns world matrices that rotate about each chain's pivot, which is not the bone head.
A quaternion drops that pivot translation and is read in the bone's own rest basis
rather than the parent's, so both the pivot and the basis come out wrong and the error
compounds down the chain. That bug shipped: the viewport showed a tree that was never
what exported. The correct form is
`rest⁻¹ @ parent_world⁻¹ @ world @ rest`, and `matrix_basis` also sidesteps the
ordering trap in `pose_bone.matrix`, which reads current parent state and so depends on
assignment order down the chain. `tests/blender/test_wind_animation.py` holds this to
1e-4 against `deform_vertices`; it currently lands at ~5e-6.

Because the poses live on the armature, the writer constraints that shape keys forced
are all lifted: triangulation, modifier evaluation and `bake_space_transform` no longer
break anything, and `use_triangles=True` is what Unity wants anyway. What the writer
does need is the armature itself — `object_types={"MESH", "ARMATURE"}`,
`export_skins` — decided from `_rigged_armatures(objects)`, never from
`bpy.context.selected_objects`, which in per-file mode sees only the current object and
would drop the rig from every file but one. Scaling is `FBX_SCALE_UNITS` with
`apply_unit_scale=True`; under `FBX_SCALE_ALL` Unreal imported a root bone scaled 100.

Actions are named `{armature_name}_Wind_{angle:03d}`, and the armature is named from
the mesh. A shared prefix would make two trees alive at once overwrite each other's
clips — the same defect a shared `ARMATURE_NAME` once was.

**Each armature exports only its own clips, and bone names carry that.** Bones are
`{armature_name}_stem{id}_seg{n}`. The FBX writer's `bake_anim_use_all_actions` offers
*every* action in `bpy.data` to an armature and keeps those whose F-curve paths
resolve on it; with bare `stem{id}_seg{n}` names, identical across variants, every
skeleton received every variant's clips — in both combined and per-file mode, and an
engine then plays one tree's clip on another's rest pose. Per-armature bone names make
foreign paths fail to resolve, so `use_all_actions` writes exactly the owner's clips,
one take each, named `{armature}|{action}`. The NLA route
(`bake_anim_use_nla_strips`) is not an alternative: it bakes every exported object
into every strip's take, so a combined file still hands each skeleton all takes,
the foreign ones frozen at rest. glTF is the other way round — its `ACTIONS` mode
exports active and NLA actions per object, but with a lone armature also every bone
action in the file — so `own_clips_on_nla` stacks the owner's clips on temporary NLA
tracks for the write, with `export_anim_single_armature=False`, and removes them after.
`tests/blender/test_wind_export.py` re-imports both formats and checks ownership, and
round-trips the full-strength FBX pose to 1e-4 m of the source; it lands at ~4e-6.

## Leaf LODs

**Build LODs** on Leaf Cards adds `<tree>_LOD1` and `<tree>_LOD2` to every tree the
mesher builds, the preview tree and each variant. Bark is identical in every LOD; only
cards change. LOD1/LOD2 keep `lod1_fraction`/`lod2_fraction` (default 0.5/0.2) of the
LOD0 cards.

- **Culling is by exposure, in C++** (`LeafCards.cpp`). Card area is splatted into a
  32-cell grid over the cards' bounds; from each card centre 32 Fibonacci-sphere
  directions march through it, transmittance `exp(-0.5 * leaf area density * path)`
  (0.5 is the mean projection of a randomly oriented card), exposure is the mean.
  Cards are kept most-exposed first, ties by index. The same grid, built from the
  mesh's leaf polygons, bakes canopy occlusion into `FoliageShading.r` for every
  vertex (`sky_exposure`, see `docs/foliage_streams.md`). Changing the estimate
  changes both LOD culling and shading.
- **LODs nest and never pop.** One exposure ranking cuts both levels, so LOD2 ⊂ LOD1 ⊂
  LOD0, and a survivor is the LOD0 card itself, same centre and orientation. Placement
  runs the unchanged RNG stream and `keep_fraction >= 1` emits exactly as before, so
  LOD0 is byte-identical with LODs on or off.
- **Survivors grow by `sqrt(N_lod0 / N_kept)` about their centre**, one global factor,
  so total leaf area is preserved exactly. Per-region scaling cannot: the interior
  regions that lose cards have no survivors left to grow.
- **One skeleton.** A LOD mesh carries `lod0_vertex`, the LOD0 vertex each of its
  vertices came from; `bind_to_armature` binds the LODs with those vertices' weights to
  the tree's own armature, so all LODs share the root bone and bone names and the wind
  clips play on every LOD. `strip_rig` unbinds them.
- LOD objects are found by name from their tree, never through parenting. Unrigged
  they are parented to the tree at identity; rigged, `bind_lods` parents them to the
  tree's armature with the tree's own parent-inverse and basis, so every exported LOD
  sits under the armature like LOD0 (glTF warns "Armature must be the parent of skinned
  mesh" otherwise, and Unreal's LOD Import expects the same hierarchy). A rebuild reuses them by name, building with LODs off removes them, and
  `variant_objects()` skips them (`is_lod_object`), so exports never pick them up.
- **Preview LOD** (Leaf Cards) shows one level for every tree in the pipeline. The tree
  is switched to `display_type = "BOUNDS"` rather than hidden: a hidden object cannot be
  selected, and the exports select it. LOD objects never render.

- **Foliage streams follow LOD0.** They are constant per vertex, so
  `copy_foliage_streams` copies the wind/height UVs (same slots) and the shading colour
  through `lod0_vertex` after every bake (Variants and the Foliage Export operator).
  Recomputing them on the LOD mesh would not match LOD0.

**Export: one file per LOD.** With Build LODs on, Mesh Export writes
`<name>_LOD0/1/2.<ext>`: in combined mode `<name>` is the source and each file holds
every tree at that level; in separate mode it is each tree. With it off the output is
exactly as before, with no suffix. Blender's FBX writer cannot emit an FbxLODGroup, so
the Unreal team attaches LOD1/LOD2 by hand through LOD Import, and Unreal accepts a LOD
only if it shares the base mesh's root bone and references no bone the skeleton lacks.
That is why every LOD file carries the tree's own armature. Wind clips go only in LOD0:
LOD1/LOD2 files are written with `bake_anim`/`export_animations` off and the armature at
`pose_position = "REST"`. Export temporarily un-hides LOD objects
(`selectable`), since the writers export the selection.

## Seeds and variants

Randomness is per-node: trunk, each branch level and growth each carry their own
`seed` socket. A different tree means offsetting all of them at once.

`MtreeFunctionNode.construct_function(seed_offset)` applies the offset in flight and
**never writes it to the sockets**. Writing seeds would make a variant a mode the
whole graph is temporarily in: `debounce.py` polls socket values at 10Hz, observes
the half-applied edit, and fires a phantom rebuild of the mesher mid-loop.

`branch_node.py` and `growth_node.py` override `construct_function`. Any new override
must thread `seed_offset` through, or the offset stops there and every variant below
it is identical — which still generates plausible trees, so it fails silently.

## Tests

Two tiers. The pure-Python tier is fast and covers the maths; the Blender tier is the
only thing that can cover object lifecycle, collections and armatures.

### Tier 1 — pure Python (no Blender), `tests/unit/`

```
cd tests
uv run --with pytest --with numpy python -m pytest -q
```

Production code under `python_classes/` mostly imports `bpy` at module scope, so
`conftest.py` puts the bpy-free subpackages on `sys.path`, and `load_node_base()`
loads `nodes/base_types/node.py` directly (it imports `bpy` only under
`TYPE_CHECKING`, but its package `__init__` does not).

### Tier 2 — inside Blender, `tests/blender/`

```
bun tools/blender-test.ts                    # all
bun tools/blender-test.ts -k asset_isolation # filtered, args pass to pytest
BLENDER=/path/to/blender bun tools/blender-test.ts
```

The launcher finds Blender, vendors pytest into a temp directory (nothing is installed
into the Blender tree), and runs `tests/blender/runner.py` inside it. Verified on
Blender 5.2.0 LTS / Python 3.13; `numpy` is bundled, `pytest` is not.

Blender is not under `Program Files` on this machine and the uninstall key's
`InstallLocation` is empty — it lives in the Steam library. Add new locations to
`CANDIDATES` in the launcher rather than sweeping a drive root.

Two flags matter and the launcher always passes them. `--factory-startup` ignores
installed addons and preferences, so the tests neither read nor disturb the real
config. `-b` is background mode, which is what makes operator-context bugs surface at
all: most `bpy.ops` calls refuse to run without an active area, so green headless
implies green UI.

`runner.py` puts the repo root at the front of `sys.path`. **This is load-bearing** —
the addon is usually also installed as an extension, and without that precedence the
tests import the installed copy and verify stale code. A probe hit exactly this and
reported a false failure.

**A native change must be installed in two places.** The tests import `m_tree` from
`%TEMP%\mtree-blender-pylibs` (`MTREE_PYLIBS`, ahead of everything), while the UI
imports it from `%APPDATA%\Blender Foundation\Blender\5.2\extensions\.local\lib\python3.13\site-packages`.
The extension in `user_default` is a symlink to the repo, so Python edits are live
and only the native module needs installing. Build the wheel from a VS dev
environment (`vcvars64.bat`, then `uv build --wheel --python 3.13 m_tree`), copy it
over `wheels/m_tree-*-cp313-*.whl` (the manifest ships it), and
`uv pip install --target <dir> --python-version 3.13 --reinstall --no-deps <wheel>`
into both directories with Blender closed. The version stays 5.5.2, so without
`--reinstall` the stale binary survives. A module missing a new binding is the stale
copy.

Tests here register the addon via `python_classes.register()` in the session fixture,
not `bpy.ops.preferences.addon_enable` — the latter would load the installed copy.

Compare Blender data by **name, not identity**: `bpy` hands out a fresh wrapper per
access, so `node_tree.nodes["X"] is node_tree.nodes["X"]` is False.

CI currently runs `ruff` and the C++ `ctest` suite only — **neither Python tier runs in
`CD.yml`**. Until that changes, both tiers are local-only gates.

### Rules for both tiers

- **Never mirror production logic into a test.** A re-declared copy of the function
  under test stays green through any production change. `test_tree_variants.py` did
  this and was worthless as a regression net. Import the real symbol, or load the
  module as `load_node_base()` does.
- Stand-in nodes must subclass the real `MtreeFunctionNode`: `construct_function`
  isinstance-checks its children, so a duck-typed fake silently builds a childless
  tree and every assertion about child nodes passes vacuously.
- **Never hold a Blender datablock reference across a mutating call.** A probe here
  held an `Armature` across `strip_rig` and died with `ReferenceError: StructRNA of
  type Armature has been removed`. Re-resolve by name after anything that removes data.
- Every test emits a deterministic artifact under `tests/artifacts/<area>/` — fixed
  precision, canonical order, no timestamps or machine paths. Snapshot text is the
  default; add images only when a shape carries the meaning.
- Prefer widening an existing parameter sweep to adding a test file.
