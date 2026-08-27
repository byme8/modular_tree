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
mesh, armature, morphs and vertex groups exclusively. **Identity flows from the
target object, never from a module-level constant.** `pivot_painter/tree_asset.py`
owns this: `armature_name_for(obj)`, `find_armature(obj)`, `strip_rig(obj)`,
`owning_collection(obj)`.

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

## Wind morphs

The skeleton is a tool, not a deliverable. A bone rig is built so the tree's response
to wind can be seen and evaluated at full strength; the resulting poses are baked to
shape keys, and **the skeleton is then deliberately not exported** — hence
`object_types={"MESH"}` for FBX and `export_skins=False` for glTF. Both are correct;
do not "fix" them by shipping the armature.

Morphs do constrain the writer, because each is a vertex-position delta against the
base mesh. Anything that changes vertex count, order or position between bake and
write invalidates every key: triangulation (`use_triangles`), modifier evaluation
(`use_mesh_modifiers` / `export_apply`) and `bake_space_transform` are all disabled
when morphs are present. That decision must come from the objects being written —
`_has_wind_morphs(objects)` — never from `bpy.context.selected_objects`, which in
per-file mode sees only the current object and would give two trees in one batch
different axis handling.

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
reported a false failure. `m_tree` still resolves from the extension's site-packages,
so the native module works without building anything.

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
