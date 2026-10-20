from __future__ import annotations

import bpy

from .. import diagnostics
from . import pipeline
from .base_types.node_tree import GRAPH_VERSION

BLOCK_FILL_SETTINGS = (
    ("density", "block_density"),
    ("scale", "block_scale"),
    ("roll_variation", "block_roll_variation"),
    ("seed", "block_seed"),
    ("max_polygons", "block_max_polygons"),
)

COLUMN_WIDTH = 260.0
ROW_HEIGHT = 320.0


def migrate_all_trees():
    for tree in list(bpy.data.node_groups):
        if tree.bl_idname == "mt_MtreeNodeTree":
            migrate_tree(tree)
    return None


@bpy.app.handlers.persistent
def migrate_loaded_trees(_file_path=None):
    migrate_all_trees()


def migrate_tree(tree) -> bool:
    try:
        if tree.graph_version >= GRAPH_VERSION:
            return False
        migrated = pipeline.needs_migration(tree)
        if migrated:
            with diagnostics.stage("migrate_graph", tree=tree.name):
                _migrate(tree)
        tree.graph_version = GRAPH_VERSION
        return migrated
    except Exception:
        diagnostics.exception("migrating node tree %s failed", tree.name)
        return False


class _OldGraph:
    def __init__(self, tree):
        self.tree = tree
        self.links = [(link.from_node.name, link.to_node.name) for link in tree.links]
        self.meshers = [node.name for node in tree.nodes if node.bl_idname == pipeline.MESHER_ID]
        self.adjacency: dict[str, list[str]] = {}
        for source, target in self.links:
            self.adjacency.setdefault(source, []).append(target)
            self.adjacency.setdefault(target, []).append(source)
        self.owners = {node.name: self._reached_mesher(node.name) for node in tree.nodes}

    def _reached_mesher(self, name):
        seen = {name}
        queue = [name]
        while queue:
            current = queue.pop(0)
            if current in self.meshers:
                return current
            for neighbour in self.adjacency.get(current, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        return None

    def idname(self, name):
        return self.tree.nodes[name].bl_idname

    def consumers(self, mesher):
        return [target for source, target in self.links if source == mesher]

    def owned(self, mesher, bl_idname, adopt_orphans=False):
        adopts = adopt_orphans and self.meshers and mesher == self.meshers[0]
        return [
            node.name
            for node in self.tree.nodes
            if node.bl_idname == bl_idname
            and (self.owners[node.name] == mesher or (adopts and self.owners[node.name] is None))
        ]

    def acting(self, mesher, bl_idname):
        direct = [name for name in self.consumers(mesher) if self.idname(name) == bl_idname]
        if direct:
            return direct[0]
        owned = self.owned(mesher, bl_idname)
        return owned[0] if owned else None

    def structure_root(self, mesher):
        return next(
            (
                name
                for name in self.consumers(mesher)
                if self.idname(name) in pipeline.STRUCTURE_IDS
            ),
            None,
        )


def _migrate(tree):
    old = _OldGraph(tree)

    plans = []
    claimed = set()

    def claim(names):
        free = [name for name in names if name not in claimed]
        claimed.update(free)
        return free

    for mesher in old.meshers:
        acting_cards = old.acting(mesher, pipeline.LEAF_CARDS_ID)
        leaf_cards = claim([acting_cards] if acting_cards is not None else [])
        wind = claim(old.owned(mesher, pipeline.FOLIAGE_WIND_ID, adopt_orphans=True)[:1])
        plans.append(
            {
                "mesher": mesher,
                "root": old.structure_root(mesher),
                "block_fill": old.acting(mesher, pipeline.BLOCK_FILL_ID),
                "stages": leaf_cards + wind,
                "variants": claim(old.owned(mesher, pipeline.VARIANTS_ID)),
                "mesh_exports": claim(
                    old.owned(mesher, pipeline.MESH_EXPORT_ID, adopt_orphans=True)
                ),
                "foliage_exports": claim(old.owned(mesher, pipeline.FOLIAGE_EXPORT_ID)),
            }
        )

    for plan in plans:
        if plan["block_fill"] is not None:
            _adopt_block_fill(tree.nodes[plan["mesher"]], tree.nodes[plan["block_fill"]])

    for node in list(tree.nodes):
        if node.bl_idname == pipeline.BLOCK_FILL_ID:
            diagnostics.log("  removed Block Fill node %s", node.name)
            tree.nodes.remove(node)

    for node in tree.nodes:
        if node.bl_idname.startswith("mt_"):
            pipeline.conform_chain_sockets(node)

    for plan in plans:
        try:
            _rebuild_chain(tree, plan)
        except Exception:
            diagnostics.exception("  rebuilding the chain of %s failed", plan["mesher"])

    for link in pipeline.rejected_links(tree):
        tree.links.remove(link)


def _adopt_block_fill(mesher, block_fill):
    for source, target in BLOCK_FILL_SETTINGS:
        setattr(mesher, target, getattr(block_fill, source))
    mesher.mesh_mode = "BLOCKS"
    diagnostics.log("  %s takes Block Fill settings from %s", mesher.name, block_fill.name)


def _deepest_structure(node):
    seen = {node.name}
    while True:
        child = next(
            (
                link.to_node
                for link in node.outputs[0].links
                if link.to_node.bl_idname in pipeline.STRUCTURE_IDS
                and link.to_node.name not in seen
            ),
            None,
        )
        if child is None:
            return node
        seen.add(child.name)
        node = child


def _rebuild_chain(tree, plan):
    nodes = tree.nodes
    mesher = nodes[plan["mesher"]]

    if plan["root"] is not None:
        deepest = _deepest_structure(nodes[plan["root"]])
        tree.links.new(deepest.outputs[0], mesher.inputs[0])
        mesher.location = (deepest.location.x + COLUMN_WIDTH, deepest.location.y)

    fan_out: dict[str, int] = {}

    def attach(upstream, name):
        node = nodes[name]
        tree.links.new(upstream.outputs[0], node.inputs[0])
        row = fan_out.get(upstream.name, 0)
        fan_out[upstream.name] = row + 1
        node.location = (
            upstream.location.x + COLUMN_WIDTH,
            upstream.location.y - row * ROW_HEIGHT,
        )
        return node

    last = mesher
    for name in plan["stages"]:
        last = attach(last, name)

    for name in plan["variants"]:
        attach(last, name)

    export_source = nodes[plan["variants"][0]] if plan["variants"] else last
    for name in plan["mesh_exports"]:
        attach(export_source, name)

    for name in plan["foliage_exports"]:
        attach(last, name)
