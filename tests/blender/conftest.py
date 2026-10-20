import bpy
import pytest


@pytest.fixture(scope="session", autouse=True)
def registered_addon():
    """Register the working-tree addon directly.

    addon_enable would load the installed extension instead, which is a
    different copy of the code and defeats the point of the tier.
    """
    import python_classes

    already = bpy.data.node_groups.get("__mtree_probe__") is not None
    if not already:
        try:
            python_classes.register()
        except Exception as error:
            if "already registered" not in str(error):
                raise

    yield


class Graph:
    def __init__(self, tree, mesher, trunk, variants):
        self.tree = tree
        self.mesher = mesher
        self.trunk = trunk
        self.variants = variants
        self.branch = None

    def __iter__(self):
        return iter((self.mesher, self.variants))

    def add_branch(self):
        """Attach a branch level, sparse enough to stay fast.

        A trunk-only tree does not bend: every bone is trunk, and the default
        stiffness pins it. Any test about deformation needs branches or it
        measures zero and passes for the wrong reason.
        """
        branch = self.tree.nodes.new("mt_BranchNode")
        self.tree.links.new(self.trunk.outputs[0], branch.inputs[0])

        for socket in branch.inputs:
            name = getattr(socket, "property_name", "")
            if name == "branches_density":
                socket.property_value = 0.05
            elif name == "resolution":
                socket.property_value = 1

        self.branch = branch
        return branch


@pytest.fixture
def node_tree():
    tree = bpy.data.node_groups.new("mtree_test", "mt_MtreeNodeTree")

    mesher = tree.nodes.new("mt_MesherNode")
    trunk = tree.nodes.new("mt_TrunkNode")
    tree.links.new(trunk.outputs[0], mesher.inputs[0])

    variants = tree.nodes.new("mt_VariantsNode")
    tree.links.new(mesher.outputs[0], variants.inputs[0])

    yield Graph(tree, mesher, trunk, variants)

    bpy.data.node_groups.remove(tree)
