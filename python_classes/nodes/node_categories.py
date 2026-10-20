import nodeitems_utils
from nodeitems_utils import NodeCategory, NodeItem


class MTreeNodeCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == "mt_MtreeNodeTree"


node_categories = [
    MTreeNodeCategory(
        "Structure",
        "Structure",
        items=[
            NodeItem("mt_TrunkNode"),
            NodeItem("mt_BranchNode"),
            NodeItem("mt_GrowthNode"),
            NodeItem("mt_PipeRadiusNode"),
        ],
    ),
    MTreeNodeCategory(
        "Mesh",
        "Mesh",
        items=[
            NodeItem("mt_MesherNode"),
            NodeItem("mt_LeafCardsNode"),
            NodeItem("mt_LeafClumpsNode"),
        ],
    ),
    MTreeNodeCategory(
        "Wind",
        "Wind",
        items=[
            NodeItem("mt_FoliageWindNode"),
        ],
    ),
    MTreeNodeCategory(
        "Output",
        "Output",
        items=[
            NodeItem("mt_VariantsNode"),
            NodeItem("mt_MeshExportNode"),
            NodeItem("mt_FoliageExportNode"),
        ],
    ),
    MTreeNodeCategory(
        "Leaf",
        "Leaf",
        items=[
            NodeItem("mt_LeafShapeNode"),
        ],
    ),
    MTreeNodeCategory(
        "Property",
        "Property",
        items=[
            NodeItem("mt_RandomPropertyNode"),
            NodeItem("mt_RampPropertyNode"),
        ],
    ),
]


def register():
    nodeitems_utils.register_node_categories("MTREE_NODES", node_categories)


def unregister():
    nodeitems_utils.unregister_node_categories("MTREE_NODES")
