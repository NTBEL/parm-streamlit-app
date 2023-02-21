import networkx as nx

# Special Keys
ID = 'id'
NAME = 'name'
DATA = 'data'
ELEMENTS = 'elements'

NODES = 'nodes'
EDGES = 'edges'

SOURCE = 'source'
TARGET = 'target'

DEF_SCALE = 100

CY_GML_NODE_STYLE = {'ellipse': 'ellipse', 'roundrectangle': 'round-rectangle'}
CY_GML_ARROWS = {'standard': 'triangle', 'none': 'none'}
CY_GML_LINE_STYLE = {'line': 'solid', 'dotted': 'dotted', 'dashed': 'dashed'}

def to_networkx(cyjs, directed=True):
    """
    Convert Cytoscape.js-style JSON object into NetworkX object.
    By default, data will be handles as a directed graph.
    """

    if directed:
        g = nx.MultiDiGraph()
    else:
        g = nx.MultiGraph()

    network_data = cyjs[DATA]
    if network_data is not None:
        for key in network_data.keys():
            g.graph[key] = network_data[key]

    nodes = cyjs[ELEMENTS][NODES]
    edges = cyjs[ELEMENTS][EDGES]

    for node in nodes:
        data = node[DATA]
        print(data)
        g.add_node(data[ID], **data)

    for edge in edges:
        data = edge[DATA]
        source = data[SOURCE]
        target = data[TARGET]

        g.add_edge(source, target, attr_dict=data)

    return g