import src
from langchain_core.messages import HumanMessage

queries = [
    """
Write a function that reads a png file and return a NumPy array with YUV format using PIL package and nothing else. You can use the image in 'data/diamond_sword.png'.
    """,
    """
Write a function that takes a NumPy array with YUV format and return a graph using networkx package using the following constraints:
- for each pixel, you look at all its neighbors (diagonal included)
- if the absolute difference of the YUV channels of the current pixel and the neighbor in Y, U, V is less than 48, 7, or 6 respectively, then they are connected.
- you are obliged to use the `networkx` package by using the following import `import networkx as netx`.
- You have to use the image in 'data/diamond_sword.png'.

    """,
    """
Write a function that plots the graph and the image using matplotlib with the following constraints.
- Both are on a unique figure.
- The most important is to see the edges of the graph and the pixels of images and understand the connections.
- the graph and the image must have the same orientation.
- the `node_size` must be equal to 1.
- You can use the image in 'data/diamond_sword.png'.
    """,
    """
Simplify the graph using these 4 criteria:
1. If a 2x2 block is fully connected, which means that we have all the edges of the blocks and the diagonal, then it is part of a continuously shaded region. In this case the two diagonal connections can be safely removed without affecting the final result.
2. If two pixels are part of a long curve feature, they should be connected. A curve is a sequence of edges in the similarity graph that only connects valence-2 nodes (i.e., it does not contain junctions). We compute the length of the two curves that each of the diagonals is part of. The shortest possible length is 1, if neither end of the diagonal has valence of 2. This heuristic votes for keeping the longer curve of the two connected, with the weight of the vote defined as the difference between the curve lengths.
3. humans tend to perceive the sparser color as foreground and the other color as background. In this case we perceive the foreground pixels as connected (e.g., think of a dotted pencil line). We turn this into a heuristic by measuring the size of the component connected to the diagonals. We only consider an 8x8 window centered around the diagonals in question. This heuristic votes for connecting the pixels with the smaller connected component. The weight is the difference between the sizes of the components.
4. we attempt to avoid fragmentation of the figure into too many small components. Therefore, we avoid creating small disconnected islands. If one of the two diagonals has a valence-1 node, it means cutting this connection would create a single disconnected pixel. We prefer this not to happen, and therefore vote for keeping this connection with a fixed weight, with an empirically determined value of 5.
    """,
]

graph = src.FeatureGraph(nb_juniors=1)

for query in queries[:2]:
    print("query: ", query)
    result = graph.invoke(query)
