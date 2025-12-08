from pose_table import PoseTable
from mobile_robot_sim.utils import Pose
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def trajectory_comparison(pose_table: PoseTable):
    for series_name in [
        "CommandPoses",
        "OdomPoses",
        "GroundTruthPoses",
        "CalcGroundTruthPoses",
    ]:
        poses: list[Pose] = pose_table[series_name].to_list()

        x = [pose.pos.x for pose in poses]
        y = [pose.pos.y for pose in poses]

        # plt.scatter(x, y)
        plt.plot(x, y)

    plt.legend(
        ["CommandPoses", "OdomPoses", "GroundTruthPoses", "CalcGroundTruthPoses"]
    )
    plt.axis("equal")
    plt.show()


def display_graph(self, graph: nx.Graph):
    """
    Function to display the created graph
    """
    # NetworkX needs a dictionary of positions {node: (x,y)} to draw
    pos_layout = nx.get_node_attributes(graph, "pos")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Using a list of colors to differentiate odom vs beacon edges
    edges = self.graph.edges(data=True)
    odom_edges = [(u, v) for u, v, d in edges if d.get("type") == "odom"]
    beacon_edges = [(u, v) for u, v, d in edges if d.get("type") == "beacon"]

    nx.draw_networkx_edges(
        graph,
        pos_layout,
        edgelist=odom_edges,
        edge_color="black",
        ax=ax,
        label="Odom",
    )
    nx.draw_networkx_edges(
        graph,
        pos_layout,
        edgelist=beacon_edges,
        edge_color="red",
        style="dashed",
        ax=ax,
        label="Beacon",
    )

    all_nodes = graph.nodes(data=True)
    odom_nodes = [n for n, d in all_nodes if d.get("node_type") == "odom"]
    beacon_nodes = [n for n, d in all_nodes if d.get("node_type") == "beacon"]

    nx.draw_networkx_nodes(
        graph,
        pos_layout,
        nodelist=odom_nodes,
        node_size=5,
        node_color="blue",
        ax=ax,
        label="Robot Pose",
    )

    nx.draw_networkx_nodes(
        graph,
        pos_layout,
        nodelist=beacon_nodes,
        node_size=5,
        node_color="green",
        node_shape="s",
        ax=ax,
        label="Landmark",
    )

    X, Y, U, V = [], [], [], []

    for node in odom_nodes:
        X.append(node.position[0])
        Y.append(node.position[1])
        U.append(np.cos(node.orientation))
        V.append(np.sin(node.orientation))

    # draw robot poses as arrows
    ax.quiver(X, Y, U, V, pivot="mid", color="blue", scale=50, headwidth=4)

    ax.set_aspect("equal")
    plt.legend()
    plt.grid(True)
    plt.title("Pose Graph")
    plt.show()
