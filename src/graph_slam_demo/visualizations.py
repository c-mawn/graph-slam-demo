from mobile_robot_sim.utils import Pose, Position, Bounds, Landmark
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pose_table import SimData


def trajectory_comparison(pose_table: SimData):
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


def display_graph(graph: nx.Graph, pose_table: pd.DataFrame | None):
    """
    Function to display the created graph
    """
    # NetworkX needs a dictionary of positions {node: (x,y)} to draw
    pos_layout = nx.get_node_attributes(graph, "pos")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Using a list of colors to differentiate odom vs beacon edges
    edges = graph.edges(data=True)
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
        edge_color="pink",
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
        node_color="orange",
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

    landmarks = [
        Landmark(Position(5, 5), 0),
        Landmark(Position(20, 5), 1),
        Landmark(Position(10, 15), 2),
        Landmark(Position(25, 20), 3),
        Landmark(Position(15, 25), 4),
    ]

    # plot landmarks as point in diff color
    landmark_x = [i.pos.x for i in landmarks]
    landmark_y = [i.pos.y for i in landmarks]
    ax.scatter(landmark_x, landmark_y, c="red", label="True Landmark", zorder=5, s=70)
    if pose_table is not None:
        # plot line of ground truth odoms from the pose table
        gt = pose_table["GroundTruthPoses"].tolist()
        gt_x = [i.pos.x for i in gt]
        gt_y = [i.pos.y for i in gt]
        ax.plot(gt_x, gt_y, c="green", label="True Poses", zorder=2)

    ax.set_aspect("equal")
    plt.legend()
    plt.grid(True)
    plt.title("Pose Graph")
    plt.show()
