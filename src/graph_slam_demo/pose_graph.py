import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from math import pi
from utils import poseStamped
from utils import sensorEdge
from utils import pose_from_odom, unpack_bearing_range


class poseGraph:
    def __init__(
        self,
        sensor_table: (
            list[
                float, list[float, float, float], list[float, float], list[float, float]
            ]
            | pd.DataFrame
        ),
    ):
        self.sensor_table = sensor_table
        self.graph = nx.Graph()

        self.generate_graph()
        self.display_graph()

    def generate_graph(self):
        """
        Generates a networkx graph where nodes are robot poses and edges are constraints.
        """

        prev_node = None

        table_col = self.sensor_table.columns
        beacon_count = 0
        for col in table_col:
            beacon_count += (
                1 if re.search(re.escape("LandmarkPinger_Landmark"), col) else 0
            )

        for index, row in self.sensor_table.iterrows():
            # goes thru each row in table
            # [0] index, [1]: time, [2:5]: beacons, [6]: gps, [7,8]: odom, [9,10], cmd vel
            timestamp = row["Time"]
            # getting pose estimate
            if index == 1 or index == 0:
                x, y, theta = (0.0, 0.0, 0.0)
            else:
                # TODO: non constant dt (how to generate it?)
                updated_pose = pose_from_odom(
                    poseStamped((x, y), theta, timestamp),
                    0.1,
                    row["Odometry_LinearVelocity"],
                    row["Odometry_AngularVelocity"],
                )
                x, y = updated_pose.position
                theta = updated_pose.orientation

            # creates a new node
            current_node = poseStamped((x, y), theta, timestamp)
            self.graph.add_node(current_node, pos=(x, y), node_type="odom")

            # connecting pose nodes
            if prev_node is not None:
                # Extract odom data if available
                lin_v = row["Odometry_LinearVelocity"]
                ang_v = row["Odometry_AngularVelocity"]

                # create the edge dict
                odom_edge = sensor_edge(
                    time=timestamp, type="odom", b=None, r=None, lv=lin_v, av=ang_v
                )

                # connect previous node to current node
                self.graph.add_edge(prev_node, current_node, **odom_edge)

            # Check if beacon data exists and is valid

            for i in range(beacon_count):
                b, r = unpack_bearing_range(row[f"LandmarkPinger_Landmark{i}"])
                if b is not None and r is not None:
                    # calculate absolute position of beacon
                    x_b = x + r * np.cos(theta + b)
                    y_b = y + r * np.sin(theta + b)

                    beacon_node = poseStamped((x_b, y_b), 0.0, timestamp)

                    # Add beacon node
                    self.graph.add_node(beacon_node, pos=(x_b, y_b), node_type="beacon")

                    # Create edge between current pose and beacon
                    beacon_edge_attr = sensor_edge(
                        time=timestamp, type="beacon", r=r, b=b, lv=None, av=None
                    )

                    self.graph.add_edge(current_node, beacon_node, **beacon_edge_attr)

            # Update previous node for the next iteration
            prev_node = current_node

    def display_graph(self):
        """
        Function to display the created graph
        """
        # NetworkX needs a dictionary of positions {node: (x,y)} to draw
        pos_layout = nx.get_node_attributes(self.graph, "pos")

        fig, ax = plt.subplots(figsize=(8, 8))

        # Using a list of colors to differentiate odom vs beacon edges
        edges = self.graph.edges(data=True)
        odom_edges = [(u, v) for u, v, d in edges if d.get("type") == "odom"]
        beacon_edges = [(u, v) for u, v, d in edges if d.get("type") == "beacon"]

        nx.draw_networkx_edges(
            self.graph,
            pos_layout,
            edgelist=odom_edges,
            edge_color="black",
            ax=ax,
            label="Odom",
        )
        nx.draw_networkx_edges(
            self.graph,
            pos_layout,
            edgelist=beacon_edges,
            edge_color="red",
            style="dashed",
            ax=ax,
            label="Beacon",
        )

        all_nodes = self.graph.nodes(data=True)
        odom_nodes = [n for n, d in all_nodes if d.get("node_type") == "odom"]
        beacon_nodes = [n for n, d in all_nodes if d.get("node_type") == "beacon"]

        nx.draw_networkx_nodes(
            self.graph,
            pos_layout,
            nodelist=odom_nodes,
            node_size=5,
            node_color="blue",
            ax=ax,
            label="Robot Pose",
        )

        nx.draw_networkx_nodes(
            self.graph,
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


# exmaple sensor table for testing
exmaple_sensor_table = [
    [0.0, [0.0, 0.0, 0.0], [np.sqrt(0.5), pi / 4], [0.0, 0.0]],
    [0.5, [0.5, 0.0, 0.0], [None, None], [1.0, 0.0]],
    [1.5, [1.0, 0.0, pi / 2], [np.sqrt(0.5), pi / 4], [1.0, 1.0]],
    [2.0, [1.0, 0.5, pi / 2], [None, None], [1.0, 0.0]],
    [2.5, [1.0, 1.0, pi / 2], [np.sqrt(0.5), 3 * pi / 4], [0.0, 0.0]],
]

df = pd.read_csv("graph/sensor_log.csv")


def main():
    pg = poseGraph(df)


if __name__ == "__main__":
    main()
