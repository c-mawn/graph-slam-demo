import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pose_stamped import poseStamped
from gen_edge import sensor_edge


class poseGraph:
    def __init__(
        self,
        sensor_table: list[
            float, list[float, float, float], list[float, float], list[float, float]
        ],
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

        for row in self.sensor_table:
            # [0]: time, [1]: pose [x, y, theta], [2]: beacon [dist, bearing], [3]: odom [lin_v, ang_v]
            timestep = row[0]
            x, y, theta = row[1]

            # creates a node
            current_node = poseStamped((x, y), theta, timestep)

            # We use the node object itself as the key in networkx
            self.graph.add_node(current_node, pos=(x, y))

            # 2. Add odom edge constraint
            if prev_node is not None:
                # Extract odom data if available
                lin_v, ang_v = row[3] if len(row) > 3 else (0.0, 0.0)

                # Create the edge dict
                odom_edge = sensor_edge(
                    time=timestep, type="odom", b=None, r=None, lv=lin_v, av=ang_v
                )

                # connect previous node to current node
                self.graph.add_edge(prev_node, current_node, **odom_edge)

            # Check if beacon data exists and is valid
            if row[2] and row[2][0] is not None:
                r = row[2][0]
                b = row[2][1]

                # calculate absolute position of beacon
                x_b = x + r * np.cos(theta + b)
                y_b = y + r * np.sin(theta + b)

                beacon_node = poseStamped((x_b, y_b), 0.0, timestep)

                # Add beacon node
                self.graph.add_node(beacon_node, pos=(x_b, y_b))

                # Create edge between current pose and beacon
                beacon_edge_attr = sensor_edge(
                    time=timestep, type="beacon", r=r, b=b, lv=None, av=None
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

        X, Y, U, V = [], [], [], []

        for node in self.graph.nodes():
            X.append(node.position[0])
            Y.append(node.position[1])
            # beacon nodes might define orientation as 0, which is fine
            U.append(np.cos(node.orientation))
            V.append(np.sin(node.orientation))

        # draw robot poses as arrows
        ax.quiver(X, Y, U, V, pivot="mid", color="blue", scale=20, headwidth=5)

        nx.draw_networkx_nodes(
            self.graph, pos_layout, node_size=20, node_color="blue", ax=ax
        )

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


def main():
    pg = poseGraph(exmaple_sensor_table)


if __name__ == "__main__":
    main()
