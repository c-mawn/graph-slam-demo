import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from math import pi
from utils import poseStamped
from utils import sensorEdge
from utils import pose_from_odom, unpack_bearing_range


class bayesNet:
    def __init__(
        self,
        sensor_table: pd.DataFrame,
    ):
        self.sensor_table = sensor_table
        self.graph = nx.Graph()

        self.generate_graph()
        self.display_graph()

    def generate_graph(self):
        """
        Generates a networkx graph where nodes are robot poses and edges are constraints.
        """
        all_odoms = []
        all_beacons = []

        # creates a node for each pose
        for index, row in self.sensor_table.iterrows():
            all_odoms.append(f"x{index}")

        self.graph.add_nodes_from(all_odoms, node_type="odom")

        # creates a node for each beacon
        table_col = self.sensor_table.columns
        beacon_count = 0
        for col in table_col:
            beacon_count += (
                1 if re.search(re.escape("LandmarkPinger_Landmark"), col) else 0
            )

        for i in range(beacon_count):
            all_beacons.append(f"l{i}")
        self.graph.add_nodes_from(all_beacons, node_type="beacon")

        # creates edges between each pose
        for i in range(len(all_odoms)):
            # Replace 'row' with direct DataFrame access using [i]
            lin_v = self.sensor_table["Odometry_LinearVelocity"][i]
            ang_v = self.sensor_table["Odometry_AngularVelocity"][i]

            odom_edge = sensorEdge(
                time=0, type="odom", b=None, r=None, lv=lin_v, av=ang_v
            )
            if i != 0:
                self.graph.add_edge(f"x{i-1}", f"x{i}", **odom_edge)

        # creates edges between odoms and beacons
        for num, beacon in enumerate(all_beacons):
            for i, odom_node in enumerate(all_odoms):
                b, r = unpack_bearing_range(
                    self.sensor_table[f"LandmarkPinger_Landmark{num}"][i]
                )
                if b is not None and r is not None:
                    # this means that this pose saw beacon{num}
                    beacon_edge_attr = sensorEdge(
                        time=0, type="beacon", r=r, b=b, lv=None, av=None
                    )
                    self.graph.add_edge(odom_node, beacon, **beacon_edge_attr)

    def display_graph(self):
        """
        Function to display the created graph.
        Distinguishes between robot poses (odom) and landmarks (beacon).
        """
        plt.figure(figsize=(12, 8))

        pos = nx.spring_layout(self.graph, seed=69)

        odom_nodes = [
            n
            for n, attr in self.graph.nodes(data=True)
            if attr.get("node_type") == "odom"
        ]
        beacon_nodes = [
            n
            for n, attr in self.graph.nodes(data=True)
            if attr.get("node_type") == "beacon"
        ]

        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=odom_nodes,
            node_color="skyblue",
            node_size=5,
            label="Robot Poses",
        )

        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=beacon_nodes,
            node_color="orange",
            node_shape="s",
            node_size=70,
            label="Landmarks",
        )

        odom_edges = [
            (u, v)
            for u, v, attr in self.graph.edges(data=True)
            if attr.get("type") == "odom"
        ]
        beacon_edges = [
            (u, v)
            for u, v, attr in self.graph.edges(data=True)
            if attr.get("type") == "beacon"
        ]

        nx.draw_networkx_edges(
            self.graph, pos, edgelist=odom_edges, width=0.05, edge_color="black"
        )

        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=beacon_edges,
            width=0.5,
            edge_color="gray",
            style="dashed",
            alpha=0.7,
        )

        nx.draw_networkx_labels(self.graph, pos, font_size=4, font_family="sans-serif")

        plt.title("Bayes Net Pose Graph")
        plt.legend(loc="upper right")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


df = pd.read_csv("graph/sensor_log.csv")


def main():
    pg = bayesNet(df)


if __name__ == "__main__":
    main()
