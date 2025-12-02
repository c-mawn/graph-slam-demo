"""
Contains an Abstract Base class for the pose graph classes to inherit from
"""

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pose_stamped import poseStamped


class poseGraph:
    """
    Class to take in the sensor data table and create a graphical representation
    of that data.

    Nodes on the graph represent the robot poses (positions and orientations) at each timestep
    Edges on the graph represent the sensor measurements at each timestep

    attributes:
        - sensor_table(type? pandas table?): table containing timesteps and sensor measurements
        - graph (nx.Graph): graph representing the pose graph

    """

    def __init__(
        self, sensor_table: list[float, list[float, float, float], list[float, float]]
    ):
        """
        initializes an instance of the poseGraph class

        """
        # assuming that the sensor table is a list of lists:
        # list containing lists, each inner list is one row in the table
        # each row has
        # [0]: time
        # [1]: [x, y, theta]
        # [2]: sensor stuff [dist, bearing]
        self.sensor_table = sensor_table
        self.graph = nx.Graph()

        self.generate_graph()
        self.display_graph()

    # input of sensor table
    # outputs some representation of the graph-- networkx stuff? adjacency matrix?

    def generate_graph(self):
        """
        generates a networkx graph where nodes are robot poses at each timestep
        """

        # nodes can be any hashable object
        # we need to make an object that can represent one node
        # one node needs to have position, orientation, timestamp
        # make sure that the pose stamped object is hashable

        # Take each row of the table, split into its 3(?) components
        # timestamps, odom, beacon data. First take times and odom to make
        # poseStamped objects... pass those into the graph as nodes

        for row in self.sensor_table:
            # each row is list
            # each list is
            # [0]: time, [1]: pose, [2]: sensor stuff

            # unpacking data from table
            timestep = row[0]
            x = row[1][0]
            y = row[1][1]
            theta = row[1][2]

            # creates a new poseStamped object to use as a node
            # appends to total list of nodes
            node = poseStamped((x, y), theta, timestep)
            self.graph.add_node(node, pos=(x, y))

    def display_graph(self):
        """
        function to display the created graph
        """

        pos = nx.get_node_attributes(self.graph, "pos")
        fig, ax = plt.subplots()

        nx.draw(self.graph, pos, ax=ax)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.draw()
        plt.axis("on")
        plt.show()


##################
##################
##################


# exmaple sensor table for testing
exmaple_sensor_table = [
    [0.0, [0.0, 0.0, 0.0], [2.4, pi / 2]],
    [0.5, [0.5, 0.0, 0.0], [2.4, pi / 2]],
    [1.5, [1.0, 0.0, pi / 2], [2.4, pi / 2]],
    [2.0, [1.0, 0.5, pi / 2], [2.4, pi / 2]],
    [2.5, [1.0, 1.0, pi / 2], [2.4, pi / 2]],
]


def main():
    pg = poseGraph(exmaple_sensor_table)


if __name__ == "__main__":
    main()
