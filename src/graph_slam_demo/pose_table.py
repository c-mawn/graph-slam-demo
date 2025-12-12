"""
Class for generating a table of robot poses from different sources, including ground truth data,
motor commands, and sensor readings
"""

from copy import deepcopy
from math import sin, cos, pi
from mobile_robot_sim.utils import Pose, Position, NEAR_ZERO
import pandas as pd
import re
from typing import Sequence
from utils import unpack_bearing_range, poseStamped, sensorEdge
import networkx as nx
import numpy as np


class SimData:
    """
    An object to configure raw simulator data into useful structures.

    Attributes:
        sensor_data (DataFrame): the table of sensor data from running the robot
        ground_truth_data (DataFrame): the table of ground truth data from running the robot
    """

    def __init__(
        self,
        sensor_data: pd.DataFrame,
        ground_truth_data: pd.DataFrame,
    ) -> None:
        self.sensor_data = sensor_data
        self.ground_truth_data = ground_truth_data
        self.pose_table = self.generate_pose_table()

    def generate_graph(self) -> nx.Graph:
        """
        Generates a networkx graph where nodes are robot poses and edges are constraints.
        """
        graph = nx.Graph()
        prev_node = None

        table_col = self.sensor_data.columns
        beacon_count = 0
        for col in table_col:
            beacon_count += (
                1 if re.search(re.escape("LandmarkPinger_Landmark"), col) else 0
            )

        for index, row in self.pose_table.iterrows():
            # goes thru each row in table
            timestamp = row["Time"]
            # getting pose estimate
            if index == 1 or index == 0:
                x, y, theta = (0.0, 0.0, 0.0)
            else:
                updated_pose = row["OdomPoses"]
                x = updated_pose.pos.x
                y = updated_pose.pos.y
                theta = updated_pose.theta

            # creates a new node
            current_node = poseStamped((x, y), theta, timestamp)
            graph.add_node(current_node, pos=(x, y), node_type="odom")

            # connecting pose nodes
            if prev_node is not None:
                # Extract odom data if available
                lin_v = self.sensor_data["Odometry_LinearVelocity"][index]
                ang_v = self.sensor_data["Odometry_AngularVelocity"][index]

                # create the edge dict
                odom_edge = sensorEdge(
                    time=timestamp, type="odom", b=None, r=None, lv=lin_v, av=ang_v
                )

                # connect previous node to current node
                graph.add_edge(prev_node, current_node, **odom_edge)

            # Check if beacon data exists and is valid

            for i in range(beacon_count):
                b, r = unpack_bearing_range(
                    self.sensor_data[f"LandmarkPinger_Landmark{i}"][index]
                )
                if b is not None and r is not None:
                    # calculate absolute position of beacon
                    x_b = x + r * np.cos(theta + b)
                    y_b = y + r * np.sin(theta + b)

                    beacon_node = poseStamped((x_b, y_b), 0.0, timestamp)

                    # Add beacon node
                    graph.add_node(beacon_node, pos=(x_b, y_b), node_type="beacon")

                    # Create edge between current pose and beacon
                    beacon_edge_attr = sensorEdge(
                        time=timestamp, type="beacon", r=r, b=b, lv=None, av=None
                    )

                    graph.add_edge(current_node, beacon_node, **beacon_edge_attr)

            # Update previous node for the next iteration
            prev_node = current_node

        return graph

    def generate_pose_table(self) -> pd.DataFrame:
        """
        Generate the pose table

        Returns:
            DataFrame: The pose table, with coumns "Time", "CommandPoses", "OdomPoses", "GroundTruthPoses", and "CalcGroundTruthPoses".
        """

        dataframe_dict: dict[str, list | pd.Series] = {}

        if self.sensor_data is not None:
            # Calculate the poses for the command data
            dataframe_dict["CommandPoses"] = self.calculate_poses(
                self.sensor_data["CMD_LinearVelocity"],
                self.sensor_data["CMD_AngularVelocity"],
                self.sensor_data["Time"],
            )

            # Calculate the poses for the odom data
            dataframe_dict["OdomPoses"] = self.calculate_poses(
                self.sensor_data["Odometry_LinearVelocity"],
                self.sensor_data["Odometry_AngularVelocity"],
                self.sensor_data["Time"],
            )

            # Set the Time for the dict
            dataframe_dict["Time"] = self.sensor_data["Time"]

        if self.ground_truth_data is not None:
            ground_truth_poses = self.ground_truth_data["RobotPose"]

            poses: list[Pose] = []

            for pose in ground_truth_poses:
                # Parse the poses
                match = re.match(r"X([-.\d]*)Y([-.\d]*)T([-.\d]*)", pose)
                if match is None:
                    print("regex failed")
                    poses.append(Pose())
                else:
                    gt_x, gt_y, gt_t = [float(group) for group in match.groups()]
                    poses.append(Pose(Position(gt_x, gt_y), gt_t))

            dataframe_dict["GroundTruthPoses"] = poses

            # Calculate the poses for the ground truth data
            dataframe_dict["CalcGroundTruthPoses"] = self.calculate_poses(
                self.ground_truth_data["Actual_LinearVelocity"],
                self.ground_truth_data["Actual_AngularVelocity"],
                self.ground_truth_data["Time"],
            )

            # Set the Time for the dict, overriding the sensor one if it's there
            dataframe_dict["Time"] = self.ground_truth_data["Time"]

        return pd.DataFrame(dataframe_dict)

    def calculate_poses(
        self,
        linear_velocity: pd.Series | Sequence[float],
        angular_velocity: pd.Series | Sequence[float],
        time: pd.Series | Sequence[float],
        prior: Pose = Pose(),
    ) -> list[Pose]:
        """
        Calculate robot poses for a specific sequence of linear and angular velocities

        Arguments:
            linear_velocity (pd.Series | Sequence[float]): The linear velocity of the robot over
            time. The first value will be ignored, as the start pose is dictated by the prior
            angular_velocity (pd.Series | Sequence[float]): The angular velocity of the robot over
            time. The first value will be ignored, as the start pose is dictated by the prior
            time (pd.Series | Sequence[float]): The time for each linear and angular velocity value
            prior (Pose): The starting pose of the robot. Defaults to `Pose(Position(0, 0), 0)`

        Returns:
            list[Pose]: The pose of the robot for each timestep
        """

        # Enforce that velocities should be lists
        if isinstance(linear_velocity, pd.Series):
            linear_velocity = linear_velocity.to_list()

        if isinstance(angular_velocity, pd.Series):
            angular_velocity = angular_velocity.to_list()

        # Enforce that time should be a sequence (for now)
        if isinstance(time, Sequence):
            time = pd.Series(time)

        # Calculate the diff of time so we get the length of each time step
        delta_time_series = time.diff()
        delta_time = delta_time_series.to_list()

        # The robot starts at the prior Pose
        poses = [prior]

        # For each time step, ignoring the first. We ignore the first because the first pose is the prior
        # Pose, and the first element of delta_time is null since we took the diff.
        for lin_vel, ang_vel, dt in zip(
            linear_velocity[1:], angular_velocity[1:], delta_time[1:]
        ):
            next_pose = deepcopy(poses[-1])

            # Calculate the change in position and angle
            # Only linear; drive in a line
            if abs(ang_vel) < NEAR_ZERO:
                dx = lin_vel * dt * cos(next_pose.theta)
                dy = lin_vel * dt * sin(next_pose.theta)
                dtheta = 0.0
            # linear and angular; drive in an arc
            else:
                r = lin_vel / ang_vel
                dtheta = ang_vel * dt
                dx = r * (sin(next_pose.theta + dtheta) - sin(next_pose.theta))
                dy = -r * (cos(next_pose.theta + dtheta) - cos(next_pose.theta))

            # Move the robot
            next_pose.pos.x += dx
            next_pose.pos.y += dy

            # New theta, free rotation is always allowed
            next_pose.theta = next_pose.theta + dtheta
            next_pose.theta = (next_pose.theta + pi) % (2 * pi) - pi

            poses.append(next_pose)

        return poses
