"""
Class for generating a table of robot poses from different sources, including ground truth data, 
motor commands, and sensor readings
"""

from copy import deepcopy
from math import sin, cos, pi
from mobile_robot_sim.utils import Pose, NEAR_ZERO
import pandas as pd
from typing import Sequence


class PoseTable():
    """
    An object to generate robot poses from sensor and groud truth data. 

    Attributes:
        sensor_data (DataFrame): the table of sensor data from running the robot
        ground_truth_data (DataFrame): the table of ground truth data from running the robot
    """
    def __init__(self, sensor_data: pd.DataFrame, ground_truth_data: pd.DataFrame) -> None:
        self.sensor_data = sensor_data
        self.ground_truth_data = ground_truth_data


    def generate(self) -> pd.DataFrame:
        """
        Generate the pose table
        
        Returns:
            DataFrame: The pose table, with coumns "Time", "CommandPoses", "OdomPoses", and "GroundTruthPoses".
        """

        # Calculate the poses for the command data
        command_poses = self.calculate_poses(
            self.sensor_data['CMD_LinearVelocity'],
            self.sensor_data['CMD_AngularVelocity'],
            self.sensor_data['Time']
        )

        # Calculate the poses for the odom data
        odom_poses = self.calculate_poses(
            self.sensor_data['Odometry_LinearVelocity'],
            self.sensor_data['Odometry_AngularVelocity'],
            self.sensor_data['Time']
        )
        
        # Calculate the poses for the ground truth data
        ground_truth_poses = self.calculate_poses(
            self.ground_truth_data['Actual_LinearVelocity'],
            self.ground_truth_data['Actual_AngularVelocity'],
            self.ground_truth_data['Time']
        )

        return pd.DataFrame({
            'Time': self.ground_truth_data['Time'],
            'CommandPoses': command_poses,
            'OdomPoses': odom_poses,
            'GroundTruthPoses': ground_truth_poses,
        })


    def calculate_poses(self, 
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
        for (lin_vel, ang_vel, dt) in zip(linear_velocity[1:], angular_velocity[1:], delta_time[1:]):
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
                dx = r * (
                    sin(next_pose.theta + dtheta)
                    - sin(next_pose.theta)
                )
                dy = -r * (
                    cos(next_pose.theta + dtheta)
                    - cos(next_pose.theta)
                )

            # Move the robot
            next_pose.pos.x += dx
            next_pose.pos.y += dy

            # New theta, free rotation is always allowed
            next_pose.theta = next_pose.theta + dtheta
            next_pose.theta = (next_pose.theta + pi) % (2 * pi) - pi

            poses.append(next_pose)

        return poses
