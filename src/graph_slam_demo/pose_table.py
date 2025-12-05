"""
Class for generating a table of robot poses from different sources, including ground truth data, motor commands, and sensor readings
"""

from copy import deepcopy
from math import sin, cos, pi
from mobile_robot_sim.utils import Pose, Position, NEAR_ZERO
import numpy as np
import pandas as pd
from typing import Sequence



class PoseTable():
    # def __init__(self, sensor_data: pd.DataFrame | None = None, ground_truth_data: pd.DataFrame | None = None) -> None:
    def __init__(self, sensor_data: pd.DataFrame, ground_truth_data: pd.DataFrame) -> None:
        self.sensor_data = sensor_data
        self.ground_truth_data = ground_truth_data


    def generate(self) -> pd.DataFrame:

        command_poses = self.calculate_poses(
            self.sensor_data['CMD_LinearVelocity'],
            self.sensor_data['CMD_AngularVelocity'],
            self.sensor_data['Time']
        )

        odom_poses = self.calculate_poses(
            self.sensor_data['Odometry_LinearVelocity'],
            self.sensor_data['Odometry_AngularVelocity'],
            self.sensor_data['Time']
        )
        
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
        
        if isinstance(linear_velocity, pd.Series):
            linear_velocity = linear_velocity.to_list()

        if isinstance(angular_velocity, pd.Series):
            angular_velocity = angular_velocity.to_list()

        if isinstance(time, Sequence):
            time = pd.Series(time)

        delta_time_series = time.diff()

        delta_time = delta_time_series.to_list()

        poses = [prior]
        
        for (lin_vel, ang_vel, dt) in zip(linear_velocity[1:], angular_velocity[1:], delta_time[1:]):
            next_pose = deepcopy(poses[-1])

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

            next_pose.pos.x += dx
            next_pose.pos.y += dy

            # new theta, free rotation is always allowed
            next_pose.theta = next_pose.theta + dtheta
            next_pose.theta = (next_pose.theta + pi) % (2 * pi) - pi

            poses.append(next_pose)

        return poses