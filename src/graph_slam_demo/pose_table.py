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

        with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
            print(self.sensor_data)
            print(self.ground_truth_data)

        lin = self.ground_truth_data['Actual_LinearVelocity']
        ang = self.ground_truth_data['Actual_AngularVelocity'].to_list()
        dt = self.ground_truth_data['Time'].diff().to_list()

        print(lin, ang, dt)
        
        poses = self.calculate_poses(lin[1:], ang[1:], dt[1:])

        print(poses)

        return pd.DataFrame()


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

        print(len(linear_velocity), len(angular_velocity), len(delta_time))

        poses = [prior]
        
        for (lin_vel, ang_vel, dt) in zip(linear_velocity[1:], angular_velocity[1:], delta_time[1:]):
            next_pose = deepcopy(poses[-1])

            print(lin_vel, ang_vel, dt, next_pose)

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