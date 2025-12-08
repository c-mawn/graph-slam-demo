"""
Main file for running the simulator.
"""

from mobile_robot_sim.environment import Environment
from mobile_robot_sim.robot import Robot
from mobile_robot_sim.utils import Pose, Position, Bounds, Landmark
import pandas as pd
import pickle

COMMANDS: list[tuple[float, float, float]] = [
    # timestamp, linear vel, angular vel
    ( 0.0,       0.0,        0.0         ),
    ( 1.0,       1.0,        0.0         ),
    ( 10.0,      1.0,        0.1         ),
    ( 20.0,      1.0,        0.0         ),
    ( 35.0,      1.0,        0.1         ),
    ( 75.0,      1.0,        0.0         ),
    ( 85.0,      1.0,        0.2         ),
    ( 95.0,      1.0,        0.0         ),
]

TOTAL_SECONDS = 105

if __name__ == "__main__":
    # set up the sim
    env = Environment(
        dimensions=Bounds(0, 30, 0, 30),
        agent_pose=Pose(Position(0, 0), 0),
        obstacles=[],
        landmarks=[
            Landmark(Position(5, 5), 0),
            Landmark(Position(20, 5), 1),
            Landmark(Position(10, 15), 2),
            Landmark(Position(25, 20), 3),
            Landmark(Position(15, 25), 4),
        ],
        timestep=1.0,
    )
    robot = Robot(env)

    # set up timekeeping
    total_timesteps = TOTAL_SECONDS / env.DT
    terminal = False

    # set up logging
    ground_truth_history = pd.DataFrame()
    sensor_data_history = pd.DataFrame()

    # start popping velocity commands
    command_generator = (command for command in COMMANDS)
    next_cmd = next(command_generator)
    current_lin_vel = float(next_cmd[1])
    current_ang_vel = float(next_cmd[2])

    # hit it!
    for step in range(int(total_timesteps) + 1):
        # first, sample the environment
        print()
        print(f"***TEST AT T{env.time}***")
        print()
        gt_meas = robot.take_gt_snapshot()
        ground_truth_history = pd.concat(
            [ground_truth_history, gt_meas],
            ignore_index=True,
        )
        print("--> Ground Truth Data")
        print(gt_meas.columns)
        print(gt_meas.values)
        print()
        sensor_meas = robot.take_sensor_measurements()
        sensor_data_history = pd.concat(
            [sensor_data_history, sensor_meas],
            ignore_index=True,
        )
        print("--> Sensor Data")
        print(sensor_meas.columns)
        print(sensor_meas.values)

        # # retrieve new command if available or passed
        if round(float(next_cmd[0]), 3) <= env.DT * step and not terminal:
            current_lin_vel = float(next_cmd[1])
            current_ang_vel = float(next_cmd[2])
            # out of commands
            try:
                next_cmd = next(command_generator)
            except StopIteration:
                terminal = True

        # move the robot with current commands
        robot.agent_step_differential(current_lin_vel, current_ang_vel)

    # log the results
    ground_truth_history.to_csv("./data/groundtruth_logs.csv")
    pickle.dump(
        ground_truth_history, 
        open("./data/groundtruth_logs.pkl", "wb")
    ) 

    sensor_data_history.to_csv("./data/sensor_logs.csv")  
    pickle.dump(
        sensor_data_history, 
        open("./data/sensor_logs.pkl", "wb")
    )       
