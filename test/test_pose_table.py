from graph_slam_demo.pose_table import PoseTable
import pandas as pd
import pickle
import re

SENSOR_DATA_PATH = "data/sensor_logs.pkl"
GROUND_TRUTH_DATA_PATH = "data/groundtruth_logs.pkl"

TOLERANCE = 1e-6

def testCalculatePose():
    sensor_data: pd.DataFrame = pickle.load(open(SENSOR_DATA_PATH, "rb"))
    ground_truth_data: pd.DataFrame = pickle.load(open(GROUND_TRUTH_DATA_PATH, "rb"))

    pose_table = PoseTable(sensor_data, ground_truth_data)

    lin = ground_truth_data['Actual_LinearVelocity']
    ang = ground_truth_data['Actual_AngularVelocity']
    dt = ground_truth_data['Time']

    poses = pose_table.calculate_poses(lin, ang, dt)
    ground_truth_poses = ground_truth_data["RobotPose"]

    print(len(poses), len(ground_truth_poses))

    for (calc, gt) in zip(poses, ground_truth_poses):
        match = re.match(r"X([-.\d]*)Y([-.\d]*)T([-.\d]*)", gt)

        assert match is not None

        gt_x, gt_y, gt_t = [float(group) for group in match.groups()]

        assert abs(calc.pos.x - gt_x) <= TOLERANCE
        assert abs(calc.pos.y - gt_y) <= TOLERANCE
        assert abs(calc.theta - gt_t) <= TOLERANCE
        