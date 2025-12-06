from graph_slam_demo.pose_table import PoseTable
import pandas as pd
import pickle
import re

SENSOR_DATA_PATH = "data/sensor_logs.pkl"
GROUND_TRUTH_DATA_PATH = "data/groundtruth_logs.pkl"

TOLERANCE = 1e-6

def testCalculatePose():
    """
    Test to make sure the math used to calculate poses is consistent
    """
    # Load sensor data
    ground_truth_data: pd.DataFrame = pickle.load(open(GROUND_TRUTH_DATA_PATH, "rb"))

    # Load the ground truth data to calculate the poses
    lin = ground_truth_data['Actual_LinearVelocity']
    ang = ground_truth_data['Actual_AngularVelocity']
    dt = ground_truth_data['Time']

    # calculate the poses
    poses = PoseTable.calculate_poses(PoseTable(pd.DataFrame(), pd.DataFrame()), lin, ang, dt)

    # Get the ground truth poses directly from the ground truth data
    ground_truth_poses = ground_truth_data["RobotPose"]

    # For each pair of poses, make sure they are the same
    for (calc, gt) in zip(poses, ground_truth_poses):
        # Parse the poses
        match = re.match(r"X([-.\d]*)Y([-.\d]*)T([-.\d]*)", gt)
        assert match is not None
        gt_x, gt_y, gt_t = [float(group) for group in match.groups()]

        # Make sure the poses are the same (within tolerance to account for different float precisions)
        assert abs(calc.pos.x - gt_x) <= TOLERANCE
        assert abs(calc.pos.y - gt_y) <= TOLERANCE
        assert abs(calc.theta - gt_t) <= TOLERANCE
