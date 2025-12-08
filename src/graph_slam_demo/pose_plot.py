import matplotlib.pyplot as plt
from mobile_robot_sim.utils import Pose
from pose_table import PoseTable
import pandas as pd
import pickle

# Paths to the data
SENSOR_DATA_PATH = "data/sensor_logs.pkl"
GROUND_TRUTH_DATA_PATH = "data/groundtruth_logs.pkl"

if __name__ == "__main__":
    # Load the data
    sensor_data: pd.DataFrame = pickle.load(open(SENSOR_DATA_PATH, "rb"))
    ground_truth_data: pd.DataFrame = pickle.load(open(GROUND_TRUTH_DATA_PATH, "rb"))

    # Create a PoseTable object
    pose_table = PoseTable(sensor_data, ground_truth_data).generate()
