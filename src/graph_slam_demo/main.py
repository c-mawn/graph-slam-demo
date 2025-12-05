from pose_table import PoseTable
import pandas as pd
import pickle

SENSOR_DATA_PATH = "data/sensor_logs.pkl"
GROUND_TRUTH_DATA_PATH = "data/groundtruth_logs.pkl"

if __name__ == "__main__":
    sensor_data: pd.DataFrame = pickle.load(open(SENSOR_DATA_PATH, "rb"))
    ground_truth_data: pd.DataFrame = pickle.load(open(GROUND_TRUTH_DATA_PATH, "rb"))

    pose_table = PoseTable(sensor_data, ground_truth_data)

    print(pose_table.generate())