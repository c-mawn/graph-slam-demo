import pandas as pd
import pose_table as pt
from visualizations import display_graph


def main():
    # file for testing the graphing
    paths = ["data/sensor_logs.csv", "data/groundtruth_logs.csv"]
    sim = pd.read_csv(paths[0])
    gt = pd.read_csv(paths[1])
    guh = pt.SimData(sim, gt)
    pose_table = guh.generate_pose_table()
    graph = guh.generate_graph()
    display_graph(graph, pose_table)


if __name__ == "__main__":
    main()
