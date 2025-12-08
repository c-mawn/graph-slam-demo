from pose_table import PoseTable
from mobile_robot_sim.utils import Pose
import matplotlib.pyplot as plt


def trajectory_comparison(pose_table: PoseTable):
    for series_name in [
        "CommandPoses",
        "OdomPoses",
        "GroundTruthPoses",
        "CalcGroundTruthPoses",
    ]:
        poses: list[Pose] = pose_table[series_name].to_list()

        x = [pose.pos.x for pose in poses]
        y = [pose.pos.y for pose in poses]

        # plt.scatter(x, y)
        plt.plot(x, y)

    plt.legend(
        ["CommandPoses", "OdomPoses", "GroundTruthPoses", "CalcGroundTruthPoses"]
    )
    plt.axis("equal")
    plt.show()
