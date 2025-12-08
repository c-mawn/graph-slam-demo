from typing import TypedDict, Optional, Tuple
import regex as re
import numpy as np
from math import pi

SENSOR_DATA_PATH = "data/sensor_logs.pkl"
GROUND_TRUTH_DATA_PATH = "data/groundtruth_logs.pkl"

# exmaple sensor table for testing
example_sensor_table = [
    [0.0, [0.0, 0.0, 0.0], [np.sqrt(0.5), pi / 4], [0.0, 0.0]],
    [0.5, [0.5, 0.0, 0.0], [None, None], [1.0, 0.0]],
    [1.5, [1.0, 0.0, pi / 2], [np.sqrt(0.5), pi / 4], [1.0, 1.0]],
    [2.0, [1.0, 0.5, pi / 2], [None, None], [1.0, 0.0]],
    [2.5, [1.0, 1.0, pi / 2], [np.sqrt(0.5), 3 * pi / 4], [0.0, 0.0]],
]


class sensorEdge(TypedDict):
    """
    Creates a generalized dictionary for all edges in the pose graph

    Contains values taken from the sensor readings, stores them in the dict, includes:
    - time(float) - time at sensor reading
    - type(string) - type of sensor reading (odom, beacon)
    - b(float) - bearing difference between robot and beacon
    - r(float) - range difference between robot and beacon
    - lv(float) - linear velocity of the robot
    - av(float) - angular velocity of the robot
    """

    time: float
    type: str
    b: Optional[float]
    r: Optional[float]
    lv: Optional[float]
    av: Optional[float]


class poseStamped:
    """ """

    def __init__(self, position: Tuple[float, float], orientation: float, time: float):
        """
        Creates an instance of a poseStamped object

        Args:
            position(Tuple[float, float]): x and y coordinates of the object
            orientation(float): orientation of the object in radians
            time(float): time in seconds of when the object was in this pose
        """
        self.position = position
        self.orientation = orientation
        self.time = time

    def __eq__(self, other):
        """
        ensures equivalent instances of the poseStamped object are
        representable by the same hash
        """
        return (
            isinstance(other, poseStamped)
            and self.position == other.position
            and self.orientation == other.orientation
            and self.time == other.time
        )

    def __hash__(self):
        """
        returns a hash value for the poseStamped object

        used in order to ensure that the object can be used as a node in the
        networkx graph
        """
        return hash((self.position, self.orientation, self.time))


def pose_from_odom(
    current_pose: poseStamped, dt: float, lin_vel: float, ang_vel: float
) -> poseStamped:
    """
    computes a pose based on current pose and odom data

    args:
        current_pose(poseStamped): current pose (before update)
        dt(float): timestep
        lin_vel(float): commanded linear velocity
        ang_vel(float): commanded angular velocity

    returns:
        updated_pose(poseStamped): updated pose
    """
    c_x, c_y = current_pose.position
    c_t = current_pose.orientation

    theta = (c_t + ang_vel * dt) / 2

    updated_pose = poseStamped(
        (c_x + (lin_vel * np.cos(theta) * dt), c_y + (lin_vel * np.sin(theta) * dt)),
        c_t + ang_vel * dt,
        current_pose.time + dt,
    )

    return updated_pose


def unpack_bearing_range(bearing_range: str) -> tuple[float | None, float | None]:
    """
    Takes the string inside the sensor log representing the BearingRange object
    and returns the bearing and range floats store inside it

    ex str input: BearingRange(bearing=0.9087192601314104, range=6.658699423125392)
        ret: (0.9087192601314104, 6.658699423125392)

    args:
        bearing_range(string): bearing range string from sensor log

    returns:
        bearing(float): angle between robot and beacon
        range(float): distance between robot and beacon
    """
    if type(bearing_range) == float:
        return None, None
    bearing_pattern = r"bearing=([^,)]+)"
    range_pattern = r"range=([^,)]+)"

    b_match = re.search(bearing_pattern, bearing_range)
    r_match = re.search(range_pattern, bearing_range)

    bearing_val = convert_to_float(b_match.group(1)) if b_match else None
    range_val = convert_to_float(r_match.group(1)) if r_match else None

    return bearing_val, range_val


def convert_to_float(val_str):
    if not val_str:
        return None
    # check specifically for "inf"
    if "inf" in val_str:
        return None
    try:
        return float(val_str)
    except ValueError:
        return None
