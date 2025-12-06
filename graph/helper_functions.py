import numpy as np
import re
from pose_stamped import poseStamped


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


def unpack_bearing_range(bearing_range: str) -> tuple[float, float]:
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
