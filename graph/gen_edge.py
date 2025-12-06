from typing import TypedDict, Optional


class sensor_edge(TypedDict):
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
