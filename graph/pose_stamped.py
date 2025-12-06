"""
Defines a poseStamped object for use in the pose graph
"""

from typing import Tuple


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
