import numpy as np
import pyransac3d as pyrsc
from geometry_msgs.msg import Point
import warnings
from message_filters import Subscriber

from stalk_detect.config import INLIER_THRESHOLD, MAX_LINE_RANSAC_ITERATIONS, MAX_X, MIN_X, MAX_Y, MIN_Y


# Line object, with a slope and intercept
class Line:
    def __init__(self, slope, intercept):
        if len(slope) != 3 or len(intercept) != 3:
            raise ValueError('Slope and intercept must be of length 3')
        self.slope = slope / np.linalg.norm(slope)
        self.intercept = intercept


def ransac_3d(points):
    '''
    Perform RANSAC line detection on a set of 3D points

    Parameters
        points (list[Point]): The points to perform RANSAC on

    Returns
        best_line (np.ndarray[Point]): The best line found
    '''
    # Convert np.ndarray[Point] to np.array (N, 3)
    points = np.array([[p.x, p.y, p.z] for p in points])

    # Catch the RuntimeWarning that pyransac3d throws when it fails to find a line
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        line = pyrsc.Line().fit(points, thresh=INLIER_THRESHOLD, maxIteration=MAX_LINE_RANSAC_ITERATIONS)
        if len(w) > 0 and not issubclass(w[-1].category, RuntimeWarning):
            warnings.warn(w[-1].message, w[-1].category)

    line = Line(line[0], line[1])

    # TODO: Use refinement using least squares

    return line


def find_xy_from_z(line: Line, z):
    '''
    Find the x and y coordinates on a line given a z coordinate

    Parameters
        line (Line): The line to find the x and y coordinates on
        z (float): The z coordinate of the point

    Returns
        x (float): The x coordinate of the point
        y (float): The y coordinate of the point
    '''
    t = (z - line.intercept[2]) / line.slope[2]

    return line.intercept[0] + t * line.slope[0], line.intercept[1] + t * line.slope[1]


class Stalk:
    '''
    Helper class for storing a 3D stalk

    A Stalk is always in world frame
    '''
    def __init__(self, points: 'list[Point]', cam_points: 'list[Point]', score: float, mask: np.ndarray):
        self.points = points
        self.cam_points = cam_points
        self.valid = True
        try:
            self.line = ransac_3d(points)
        except ValueError:
            self.valid = False
        self.score = score
        self.mask = mask

    def is_valid(self):
        return self.valid

    def is_within_bounds(self) -> bool:
        return max([p.x for p in self.points]) <= MAX_X and min([p.x for p in self.points]) >= MIN_X and \
            max([p.y for p in self.points]) <= MAX_Y and min([p.y for p in self.points]) >= MIN_Y


class KillableSubscriber(Subscriber):
    '''
    A message filter subscriber that can unregister to the topic
    '''
    def unregister(self):
        self.sub.unregister()
