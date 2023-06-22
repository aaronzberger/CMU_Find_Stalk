import numpy as np
import pyransac3d as pyrsc
from geometry_msgs.msg import Point
import warnings
from message_filters import Subscriber

from stalk_detect.config import INLIER_THRESHOLD, MAX_LINE_RANSAC_ITERATIONS, MAX_X, MIN_X, MAX_Y, MIN_Y, OPTIMAL_STALK_HEIGHT


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

    return line


def find_xy_from_z(line, z):
    '''
    Find the x and y coordinates on a line given a z coordinate

    Parameters
        line (np.ndarray[Point]): The line to find the point on
        z (float): The z coordinate of the point

    Returns
        x (float): The x coordinate of the point
        y (float): The y coordinate of the point
    '''
    normalized_direction = line[0] / np.linalg.norm(line[0])
    t = (z - line[1][2]) / normalized_direction[2]

    return line[1][0] + t * normalized_direction[0], line[1][1] + t * normalized_direction[1]


class Stalk:
    '''
    Helper class for storing a 3D stalk

    A Stalk is always in world frame
    '''
    def __init__(self, points: 'list[Point]', score: float, mask: np.ndarray):
        self.points = points
        self.line = ransac_3d(points)
        self.score = score
        self.mask = mask

        self.grasp_point = None
        self.cam_grasp_point = None

    def set_grasp_point(self, min_height=0):
        '''
        Get the point on the stalk to grasp

        Parameters
            min_height (float): The minimum height the stalk can touch
        '''
        if self.grasp_point is not None:
            return self.grasp_point

        # Retrieve the point above the lowest point
        goal_height = min_height + OPTIMAL_STALK_HEIGHT

        # Find the point on the line at this height
        x, y = find_xy_from_z(self.line, goal_height)

        self.grasp_point = Point(x=x, y=y, z=goal_height)

    def is_valid(self):
        return len(self.line[0]) > 0
    
    def is_within_bounds(self) -> bool:
        if self.grasp_point is None:
            raise ValueError('Grasp point not set')
        return self.cam_grasp_point.x <= MAX_X and self.cam_grasp_point.x >= MIN_X and self.cam_grasp_point.y <= MAX_Y and self.cam_grasp_point.y >= MIN_Y
    
    def set_cam_grasp_point(self, cam_grasp_point):
        self.cam_grasp_point = cam_grasp_point


class KillableSubscriber(Subscriber):
    '''
    A message filter subscriber that can unregister to the topic
    '''
    def unregister(self):
        self.sub.unregister()
