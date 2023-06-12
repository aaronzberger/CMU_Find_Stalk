import numpy as np
import pyransac3d as pyrsc
from geometry_msgs.msg import Point
import warnings
from message_filters import Subscriber

from config import INLIER_THRESHOLD, MAX_LINE_RANSAC_ITERATIONS


def ransac_3d(points):
    '''
    Perform RANSAC line detection on a set of 3D points

    Parameters
        points (np.ndarray[Point]): The points to perform RANSAC on

    Returns
        best_line (np.ndarray[Point]): The best line found
    '''
    # Convert np.ndarray[Point] to np.array (N, 3)
    points = np.array([[p.x, p.y, p.z] for p in points])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        line = pyrsc.Line().fit(points, thresh=INLIER_THRESHOLD, maxIteration=MAX_LINE_RANSAC_ITERATIONS)
        if len(w) > 0 and not issubclass(w[-1].category, RuntimeWarning):
            warnings.warn(w[-1].message, w[-1].category)

    # print(colored('RANSAC line: {}'.format(line), 'blue'))

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
    '''
    def __init__(self, points: 'np.ndarray[Point]'):
        self.points = points
        self.line = ransac_3d(points)

    def get_grasp_point(self, min_height=0):
        '''
        Get the point on the stalk to grasp

        Returns
            Point: The point to grasp
        '''
        # Retrieve the point 0.05m above the lowest point
        goal_height = min_height + 0.05

        # Find the point on the line at this height
        x, y = find_xy_from_z(self.line, goal_height)
        return Point(x=x, y=y, z=goal_height)


class KillableSubscriber(Subscriber):
    '''
    A message filter subscriber that can unregister to the topic
    '''
    def unregister(self):
        self.sub.unregister()
