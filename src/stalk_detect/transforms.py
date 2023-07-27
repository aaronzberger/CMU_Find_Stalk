import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo

from stalk_detect.config import CAMERA_INFO, DEPTH_CAMERA_INFO


class TfBuffer():
    '''
    Helper class for storing the tf buffer

    This class needs to be initialized before looking up any transforms in the TF tree below
    '''
    @classmethod
    def __init__(cls):
        cls.tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(cls.tf_buffer, queue_size=1)

    @classmethod
    def get_tf_buffer(cls):
        return cls.tf_buffer


class Transformer():
    '''
    Helper class for storing the camera to base transformations at a specfic time
    '''
    def __init__(self, tf_buffer: tf2_ros.Buffer):
        # Get the camera intrinsics
        camera_info = rospy.wait_for_message(CAMERA_INFO, CameraInfo, timeout=0.5)
        depth_info = rospy.wait_for_message(DEPTH_CAMERA_INFO, CameraInfo, timeout=0.5)

        if camera_info is None or depth_info is None:
            raise RuntimeError(f'Failed to get camera and depth camera info on topics {camera_info} and {depth_info}')

        self.intrinsic = np.array(camera_info.K).reshape((3, 3))
        self.depth_intrinsic = np.array(depth_info.K).reshape((3, 3))

        self.width, self.height = camera_info.width, camera_info.height

    def transform_instrinsic(self, pt: Point) -> Point:
        '''
        Transform a point from the camera frame to the robot frame for this transform

        Parameters
            pt (geometry_msgs.msg.Point): The point to transform

        Returns
            geometry_msgs.msg.Point: The transformed point
        '''
        # Normalize the point
        x = (pt.x - self.intrinsic[0, 2]) / self.intrinsic[0, 0]
        y = (pt.y - self.intrinsic[1, 2]) / self.intrinsic[1, 1]

        # Scale with depth
        x *= pt.z
        y *= pt.z

        return Point(x=pt.z, y=x, z=y)
