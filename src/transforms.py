import numpy as np
import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo


class Transformer():
    '''
    Helper class for transformations at a specific time
    '''
    def __init__(self):
        # Get the transformation from cam to robot
        tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tf_buffer)

        trans = tf_buffer.lookup_transform('cam_link', 'base_link', rospy.Time(0), rospy.Duration(1))
        self.E = tf_conversions.toMatrix(tf_conversions.fromMsg(trans.transform))

        camera_info = rospy.wait_for_message('device_0/sensor_0/Color_0/info/camera_info', CameraInfo)
        depth_info = rospy.wait_for_message('device_0/sensor_0/Depth_0/info/camera_info', CameraInfo)
        self.intrinsic = np.array(camera_info.K).reshape((3, 3))
        self.depth_intrinsic = np.array(depth_info.K).reshape((3, 3))

        self.width, self.height = camera_info.width, camera_info.height

    def transform_stalk(self, stalk: Point) -> Point:
        '''
        Transform the stalk from the camera frame to the robot frame.

        Parameters
            stalk (geometry_msgs.msg.Point): The stalk to transform

        Returns
            transformed_stalk (geometry_msgs.msg.Point): The transformed stalk
        '''
        # Normalize the stalk
        x = (stalk.x - self.intrinsic[0, 2]) / self.intrinsic[0, 0]
        y = (stalk.y - self.intrinsic[1, 2]) / self.intrinsic[1, 1]

        # Scale with depth
        x *= stalk.z
        y *= stalk.z

        # Transform
        transformed_stalk = np.matmul(self.E, np.array([x, y, stalk.z, 1]))

        return Point(x=transformed_stalk[0], y=transformed_stalk[1], z=transformed_stalk[2])
