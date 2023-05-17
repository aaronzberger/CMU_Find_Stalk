import numpy as np
import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import Point

INTRINSIC = np.array([[431.6290588378906, 0.0, 421.2195739746094, 0.0],
                      [0.0, 430.9979248046875,  241.19505310058594, 0.0],
                      [0.0,       0.0,        1.0, 0.0]])


class Transformer():
    def __init__(self):
        tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tf_buffer)

        trans = tf_buffer.lookup_transform('cam_link', 'base_link', rospy.Time(0), rospy.Duration(1))
        self.E = tf_conversions.toMatrix(tf_conversions.fromMsg(trans.transform))

    def transform_stalk(self, stalk: Point) -> Point:
        '''
        Transform the stalk from the camera frame to the robot frame.

        Parameters
            stalk (geometry_msgs.msg.Point): The stalk to transform

        Returns
            transformed_stalk (geometry_msgs.msg.Point): The transformed stalk
        '''
        # Normalize the stalk
        x = (stalk.x - INTRINSIC[0, 2]) / INTRINSIC[0, 0]
        y = (stalk.y - INTRINSIC[1, 2]) / INTRINSIC[1, 1]

        # Scale with depth
        x *= stalk.z
        y *= stalk.z

        # Transform
        transformed_stalk = np.matmul(self.E, np.array([x, y, stalk.z, 1]))

        return Point(x=transformed_stalk[0], y=transformed_stalk[1], z=transformed_stalk[2])
