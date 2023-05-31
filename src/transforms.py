import numpy as np
import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import Point, Pose, Pose2D
from sensor_msgs.msg import CameraInfo

from config import BASE_FRAME, CAMERA_FRAME, CAMERA_INFO, DEPTH_CAMERA_INFO


class Transformer():
    '''
    Helper class for storing the camera to base transformations at a specfic time
    '''
    def __init__(self):
        # Get the transformation from camera to base
        tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tf_buffer)

        trans = tf_buffer.lookup_transform(CAMERA_FRAME, BASE_FRAME, rospy.Time(0), rospy.Duration(1))

        # Convert the Transform msg to a Pose msg
        pose = Pose(position=Point(
                        x=trans.transform.translation.x, y=trans.transform.translation.y, z=trans.transform.translation.z),
                    orientation=trans.transform.rotation)

        self.E = tf_conversions.toMatrix(tf_conversions.fromMsg(pose))

        # Get the camera intrinsics
        camera_info = rospy.wait_for_message(CAMERA_INFO, CameraInfo)
        depth_info = rospy.wait_for_message(DEPTH_CAMERA_INFO, CameraInfo)
        self.intrinsic = np.array(camera_info.K).reshape((3, 3))
        self.depth_intrinsic = np.array(depth_info.K).reshape((3, 3))

        self.width, self.height = camera_info.width, camera_info.height

    def transform_point(self, pt: Point) -> Point:
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

        # Transform
        transformed_pt = np.matmul(self.E, np.array([x, y, pt.z, 1]))

        return Point(x=transformed_pt[0], y=transformed_pt[1], z=transformed_pt[2])
