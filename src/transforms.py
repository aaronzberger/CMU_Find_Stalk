import numpy as np
import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import Point, Pose
from sensor_msgs.msg import CameraInfo

from config import BASE_FRAME, CAMERA_ALIGNED_FRAME, CAMERA_FRAME, CAMERA_INFO, DEPTH_CAMERA_INFO, CAMERA_COLOR_FRAME, WORLD_FRAME


class TfBuffer():
    @classmethod
    def __init__(cls):
        cls.tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(cls.tf_buffer)

    @classmethod
    def get_tf_buffer(cls):
        return cls.tf_buffer


class Transformer():
    '''
    Helper class for storing the camera to base transformations at a specfic time
    '''
    def __init__(self, tf_buffer: tf2_ros.Buffer):
        # Get the transformation from camera to world
        trans_cam_to_world = tf_buffer.lookup_transform(WORLD_FRAME, CAMERA_ALIGNED_FRAME, rospy.Time(0))
        trans_world_to_cam = tf_buffer.lookup_transform(CAMERA_ALIGNED_FRAME, WORLD_FRAME, rospy.Time(0))

        # Convert the Transform msg to a Pose msg
        pose = Pose(position=Point(
                        x=trans_cam_to_world.transform.translation.x, y=trans_cam_to_world.transform.translation.y, z=trans_cam_to_world.transform.translation.z),
                    orientation=trans_cam_to_world.transform.rotation)

        self.E_cam_to_world = tf_conversions.toMatrix(tf_conversions.fromMsg(pose))

        pose = Pose(position=Point(
                        x=trans_world_to_cam.transform.translation.x, y=trans_world_to_cam.transform.translation.y, z=trans_world_to_cam.transform.translation.z),
                    orientation=trans_world_to_cam.transform.rotation)

        self.E_world_to_cam = tf_conversions.toMatrix(tf_conversions.fromMsg(pose))

        # Get the camera intrinsics
        camera_info = rospy.wait_for_message(CAMERA_INFO, CameraInfo)
        depth_info = rospy.wait_for_message(DEPTH_CAMERA_INFO, CameraInfo)
        self.intrinsic = np.array(camera_info.K).reshape((3, 3))
        self.depth_intrinsic = np.array(depth_info.K).reshape((3, 3))

        self.width, self.height = camera_info.width, camera_info.height

    def transform_cam_to_world(self, pt: Point) -> Point:
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
        transformed_pt = np.matmul(self.E_cam_to_world, np.array([pt.z, x, y, 1]))

        return Point(x=transformed_pt[0], y=transformed_pt[1], z=transformed_pt[2])

    def transform_world_to_cam(self, pt: Point) -> Point:
        '''
        Transform a point from the robot frame to the camera frame for this transform

        Parameters
            pt (geometry_msgs.msg.Point): The point to transform

        Returns
            geometry_msgs.msg.Point: The transformed point
        '''
        # Transform
        transformed_pt = np.matmul(self.E_world_to_cam, np.array([pt.x, pt.y, pt.z, 1]))

        return Point(x=transformed_pt[0], y=transformed_pt[1], z=transformed_pt[2])
