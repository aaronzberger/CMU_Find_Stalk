import numpy as np
import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import Point, Pose
from sensor_msgs.msg import CameraInfo

from stalk_detect.config import (CAMERA_INFO,
                                 DEPTH_CAMERA_INFO, WORLD_FRAME, CAMERA_COLOR_FRAME)


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
        # Get the transformation from camera to world
        cam_to_world = tf_buffer.lookup_transform(WORLD_FRAME, CAMERA_COLOR_FRAME, rospy.Time(0), rospy.Duration.from_sec(0.5)).transform
        world_to_cam = tf_buffer.lookup_transform(CAMERA_COLOR_FRAME, WORLD_FRAME, rospy.Time(0), rospy.Duration.from_sec(0.5)).transform

        # Convert the Transform msg to a Pose msg
        pose = Pose(position=Point(
                        x=cam_to_world.translation.x, y=cam_to_world.translation.y, z=cam_to_world.translation.z),
                    orientation=cam_to_world.rotation)

        self.E_cam_to_world = tf_conversions.toMatrix(tf_conversions.fromMsg(pose))

        pose = Pose(position=Point(
                        x=world_to_cam.translation.x, y=world_to_cam.translation.y, z=world_to_cam.translation.z),
                    orientation=world_to_cam.rotation)

        self.E_world_to_cam = tf_conversions.toMatrix(tf_conversions.fromMsg(pose))

        # Get the camera intrinsics
        camera_info = rospy.wait_for_message(CAMERA_INFO, CameraInfo, timeout=0.5)
        depth_info = rospy.wait_for_message(DEPTH_CAMERA_INFO, CameraInfo, timeout=0.5)

        if camera_info is None or depth_info is None:
            raise RuntimeError(f'Failed to get camera and depth camera info on topics {camera_info} and {depth_info}')

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

    def transform_world_to_cam_frame(self, pt: Point) -> Point:
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
