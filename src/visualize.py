import numpy as np
import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose
from sensor_msgs.msg import Image, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
import ros_numpy
from termcolor import colored


# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8


class Visualizer:
    '''
    Publish a variety of visualizations for the pipeline
    '''
    @classmethod
    def __init__(cls):
        cls.publishers: list[rospy.Publisher] = []
        cls.ids = []

    @classmethod
    def _point_to_marker(cls, point: Point) -> Marker:
        '''
        Convert a point to a marker

        Parameters
            point (geometry_msgs.msg.Point): the point to convert

        Returns
            visualization_msgs.msg.Marker: the marker
        '''
        return Marker(pose=Pose(position=point), duration=rospy.Duration(0), type=Marker.SPHERE,
                      scale=Point(x=0.05, y=0.05, z=0.05), color=Marker(color=Point(r=1, g=0, b=0, a=1)))

    @classmethod
    def _o3d_to_pcl_ros(cls, o3d_pcl: o3d.geometry.PointCloud) -> PointCloud2:
        '''
        Convert an Open3D point cloud to a ROS point cloud

        Parameters
            o3d_pcl (open3d.geometry.PointCloud): the point cloud to convert

        Returns
            sensor_msgs.msg.PointCloud2: the converted point cloud
        '''
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "base_link"

        points = np.asarray(o3d_pcl.points)

        o3d_x = points[:, 0]
        o3d_y = points[:, 1]
        o3d_z = points[:, 2]

        cloud_data = np.core.records.fromarrays([o3d_x, o3d_y, o3d_z], names='x,y,z')

        if not o3d_pcl.colors:  # XYZ only
            fields = FIELDS_XYZ
        else:  # XYZ + RGB
            fields = FIELDS_XYZRGB
            color_array = np.array(np.floor(np.asarray(o3d_pcl.colors) * 255), dtype=np.uint8)

            o3d_r = color_array[:, 0]
            o3d_g = color_array[:, 1]
            o3d_b = color_array[:, 2]

            cloud_data = np.lib.recfunctions.append_fields(cloud_data, ['r', 'g', 'b'], [o3d_r, o3d_g, o3d_b])

            cloud_data = ros_numpy.point_cloud2.merge_rgb_fields(cloud_data)

        return pc2.create_cloud(header, fields, cloud_data)

    @classmethod
    def publish_item(cls, id, item, delete_old_markers=True):
        '''
        Publish an item to the visualization topic

        Parameters
            id (str): the id of the item to publish
            item (np.ndarray): the item to publish
        '''
        if id not in cls.ids:
            if isinstance(item, np.ndarray):
                topic_type = Image
            elif isinstance(item, Point):
                topic_type = Marker
            elif isinstance(item, list) and isinstance(item[0], Point):
                topic_type = MarkerArray
            elif isinstance(item, o3d.geometry.PointCloud):
                topic_type = PointCloud2
            else:
                print(colored('Invalid visualization type {} when trying to publish with id {}'.format(
                    type(item), id), 'red'))
                return

            cls.publishers.append(rospy.Publisher('stalk_detect/viz/{}'.format(id), topic_type, queue_size=10))
            cls.ids.append(id)

        if isinstance(item, np.ndarray):
            msg = CvBridge().cv2_to_imgmsg(item, encoding='bgr8')

        elif isinstance(item, Point):
            if delete_old_markers:
                # Delete all old markers
                cls.publishers[cls.ids.index(id)].publish(Marker(action=Marker.DELETEALL))

            msg = cls._point_to_marker(item)

        elif isinstance(item, list) and isinstance(item[0], Point):
            if delete_old_markers:
                # Delete all old markers
                cls.publishers[cls.ids.index(id)].publish(Marker(action=Marker.DELETEALL))

            msg = MarkerArray(markers=[cls._point_to_marker(i) for i in item])

        elif isinstance(item, o3d.geometry.PointCloud):
            msg = cls._o3d_to_pcl_ros(item)

        else:
            print(colored('Invalid visualization type {} when trying to publish with id {}'.format(
                type(item), id), 'red'))
            return

        cls.publishers[cls.ids.index(id)].publish(msg)
