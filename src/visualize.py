import numpy as np
import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose
from sensor_msgs.msg import Image, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
import cv2 as cv

# Unfortunate hack to fix a bug in ROS Noetic
np.float = np.float64

from std_msgs.msg import Header
from open3d_ros_helper import open3d_ros_helper as orh
from termcolor import colored
from std_msgs.msg import ColorRGBA


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
        cls.counter = 0

    @classmethod
    def _point_to_marker(cls, point: Point, color, id: int) -> Marker:
        '''
        Convert a point to a marker

        Parameters
            point (geometry_msgs.msg.Point): the point to convert

        Returns
            visualization_msgs.msg.Marker: the marker
        '''
        header = Header(frame_id='link_base', stamp=rospy.Time.now())

        return Marker(header=header, id=id, pose=Pose(position=point), lifetime=rospy.Duration(0), type=Marker.SPHERE,
                      scale=Point(x=0.025, y=0.025, z=0.025),
                      color=ColorRGBA(r=color[0] / 255., g=color[1] / 255., b=color[2] / 255., a=1.), action=Marker.ADD)

    @classmethod
    def _o3d_to_pcl_ros(cls, o3d_pcl: o3d.geometry.PointCloud) -> PointCloud2:
        '''
        Convert an Open3D point cloud to a ROS point cloud

        Parameters
            o3d_pcl (open3d.geometry.PointCloud): the point cloud to convert

        Returns
            sensor_msgs.msg.PointCloud2: the converted point cloud
        '''
        return orh.o3dpc_to_rospc(o3d_pcl)

    @classmethod
    def publish_item(cls, id, item, delete_old_markers=True, marker_color=(255, 0, 0)):
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
            elif isinstance(item, list) and isinstance(item[0], Point) or isinstance(item[0], np.ndarray):
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
            cv.imwrite('viz/{}.png'.format(cls.counter), item)
            try:
                msg = CvBridge().cv2_to_imgmsg(item, encoding='bgr8')
            except Exception as e:
                print(colored('Error converting image to ROS message: {}'.format(e), 'red'))
                return

        elif isinstance(item, list):
            if delete_old_markers:
                # Delete all old markers
                cls.publishers[cls.ids.index(id)].publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))

            if isinstance(item[0], np.ndarray) and isinstance(item[0][0], Point):
                # Combine all the markers across multiple stalks into one MarkerArray
                markers = []
                counter = 0
                for sub_array in item:
                    for j in sub_array:
                        markers.append(cls._point_to_marker(j, marker_color, counter))
                        counter += 1
                msg = MarkerArray(markers=markers)

                print('Publishing {} markers'.format(len(markers)))

            elif isinstance(item[0], Point):
                msg = MarkerArray(markers=[cls._point_to_marker(i, marker_color, j) for j, i in enumerate(item)])
            else:
                print(colored('Invalid visualization of type list of {} when trying to publish with id {}'.format(
                    type(item[0]), id), 'red'))
                return

        elif isinstance(item, o3d.geometry.PointCloud):
            msg = cls._o3d_to_pcl_ros(item, marker_color)

        else:
            print(colored('Invalid visualization type {} when trying to publish with already established id {}'.format(
                type(item), id), 'red'))
            return

        cls.publishers[cls.ids.index(id)].publish(msg)
