#!/usr/bin/env python3

# -*- encoding: utf-8 -*-

import math

import cv2 as cv
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer
from sensor_msgs.msg import Image
from sklearn.cluster import DBSCAN

from model import Mask_R_CNN
from stalk_detect.srv import GetStalk, GetStalkRequest, GetStalkResponse
from transforms import Transformer, INTRINSIC
import pcl
import open3d as o3d


DEPTH_SCALE = 1000.0
DEPTH_TRUNC = 10


class DetectNode:
    @classmethod
    def __init__(cls):
        cls.model = Mask_R_CNN()
        cls.cv_bridge = CvBridge()

        cls.get_stalk_service = rospy.Service('get_stalk', GetStalk, cls.find_stalk)

        cls.call_index = -1
        cls.image_index = -1

    @classmethod
    def run_detection(cls, image) -> np.ndarray:
        '''
        Run the Mask R-CNN model on the given image

        Parameters
            image (sensor_msgs.msg.Image): The image to run the model on

        Returns
            masks (np.ndarray): The masks of the detected stalks
        '''
        cv_image = cv.cvtColor(cls.cv_bridge.imgmsg_to_cv2(
            image, desired_encoding='bgr8'), cv.COLOR_BGR2RGB)

        # Run the model
        scores, bboxes, masks, output = cls.model.forward(cv_image)
        masks = masks.astype(np.uint8) * 255

        # Save the image
        visualized = cls.model.visualize(cv_image, output)
        cv.imwrite(f'viz/{cls.call_index}-{cls.image_index}.jpg', visualized)

        return masks

    @classmethod
    def get_stalk_features(cls, masks) -> np.ndarray:
        '''
        Get the center points going up each stalk

        Parameters
            masks (np.ndarray): The masks of the detected stalks

        Returns
            stalk_features (np.ndarray): The center points of the stalks
        '''
        # Get the center points of the stalks
        stalk_features = []
        for mask in masks:
            # Get the top and bottom height values of the stalk
            bottom_y, top_y = np.nonzero(mask)[1].min(), np.nonzero(mask)[1].max()

            stalk_features.append([np.nonzero(mask[:, bottom_y])[0].mean(), bottom_y])

            # For every 10 pixels, get the center point
            for y in range(bottom_y, top_y, 10):
                # Find the average x value for nonzero pixels at this y value
                stalk_features.append([np.nonzero(mask[:, y])[0].mean(), y])

            stalk_features.append([np.nonzero(mask[:, top_y])[0].mean(), top_y])

        return np.array(stalk_features)

    @classmethod
    def determine_best_stalk(cls, positions) -> Point:
        '''
        Find the best stalk from a list of positions.

        Parameters
            positions (list): A list of positions to cluster and average

        Returns
            best_stalk (geometry_msgs.msg.Point): The best stalk found
        '''
        STORING_FAC = 100000  # Allowing for 2 decimal places

        # Transform to list of 2D points
        positions = [[p.x * STORING_FAC + p.y, p.z] for p in positions]

        # Use a custom metric to allow 3-dimensional clustering
        def three_dim(a, b):
            return math.sqrt((a[0] / STORING_FAC - b[0] / STORING_FAC)**2 +
                             (a[0] % STORING_FAC - b[0] % STORING_FAC)**2 +
                             (a[1] - b[1])**2)

        clustering = DBSCAN(eps=0.1, min_samples=1, metric=three_dim).fit(positions)

        # Find the cluster with the most points
        best_cluster = None
        best_cluster_size = 0
        for cluster in set(clustering.labels_):
            cluster_size = np.count_nonzero(clustering.labels_ == cluster)
            if cluster_size > best_cluster_size:
                best_cluster_size = cluster_size
                best_cluster = cluster

        # Average the points in the best cluster
        best_stalk = np.mean(np.array(positions)[np.nonzero(clustering.labels_ == best_cluster)], axis=0)

        return Point(x=best_stalk[0] / STORING_FAC, y=best_stalk[0] % STORING_FAC, z=best_stalk[1])

    @classmethod
    def get_pcl(cls, image, depth_image, transformer):
        '''
        Get the point cloud

        Parameters
            image (sensor_msgs.msg.Image): The depth image

        Returns
            point_cloud (pcl PointCloud): The point cloud
        '''
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(cls.cv_bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')),
                o3d.geometry.Image(cls.cv_bridge.imgmsg_to_cv2(depth_image, desired_encoding='mono8')),
                depth_scale=DEPTH_SCALE, depth_trunc=DEPTH_TRUNC, convert_rgb_to_intensity=False),
            o3d.camera.PinholeCameraIntrinsic(
                transformer.width, transformer.height, transformer.intrinsic))

        return pcd

    @classmethod
    def find_stalk(cls, req: GetStalkRequest) -> GetStalkResponse:
        cls.call_index += 1
        cls.image_index = -1
        start = rospy.get_rostime()
        frame_count = 0
        positions = []

        def image_depth_callback(image, depth_image):
            nonlocal frame_count

            transformer = Transformer()

            cls.image_index += 1
            masks = cls.run_detection(image, depth_image)
            stalk_features = cls.get_stalk_features(masks)

            pointcloud = cls.get_pcl(image, depth_image, transformer)

            frame_count += 1

            # TODO: Project stalk_features onto pointcloud
            # TODO: Find the best stalk, and add it to positions
            positions.append(detected)

        cls.sub_images = message_filters.Subscriber(
            '/device_0/sensor_0/Color_0/image/data', Image)
        cls.sub_depth = message_filters.Subscriber(
            '/device_0/sensor_0/Depth_0/image/data', Image)
        ts = ApproximateTimeSynchronizer(
            [cls.sub_images, cls.sub_depth], queue_size=20, slop=0.2)
        ts.registerCallback(image_depth_callback)

        # Wait until enough frames have been gathered, or the timeout has been reached
        while frame_count <= req.num_frames and (rospy.get_rostime() - start).to_sec() < req.timeout:
            rospy.sleep(0.1)

        #  Cluster the positions, find the best one, average it
        if len(positions) == 0:
            rospy.logwarn('No stalks detected')
            return GetStalkResponse()
        best_stalk = cls.determine_best_stalk(positions)

        rospy.loginfo('Found stalk at (%f, %f, %f)', best_stalk.x, best_stalk.y, best_stalk.z)

        return best_stalk


if __name__ == '__main__':
    rospy.init_node('detect', log_level=rospy.INFO)

    detect_node = DetectNode()

    rospy.loginfo('started detect node')

    rospy.spin()
