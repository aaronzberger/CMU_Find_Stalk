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
from transforms import Transformer


DEPTH_SCALE = 1000.0


class DetectNode:
    @classmethod
    def __init__(cls):
        cls.model = Mask_R_CNN()
        cls.cv_bridge = CvBridge()

        cls.get_stalk_service = rospy.Service('get_stalk', GetStalk, cls.find_stalk)

        cls.call_index = -1
        cls.image_index = -1

    @classmethod
    def run_detection(cls, image, depth_image) -> Point:
        '''
        Run the Mask R-CNN model on the given image and project the result onto the depth image.

        Parameters
            image (sensor_msgs.msg.Image): The image to run the model on
            depth_image (sensor_msgs.msg.Image): The depth image to project the result onto

        Returns
            prediction (geometry_msgs.msg.Point): The predicted position of the forefront stalk (x, y, z)
        '''
        cv_image = cv.cvtColor(cls.cv_bridge.imgmsg_to_cv2(
            image, desired_encoding='bgr8'), cv.COLOR_BGR2RGB)

        # Run the model
        scores, bboxes, masks, output = cls.model.forward(cv_image)
        masks = masks.astype(np.uint8) * 255

        # Save the image
        visualized = cls.model.visualize(cv_image, output)

        # Determine the largest mask
        largest_mask = None
        largest_mask_area = 0
        for i in range(masks.shape[0]):
            mask = masks[i]
            mask_area = np.count_nonzero(mask)
            if mask_area > largest_mask_area:
                largest_mask_area = mask_area
                largest_mask = mask

        # Find the center of the largest mask, which is the average of all points in each dimension
        mask_center_x = np.mean(np.nonzero(largest_mask)[1])
        mask_center_y = np.mean(np.nonzero(largest_mask)[0])

        # NOTE: If needed, align the depth and color images here
        # Project the center onto the depth image
        depth_image = cls.cv_bridge.imgmsg_to_cv2(depth_image, desired_encoding="mono8")
        stalk_depth = np.mean(depth_image[np.nonzero(largest_mask)])
        stalk_depth *= DEPTH_SCALE

        depth_visualized = cv.circle(depth_image, (int(mask_center_x), int(mask_center_y)), 5, (0, 0, 255), -1)
        cv.imwrite(f'viz/{cls.call_index}-{cls.image_index}-depth.jpg', depth_visualized)

        # Add a dot for the mask center
        visualized = cv.circle(np.array(visualized), (int(mask_center_x), int(mask_center_y)), 5, (0, 0, 255), -1)

        # Add text at the dot
        cv.putText(visualized, f'{stalk_depth:.2f}', (int(mask_center_x) + 10, int(mask_center_y) + 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv.imwrite(f'viz/{cls.call_index}-{cls.image_index}.jpg', visualized)

        return Point(x=mask_center_x, y=mask_center_y, z=stalk_depth)

    @classmethod
    def find_best_stalk(cls, positions) -> Point:
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
    def find_stalk(cls, req: GetStalkRequest) -> GetStalkResponse:
        cls.call_index += 1
        cls.image_index = -1
        start = rospy.get_rostime()
        frame_count = 0
        positions = []

        def image_depth_callback(image, depth_image):
            nonlocal frame_count

            try:
                cls.image_index += 1
                detected = cls.run_detection(image, depth_image)
            except RuntimeError:
                rospy.logwarn('Segmentation failed on image')
                return

            frame_count += 1
            positions.append(detected)

        cls.sub_images = message_filters.Subscriber(
            '/device_0/sensor_0/Color_0/image/data', Image)
        cls.sub_depth = message_filters.Subscriber(
            '/device_0/sensor_0/Depth_0/image/data', Image)
        ts = ApproximateTimeSynchronizer(
            [cls.sub_images, cls.sub_depth], queue_size=20, slop=0.2)
        ts.registerCallback(image_depth_callback)

        # transform = Transformer()  # NOTE: Temporary

        # Wait until enough frames have been gathered, or the timeout has been reached
        while frame_count <= req.num_frames and (rospy.get_rostime() - start).to_sec() < req.timeout:
            rospy.sleep(0.1)

        #  Cluster the positions, find the best one, average it
        if len(positions) == 0:
            rospy.logwarn('No stalks detected')
            return GetStalkResponse()
        best_stalk = cls.find_best_stalk(positions)

        # Transform
        # best_stalk = transform.transform_stalk(best_stalk)

        rospy.loginfo('Found stalk at (%f, %f, %f)', best_stalk.x, best_stalk.y, best_stalk.z)

        return best_stalk


if __name__ == '__main__':
    rospy.init_node('detect', log_level=rospy.INFO)

    detect_node = DetectNode()

    rospy.loginfo('started detect node')

    rospy.spin()
