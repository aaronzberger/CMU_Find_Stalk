#!/usr/bin/env python3

# -*- encoding: utf-8 -*-

import math

import cv2 as cv
import numpy as np
import open3d as o3d
import pyransac3d as pyrsc
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose2D
from message_filters import ApproximateTimeSynchronizer
from sensor_msgs.msg import Image
from sklearn.cluster import DBSCAN
from termcolor import colored

from config import (BEST_STALK_ALGO, DEPTH_SCALE, DEPTH_TOPIC, DEPTH_TRUNC,
                    GRASP_POINT_ALGO, IMAGE_TOPIC, INLIER_THRESHOLD,
                    MAX_RANSAC_ITERATIONS, MAX_X, MAX_Y, MIN_X, MIN_Y,
                    OPTIMAL_STALK_DISTANCE, VISUALIZE, BestStalkOptions,
                    GraspPointFindingOptions)
from model import MaskRCNN
from stalk_detect.srv import GetStalk, GetStalkRequest, GetStalkResponse
from transforms import TfBuffer, Transformer
from utils import KillableSubscriber, Stalk
from visualize import Visualizer


class DetectNode:
    @classmethod
    def __init__(cls):
        cls.model = MaskRCNN()
        cls.cv_bridge = CvBridge()

        cls.visualizer = Visualizer()

        cls.tf_buffer = TfBuffer()

        cls.get_stalk_service = rospy.Service('get_stalk', GetStalk, cls.find_stalk)

        # Count the number of total service calls (for visualization, etc)
        cls.call_index = -1

        # Count the number of images processed in the current service call
        cls.image_index = -1

    @classmethod
    def run_detection(cls, image):
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

        return masks, output

    @classmethod
    def get_stalks_features(cls, masks) -> np.ndarray:
        '''
        Get the center points going up each stalk

        Parameters
            masks (np.ndarray): The masks of the detected stalks

        Returns
            stalks_features (list[np.ndarray[Pose2D]]): The center points of the stalks, for each stalk
        '''
        # Get the center points of the stalks
        stalks_features = []
        for mask in masks:
            # Swap x and y in the mask
            mask = np.swapaxes(mask, 0, 1)
            nonzero = np.nonzero(mask)

            # Get the top and bottom height values of the stalk
            top_y, bottom_y = nonzero[1].min(), nonzero[1].max()

            stalk_features = [Pose2D(x=np.nonzero(mask[:, top_y])[0].mean(), y=top_y)]

            # For every 10 pixels, get the center point
            for y in range(top_y, bottom_y, 10):
                # Find the average x value for nonzero pixels at this y value
                stalk_features.append(Pose2D(x=np.nonzero(mask[:, y])[0].mean(), y=y))

            stalk_features.append(Pose2D(x=np.nonzero(mask[:, bottom_y])[0].mean(), y=bottom_y))

            stalks_features.append(np.array(stalk_features))

        return stalks_features

    @classmethod
    def determine_best_stalk(cls, positions) -> Point:
        '''
        Find the best stalk from a list of positions.

        Parameters
            positions (list): A list of positions to cluster and average

        Returns
            best_stalk (geometry_msgs.msg.Point): The best stalk found
        '''
        STORING_FAC = 100000.0  # Allowing for 2 decimal places

        original_positions = positions

        # Transform to list of 2D points
        positions = [[float(int(p.x * STORING_FAC)) + p.y, p.z] for p in positions]

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
        best_stalk_average = np.mean(np.array(positions)[np.nonzero(clustering.labels_ == best_cluster)], axis=0)
        best_stalk_average = [round(best_stalk_average[0] / STORING_FAC, 2), best_stalk_average[0] % 1, best_stalk_average[1]]

        # Find the point in the best cluster closest to the average
        best_stalk = None
        best_stalk_dist = float('inf')
        for i in range(len(original_positions)):
            if clustering.labels_[i] == best_cluster:
                dist = math.sqrt((original_positions[i].x - best_stalk_average[0])**2 +
                                 (original_positions[i].y - best_stalk_average[1])**2 +
                                 (original_positions[i].z - best_stalk_average[2])**2)
                if dist < best_stalk_dist:
                    best_stalk_dist = dist
                    best_stalk = [original_positions[i].x, original_positions[i].y, original_positions[i].z]

        return Point(*best_stalk)

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
    def transform_points(cls, stalks_features, transformer: Transformer):
        '''
        Transform the stalk features to the ground plane (using the robot base transform)

        Parameters
            stalk_features (list[np.ndarray[Point]]): The center points of the stalks
            transformer (Transformer): The transformer

        Returns
            transformed_features (list[np.ndarray[Point]]): The transformed features
        '''
        transformed_features = []
        for stalk in stalks_features:
            transformed_features.append(np.array([transformer.transform_cam_to_world(p) for p in stalk]))

        return transformed_features

    @classmethod
    def ransac_ground_plane(cls, pointcloud):
        '''
        Find the ground plane using RANSAC

        Parameters
            pointcloud (o3d.geometry.PointCloud): The point cloud

        Returns
            plane (np.ndarray): The plane coefficients [A,B,C,D] for Ax+By+Cz+D=0
        '''
        # NOTE: Alternatively, use pointcloud.segment_plane from open3d

        A = 150
        B = 100
        C = 10
        colors = np.asarray(pointcloud.colors)
        # Find the brown points in the point cloud
        brown_points = np.array(pointcloud.points)[np.argwhere(np.logical_and(
            colors[2] < A, np.logical_and(abs(colors[0] - colors[1]) < B, np.maximum(colors[0], colors[1]) > C)))]

        # Find the plane using RANSAC
        plane = pyrsc.Plane()
        best_eq, best_inliers = plane.fit(brown_points, INLIER_THRESHOLD, MAX_RANSAC_ITERATIONS)

        return best_eq

    @classmethod
    def get_grasp_points_by_ransac(cls, stalks_features, pointcloud):
        '''
        Find a point closest to 1-3" from the ground plane for each stalk

        Parameters
            stalks_features (list[np.ndarray[Point]]): The center points of the stalks
            pointcloud (o3d.geometry.PointCloud): The point cloud

        Returns
            grasp_points ([[geometry_msgs.msg.Point, distance_to_ground]]):
                The best grasp points found for each stalk, and their distance to the ground plane
        '''
        # Find the ground plane
        plane = cls.ransac_ground_plane(pointcloud)

        # Find the point closest to 2" from the ground plane for each stalk
        # NOTE: This relies on the ground plane being horizontal. For non-horizontal, use point to plane distance
        goal_height = -plane[3] + 2
        grasp_points = []
        for stalk in stalks_features:
            best_point = min(stalk, key=lambda pt, goal_height=goal_height: abs(pt.z - goal_height))
            distance_to_ground = best_point.z - (-plane[3])
            grasp_points.append([best_point, distance_to_ground])

        return grasp_points

    @classmethod
    def combine_pointclouds(cls, pointclouds, transformers):
        '''
        Combine multiple point clouds into one using ICP and the given transformations

        Parameters
            pointclouds (list[o3d.geometry.PointCloud]): The point clouds

        Returns
            pointcloud (o3d.geometry.PointCloud): The combined point cloud
        '''
        # If ICP is needed
        transformations = []
        for pcl, transformer in zip(pointclouds[1:], transformers[1:]):
            # Find the predicted transformations between pcl and the first pointcloud
            trans_init = transformer.E @ np.linalg.inv(transformers[0].E)

            result = o3d.pipelines.registration.registration_icp(
                pcl, pointclouds[0], 0.05, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            transformations.append(result.transformation)

        # Combine the point clouds
        pointcloud = pointclouds[0]
        for pcl in pointclouds[1:]:
            # Transform the pointcloud to the frame of the first one if ICP is being used
            pcl.transform(transformations.pop())
            pointcloud.points.extend(pcl.points)

    @classmethod
    def find_stalk(cls, req: GetStalkRequest) -> GetStalkResponse:
        '''
        Process a request to find the best stalk to grab

        Parameters
            req (GetStalkRequest): The request, with a number of frames to process and a timeout

        Returns
            GetStalkResponse: The response, with a message, the best stalk found, and # frames processed
        '''
        print('Received a request for {} frames with timeout {} seconds'.format(req.num_frames, req.timeout))
        cls.call_index += 1
        cls.image_index = -1
        start = rospy.get_rostime()
        frame_count = 0
        positions = []
        all_masks = []
        pcls = []
        transformers = []

        def image_depth_callback(image, depth_image):
            nonlocal frame_count, positions, all_masks, pcls, transformers

            transformer = Transformer(cls.tf_buffer.get_tf_buffer())
            cls.image_index += 1

            # Run the model
            masks, output = cls.run_detection(image)

            cv_image = cv.cvtColor(cls.cv_bridge.imgmsg_to_cv2(image, desired_encoding='bgr8'), cv.COLOR_BGR2RGB)
            cv_depth_image = cls.cv_bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")
            cv_depth_image = np.array(cv_depth_image, dtype=np.float32)

            # Scale the image to 560 x 480 (TODO: Remove once using new topic)
            cv_depth_image = cv.resize(cv_depth_image, (560, 480))

            # Add the 10 pixels back to the left and right
            cv_depth_image = cv.copyMakeBorder(cv_depth_image, 0, 0, 40, 40, cv.BORDER_CONSTANT, value=0)

            # region Grasp Point Finding
            stalks_features = cls.get_stalks_features(masks)

            # Visualize the stalks features on the image
            if VISUALIZE:
                features_image = cv.cvtColor(cls.model.visualize(cv_image, output), cv.COLOR_RGB2BGR)
                for stalk in stalks_features:
                    for point in stalk:
                        try:
                            cv.circle(features_image, (int(point.x), int(point.y)), 2, (0, 0, 255), -1)
                        except Exception as e:
                            print(stalks_features, point, e)
                cls.visualizer.publish_item('masks', features_image)

            # Add the depths to the stalk features
            for i, stalk in enumerate(stalks_features):
                for j in range(len(stalk)):
                    # Get the depth from at this point
                    stalks_features[i][j] = Point(x=640 - stalk[j].x, y=480 - stalk[j].y,
                                                  z=cv_depth_image[int(stalk[j].y), int(stalk[j].x)] / DEPTH_SCALE)

            # Transform the points
            stalks_features = cls.transform_points(stalks_features, transformer)

            # TODO eliminate stalks with too few points (too few mask pixels, probably)

            # Turn these features into stalks
            stalks = [Stalk(points=stalk) for stalk in stalks_features]

            if VISUALIZE:
                cls.visualizer.new_frame()
                cls.visualizer.publish_item('features', stalks_features, marker_color=[255, 0, 0])
                for stalk in stalks:
                    cls.visualizer.publish_item('stalk_lines', stalk, marker_color=[255, 255, 0], marker_size=0.01)

            if GRASP_POINT_ALGO == GraspPointFindingOptions.mask_only:
                grasp_points = [stalk.get_grasp_point(min_height=min([p.z for p in stalk.points])) for stalk in stalks]

            elif GRASP_POINT_ALGO == GraspPointFindingOptions.mask_projection:
                grasp_points = [stalk.get_grasp_point(min_height=0) for stalk in stalks]

            elif GRASP_POINT_ALGO == GraspPointFindingOptions.ransac_ground_plane:
                pointcloud = cls.get_pcl(image, depth_image, transformer)

                if VISUALIZE:
                    cls.visualizer.publish_item('pointcloud', pointcloud)

                grasp_points = cls.get_grasp_points_by_ransac(stalks_features, pointcloud, transformer)
            # endregion

            # region Best Stalk Determination
            valid_grasp_points = [g for g in grasp_points if g.x <= MAX_X and g.x >= MIN_X and g.y <= MAX_Y and g.y >= MIN_Y]
            valid_masks = [masks[grasp_points.index(g)] for g in valid_grasp_points]

            if len(valid_masks) == 0:
                rospy.logwarn('No stalks detected on frame {}'.format(cls.image_index))
                frame_count += 1
                return

            if BEST_STALK_ALGO == BestStalkOptions.largest:
                largest_mask = max(valid_masks, key=lambda mask: np.count_nonzero(mask))
                grasp_point = valid_grasp_points[valid_masks.index(largest_mask)]
                corresponding_mask = largest_mask

            elif BEST_STALK_ALGO == BestStalkOptions.largest_favorable:
                # We have:
                #   mask_pixels: The number of pixels in the mask (order of x^2)
                #   mask_height: The average height of the mask (order of x)
                #   mask_width: The average width of the mask (order of x)
                #   distance_from_center: The distance from the center of the image (order of x)
                #   distance_from_camera: The distance from the camera (order of x)

                # So, we do a weighting function to decide the best stalk
                #   mask_pixels, mask_height, and mask_width can be multiplicative weights
                #   distance_from_center and distance_from_camera can be additive weights

                # Thus, we have:
                #   weight = mask_pixels * mask_height * mask_width -
                #       distance_from_center**3 - (distance_from_camera - optimal_distance)**3

                # Note that metrics like verticality of the masks should be taken care of by the segmentation

                mask_weights = []
                for i, mask in enumerate(valid_masks):
                    mask_pixels = np.count_nonzero(mask)
                    mask_height = np.ptp([p.z for p in stalks[i].points])
                    mask_width = np.ptp([p.x for p in stalks[i].points])
                    distance_from_center = np.mean([p.x for p in stalks[i].points]) - 320
                    distance_from_camera = valid_grasp_points[i].x

                    weight = mask_pixels * mask_height * mask_width - \
                        distance_from_center ** 3 - (distance_from_camera - OPTIMAL_STALK_DISTANCE) ** 3

                    mask_weights.append(weight)

                largest_weight = max(mask_weights)
                grasp_point = valid_grasp_points[mask_weights.index(largest_weight)]

                if VISUALIZE:
                    cls.visualizer.publish_item('grasp_points', grasp_points, marker_color=[255, 0, 255], marker_size=0.02)

                largest_valid_mask = None
                num_best_mask_pixels = 0
                grasp_point = None
                for i, mask in enumerate(valid_masks):
                    num_mask_pixels = np.count_nonzero(mask)
                    if num_mask_pixels > num_best_mask_pixels:
                        num_best_mask_pixels = num_mask_pixels
                        largest_valid_mask = mask
                        grasp_point = valid_grasp_points[i]

                corresponding_mask = largest_valid_mask

            elif BEST_STALK_ALGO == BestStalkOptions.combine_pcls:
                positions += grasp_points
                all_masks += masks
                pcls.append(pointcloud)
                transformers.append(transformer)

            if BEST_STALK_ALGO in [BestStalkOptions.largest, BestStalkOptions.largest_favorable]:
                # NOTE: Temporarily transforming to camera_frame for testing
                grasp_point = transformer.transform_world_to_cam(grasp_point)
                positions.append(grasp_point)
                all_masks.append(corresponding_mask)
            # endregion

            frame_count += 1

        # Setup the callback for the images and depth images
        # cls.sub_images = message_filters.Subscriber(IMAGE_TOPIC, Image)
        # cls.sub_depth = message_filters.Subscriber(DEPTH_TOPIC, Image)
        cls.sub_images = KillableSubscriber(IMAGE_TOPIC, Image)
        cls.sub_depth = KillableSubscriber(DEPTH_TOPIC, Image)
        ts = ApproximateTimeSynchronizer(
            [cls.sub_images, cls.sub_depth], queue_size=5, slop=0.2)
        ts.registerCallback(image_depth_callback)

        # Continue until enough frames have been gathered, or the timeout has been reached
        while frame_count <= req.num_frames and (rospy.get_rostime() - start).to_sec() < req.timeout:
            rospy.sleep(0.1)

        # Unregister the callback
        cls.sub_depth.unregister()
        cls.sub_images.unregister()
        del ts

        #  Cluster the positions, find the best one, average it
        if len(positions) == 0:
            rospy.logwarn('No stalks detected for this service request')
            return GetStalkResponse(success='REPOSITION')

        # Option 1 or 2.1: Simply find the consensus among the individual decisions
        if BEST_STALK_ALGO in [BestStalkOptions.largest, BestStalkOptions.largest_favorable]:
            best_stalk = cls.determine_best_stalk(positions)

        elif BEST_STALK_ALGO == BestStalkOptions.combine_pcls:
            # Combine point clouds with the first one
            single_pcl = cls.combine_pointclouds(pcls, transformers)

            # Find the grasp points for this point cloud, which is in the frame of the first point cloud
            grasp_points = cls.get_grasp_points_by_ransac(positions, single_pcl, transformers[0])

            best_stalk = cls.determine_best_stalk(grasp_points)

        print(colored('Found stalk at robot frame ({}, {}, {})'.format(
            best_stalk.x, best_stalk.y, best_stalk.z), 'green'))

        return GetStalkResponse(success='DONE', position=best_stalk, num_frames=cls.image_index)


if __name__ == '__main__':
    print('Starting stalk_detect node.\n\tUsing grasp point algorithm: {}\n\tUsing best stalk algorithm: {}'.format(
        GRASP_POINT_ALGO.name, BEST_STALK_ALGO.name))

    rospy.init_node('stalk_detect')

    detect_node = DetectNode()

    print(colored('Loaded model. Waiting for service calls...', 'green'))

    rospy.spin()
