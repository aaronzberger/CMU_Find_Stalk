#!/usr/bin/env python3

# -*- encoding: utf-8 -*-

import math
from typing import Optional

import cv2 as cv
import message_filters
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

from config import (BEST_STALK_ALGO, DEPTH_SCALE, DEPTH_TRUNC,
                    GRASP_POINT_ALGO, INLIER_THRESHOLD, MAX_RANSAC_ITERATIONS,
                    VISUALIZE, IMAGE_TOPIC, DEPTH_TOPIC, GraspPointFindingOptions, BestStalkOptions)
from model import MaskRCNN
from stalk_detect.srv import GetStalk, GetStalkRequest, GetStalkResponse
from transforms import Transformer, TfBuffer
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
    def run_detection(cls, image, visualize: bool) -> np.ndarray:
        '''
        Run the Mask R-CNN model on the given image

        Parameters
            image (sensor_msgs.msg.Image): The image to run the model on
            visualize (bool): Whether to visualize the results

        Returns
            masks (np.ndarray): The masks of the detected stalks
        '''
        cv_image = cv.cvtColor(cls.cv_bridge.imgmsg_to_cv2(
            image, desired_encoding='bgr8'), cv.COLOR_BGR2RGB)

        # Run the model
        scores, bboxes, masks, output = cls.model.forward(cv_image)
        masks = masks.astype(np.uint8) * 255

        if visualize:
            # Save the image
            visualized = cls.model.visualize(cv_image, output)
            cls.visualizer.publish_item('masks', visualized)

        return masks

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
            # Get the top and bottom height values of the stalk
            bottom_y, top_y = np.nonzero(mask)[1].min(), np.nonzero(mask)[1].max()

            stalk_features = [Pose2D(x=np.nonzero(mask[:, bottom_y])[0].mean(), y=bottom_y)]

            # For every 10 pixels, get the center point
            for y in range(bottom_y, top_y, 10):
                # Find the average x value for nonzero pixels at this y value
                stalk_features.append(Pose2D(x=np.nonzero(mask[:, y])[0].mean(), y=y))

            stalk_features.append(Pose2D(x=np.nonzero(mask[:, top_y])[0].mean(), y=top_y))

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
    def get_grasp_pts_by_features(cls, stalks_features, transformer: Optional[Transformer] = None):
        '''
        Find a point closest to 1-3" from the bottom of the feature points

        Parameters
            stalk_features (list[np.ndarray[Point]]): The center points of the stalks
            transformer (Transformer): The transformer

        Returns
            grasp_point (geometry_msgs.msg.Point): The best grasp point found
        '''
        world_pts = []
        for stalk in stalks_features:
            if transformer is not None:
                world_pts.append([transformer.transform_point(p) for p in stalk])
            else:
                world_pts.append(stalk)

        # Find the point closest to 2" from the bottom of the stalk
        grasp_points = []
        for stalk in world_pts:
            goal_height = stalk[0].z + 2
            grasp_points.append(min(stalk, key=lambda pt, goal_height=goal_height: abs(pt.z - goal_height)))

        return grasp_points

    @classmethod
    def project_features_to_base(cls, stalks_features, transformer):
        '''
        Project the stalk features to the ground plane (using the robot base transform)

        Parameters
            stalk_features (list[np.ndarray[Point]]): The center points of the stalks
            transformer (Transformer): The transformer

        Returns
            projected_features (list[np.ndarray[Point]]): The projected features
        '''
        world_pts = []
        for stalk in stalks_features:
            world_pts.append(np.array([transformer.transform_point(p) for p in stalk]))

        # Assume the ground plane is at z=0, so add points going in the same line as the stalk to the ground
        projected_features = []
        for i, stalk in enumerate(world_pts):
            single_features = world_pts[i]
            # Find the median height distance between consecutive points
            median_height_diff = np.median([stalk[i].z - stalk[i - 1].z for i in range(1, len(stalk))])
            min_height = min([p.z for p in stalk])

            array_stalk = np.array(np.array([[p.x, p.y, p.z] for p in stalk]))

            # Calculate the center of the points
            center = np.mean(array_stalk, axis=0)

            # Fit a line to the 3D points
            _, _, vv = np.linalg.svd(array_stalk - center)

            # Add points going down to the ground
            try:
                num_points = int(abs(min_height) / median_height_diff)
            except ZeroDivisionError:
                return None

            for height in np.linspace(min_height, 0, num_points):
                single_features = np.append(single_features, [Point(x=center[0] + vv[0][0] * height,
                                                                    y=center[1] + vv[0][1] * height,
                                                                    z=center[2] + vv[0][2] * height)], axis=0)
                # center + vv[2] * height  # This is what was recommended, but above is copilot's version

            projected_features.append(single_features)

        return projected_features

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
    def get_grasp_points_by_ransac(cls, stalks_features, pointcloud, transformer):
        '''
        Find a point closest to 1-3" from the ground plane for each stalk

        Parameters
            stalks_features (list[np.ndarray[Point]]): The center points of the stalks
            pointcloud (o3d.geometry.PointCloud): The point cloud
            transformer (Transformer): The transformer

        Returns
            grasp_points ([[geometry_msgs.msg.Point, distance_to_ground]]):
                The best grasp points found for each stalk, and their distance to the ground plane
        '''
        world_pts = []
        for stalk in stalks_features:
            world_pts.append([transformer.transform_point(p) for p in stalk])

        # Find the ground plane
        plane = cls.ransac_ground_plane(pointcloud)

        # Find the point closest to 2" from the ground plane for each stalk
        # NOTE: This relies on the ground plane being horizontal. For non-horizontal, use point to plane distance
        goal_height = -plane[3] + 2
        grasp_points = []
        for stalk in world_pts:
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

            print('Processing frame {}'.format(frame_count))

            transformer = Transformer(cls.tf_buffer.get_tf_buffer())
            cls.image_index += 1

            # Run the model
            masks = cls.run_detection(image, VISUALIZE)

            print('Found {} stalks'.format(len(masks)))

            # region Grasp Point Finding
            stalks_features = cls.get_stalks_features(masks)

            print(len(stalks_features), [len(stalk) for stalk in stalks_features])

            depth_image_cv = cls.cv_bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")

            # Add the depths to the stalk features
            for i, stalk in enumerate(stalks_features):
                for j in range(len(stalk)):
                    stalks_features[i][j] = Point(x=stalk[j].x, y=stalk[j].y,
                                                  z=depth_image_cv[int(stalk[j].y)][int(stalk[j].x)] / DEPTH_SCALE)

            if GRASP_POINT_ALGO == GraspPointFindingOptions.mask_only:
                grasp_points = cls.get_grasp_pts_by_features(stalks_features, transformer)

            elif GRASP_POINT_ALGO == GraspPointFindingOptions.mask_projection:
                stalks_features = cls.project_features_to_base(stalks_features, transformer)

                if stalks_features is None:
                    rospy.logwarn('Error projecting the features to base')
                    frame_count += 1
                    return

                grasp_points = cls.get_grasp_pts_by_features(stalks_features, transformer=None)

            elif GRASP_POINT_ALGO == GraspPointFindingOptions.ransac_ground_plane:
                pointcloud = cls.get_pcl(image, depth_image, transformer)

                if VISUALIZE:
                    cls.visualizer.publish_item('pointcloud', pointcloud)

                grasp_points = cls.get_grasp_points_by_ransac(stalks_features, pointcloud, transformer)

            if VISUALIZE:
                cls.visualizer.publish_item('features', cls.visualizer.publish_item(
                    'features', stalks_features, marker_color=(255, 0, 0)))
                cls.visualizer.publish_item('grasp_points', cls.visualizer.publish_item(
                    'grasp_points', grasp_points, delete_old_markers=False, marker_color=(0, 255, 0)))
            # endregion

            # region Best Stalk Determination
            if BEST_STALK_ALGO == BestStalkOptions.largest:
                largest_mask = max(masks, key=lambda mask: np.count_nonzero(mask))
                grasp_point = grasp_points[masks.index(largest_mask)][0]
                corresponding_mask = largest_mask

            elif BEST_STALK_ALGO == BestStalkOptions.largest_favorable:
                valid_grasp_points = [g for g in grasp_points if g.z <= 3 and g.z >= 1]
                valid_masks = [masks[grasp_points.index(g)] for g in valid_grasp_points]

                if len(valid_masks) == 0:
                    rospy.logwarn('No stalks detected')
                    frame_count += 1
                    return

                largest_valid_mask = max(valid_masks, key=lambda mask: np.count_nonzero(mask))
                grasp_point = valid_grasp_points[valid_masks.index(largest_valid_mask)][0]
                corresponding_mask = largest_valid_mask

            elif BEST_STALK_ALGO == BestStalkOptions.combine_pcls:
                positions += grasp_points
                all_masks += masks
                pcls.append(pointcloud)
                transformers.append(transformer)

            if BEST_STALK_ALGO in [BestStalkOptions.largest, BestStalkOptions.largest_favorable]:
                positions.append(grasp_point)
                all_masks.append(corresponding_mask)
                pcls.append(pointcloud)
            # endregion

            frame_count += 1

        # Setup the callback for the images and depth images
        cls.sub_images = message_filters.Subscriber(IMAGE_TOPIC, Image)
        cls.sub_depth = message_filters.Subscriber(DEPTH_TOPIC, Image)
        ts = ApproximateTimeSynchronizer(
            [cls.sub_images, cls.sub_depth], queue_size=20, slop=0.2)
        ts.registerCallback(image_depth_callback)

        # Continue until enough frames have been gathered, or the timeout has been reached
        while frame_count <= req.num_frames and (rospy.get_rostime() - start).to_sec() < req.timeout:
            rospy.sleep(0.1)

        #  Cluster the positions, find the best one, average it
        if len(positions) == 0:
            rospy.logwarn('No stalks detected')
            return GetStalkResponse()

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

        return GetStalkResponse(success='Done', position=best_stalk, num_frames=cls.image_index)


if __name__ == '__main__':
    print('Starting stalk_detect node.\n\tUsing grasp point algorithm: {}\n\tUsing best stalk algorithm: {}'.format(
        GRASP_POINT_ALGO.name, BEST_STALK_ALGO.name))

    rospy.init_node('stalk_detect')

    detect_node = DetectNode()

    print(colored('Loaded model. Waiting for service calls...', 'green'))

    rospy.spin()
