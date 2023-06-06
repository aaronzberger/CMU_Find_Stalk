from enum import Enum

# Currently, the only option for stalk detection is Mask R-CNN


GraspPointFindingOptions = Enum(
    'GraspPointFindingOptions',
    ['mask_only', 'mask_projection', 'ransac_ground_plane', 'segment_ground_plane'])

BestStalkOptions = Enum(
    'BestStalkOptions', ['largest', 'largest_favorable', 'combine_pcls'])


# Camera parameters
DEPTH_SCALE = 1000.0
DEPTH_TRUNC = 10
INLIER_THRESHOLD = 0.05
MAX_RANSAC_ITERATIONS = 1000

# Model parameters
MODEL_PATH = '/home/frc/catkin_ws/src/stalk_detect/model_final.pth'
SCORE_THRESHOLD = 0.6
CUDA_DEVICE_NO = 0

VISUALIZE = True

IMAGE_TOPIC = '/camera/color/image_raw'
DEPTH_TOPIC = '/camera/aligned_depth_to_color/image_raw'
CAMERA_INFO = '/camera/color/camera_info'
DEPTH_CAMERA_INFO = '/camera/aligned_depth_to_color/camera_info'
CAMERA_FRAME = 'camera_link'
BASE_FRAME = 'link_base'
WORLD_FRAME = 'world'
CAMERA_COLOR_FRAME = 'camera_color_frame'
CAMERA_ALIGNED_FRAME = 'camera_aligned_depth_to_color_frame'
POINTCLOUD_TOPIC = '/camera/pointcloud/points'

# Select the algorithms to use
GRASP_POINT_ALGO = GraspPointFindingOptions.mask_projection
BEST_STALK_ALGO = BestStalkOptions.largest_favorable
