from enum import Enum

# Currently, the only option for stalk detection is Mask R-CNN


class GraspPointFindingOptions(Enum):
    'mask-only',
    'mask_projection',
    'ransac_ground_plane',
    'segment_ground_plane'


class BestStalkOptions(Enum):
    'largest',
    'largest_favorable',
    'combine_pcls',


# Camera parameters
DEPTH_SCALE = 1000.0
DEPTH_TRUNC = 10
INLIER_THRESHOLD = 0.05
MAX_RANSAC_ITERATIONS = 1000

# Model parameters
MODEL_PATH = '/home/frc/catkin_ws/src/stalk_detect/model_final.pth'
SCORE_THRESHOLD = 0.6
CUDA_DEVICE_NO = 0

# Select the algorithms to use
GRASP_POINT_ALGO = GraspPointFindingOptions.mask_projection
BEST_STALK_ALGO = BestStalkOptions.largest_favorable
