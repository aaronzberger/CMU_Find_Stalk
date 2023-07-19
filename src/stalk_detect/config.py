RUN_REALSENSE_ON_REQUEST = False
DRIVER_COMMAND = ['roslaunch', 'realsense2_camera', 'rs_camera_pcloud.launch']

# Camera parameters
DEPTH_SCALE = 1000.0
DEPTH_TRUNC = 10
INLIER_THRESHOLD = 0.025
MAX_RANSAC_ITERATIONS = 1000

# Used for RANSAC line detection for the stalks
MAX_STALK_THICKNESS = 0.005  # Used for RANSAC Inlier threshold
MAX_LINE_RANSAC_ITERATIONS = 1000
GRIPPER_WIDTH = 0.15  # 6 inches
GRIPPER_LENGTH_PAST_STALK = 0.075  # 3 inches

OPTIMAL_STALK_DISTANCE = 0.20  # 8 inches
OPTIMAL_STALK_HEIGHT = 0.10  # 4 inches
MINIMUM_MASK_AREA = 30

# Distance in pixels between each feature point in a stalk
# Measured in pixels (and not meters) since feature point accuracy relies on image and depth image resolution
FEATURE_POINT_PIXEL_OFFSET = 10

MIN_X = 0.1
MAX_X = 0.5
MIN_Y = -0.07
MAX_Y = 100

# Model parameters
MODEL_PATH = '/home/frc/catkin_ws/src/stalk_detect/model_field_day1.pth'
SCORE_THRESHOLD = 0.6
CUDA_DEVICE_NO = 0

VISUALIZE = True

IMAGE_TOPIC = '/camera/color/image_raw'
DEPTH_TOPIC = '/camera/depth/image_rect_raw'
CAMERA_INFO = '/camera/color/camera_info'
DEPTH_CAMERA_INFO = '/camera/depth/camera_info'
CAMERA_FRAME = 'camera_link'
BASE_FRAME = 'link_base'
WORLD_FRAME = 'world'
CAMERA_COLOR_FRAME = 'camera_color_frame'
POINTCLOUD_TOPIC = '/camera/pointcloud/points'
