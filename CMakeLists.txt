cmake_minimum_required(VERSION 3.0.2)
project(stalk_detect)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
  message_generation
)

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate services in the 'srv' folder
add_service_files(
  FILES
  GetStalk.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs   sensor_msgs   std_msgs
)

###################################
## catkin specific configuration ##
###################################

catkin_package(
 CATKIN_DEPENDS geometry_msgs rospy sensor_msgs std_msgs message_runtime
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
