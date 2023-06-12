# Corn Stalk Grasping Pose Estimation

Detect corn stalks in an environment and calculate the best grasping point for a robot arm to insert a needle sensor.

![Visualization](https://github.com/aaronzberger/CMU_Find_Stalk/assets/35245591/c0195655-8a59-4e6c-80ba-e67ca75037da)

## Table of Contents
- [Details](#Details)
  - [Service](#Service)
- [Pipeline](#Pipeline)
- [Usage](#Usage)
- [Dataset](#Dataset)
  - [Labeling](#Labeling)
  - [Training](#Training)
- [Acknowledgements](#Acknowledgements)

## Details
This pipeline is meant as a plug-and-play system, meaning it can be applied to varying similar problems with little interference. Each part of the pipeline can be swapped out with different alternatives, which may work better under certain environment conditions, like camera field of view, ground plane visibility, ground plane smoothness, point cloud density and accuracy, etc.

You can select the desired options for each pipeline step in [`src/config.py`](#https://github.com/aaronzberger/CMU_Find_Stalk/blob/main/src/config.py).

The options are described in more detail [below](#pipeline).

### Service
This ROS node registers a ROS [service](#http://wiki.ros.org/Services), which allows a request-reply communication between two nodes. In this case, this node publishes the service `get_stalk`, which receives a number of frames and a timeout, and returns the stalk positions and other information. This service is defined [here](#https://github.com/aaronzberger/CMU_Find_Stalk/blob/main/srv/GetStalk.srv).

A node can wait for this service to be available and then "call" it and await a response. Alternatively, a node can make a *persistent connection* to a service, which maintains the communication between nodes and does not re-search for the node at each subsequent service request (this can increase performance when the service is being called multiple times from the same node).

For example, a node could call this service once using the following code:
```
rospy.wait_for_service('get_stalk')
get_stalk = rospy.ServiceProxy('get_stalk', GetStalk)
try:
    resp1 = get_stalk(num_frames=5, timeout=10.0)
except rospy.ServiceException as exc:
    print("Service did not process request: " + str(exc))
```

Alternatively, you can call the service from the terminal:
```
rosservice call /get_stalk "num_frames: 5
timeout: 10.0"
```

## Pipeline
The code is separated into 3 steps, as shown below.

![Pipeline](https://github.com/aaronzberger/CMU_Find_Stalk/assets/35245591/4ce6a61d-f59d-4b20-9d7b-c98ceec4ca0a)

- In "Stalk Detection", the images are converted to image masks representing the stalk locations in 2D. Currently, the only option available is Mask-R-CNN, which performs well on this task. The model is run in [`model.py`](#https://github.com/aaronzberger/CMU_Find_Stalk/blob/main/src/model.py) and called by the ROS service [here](#https://github.com/aaronzberger/CMU_Find_Stalk/blob/78da4aee769fc75f414fe1d12053476434de4b5e/src/main.py#LL331C7-L331C7). See [Dataset](#dataset), [Training](#training), and [Labeling](#labeling) for tools on how to get your model trained easily.


## Usage
To get this node up and running, you'll need to prepare the following:
1. Install [Detectron2](#https://detectron2.readthedocs.io/en/latest/tutorials/install.html), which requires CUDA and torch. Detectron2 also works with torch installed for CPU, if you do not have a GPU. We recommend building Detectron2 from source.
2. Install all package dependencies, which are in requirements.txt.
3. If desired, replace the [first line](https://github.com/aaronzberger/CMU_UNet_Node/blob/main/src/main.py#L1) of `main.py` with your Python interpreter path
4. Fill in the configuration file with your desired parameters: specifically, make sure to edit the [`MODEL_PATH`](#https://github.com/aaronzberger/CMU_Find_Stalk/blob/fca1f3f9c3d962b5cb712d720bd9cb57dc0e9a0c/src/config.py#L36), the [TF frames](#https://github.com/aaronzberger/CMU_Find_Stalk/blob/fca1f3f9c3d962b5cb712d720bd9cb57dc0e9a0c/src/config.py#L46:L50), the image and depth image [topics](#https://github.com/aaronzberger/CMU_Find_Stalk/blob/fca1f3f9c3d962b5cb712d720bd9cb57dc0e9a0c/src/config.py#L42:L45), and the constraints on stalk distance.
5. Ensure the `main.py` script is executable, run `catkin_make`, start up your `roscore`, and run:
  
  `rosrun stalk_detect main.py`

## Dataset
We recommend training on a large number of images for robustness in deployment. With the labeling tool provided below, labeling is very quick, and 500 images should take no more than a few hours.

### Labeling
For labeling, we use the [Segment Anything](#https://segment-anything.com/) model to predict masks around objects given a position contained within the mask. This [tool](#https://github.com/aaronzberger/salt) is forked from the commonly-used [SALT](#https://github.com/anuragxel/salt) package (Segment Anything Labeling Tool), and fixes some small bugs and adds progress bars.

Follow instructions described in the tool's README.

### Training
To train and test your Mask-R-CNN model, you may use [this](#https://github.com/aaronzberger/CMU_Mask-R-CNN_Trainer) tool, which describes the needed data format and other configuration. Training is very quick with a decent GPU, since we use the pre-trained checkpoint to start.


## Acknowledgements
- [Mark Lee](#https://github.com/MarkLee634) for assistance in testing, data collection, drivers, and overall assistance
- [Kantor Lab](#https://www.ri.cmu.edu/robotics-groups/kantorlab) for general assistance.
