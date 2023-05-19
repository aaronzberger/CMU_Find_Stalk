# Label images with either stalk or not stalk
# User presses the up arrow for stalk, down arrow for not stalk
# An output folder is created, and the image is copied into it, and a json file is also saved with all labels

import os
import matplotlib.pyplot as plt
from pynput import keyboard
import json
import numpy as np


data_dir = 'masks'

if not os.path.exists(os.path.join(data_dir, 'stalk')):
    os.mkdir(os.path.join(data_dir, 'stalk'))

if not os.path.exists(os.path.join(data_dir, 'not_stalk')):
    os.mkdir(os.path.join(data_dir, 'not_stalk'))

labels = {}

for img in os.listdir(data_dir):
    # Display the image
    img_array = plt.imread(os.path.join(data_dir, img))

    mask = np.argwhere(img_array[:, img_array.shape[1] // 2:, :] != 0)

    # Remove the third axis
    mask = mask[:, :2]

    print(mask)

    # Make the pixels in the mask white on the left half of the image
    img_array[mask[:, 0], mask[:, 1], :] = [255, 255, 255]

    plt.imshow(img_array)
    plt.show()

    # Wait for up or down arrow pressed
    print('Press up arrow for stalk, down arrow for not stalk')

    while True:
        if keyboard.is_pressed('up'):
            labels[img] = 'stalk'
            os.rename(os.path.join(data_dir, img), os.path.join(data_dir, 'stalk', img))
            break
        elif keyboard.is_pressed('down'):
            labels[img] = 'not_stalk'
            os.rename(os.path.join(data_dir, img), os.path.join(data_dir, 'not_stalk', img))
            break

    # Save current labels to json file
    with open('labels.json', 'w') as f:
        json.dump(labels, f)
