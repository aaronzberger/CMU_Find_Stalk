from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys


sam_checkpoint = "/home/frc/catkin_ws/src/stalk_detect/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)


image = cv2.imread(sys.argv[1])
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

green_pixels = np.empty((0, 2))

# Get the indices of up to 1000 random green pixels in image: split the image into 100 boxes and get 10 random pixels from each box
x_splits = np.array_split(np.arange(image.shape[0]), 10)
y_splits = np.array_split(np.arange(image.shape[1]), 10)
for x in x_splits:
    for y in y_splits:
        # Green pixels are between 40 and 100 in H, and between 100 and 255 in S
        new_pixels = np.argwhere(np.logical_and(image[x[0]:x[-1], y[0]:y[-1], 1] >= 100,
                                 np.logical_and(image[x[0]:x[-1], y[0]:y[-1], 0] >= 40, image[x[0]:x[-1], y[0]:y[-1], 0] <= 100)))

        # Choose no more than 10 random pixels from this box
        if len(new_pixels) > 10:
            new_pixels = new_pixels[np.random.choice(len(new_pixels), 10, replace=False)]

        # Add the pixels to the list
        green_pixels = np.concatenate((green_pixels, new_pixels + np.array([x[0], y[0]])))


# Switch the axis (swap x and y)
green_pixels = np.flip(green_pixels, axis=1)
print(green_pixels.shape, green_pixels)

# Normalize the green pixels
green_pixels = green_pixels / np.array([image.shape[1], image.shape[0]])

image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

# Display the pixels on the image
plt.figure(figsize=(20, 20))
plt.imshow(image)

# Display the grid from x and y splits
for x in x_splits:
    plt.plot([0, image.shape[1]], [x[0], x[0]], color='red')
for y in y_splits:
    plt.plot([y[0], y[0]], [0, image.shape[0]], color='red')

# Display the green pixels in black
plt.scatter(green_pixels[:, 0] * image.shape[1], green_pixels[:, 1] * image.shape[0], color='black', s=1)
plt.axis('off')
plt.show()

# Add an axis to the green pixels
green_pixels = np.expand_dims(green_pixels, axis=0)

print(green_pixels.shape)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=None,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=50,  # Requires open-cv to run post-processing
    point_grids=green_pixels
)


start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

start_time.record()
torch.cuda.synchronize()
masks = mask_generator.generate(image)
end_time.record()
print(f"Time taken: {start_time.elapsed_time(end_time)}ms")


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


plt.figure(figsize=(20, 20))

# Grayscale the image
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(image, cmap='gray')
show_anns(masks)
plt.axis('off')
plt.show()
