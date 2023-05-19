import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from skimage import measure
import sys
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


SAM_CHECKPOINT = "/home/frc/catkin_ws/src/stalk_detect/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

DEVICE = "cuda"
BOXES_PER_SIDE = 10
POINTS_PER_BOX = 10


class Detector:
    @classmethod
    def __init__(cls):
        cls.sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        cls.sam.to(device=DEVICE)

    @classmethod
    def get_pixel_queries(cls, image, viz_path=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_pixels = np.empty((0, 2))

        # Get indices of some green pixels: split the image into BOXES_PER_SIDE**2 boxes
        # and get POINTS_PER_BOX random pixels from each
        x_splits = np.array_split(np.arange(image.shape[0]), BOXES_PER_SIDE)
        y_splits = np.array_split(np.arange(image.shape[1]), BOXES_PER_SIDE)
        for x in x_splits:
            for y in y_splits:
                # Green pixels are between 40 and 100 in H, and between 100 and 255 in S
                new_pixels = np.argwhere(np.logical_and(image[x[0]:x[-1], y[0]:y[-1], 1] >= 100,
                                         np.logical_and(image[x[0]:x[-1], y[0]:y[-1], 0] >= 40,
                                                        image[x[0]:x[-1], y[0]:y[-1], 0] <= 100)))

                # Choose no more than 10 random pixels from this box
                if len(new_pixels) > POINTS_PER_BOX:
                    new_pixels = new_pixels[np.random.choice(len(new_pixels), POINTS_PER_BOX, replace=False)]

                # Add the pixels to the list
                green_pixels = np.concatenate((green_pixels, new_pixels + np.array([x[0], y[0]])))

        # Switch the axis (swap x and y)
        green_pixels = np.flip(green_pixels, axis=1)

        # Normalize the green pixels
        green_pixels = green_pixels / np.array([image.shape[1], image.shape[0]])

        # Add an axis at the beginning for cropped points
        # NOTE: if the crop_n_layers parameter is changed, this will need to be changed as well
        green_pixels = np.expand_dims(green_pixels, axis=0)

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        if viz_path is None:
            return green_pixels

        # Visualize
        plt.figure(figsize=(20, 20))
        plt.imshow(image)

        # Display the grid from x and y splits
        for x in x_splits:
            plt.plot([0, image.shape[1]], [x[0], x[0]], color='red')
        for y in y_splits:
            plt.plot([y[0], y[0]], [0, image.shape[0]], color='red')

        # Display the green pixels in black
        plt.scatter(green_pixels[0, :, 0] * image.shape[1], green_pixels[0, :, 1] * image.shape[0], color='black', s=1)
        plt.axis('off')
        plt.savefig("green_pixels.png")

        return green_pixels

    @classmethod
    def run_model(cls, image, queries):
        mask_generator = SamAutomaticMaskGenerator(
            model=cls.sam,
            points_per_side=None,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=50,  # Requires open-cv to run post-processing
            point_grids=queries
        )

        masks = mask_generator.generate(image)

        return masks

    @classmethod
    def heuristic_fit(cls, masks):
        labels = np.zeros((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1]))
        for i in range(len(masks)):
            labels[masks[i]['segmentation'] == 1] = i + 1

        properties = measure.regionprops(labels.astype(int))

        # For each mask, print the center and all the properties
        for i in range(len(properties)):
            print("Center: {}".format(properties[i]['centroid']))
            print("Orientation: {}".format(properties[i]['orientation']))
            print("Eccentricity: {}".format(properties[i]['eccentricity']))
            print("")

        # For vertical-facing objects, orientation should be close to 90 or -90
        # For tall/long objects, eccentricity should be close to 1
        # For non-disjointed and non-irregular shaped objects, solidity should be close to 1, but stalks can be disconnected by leaves
        new_masks = []
        for i in range(len(properties)):
            if abs(properties[i]['orientation']) < 0.2 and \
                    properties[i]['eccentricity'] > 0.9:
                new_masks.append(masks[properties[i]['label'] - 1])

        return new_masks

    @classmethod
    def visualize(cls, image, masks):
        plt.figure(figsize=(20, 20))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cmap='gray')
        plt.axis('off')

        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        # Create a mask image
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0

        # Draw the masks in different colors
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask

        ax.imshow(img)

        # Convert to openCV
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image_cv = image_flat.reshape(canvas.get_width_height()[::-1] + (3,))
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        return image_cv

    @classmethod
    def forward(cls, image):
        pixel_queries = cls.get_pixel_queries(image, viz_path='green_pixels.png')
        masks = cls.run_model(image, pixel_queries)

        image = Detector.visualize(image, masks)
        cv2.imwrite('before.png', image)

        # NOTE: Here, we have two options to find which masks are stalks
        #     (1) Use heuristics about the mask
        #     (2) Train a classification model, using heuristics for labeling
        # stalks = cls.heuristic_fit(masks)

        return masks

    @classmethod
    def save_masks(cls, image, masks, index):
        '''
        Save the masks for SVM training

        Parameters
            image (numpy.ndarray): The image
            masks (list): The list of masks (T/F) for the image
        '''
        for i in range(len(masks)):
            mask = masks[i]['segmentation']

            # Crop the image
            mask_image = np.zeros((image.shape[0], image.shape[1], 3))

            mask_image[mask] = (255, 255, 255)

            # Combine the image and mask horizontally
            combined_image = np.concatenate((image, mask_image), axis=1)

            # Save the image
            cv2.imwrite('masks/{}-{}.png'.format(index, i), combined_image)


if __name__ == '__main__':
    # Load the image
    image = cv2.imread(sys.argv[1])

    original_image = image.copy()

    # Run the model
    Detector()
    stalks = Detector.forward(image)

    print("Found {} stalks".format(len(stalks)))

    # Visualize the results
    image = Detector.visualize(image, stalks)
    cv2.imwrite('after.png', image)

    # Save the masks
    Detector.save_masks(original_image, stalks, 1)
