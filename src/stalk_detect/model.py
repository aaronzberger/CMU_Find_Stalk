from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.catalog import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from stalk_detect.config import CUDA_DEVICE_NO, MODEL_PATH, SCORE_THRESHOLD


class MaskRCNN:
    def __init__(self):
        self.model_cfg = get_cfg()
        self.model_cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )
        self.model_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
        self.model_cfg.MODEL.WEIGHTS = MODEL_PATH
        self.model_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # background, stalk
        self.model_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1
        self.model_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)
        self.model_cfg.MODEL.DEVICE = 'cuda:{}'.format(CUDA_DEVICE_NO) if CUDA_DEVICE_NO >= 0 else 'cpu'

        try:
            self.model = DefaultPredictor(self.model_cfg)
        except AssertionError:
            # If the model couldn't be loaded, try to load it from the CPU
            self.model_cfg.MODEL.DEVICE = 'cpu'
            self.model = DefaultPredictor(self.model_cfg)

    def forward(self, image):
        '''
        Forward Prop for the Mask-R-CNN model

        Parameters
            image (cv.Mat/np.ndarray): the input image

        Returns
            np.ndarray: confidence scores for each prediction
            np.ndarray: (N X 4) array of bounding box corners
            np.ndarray: (N X H X W) masks for each prediction,
                where the channel 0 index corresponds to the prediction index
                and the other channels are boolean values representing whether
                that pixel is inside that prediction's mask
            dict: output object from detectron2 predictor
        '''
        outputs = self.model(image)

        scores = outputs['instances'].scores.to('cpu').numpy()

        bboxes = outputs['instances'].pred_boxes.tensor.to('cpu').numpy()
        masks = outputs['instances'].pred_masks.to('cpu').numpy()

        # Convert xyxy corner format to xywh center format
        bboxes_center = [[(i[0] + i[2]) / 2, (i[1] + i[3]) / 2, abs(i[2] - i[0]), abs(i[3] - i[1])] for i in bboxes]

        return scores, bboxes_center, masks, outputs

    def visualize(self, input_image, output):
        '''
        Visualize the results of the model

        Parameters
            image (cv.Mat/np.ndarray): the input image
            output (dict): output from the detectron2 predictor

        Returns
            np.ndarray: visualized results
        '''

        v = Visualizer(input_image[:, :, ::-1],
                       metadata=MetadataCatalog.get('empty'),
                       instance_mode=ColorMode.IMAGE_BW)  # remove the colors of unsegmented pixels

        v = v.draw_instance_predictions(output['instances'].to('cpu'))

        return v.get_image()[:, :, ::-1]
