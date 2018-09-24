
"""

# Run COCO evaluatoin on the last model you trained
python3 LoopEvalRun.py evaluate --dataset=/host/Mask_RCNN/datasets/raspberry/ --limit=100
"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
import os
import sys
import time
import json as js
import numpy as np
import skimage.draw
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



############################################################
#  Configurations
############################################################


class RaspberryConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "raspberry"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    #GPU_COUNT = 1
    #DEVICE = "/gpu:0"
    
    #USE_MINI_MASK = False
    
    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001 # Default is 0.001
    LEARNING_MOMENTUM = 0.9 # Default is 0.9
    
    # Weight decay regularization
    #WEIGHT_DECAY = 0.0001

    # Number of classes (including background) 
    NUM_CLASSES = 1 + 3  
    
    #DETECTION_MIN_CONFIDENCE = 0.7  #tried 0.9, 0.7
    
    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    #BACKBONE_STRIDES = [13, 19, 25, 32, 38] #2, 4, 8, 16, 32
    
    # Non-maximum suppression threshold for detection
    #DETECTION_NMS_THRESHOLD = 0.2  #0.3
    
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9  #0.7
    
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 40
    
    STEPS_PER_EPOCH = 20
    
    VALIDATION_STEPS = 10
    
    BACKBONE = "resnet50"
    
    #MAX_GT_INSTANCES = 7
    #DETECTION_MAX_INSTANCES = 12
    
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (208, 304, 400, 512, 608)  #32, 64,128,256,512
    
    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.75, 1, 1.5]
    
    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 2

    
    #IMAGE_RESIZE_MODE = "square"
    #IMAGE_MIN_DIM = 576
    #IMAGE_MAX_DIM = 1024
    #IMAGE_MIN_SCALE = 0
    
    GRADIENT_CLIP_NORM = 10.0  #tried 5, 10, 15 default:5
    
    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.0,
        "rpn_bbox_loss": 1.0,
        "mrcnn_class_loss": 1.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 1.0
    }
    



############################################################
#  Dataset
############################################################

class RaspberryDataset(utils.Dataset):
    def load_raspberry(self, dataset_dir, subset):
        """Load a subset of the COCO dataset.
        """
        self.add_class("object", 1, "raspberry_1")
        self.add_class("object", 2, "raspberry_2")
        self.add_class("object", 3, "raspberry_3")
        #self.add_class("object", 4, "raspberry_4")
        
        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
       
        annotations = js.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            objects = [s['region_attributes'] for s in a['regions'].values()]
            num_ids = [int(n['object']) for n in objects]
            
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)
            

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        num_ids = info['num_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # print("info['num_ids']=", info['num_ids'])
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids
    
        
    def image_reference(self, image_id):
        """Return the path of the image.
        info = self.image_info[image_id]
        if info["source"] == "raspberry":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)"""
       
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Evaluation
############################################################


def evaluate_raspberry(model, dataset, annotations, eval_type="segm", limit=100, image_ids=None):
    
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    inference_config = RaspberryConfig()
    image_ids = np.random.choice(dataset_val.image_ids, limit)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        """AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])"""
        AP = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
    
    #print(str(APs))
    print("mAP: ", np.mean(APs))
    mAP=np.mean(APs)
    average.append(mAP)
    mAP=str(mAP)
    f.write("%s, " % mAP)
    print("evaluation completed and recorded")
    


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        #default=500,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    """parser.add_argument('--target', required=True,
                        default=[180],
                        metavar="[list of values]",
                        help='Weights to evaluate in folder')"""
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    #print("Weights to evaluate: ", args.target)
    
   
    class InferenceConfig(RaspberryConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0
        #DEVICE = "/cpu:0"
    config = InferenceConfig()
    config.display() 
    #DEVICE = "/cpu:0" 
    
    #directory = os.fsencode(args.model)  # gives us the directory to look in for the weights
    targets = range(180,169,-1) #[ '105.h5', '157.h5', '046.h5']
    #'/host/Mask_RCNN/logs/raspberry20180827T1524', '/host/Mask_RCNN/logs/raspberry20180828T1158', '/host/Mask_RCNN/logs/raspberry20180830T0718', '/host/Mask_RCNN/logs/raspberry20180830T1521', '/host/Mask_RCNN/logs/raspberry20180830T2043', '/host/Mask_RCNN/logs/raspberry20180831T0350',
    logs = [ '/host/Mask_RCNN/logs/raspberry20180901T0204', '/host/Mask_RCNN/logs/raspberry20180901T0658', '/host/Mask_RCNN/logs/raspberry20180901T1141', '/host/Mask_RCNN/logs/raspberry20180901T1634']
    #logs = ['/host/Mask_RCNN/logs/raspberry20180828T1158']
    #logs = ['/host/Mask_RCNN/logs/raspberry20180830T0718']
    #logs = ['/host/Mask_RCNN/logs/raspberry20180830T1521']
    #logs = ['/host/Mask_RCNN/logs/raspberry20180831T2043']
    #logs = ['/host/Mask_RCNN/logs/raspberry20180831T0350']
    
    
    for log in logs:
        for target in targets:
            target=str(target)
            weight = 'mask_rcnn_raspberry_0'+target+'.h5'
            model_path = os.path.join(log, weight)
            print(model_path)
            f=open("EvalResults.txt", "a+")
            model = modellib.MaskRCNN(mode="inference", config=config,
                                      model_dir=args.logs)

            print("Loading weights ", model_path)
            model.load_weights(model_path, by_name=True)

            dataset_val = RaspberryDataset()
            raspberry = dataset_val.load_raspberry(args.dataset, "test")
            dataset_val.prepare()
            average=[]
            f.write("Model = %s ,mAP, " % model_path)
            print("Running RASP evaluation on {} images.".format(args.limit))
            evaluate_raspberry(model, dataset_val, raspberry, "segm", limit=100)
            evaluate_raspberry(model, dataset_val, raspberry, "segm", limit=100)
            evaluate_raspberry(model, dataset_val, raspberry, "segm", limit=100)
            evaluate_raspberry(model, dataset_val, raspberry, "segm", limit=100)
            evaluate_raspberry(model, dataset_val, raspberry, "segm", limit=100)
            evaluate_raspberry(model, dataset_val, raspberry, "segm", limit=100)
            evaluate_raspberry(model, dataset_val, raspberry, "segm", limit=100)
            evaluate_raspberry(model, dataset_val, raspberry, "segm", limit=100)
            evaluate_raspberry(model, dataset_val, raspberry, "segm", limit=100)
            evaluate_raspberry(model, dataset_val, raspberry, "segm", limit=100)

            average=str(np.mean(average))
            f.write("Mean, %s, \n" % average)
            f.close()






