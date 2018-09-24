"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 raspberriesV2Classes.py train --dataset=/host/Mask_RCNN/datasets/raspberry2/ --model=imagenet 

    # Continue training a model that you had trained earlier
    python3 raspberriesV2Classes.py train --dataset=/host/Mask_RCNN/datasets/raspberry2/ --model=/host/Mask_RCNN/logs/raspberry20180917T0037/mask_rcnn_raspberry_0165.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 raspberriesV2Classes.py evaluate --dataset=/host/Mask_RCNN/datasets/raspberry/ --model=/host/Mask_RCNN/logs/raspberry20180916T0708/mask_rcnn_raspberry_0011.h5 --limit=100
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
    # Give the configuration a recognizable name
    NAME = "raspberry"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1
    DEVICE = "/gpu:0"
    
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
    NUM_CLASSES = 1 + 2  
    
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
    
    STEPS_PER_EPOCH = 200 #20
    
    VALIDATION_STEPS = 20  #10
    
    BACKBONE = "resnet50"
    
    #MAX_GT_INSTANCES = 7
    #DETECTION_MAX_INSTANCES = 12
    
    # Length of square anchor side in pixels
    #RPN_ANCHOR_SCALES = (208, 304, 400, 512, 608)  #32, 64,128,256,512
    
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
    
    GRADIENT_CLIP_NORM = 5.0  #tried 5, 10, 15 default:5
    
    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.0,
        "rpn_bbox_loss": 1.0,
        "mrcnn_class_loss": 2.0,
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
        #self.add_class("object", 3, "raspberry_3")
        #self.add_class("object", 4, "raspberry_4")
        
        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
       
        #annotations = js.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        #annotations = js.load(open(os.path.join(dataset_dir, "via_region_data_400.json")))
        #annotations = js.load(open(os.path.join(dataset_dir, "via_region_data_200.json")))
        annotations = js.load(open(os.path.join(dataset_dir, "via_region_data_50.json")))
        #annotations = js.load(open(os.path.join(dataset_dir, "via_region_data_10.json")))
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
#  COCO Evaluation
############################################################

def build_raspberry_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []
    

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "object"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    #print("should have results to load now")
    return results


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
    
    print("mAP: ", np.mean(APs))
    print("mAP's: ", APs)
    #print("precisions: ", precisions)
    #print("recalls: ", recalls)
    #print("overlaps: ", overlaps)


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
    parser.add_argument('--model', required=True,
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
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        config = RaspberryConfig()
    else:
        class InferenceConfig(RaspberryConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "raspberry":
        model_path = RASP_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        print("Loading training dataset")
        dataset_train = RaspberryDataset()
        dataset_train.load_raspberry(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        print("Loading validation dataset")
        dataset_val = RaspberryDataset()
        dataset_val.load_raspberry(args.dataset, "val")
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        print("Augmenting images")
        augmentation0 = imgaug.augmenters.Sequential([imgaug.augmenters.Sometimes(0.5,imgaug.augmenters.Fliplr(0.5)),  # Flip
                        imgaug.augmenters.Sometimes(0.5,imgaug.augmenters.Flipud(0.5)), # Flip half of images vertically
                        imgaug.augmenters.Sometimes(0.5,imgaug.augmenters.Affine(rotate=(-20, 20))),  #rotate
                        imgaug.augmenters.Sometimes(0.5,imgaug.augmenters.Affine(shear=(-10, 10)))])  #Shear
        
        # Just flip L/R
        augmentation1 = imgaug.augmenters.Fliplr(0.5)  # Flip half of images horizontally
                        
        # Flip horizontal and vertical
        augmentation2 = imgaug.augmenters.Sequential([imgaug.augmenters.Fliplr(0.5),
                        imgaug.augmenters.Flipud(0.5)])
        
        # Just Rotate
        augmentation3 = imgaug.augmenters.Affine(rotate=(-20, 20))  #rotate
            
        # Just Shear
        augmentation4 = imgaug.augmenters.Affine(shear=(-10, 10))  #Shear
        
        # rotate and shear
        augmentation5 = imgaug.augmenters.Sequential([imgaug.augmenters.Affine(rotate=(-20, 20)),
                        imgaug.augmenters.Affine(shear=(-10, 10))])

        # *** This training schedule is an example. Update to your needs ***
        augmentations = augmentation0
        # Training - Stage 1 Network heads only
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=20, #40
                    layers='heads',
                    augmentation=augmentations)
        
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40, #40
                    layers='heads',
                    augmentation=augmentations)

        # Training - Stage 2 Resnet stages 4 and up
        print("Train Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=60, #120
                    layers='4+',
                    augmentation=augmentations)

        print("Train Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=80, #160
                    layers='4+',
                    augmentation=augmentations)
        
        print("Train Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=100, #120
                    layers='4+',
                    augmentation=augmentations)

        print("Train Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120, #120
                    layers='4+',
                    augmentation=augmentations)

        # Training - Stage 3 Fine tuning of all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=140, #140
                    layers='all',
                    augmentation=augmentations)
        
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 100,
                    epochs=160, #160
                    layers='all',
                    augmentation=augmentations)
        
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 100,
                    epochs=180, #180
                    layers='all',
                    augmentation=augmentations)
        
        # Save weights
        # Typically not needed because callbacks save after every epoch
        # Uncomment to save manually
        #model_path = os.path.join(DEFAULT_LOGS_DIR, "mask_rcnn_raspberry1.h5")
        #model.keras_model.save_weights(model_path)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = RaspberryDataset()
        raspberry = dataset_val.load_raspberry(args.dataset, "test")
        dataset_val.prepare()
        print("Running RASP evaluation on {} images.".format(args.limit))
        evaluate_raspberry(model, dataset_val, raspberry, "segm", limit=100)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
