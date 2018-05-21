

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import warnings

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

#extend the father class Config
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 4 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # background + 7 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8*4, 16*4, 32*4, 64*4, 128*4)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    #TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 4000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 200
    
    iter_num=0


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

class WadDataset(utils.Dataset):
    def load_mask(self,image_id):
        """
        extract the mask of each object 
        """
        class_idx = [33, 34, 35, 36, 38, 39, 40]
        '''
        class_dict = {'car':33,
                      'motorbicycle':34,
                      'bicycle':35,
                      'person':36,
                      'truck':38,
                      'bus':39,
                      'tricycle':40}
        '''
        info=self.image_info[image_id]
        mask=np.asarray(Image.open(info['mask_path']))
        index_label=np.unique(mask)[1:]
        class_ids=[]
        #print(index_label)
        mask_set=[]#np.zeros((mask.shape[0],mask.shape[1],len(index_label)))
        index=0
        for idx in index_label:
            if idx//1000 not in class_idx:
                continue
            else:
                class_ids.append(class_idx.index(idx//1000)+1)
                label = mask == idx
                label = label.astype(int)
                label = scipy.misc.imresize(label,[800,1024])
                label = label >0
                label=label.reshape(label.shape+(1,))
                mask_set.append(label)
                index+=1
        mask_set=np.concatenate(mask_set,axis=2)
        class_ids=np.array(class_ids)
        return mask_set.astype(bool),class_ids.astype(np.int32)
    def load_shapes(self, count, height, width, img_floder, mask_floder, imglist):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "car")
        self.add_class("shapes", 2, "motorbicycle")
        self.add_class("shapes", 3, "bicycle")
        self.add_class("shapes", 4, "person")
        self.add_class("shapes", 5, "truck")
        self.add_class("shapes", 6, "bus")
        self.add_class("shapes", 7, "tricycle")
        for i in range(count):
            filestr = imglist[i][:-4]+'_instanceIds'
            mask_path = mask_floder + "/" + filestr + ".png"
            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=width, height=height, mask_path=mask_path)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        
        info = self.image_info[image_id]
        image=cv2.imread(info['path'])
        image=scipy.misc.imresize(image,[800,1024])
        '''
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        '''
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y-s, x-s, y+s, x+s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes

############################################################
#  Training
############################################################
os.environ['CUDA_VISIBLE_DEVICES'] ='4,5'
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    #set up all the Config
    config = ShapesConfig()
    config.display()
    
    #train_address
    img_floder='/net/pasnas01/pool1/liuyn/competitions/data/train_color'
    mask_floder='/net/pasnas01/pool1/liuyn/competitions/data/train_label'
    imglist=[]
    for i in os.listdir(img_floder):
        imglist.append(i)

    # Validation address
    img_val_floder='/net/pasnas01/pool1/liuyn/competitions/data/val_color'
    mask_val_floder='/net/pasnas01/pool1/liuyn/competitions/data/val_label'
    vallist=[]
    for i in os.listdir(img_val_floder):
        vallist.append(i)

    # Training dataset
    dataset_train = WadDataset()
    dataset_train.load_shapes(len(imglist),config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1],img_floder,mask_floder,imglist)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = WadDataset()
    dataset_val.load_shapes(len(vallist), config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1],img_val_floder,mask_val_floder,vallist)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,model_dir=MODEL_DIR)

        # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
    

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=50, 
                layers='heads')
