import os
import sys
import random
import math
import re
import time
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
from train import ShapesConfig,WadDataset,get_ax
from tqdm import tqdm

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
#test_floder='/home/liuyn/masterthesis/kaggle_sampledata/test'
test_floder='D:/LAB/cvpr_data/test'

img_ids = next(os.walk(test_floder))[2]
print("test samples has ",len(img_ids))

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
#from mrcnn.model import log
import result

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_shapes_0022.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(MODEL_PATH):
    print("wrong")#utils.download_trained_weights(MODEL_PATH)


#detection start

os.environ['CUDA_VISIBLE_DEVICES'] ='1'

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
inference_config.display()


# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)
# Load trained weights (fill in path to trained weights here)
model.load_weights(MODEL_PATH, by_name=True)

# testing dataset
for i in tqdm(range(len(img_ids))):
    image=skimage.io.imread(os.path.join(test_floder, img_ids[i]))
    results=model.detect([image], verbose=1)
    results[0]['img_id']=img_ids[i]
    #write to csv
    result.write_csv(results,i)
'''
visulizing data
results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],dataset_test.class_names, r['scores'], ax=get_ax())
'''