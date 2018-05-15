import numpy as np
#import pickle
import re
import random
import os
from PIL import Image
from skimage.io import imread,imsave
import shutil
from tqdm import tqdm

img_floder='E:/FILES/data/val_color/image'
mask_floder='E:/FILES/data/val_label1/label'

save_img_dir='D:/LAB/cvpr_data/train/val_color'
save_mask_dir='D:/LAB/cvpr_data/train/val_label'
class_idx = [33, 34, 35, 36, 38, 39, 40]
    #clean out the non related image
for i in tqdm(os.listdir(img_floder)):
    label=i[:-4]+'_instanceIds.png'
    cls=np.unique(np.asarray(Image.open(mask_floder+'/'+label))//1000)[1:]
    for j in cls:
        if j in class_idx:
            shutil.copy(img_floder+'/'+i,save_img_dir)
            shutil.copy(mask_floder+'/'+label,save_mask_dir)
            break