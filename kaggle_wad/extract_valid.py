import numpy as np
#import pickle
import re
import random
import os
from PIL import Image
from skimage.io import imread,imsave
import shutil
from tqdm import tqdm

img_floder='/net/pasnas01/pool1/liuyn/competitions/cvpr-2018-autonomous-driving/train_color'
mask_floder='/net/pasnas01/pool1/liuyn/competitions/cvpr-2018-autonomous-driving/train_label'

save_train_img_dir='/net/pasnas01/pool1/liuyn/competitions/data/train_color'
save_train_mask_dir='/net/pasnas01/pool1/liuyn/competitions/data/train_label'
save_val_img_dir='/net/pasnas01/pool1/liuyn/competitions/data/val_color'
save_val_mask_dir='/net/pasnas01/pool1/liuyn/competitions/data/val_label'
class_idx = [33, 34, 35, 36, 38, 39, 40]

    #clean out the non related image
img_list = []
for img in tqdm(os.listdir(img_floder)):
    img_list.append(img)

random.seed(1)
random.shuffle(img_list)
length = np.int(len(img_list)*0.1)
test_set = img_list[:length]
train_set = img_list[length:]

print("train list start")
for i in tqdm(train_set):
    label=i[:-4]+'_instanceIds.png'
    cls=np.unique(np.asarray(Image.open(mask_floder+'/'+label))//1000)[1:]
    for j in cls:
        if j in class_idx:
            shutil.copy(img_floder+'/'+i,save_train_img_dir)
            shutil.copy(mask_floder+'/'+label,save_train_mask_dir)
            break

print("test list start")
for i in tqdm(test_set):
    label=i[:-4]+'_instanceIds.png'
    cls=np.unique(np.asarray(Image.open(mask_floder+'/'+label))//1000)[1:]
    for j in cls:
        if j in class_idx:
            shutil.copy(img_floder+'/'+i,save_val_img_dir)
            shutil.copy(mask_floder+'/'+label,save_val_mask_dir)
            break
