import numpy as np
#import pickle
import re
import random
import os
from PIL import Image
from skimage.io import imread,imsave
import cv2
from tqdm import tqdm
import shutil

def patch_split(base_dir,flag,save_dir,img_size=384):
        #imgs=[]
        #labels=[]
        for img in tqdm(os.listdir(base_dir+flag+"_color/image")):
            image=imread(base_dir+flag+"_color/image/"+img)
            label=np.asarray(Image.open(base_dir+flag+"_label1/label/"+img[:-4]+"_instanceIds.png"))
            #print(np.unique(label))
            x,y,z=image[:,:,:3].shape
            coords_x = x /img_size
            coords_y = y/img_size
            coords = [ (q,r) for q in range(int(coords_x)) for r in range(int(coords_y)) ]
            for coord in coords:
                imgs=image[coord[0]*img_size:(coord[0]+1)*img_size,coord[1]*img_size:(coord[1]+1)*img_size,:]
                labels=label[coord[0]*img_size:(coord[0]+1)*img_size,coord[1]*img_size:(coord[1]+1)*img_size]
                tmp=labels
                if np.unique(tmp//1000).any()==0:
                   continue
                else:
                   imsave(save_dir+flag+"_color/image/"+img[:-4]+"_"+str(coord)+".jpg",imgs)
                   result=Image.fromarray(labels)
                   result.save(save_dir+flag+"_label/label/"+img[:-4]+"_"+str(coord)+"_instanceIds.png")


patch_split("/home/liuyn/masterthesis/kaggle_sampledata/",flag="train",save_dir="/home/liuyn/masterthesis/kaggle_sampledata/new/")
print("train_patch complicated")
patch_split("/home/liuyn/masterthesis/kaggle_sampledata/",flag="val",save_dir="/home/liuyn/masterthesis/kaggle_sampledata/new/")
print("val_patch complicated")
		
		
