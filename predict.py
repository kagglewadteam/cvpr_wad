from keras.models import load_model
from model.UNET import *
import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from keras import backend as K


os.environ['CUDA_VISIBLE_DEVICES'] ='1'
for cls in range(8):
    network = create_model()
    network.load_weights('/net/pasnas01/pool1/liuyn/competitions/models/UNET_class'+str(cls+1)+'.h5')
    y=[]
    for index,img in enumerate(os.listdir("/home/liuyn/masterthesis/kaggle_sampledata/new/val_color/image")):
        if index==500:
           break
        else:
            y.append(cv2.imread("/home/liuyn/masterthesis/kaggle_sampledata/new/val_color/image/"+str(img)))
    y=np.array(y)
    #y= cv2.imread("/home/liuyn/masterthesis/kaggle_sampledata/new/val_color/image/170908_061630238_Camera_6_(4, 1).jpg")
    y_pred = network.predict(y)
    y_pred=y_pred.astype(int)
    print(np.unique(y_pred))
    cmap1= matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',['black','red'],256)
    for i in range(y_pred.shape[0]):
        #for j in range(8):
        if y_pred[i,:,:,0].any() == 1:
           y_new = y_pred[i,:,:,0]
           print(np.unique(y_new))
           print(np.sum(y_new))
           plt.figure()
           plt.imshow(y_new,cmap='gray')
           plt.show()
        else:
           continue
    K.clear_session()
'''
test = np.asarray(Image.open("/home/liuyn/masterthesis/kaggle_sampledata/new/val_label/label/170908_061545080_Camera_6_(4, 4)_instanceIds.png"))//1000
#print(np.unique(test))
class_indx =[33,34,35,36,38,39,40]
for ind in class_indx:
    mask = test == ind
    mask = mask.astype(int)
    print(np.sum(mask))
    print(np.unique(mask))
    if mask.any() == 1:
       plt.figure()
       plt.imshow(mask,cmap='gray')
       plt.show()
    else:
       continue

'''
