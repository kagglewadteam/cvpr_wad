import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

img_name = '170908_072650121_Camera_5'
l_name = img_name + '_instanceIds'
class_idx = [33, 34, 35, 36, 38, 39, 40]
    # class_dict = {'car':33,
    #               'motorbicycle':34,
    #               'bicycle':35,
    #               'person':36,
    #               'truck':38,
    #               'bus':39,
    #               'tricycle':40}

label = np.asarray(imread(l_name + '.png'))//1000
one_hot_label = []
other_label = np.zeros(shape=(2710,3384)).astype(bool)

#8 figures will popup
for idx in class_idx:
    mask = label == idx
    other_label += mask
    mask = mask.astype(int)
    one_hot_label.append(mask)
    plt.figure()
    plt.title('class '+str(idx))
    plt.imshow(mask, cmap='gray')

other_label = (1 - other_label.astype(int))
plt.figure()
plt.title('other class')
plt.imshow(other_label, cmap='gray')
plt.show()






