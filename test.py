import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.segmentation import find_boundaries
import scipy.ndimage.measurements as mnts

"""

img_name = '170908_061502825_Camera_5'
l_name = img_name + '_instanceIds'
#class_idx = [33, 34, 35, 36, 38, 39, 40]
    # class_dict = {'car':33,
    #               'motorbicycle':34,
    #               'bicycle':35,
    #               'person':36,
    #               'truck':38,
    #               'bus':39,
    #               'tricycle':40}

#label = np.asarray(imread(l_name + '.png'))//1000
#one_hot_label = []
#other_label = np.zeros(shape=(2710,3384)).astype(bool)
label = np.asarray(imread(l_name + '.png'))
#8 figures will popup
#for idx in class_idx:
mask = label == 33003
#other_label += mask
mask = mask.astype(int)
mask = find_boundaries(mask)
#one_hot_label.append(mask)
plt.figure()
plt.title('class 33000')

plt.show()
"""
a = np.array([[2, 2, 2, 0, 0, 1],
                [2, 2, 2, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0]])
structure = np.array([[1,1,1],
                      [1,1,1],
                      [1,1,1]])
b = mnts.label(a, structure=structure)[0]
print("b",b)
c = mnts.find_objects(b)
print("c:", c)


res_list = []
for i in c:
    res = np.zeros(a.shape)
    res[i] = a[i]

    res_list.append(res)

for k in res_list:
    print(k)