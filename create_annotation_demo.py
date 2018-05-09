from skimage.io import imread
import numpy as np

class_idx = [33, 34, 35, 36, 38, 39, 40]
class_dict = {33: 'car',
              34: 'motorbicycle',
              35: 'bicycle',
              36: 'person',
              38: 'truck',
              39: 'bus',
              40: 'tricycle'}

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def create_annotation(path_to_img, path_to_label):
    # img = np.asarray(imread(data_dir + img_name + '.jpg'))
    ori_label = np.asarray(imread(path_to_label))
    # instance  = np.copy(ori_label)%1000
    label = np.copy(ori_label) // 1000
    annotations = []
    for idx in class_idx:
        mask = label == idx
        masked_label = np.copy(ori_label)
        masked_label[~mask] = 0
        instance_value = np.delete(np.unique(masked_label), 0)

        if len(instance_value) > 0:
            # print instance_value
            for value in instance_value:
                instance_mask = ori_label == value
                temp = np.copy(ori_label)
                temp[~instance_mask] = 0
                ymin, ymax, xmin, xmax = bbox2(temp)
                # print path_to_img + "," + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + "," + \
                #           class_dict[idx]
                annotations.append(path_to_img + "," + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + "," + \
                          class_dict[idx])
    return annotations


if __name__ == '__main__':
    data_dir = 'train/train_color/'
    label_dir = 'train_label/'
    img_name = '170908_072650121_Camera_5'
    l_name = img_name + '_instanceIds'
    path_to_img = data_dir + img_name + '.jpg'
    path_to_label = label_dir + l_name + '.png'
    for s in create_annotation(path_to_img, path_to_label):
        print s
