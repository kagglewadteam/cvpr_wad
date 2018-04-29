"""
Instance segmentation idea:

1. do a multi-class semantic segmentation (get by neural network)
postprocessing:
2. use get_regions() to get region mask for each class mask (based on clusters), this only works under the assumption that object group is not considered
3. get instance masks from labels
4. compare each region mask with the instance mask in the same class, set the region to the instance which has a largest IoU with that instance mask

This module does this postprocessing step
"""

import numpy as np
from skimage.io import imread
import scipy.ndimage.measurements as mnts



def get_hardmax_predection(y_pred):
    y_hardmax_pred = np.zeros(y_pred.shape)
    for x, row in enumerate(y_pred):
        for y, col in enumerate(row):
            pos = np.argmax(y_pred[x,y,:])
            y_hardmax_pred[x,y,pos] = 1
    return y_hardmax_pred


def get_instance_map_dict(label_path):
    label = imread(label_path)
    instance_list = np.unique(np.nonzero(label))
    instance_map_dict = {}
    for instance in instance_list:
        instance_mask = np.zeros(label.shape)
        instance_mask[np.where(label == instance)] = 1              #each mask only contains "1" and "0"
        instance_map_dict[instance] = instance_mask

    return instance_map_dict

def get_regions(y_pred,structure):
    """
    # TODO get bounding box can be implemented based on this func
    :param y_pred: this y_pred should be a hardmax map
    :return:
    """

    object_regions_mask = mnts.label(y_pred, structure=structure)[0] # label() will give mask labels based on clusters
    object_regions_list = mnts.find_objects(object_regions_mask) #find_object will find the regions of each label

    return object_regions_list

def get_region_masks(y_pred, structure):
    class_regions_map = {}
    """
    33: [mask1, mask2, ...]
    34: [mask1, ....]
    ...
    40: [mask1, mask2, ...]
    0: [mask1, mask2, ...]  not consider this "other" class
    """
    channel_class_map = {
        0: 33,  # car
        1: 34,  # motorbicycle
        2: 35,  # bicycle
        3: 36,  # person
        4: 38,  # truck
        5: 39,  # bus
        6: 40,  # tricycle
        7: 0  # other
    }

    for i in range(y_pred.shape[-1] - 1):           #not consider other class
        class_mask = y_pred[:,:,i]
        region_list = get_regions(class_mask, structure)
        region_mask_list = []
        for region in region_list:
            mask = np.zeros(class_mask.shape)
            mask[region] = class_mask[region]
            mask_set = (mask, region)
            region_mask_list.append(mask_set)
        class_regions_map[channel_class_map[i]] = region_mask_list

    return class_regions_map

def mask_pixel_count(y_pred):
    count = np.sum(y_pred)
    return count

def compute_mask_iou(y_pred, y_true):
    intersection = np.multiply(y_pred,y_true)
    intersection_count = mask_pixel_count(intersection)
    pred_count = mask_pixel_count(y_pred)
    label_count = mask_pixel_count(y_true)
    iou = intersection_count / (pred_count + label_count - intersection_count)

    return iou


def get_instance_mask(y_pred,
                      label_path,
                      threshold,
                      structure = np.array([[1,1,1],
                                            [1,1,1],
                                            [1,1,1]])):

    y_hardmax = get_hardmax_predection(y_pred)

    instance_mask = np.zeros(y_hardmax.shape[:2])
    # compare instance mask with class mask
    instance_map_dict = get_instance_map_dict(label_path=label_path)
    class_region_masks_map = get_region_masks(y_hardmax, structure)
    for instance_class in class_region_masks_map:
        region_masks = class_region_masks_map[instance_class]
        for region_set in region_masks:
            max = 0
            instance_id = 0
            for key in filter(lambda x: x // 1000 == instance_class, instance_map_dict.keys()):
                temp = compute_mask_iou(region_set[0], instance_map_dict[key])      #region_set[0] represents the mask, region_set[1] represents the region slice
                if temp > max and temp >= threshold:
                    max = temp
                    instance_id = key

            instance_mask[np.where(region_set[0] == 1)] = instance_id

    return instance_mask



