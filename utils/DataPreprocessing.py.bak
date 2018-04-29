import random
import os
from skimage.io import imread
import numpy as np


def mask2onehot(label_path):
    class_idx = [33, 34, 35, 36, 38, 39, 40]
    # class_dict = {'car':33,
    #               'motorbicycle':34,
    #               'bicycle':35,
    #               'person':36,
    #               'truck':38,
    #               'bus':39,
    #               'tricycle':40}

    label = np.asarray(imread(label_path)) // 1000
    one_hot_label = []
    other_label = np.zeros(shape=(2710, 3384)).astype(bool)

    # 8 figures will popup
    for idx in class_idx:
        mask = label == idx
        other_label += mask
        mask = mask.astype(int)
        mask = mask.reshape(mask.shape + (1,))
        one_hot_label.append(mask)

    other_label = (1 - other_label.astype(int))
    other_label = other_label.reshape(other_label.shape + (1,))

    return one_hot_label, other_label


def load_paths(data_dir, label_dir):
    data_path_list = []
    for path in os.listdir(data_dir):
        label_name = os.path.splitext(path)[0] + "_instanceIds.png"
        data_path = data_dir + path
        label_path = label_dir + label_name
        data_path_list.append((data_path, label_path))

    return data_path_list


def dataset_split(data_dir, label_dir, valid_train_rate = 0.1, shuffle = True, seed = None):
    data_list = load_paths(data_dir, label_dir)
    if shuffle:
        if seed is not None:
            random.seed(seed)
            random.shuffle(data_list)
        else:
            random.shuffle(data_list)

    validation_patient_num = int(valid_train_rate * len(data_list))
    print('validation_patient_num', validation_patient_num)
    validation_path = [data_list[i] for i in range(validation_patient_num)]
    train_patient_num = len(data_list) - validation_patient_num
    print('train_patient_num', train_patient_num)
    train_path = [data_list[validation_patient_num + i] for i in range(train_patient_num)]

    return train_path, validation_path


def load_data(data_path):
    img = imread(data_path)
    return img


def merge_labels(label_path):
    one_hot_label, other_label = mask2onehot(label_path)
    labels = np.concatenate(one_hot_label, axis = -1)
    labels = np.concatenate([labels, other_label], axis = -1)

    return labels


def normalize(data, range):
    return (data - np.min(data))*(range[1] - range[0])/(np.max(data) - np.min(data)) + range[0]