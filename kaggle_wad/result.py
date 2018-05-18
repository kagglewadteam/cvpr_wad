import os
import sys
import pandas as pd
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

"""
This module is for change format of the prediction result from mask rcnn into the submission format
"""
"""
The prediction result format:
results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        
        
Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
"""

"""
The submission result format:
ImageId is the file name,
LabelId is the class of that object (car, person, etc),
Confidence is your confidence of the prediction,
PixelCount is the total number of pixels in that object (this is to help our evaluation execution speed),
EncodedPixels is Run-length encoded, with each pair delimited by |, for example: 1 3|10 5| 
"""


def count_mask_pixels(mask):
    return np.sum(mask)

def run_length_encoding(mask): #2D mask
    str = ""
    mask_array = mask.flatten()
    print(mask_array)
    res = np.where(mask_array == 1)[0]
    print(res)
    start = res[0]
    length = 1
    for i in range(1, res.shape[0]):
        if res[i] == res[i - 1] + 1:
            length += 1
        else:
            str = str + "{} {}|".format(start, length)
            print("str:", str)
            length = 1
            start = res[i]
    str = str + "{} {}|".format(start, length)
    return str

def get_csv_result(result):
    class_names=['BG','car','motorbicycle','bicycle','person','truck','bus','tricycle']
    csv_rows = []    #dict in a list, can use pd data frame
    """
    one dict item is just one object instance
    csv_row = [{"ImageId" : aabcdefg, "LabelId":xxx, "Confidence":xxx, "PixelCount" : xxx, "EncodedPixels" : xxx},
               {"ImageId" : aabcdefg, "LabelId":xxx, "Confidence":xxx, "PixelCount" : xxx, "EncodedPixels" : xxx}, ...]
    """
    for j in range(len(result)):
        item=result[j][0]
        for i in range(item["rois"].shape[0]):
            row_item_dict = {}
            row_item_dict["ImageId"] = item["img_id"]
            row_item_dict["LabelId"] = class_names[item["class_ids"][i]]
            row_item_dict["Confidence"] = item["scores"][i]
            #TODO check the mask implementation
            mask = item["masks"][:,:,i]
            mask = mask.reshape(mask.shape[:2])
            row_item_dict["PixelCount"] = count_mask_pixels(mask)
            row_item_dict["EncodedPixels"] = run_length_encoding(mask)
            csv_rows.append(row_item_dict)

    return csv_rows

def write_csv(result):
    csv_rows = get_csv_result(result)
    df_res = pd.DataFrame(csv_rows)
    df_res = df_res[["ImageId", "LabelId", "PixelCount", "Confidence", "EncodedPixels"]]
    df_res.to_csv("mask_rcnn.csv", index = None)

