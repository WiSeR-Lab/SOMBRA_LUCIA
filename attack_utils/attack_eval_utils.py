import numpy as np
import pandas as pd
from opencood.utils import common_utils

def update_dict(parent_dict, child_dict):
    """
    Append every non-dict entry in child_dict at same locations in parent dict, into a list
    """
    for key, item in child_dict.items():
        if key not in parent_dict:
            if isinstance(item, dict):
                parent_dict[key] = {}
                update_dict(parent_dict[key], item)
            else:
                parent_dict[key] = [item]
        else:
            if isinstance(item, dict):
                update_dict(parent_dict[key], item)
            else:
                parent_dict[key].append(item)


def dict_to_dataframe(data_dict):
    # Initialize an empty dictionary to hold the concatenated data
    concatenated_data = {}
    # Iterate over each key-value pair in the dictionary
    for parent_key, child_dict in data_dict.items():
        # Iterate over each key-value pair in the child dictionary
        for child_key, value_list in child_dict.items():
            # Create the new column name by concatenating the parent and child keys
            column_name = f"{parent_key}_{child_key}"
            # Add the value list to the concatenated data dictionary with the new column name
            concatenated_data[column_name] = value_list
    # Create a DataFrame from the concatenated data dictionary
    df = pd.DataFrame(concatenated_data)
    return df



def update_time_dict(time_dict, result_dict):
    """
    time_dict: dict that stores all running time {'inference':[0.1,0.1,...], ...}
    result_dict: result from one run {'inference': 0.1, 'postprocess': 1.0, ...}
    """
    for phase, time_taken in result_dict.items():
        if phase not in time_dict:
            time_dict[phase] = [time_taken]
        else:
            time_dict[phase].append(time_taken)


def removal_sucess(det_boxes, det_score, target_bboxes, iou_thresh=0.3):
    """
    Determine if the targeted removal is successful
    """
    if det_boxes is None:
        return True
    else:
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        target_box = common_utils.torch_tensor_to_numpy(target_bboxes)

        # sort the prediction bounding box by score
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        target_polygon_list = list(common_utils.convert_format(target_box))

        for i in range(det_boxes.shape[0]):
            det_polygon = det_polygon_list[i]
            ious = common_utils.compute_iou(det_polygon, target_polygon_list)

            if np.max(ious) > iou_thresh:
                return False
        
    return True

