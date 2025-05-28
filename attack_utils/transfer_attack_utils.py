from collections import OrderedDict

import numpy as np
import torch, time, random
import torch.nn.functional as F
from opencood.utils import eval_utils
from attack_utils.inference_attack_utils import get_attn


def add_tensors_with_fixed_width(tensor1, tensor2):
    """
    Adds two tensors by resizing both tensors to a fixed target width using nearest neighbor interpolation.
    
    Parameters:
        tensor1 (torch.Tensor): Tensor of size (C, W1, H)
        tensor2 (torch.Tensor): Tensor of size (C, W2, H)
        target_width (int): The target width (W) that both tensors will be resized to.
        
    Returns:
        torch.Tensor: The result of adding the resized tensors with shape (C, target_width, H).
    """
    
    # Get the shapes of both tensors
    C1, W1, H1 = tensor1.shape
    C2, W2, H2 = tensor2.shape
    
    if H1 != H2 or C1 != C2:
        raise ValueError(f"Tensors must have the same height and channel dimensions, "
                         f"but got {tensor1.shape} and {tensor2.shape}")
    
    # Resize tensor1 to match the target width (C, target_width, H) using nearest neighbor interpolation
    #tensor1_resized = F.interpolate(tensor1.unsqueeze(0), size=(target_width, H1), mode='nearest').squeeze(0)
    
    # Resize tensor2 to match the target width (C, target_width, H) using nearest neighbor interpolation
    if W1 != W2:
        tensor2_resized = F.interpolate(tensor2.unsqueeze(0), size=(W1, H2), mode='bilinear').squeeze(0)
    else:
        tensor2_resized = tensor2
    
    # Add the resized tensors together
    result = tensor1 + tensor2_resized
    
    return result

def mor_varying_cav(batch_data, results, model, dataset, adv_feature, target_bbox=None, cav_max=None, attacker_index=1, defender=None):
    output_dict = OrderedDict()
    cav_nums = []
    for cav_id, cav_content in batch_data.items():
        # Attack happens here
        if cav_id == 'ego':
            
            cav_num = cav_content['cav_num']
            
            total_voxel_feature_dict = {'voxel_features':cav_content['processed_lidar']['voxel_features'],
                  'voxel_coords':cav_content['processed_lidar']['voxel_coords'],
                  'voxel_num_points':cav_content['processed_lidar']['voxel_num_points'],
                  'record_len': cav_content['record_len'],
                  'pairwise_t_matrix': cav_content['pairwise_t_matrix']}
            torch.cuda.empty_cache()
            # Get Pillar feature in the dict
            model.pillar_vfe(total_voxel_feature_dict)
            # Get scattered pillar feature in the dict
            model.scatter(total_voxel_feature_dict)
            total_voxel_feature_dict['spatial_features'][attacker_index] = adv_feature

            max_cav = min(cav_num, cav_max) if cav_max is not None else cav_num
            for total_cav in range(2,max_cav):
                if total_cav not in results:
                    results[total_cav] = {'asr_0': [], 'asr_1': [], 'orr': [], 'attn': []}
                sampled_indices = sample_indices(max_cav-1, total_cav)
                voxel_feature_dict = {'spatial_features': torch.index_select(total_voxel_feature_dict['spatial_features'], 0, torch.tensor(sampled_indices).to('cuda:0')),
                      'record_len': torch.tensor([total_cav]).to('cuda:0'),
                      'pairwise_t_matrix': cav_content['pairwise_t_matrix']}
            
                if defender is not None:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    start = time.time()
                    trust_score = defender(voxel_feature_dict['spatial_features'])
                    torch.cuda.synchronize()
                    defense_time = time.time() - start
                else:
                    trust_score = None
                
                output_dict[cav_id] = model(voxel_feature_dict, trust_score)
                pred_box_tensor, pred_score, gt_box_tensor = \
                    dataset.post_process(batch_data,
                                        output_dict)
                
                cav_nums.append(total_cav)
                if target_bbox is not None:
                    num_detected = eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, target_bbox, None, 0, write=False)
                    results.append(num_detected)
                else:
                    pred_num = 0 if pred_box_tensor is None else len(pred_box_tensor)
                    gt_num = 0 if gt_box_tensor is None else len(gt_box_tensor)
                    orr = 1 - min(pred_num, gt_num) / gt_num
                    results[total_cav]['orr'].append(orr)
                    results[total_cav]['asr_0'].append(pred_num <= 0)
                    results[total_cav]['asr_1'].append(pred_num <= 1)
                    try:
                        attn = torch.mean(get_attn(model)[0][1])
                    except:
                        attn = torch.mean(get_attn(model)[0])
                    results[total_cav]['attn'].append(attn.item())
                    
                
                
def sample_indices(max_cav, num_samples):
    """
    Randomly sample indices up to max_cav (inclusive), ensuring index 1 is always included.

    Parameters:
    ----------
    max_cav : int
        The maximum index value (inclusive).
    num_samples : int
        The total number of indices to sample, including index 1.

    Returns:
    -------
    list
        A list of sampled indices.
    """
    if num_samples > max_cav:
        raise ValueError("num_samples cannot be greater than max_cav.")

    # Ensure index 0, 1 is included
    sampled_indices = {0, 1}

    # Randomly sample the remaining indices
    remaining_indices = set(range(2, max_cav + 1))
    sampled_indices.update(random.sample(remaining_indices, num_samples - 2))

    return sorted(sampled_indices)