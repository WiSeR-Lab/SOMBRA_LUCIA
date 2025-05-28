import os
from collections import OrderedDict

import numpy as np
import torch, time

from opencood.utils.common_utils import torch_tensor_to_numpy

def get_empty_target(batch_data, post_processor):
    """
    Return target label dict and related info augmented with an spoofing target
    Input:
        batch_dat
    Output:
        target_label_dict
        target_bbx_center
        target_object_ids
    """

    """
    batch_data['ego']['target_label_dict'] = target_data['ego']['label_dict']
    batch_data['ego']['target_bbx_center'] = target_data['ego']['object_bbx_center']
    batch_data['ego']['target_object_bbox'] = target_data['ego']['target_object_bbox']
    """

    target_bbox_center = np.zeros_like(batch_data['ego']['object_bbx_center'].detach().clone().cpu()[0])
    target_bbox_mask = np.zeros_like(batch_data['ego']['object_bbx_mask'].detach().clone().cpu()[0])

    target_bbox_center[-1,...] = [0, 0, 0, 0.1, 0.1, 0.1, 0]
    target_bbox_mask[-1] = 1

    target_label_dict = post_processor.generate_label(gt_box_center=target_bbox_center, 
                                                      mask=target_bbox_mask)

    batch_data['ego']['label_dict'] = post_processor.collate_batch([target_label_dict])


def inference_intermediate_fusion_attack(data_sync, model, dataset, loss_fn, iter, lr, criterion=None, cav_max=None, async_data=None, attacker_index=1, defender=None):
    """

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset
    loss_fn: corresponding attention loss
    iter: number of iterations
    lr: learning rate
    criterion: loss function for object removal
    cav_max: maximum number of CAVs that the attacker knows (if 2, then it's only the attacker and the victim)
    async_data: the dataset attacker used for perturbation generation
    attacker_index: the index of the attacker in the CAV list
    defender: whether Lucia is included

    """

    if async_data is not None:
        batch_data = async_data
    else:
        batch_data = data_sync

    output_dict = OrderedDict()
    for cav_id, cav_content in batch_data.items():
        # Attack happens here
        if cav_id == 'ego':
            
            assert cav_content['cav_num'] >= 2
            #print(cav_content['pairwise_t_matrix'].shape)
            voxel_feature_dict = {'voxel_features':cav_content['processed_lidar']['voxel_features'],
                  'voxel_coords':cav_content['processed_lidar']['voxel_coords'],
                  'voxel_num_points':cav_content['processed_lidar']['voxel_num_points'],
                  'record_len': torch.tensor([2]).to('cuda:0'),
                  'attacker_index': attacker_index,
                  'pairwise_t_matrix': cav_content['pairwise_t_matrix'][0,:2,:2,...].unsqueeze(0)}
            torch.cuda.empty_cache()
            # Get Pillar feature in the dict
            model.pillar_vfe(voxel_feature_dict)
            # Get scattered pillar feature in the dict
            model.scatter(voxel_feature_dict)
            # Initialize the perturbation
            spatial_feature = torch.clone(voxel_feature_dict['spatial_features']).detach() # limited attacker knowledge
            perturbation = torch.zeros_like(spatial_feature[attacker_index], requires_grad=True).to('cuda:0')
            
            for i in range(iter):
                voxel_feature_dict['spatial_features'] = spatial_feature[:cav_max] # limited attacker knowledge
                voxel_feature_dict['spatial_features'][attacker_index] = spatial_feature[attacker_index] + perturbation

                output_dict = {'ego':model(voxel_feature_dict)}
                target_loss = criterion(output_dict, batch_data, batch_data)

                if target_loss is None: #Empty prediction already, if using PA loss
                    break

                loss = loss_fn(model) if loss_fn is not None else 0
                alpha=1.0
                total_loss = alpha*target_loss + 1.0*loss 
                total_loss.backward(retain_graph=True)
                assert perturbation.grad.data is not None
                perturbation = perturbation.clone().detach() + lr*perturbation.grad.data.sign()
                perturbation.requires_grad = True

            torch.cuda.empty_cache()
            model.pillar_vfe(data_sync['ego']['processed_lidar'])
            model.scatter(data_sync['ego']['processed_lidar'])

            voxel_feature_dict['spatial_features'] = data_sync['ego']['processed_lidar']['spatial_features']
            adv_feature = spatial_feature[attacker_index] + perturbation
            voxel_feature_dict['spatial_features'][attacker_index] = adv_feature
            if defender is not None:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                start = time.time()
                trust_score = defender(voxel_feature_dict['spatial_features'])
                torch.cuda.synchronize()
                defense_time = time.time() - start
                #print(f"trust score: {trust_score}")
                #torch.cuda.empty_cache()
            else:
                trust_score = None
            #else:
            voxel_feature_dict['record_len'] = data_sync['ego']['record_len']
            voxel_feature_dict['pairwise_t_matrix'] = data_sync['ego']['pairwise_t_matrix']

            output_dict[cav_id] = model(voxel_feature_dict, trust_score)
        else:
            output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(data_sync,
                             output_dict)

    if defender is not None:
        return pred_box_tensor, pred_score, gt_box_tensor, torch.clone(adv_feature).detach(), trust_score, defense_time
    else:
        return pred_box_tensor, pred_score, gt_box_tensor, torch.clone(adv_feature).detach(), None, None
    
def get_attn(model, model_name='AttentiveFusion'):
    """
    Get the attention weight of the model, currently only implemented for AttentiveFusion and the first layer
    Output:
        dict: {0: attn_weight, 1: attn_weight, ...}
    """
    attn_dict = {}
    if hasattr(model, 'model_name'):
        model_name = model.model_name
    if model_name == 'AttentiveFusion':
        for i, layer in enumerate(model.backbone.fuse_modules):
            if hasattr(layer, 'attn_score'):
                #print(f"shape of attn: {layer.attn_score.shape}")
                attn_dict[i] = layer.attn_score[0,...]
    elif model_name == 'Where2comm':
        for i, layer in enumerate(model.fusion_net.fuse_modules):
            if hasattr(layer, 'attn_score'):
                #print(f"shape of attn: {layer.attn_score.shape}")
                attn_dict[i] = layer.attn_score[0,...]
    elif model_name == 'CoAlign':
        for i, layer in enumerate(model.fusion_net):
            if hasattr(layer, 'attn_score'):
                #print(f"shape of attn: {layer.attn_score.shape}")
                attn_dict[i] = layer.attn_score[0,...]
    elif model_name == 'V2VAM':
        attn_dict[0] = 1 - torch.mean(model.fusion_net.CCNet.scores)
    return attn_dict