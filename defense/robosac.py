import torch
import numpy as np
import random
from opencood.tools import inference_utils
from defense.robosac_util import associate_2_detections
from collections import OrderedDict

random.seed(0)

def cal_robosac_consensus(num_agent, step_budget, num_attackers):
    eta = num_attackers / num_agent
    s = np.floor(np.log(1-np.power(1-0.99, 1/step_budget)) / np.log(1-eta)).astype(int)
    return s

def sample_agents(agent_num, s):
    return random.sample(range(1,agent_num), s)

def get_ego_pred(batch_data, model, dataset):
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    voxel_feature_dict = {'voxel_features':cav_content['processed_lidar']['voxel_features'],
                'voxel_coords':cav_content['processed_lidar']['voxel_coords'],
                'voxel_num_points':cav_content['processed_lidar']['voxel_num_points'],
                'record_len': torch.tensor([1]).to('cuda:0'),
                'pairwise_t_matrix': cav_content['pairwise_t_matrix'][0,:2,:2,...].unsqueeze(0)}
    model.pillar_vfe(voxel_feature_dict)
    # Get scattered pillar feature in the dict
    model.scatter(voxel_feature_dict)
    voxel_feature_dict['spatial_features'] = voxel_feature_dict['spatial_features'][0].unsqueeze(0)
    output_dict['ego'] = model(voxel_feature_dict)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def robosac(batch_data, model, dataset, perturbation, attacker_idx=1, sampling_budget=10):
    agent_num = batch_data['ego']['cav_num']
    s = cal_robosac_consensus(agent_num, sampling_budget, 1)
    s = min(s, agent_num-1)
    #print(f"Selecting {s} agents for collab from {agent_num} agents")
    n = 0
    ego_pred_box_tensor, ego_pred_score, gt_box_tensor = get_ego_pred(batch_data, model, dataset)
    if s == 0:
        return ego_pred_box_tensor, ego_pred_score, gt_box_tensor
    while n < sampling_budget:
        n += 1
        cav_content = batch_data['ego']
        cav_idx = [0] + sample_agents(agent_num, s)
        indices_tensor = torch.tensor(cav_idx).to('cuda:0')
        voxel_feature_dict = {'voxel_features':cav_content['processed_lidar']['voxel_features'],
                  'voxel_coords':cav_content['processed_lidar']['voxel_coords'],
                  'voxel_num_points':cav_content['processed_lidar']['voxel_num_points'],
                  'record_len': torch.tensor([len(cav_idx)]).to('cuda:0'),
                  'pairwise_t_matrix': torch.index_select(torch.index_select(cav_content['pairwise_t_matrix'][0], dim=0, index=indices_tensor),
                                                          dim=1, index=indices_tensor).unsqueeze(0)}
        model.pillar_vfe(voxel_feature_dict)
        # Get scattered pillar feature in the dict
        model.scatter(voxel_feature_dict)
        #attacker_spatial_feature = torch.clone(voxel_feature_dict['spatial_features'][attacker_idx]).detach()
        #voxel_feature_dict['spatial_features'][attacker_idx] = attacker_spatial_feature + perturbation
        voxel_feature_dict['spatial_features'][attacker_idx] = perturbation
        #print(f'indices_tensor: {indices_tensor}')
        voxel_feature_dict['spatial_features'] = torch.index_select(voxel_feature_dict['spatial_features'], dim=0, index=indices_tensor)
        #pred_box_tensor, pred_score, _ = inference_utils.inference_intermediate_fusion(batch_data, model, dataset)
        output_dict = OrderedDict()
        output_dict['ego'] = model(voxel_feature_dict)
        pred_box_tensor, pred_score, _ = dataset.post_process(batch_data, output_dict)
        if associate_2_detections(ego_pred_box_tensor, pred_box_tensor) >= 0.3:
            return pred_box_tensor, pred_score, gt_box_tensor
        elif n == sampling_budget - 1:
            return ego_pred_box_tensor, ego_pred_score, gt_box_tensor