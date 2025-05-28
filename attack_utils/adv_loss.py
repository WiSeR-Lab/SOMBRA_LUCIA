import torch
import torch.nn as nn
import torch.nn.functional as F
from attack_utils.object_removal_loss import objectRemoval, objectRemoval_target
from attack_utils.pa_loss import targetedLoss

def attFusionLoss(model, partial=False, attacker_first=False, device="cuda:0"):
    cav_num = model.backbone.fuse_modules[0].attn_score.shape[1]
    if attacker_first:
        att_i = 0
        vic_i = 1
    else:
        att_i = 1
        vic_i = 0
    if cav_num <= 2 or partial:
        return torch.mean(model.backbone.fuse_modules[0].attn_score[0,att_i,...])-\
            torch.mean(model.backbone.fuse_modules[0].attn_score[0,vic_i,...]).to(device)
    else:
        loss = torch.mean(model.backbone.fuse_modules[0].attn_score[0,att_i,...])-\
            torch.mean(model.backbone.fuse_modules[0].attn_score[0,vic_i,...]).to(device)
        for i in range(2, cav_num):
            loss -= torch.mean(model.backbone.fuse_modules[0].attn_score[0,i,...]).to(device)
        return loss

def v2vamLoss(model, attacker_first=False, device="cuda:0"):
    return torch.mean(model.fusion_net.attn_score_W) + torch.mean(model.fusion_net.attn_score_H).to(device)

def coalignLoss(model, partial=False, attacker_first=False, device="cuda:0"):
    cav_num = model.fusion_net[0].attn_score.shape[1]
    if cav_num <= 2 or partial:
        return torch.mean(model.fusion_net[0].attn_score[0,1,...])-\
            torch.mean(model.fusion_net[0].attn_score[0,0,...]).to(device)
    else:
        loss = torch.mean(model.fusion_net[0].attn_score[0,1,...])-\
            torch.mean(model.fusion_net[0].attn_score[0,0,...]).to(device)
        for i in range(2, cav_num):
            loss -= torch.mean(model.fusion_net[0].attn_score[0,i,...]).to(device)
    return loss

def where2commLoss(model, partial=False, attacker_first=False, device="cuda:0"):
    cav_num = model.fusion_net.fuse_modules[0].attn_score.shape[1]
    if cav_num <= 2 or partial:
        return torch.mean(model.fusion_net.fuse_modules[0].attn_score[0,1,...])-\
            torch.mean(model.fusion_net.fuse_modules[0].attn_score[0,0,...]).to(device)
    else:
        loss = torch.mean(model.fusion_net.fuse_modules[0].attn_score[0,1,...])-\
            torch.mean(model.fusion_net.fuse_modules[0].attn_score[0,0,...]).to(device)
        for i in range(2, cav_num):
            loss -= torch.mean(model.fusion_net.fuse_modules[0].attn_score[0,i,...]).to(device)
        return loss
    
class focalLoss(nn.Module):
    def __init__(self, params, target=False):
        super(focalLoss, self).__init__()
        self.loss = objectRemoval_target(params) if target else objectRemoval(params)
    
    def forward(self, output_dict, dat, batch_dat):
        return -self.loss(output_dict['ego'], dat['ego']['label_dict']) 


class PALoss(nn.Module): # Loss used by prior art
    def __init__(self, targeted=False):
        super(PALoss, self).__init__()
        self.loss = targetedLoss()
        self.targeted = targeted
    
    def forward(self, output_dict, dat, batch_dat):
        if self.targeted: # Only remove the targeted objects
            return self.loss(output_dict, dat['ego']['target_object_bbox'][0], batch_dat) 
        else: # Remove all objects
            return self.loss(output_dict, dat['ego']['object_bbx_center'][0], batch_dat) # we give precise prior knowledge of the target objects
            