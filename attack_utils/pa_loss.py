import torch
import numpy as np
from opencood.utils import box_utils
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
#from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

# Utility functions to obtain the 3D bboxes and scores (prior work loss defined on IoU and cls scores)
def delta_to_boxes3d(deltas, anchors, channel_swap=True):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, W, L, 14)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr
        channel_swap : bool
            Whether to swap the channel of deltas. It is only false when using
            FPV-RCNN

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        if channel_swap:
            deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        else:
            deltas = deltas.contiguous().view(N, -1, 7)

        boxes3d = torch.zeros_like(deltas)
        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                      anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d

def convert_to_bbox_score(data_dict, output_dict):
    pred_box2d_list = []

    for cav_id, cav_content in data_dict.items():
            #assert cav_id in output_dict
            #print(output_dict.keys())
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix']

            # (H, W, anchor_num, 7)
            anchor_box = cav_content['anchor_box']

            # classification probability
            prob = output_dict[cav_id]['psm']
            prob = F.sigmoid(prob.permute(0, 2, 3, 1))
            prob = prob.reshape(1, -1)

            # regression map
            reg = output_dict[cav_id]['rm']

            # convert regression map back to bounding box
            # (N, W*L*anchor_num, 7)
            batch_box3d = delta_to_boxes3d(reg, anchor_box)
            mask = \
                torch.gt(prob, 0.10)
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0],
                                          mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = \
                    box_utils.boxes_to_corners_3d(boxes3d,
                                                  order='hwl')
                # (N, 8, 3)
                projected_boxes3d = \
                    box_utils.project_box3d(boxes3d_corner,
                                            transformation_matrix)
                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                pred_box2d_list.append(boxes2d_score)
            
            # shape: (N, 5)
            if pred_box2d_list != []:
                pred_box2d_list = torch.vstack(pred_box2d_list)
                return pred_box2d_list
            else:
                 return []
    


class targetedLoss(nn.Module):
    def __init__(self):
        super(targetedLoss, self).__init__()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, data_dict):
        """
        input: output given by the model, before postprocessing
        target: targeted_bboxes: (max_num, 4) - four corners x1y1x2y2 representation or (max_num 7)
        """
        pred_boxes = convert_to_bbox_score(data_dict, input)
        if pred_boxes == []:
             return None
        
        """
        Creation loss (Default): sum IoU(pred, target) * (1 - log (score_of_pred))
        Removal loss: - Creation loss
        """
        #print(target.shape)
        #assert target.shape[1]==7 or target.shape[1]==4
        if target.shape[0] == 7:
            target = target.unsqueeze(0)
        if target.shape[1] == 7:
            # (N, 8, 3)
            boxes3d_corner = \
                box_utils.boxes_to_corners_3d(target,
                                                order='hwl')
            # (N, 8, 3)
            #transformation_matrix = data_dict['ego']['transformation_matrix']
            #projected_boxes3d = \
            #    box_utils.project_box3d(boxes3d_corner,
            #                            transformation_matrix)
            # convert 3d bbx to 2d, (N,4)
            projected_boxes2d = \
                box_utils.corner_to_standup_box_torch(boxes3d_corner)

        scores = pred_boxes[:, 4].T
        scores = torch.log(1. - scores)

        ious = self.iou2d_batch(pred_boxes, projected_boxes2d)
        
        loss = torch.sum(torch.matmul(scores, ious))
        return loss

        
    
    def iou2d_batch(self, pred_boxes: torch.Tensor, target: torch.Tensor):
        bb_gt = torch.unsqueeze(target, 0)
        bb_test = torch.unsqueeze(pred_boxes, 1)

        xx1 = torch.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = torch.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = torch.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = torch.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = torch.maximum(torch.zeros(xx1.shape, device="cuda:0"), xx2 - xx1)
        h = torch.maximum(torch.zeros(yy1.shape, device="cuda:0"), yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
            + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)   

        assert o.shape[0] == pred_boxes.shape[0]
        assert o.shape[1] == target.shape[0]

        
        return o