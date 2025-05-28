# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, 
# License: TDG-Attribution-NonCommercial-NoDistrib


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.score = None
        self.attn = None
        self.context = None

    def forward(self, query, key, value, trust_score=None):
        self.score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        if trust_score is not None:
            self.score = self.score * trust_score
        self.attn = F.softmax(self.score, -1)
        if trust_score is not None:
            self.attn = self.attn * trust_score
            self.attn = self.attn / self.attn.sum(dim=-1, keepdim=True)
        self.context = torch.bmm(self.attn, value)
        return self.context


class AttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)
        self.attn_score = None

    def forward(self, x, record_len, trust_score=None):
        split_x = self.regroup(x, record_len)
        C, W, H = split_x[0].shape[1:]
        #print(C)
        #print(W)
        #print(H)
        out = []
        scores = []
        for xx in split_x:
            cav_num = xx.shape[0]
            xx = xx.view(cav_num, C, -1).permute(2, 0, 1)
            h = self.att(xx, xx, xx, trust_score=trust_score)
            #print(h.permute(1, 2, 0).shape)
            h = h.permute(1, 2, 0).view(cav_num, C, W, H)[0, ...]
            score = self.att.attn.permute(1, 2, 0).view(cav_num, cav_num, W, H)[0,...]
            #print(score.shape)
            scores.append(score)
            out.append(h)
        self.attn_score = torch.stack(scores)
        return torch.stack(out)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
