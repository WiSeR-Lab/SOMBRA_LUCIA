# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
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

    def __init__(self, dim, temperature=1):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.score = None
        self.attn = None
        self.context = None
        self.temperature = temperature

    def forward(self, query, key, value, trust_score=None):
        self.score = torch.bmm(query[:,0,:].unsqueeze(1), key.transpose(1, 2)) / self.sqrt_dim
        if trust_score is not None:
            self.score = self.score * trust_score
        self.attn = F.softmax(self.score / self.temperature, -1)
        if trust_score is not None:
            self.attn = self.attn * trust_score
            self.attn = self.attn / self.attn.sum(dim=-1, keepdim=True)
        self.context = torch.bmm(self.attn, value)
        return self.context
    
    def set_temperature(self, temperature):
        self.temperature = temperature


class AttFusion(nn.Module):
    def __init__(self, feature_dim, temperature=1):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim, temperature=temperature)
        self.feature_dim = feature_dim
        self.attn_score = None

    def forward(self, x, record_len, trust_score=None):
        split_x = self.regroup(x, record_len)
        C, W, H = split_x[0].shape[1:]
        out = []
        scores = []
        for i, xx in enumerate(split_x):
            cav_num = xx.shape[0]
            xx = xx.view(cav_num, C, -1).permute(2, 0, 1)
            h = self.att(xx, xx, xx, trust_score=trust_score)
            h = h.permute(1, 2, 0).view(1, C, W, H)[0, ...]
            score = self.att.attn.permute(1, 2, 0).view(1, cav_num, W, H)[0,...]
            scores.append(score)
            out.append(h)
        try:
            self.attn_score = torch.stack(scores)
        except:
            pass
        return torch.stack(out)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

