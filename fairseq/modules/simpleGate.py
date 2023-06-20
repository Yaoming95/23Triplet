# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.incremental_decoding_utils import with_incremental_state
from torch import nn

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m





@with_incremental_state
class SimpleGate(nn.Module):
    def __init__(self, vdim, intermediate_dim, output_dim, qkv_bias=True, attn_drop=0.):
        super().__init__()
        self.num_heads = 1

        self.vdim = vdim
        self.output_dim = output_dim
        self.intermediate_dim = intermediate_dim

        self.v_proj = Linear(vdim, intermediate_dim, bias=qkv_bias)   
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(intermediate_dim, output_dim)

    def forward(self, query, value, key_padding_mask=None):
        # _, batch_size, value_dim = value.shape
        text_length, batch_size, _ = query.shape
        v = self.v_proj(value)
        v = v.contiguous().transpose(0, 1)
        fake_attn = torch.ones([batch_size, text_length, 1]).to(v.device)
        # fake_attn = self.attn_drop(fake_attn)
        x = (fake_attn @ v)
        x = x.transpose(0, 1).contiguous()
        x = self.proj(x)
        return x


