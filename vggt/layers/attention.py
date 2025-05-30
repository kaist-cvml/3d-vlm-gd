# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def custom_scaled_dot_product_attention(
        self, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor,
        dropout_p: float = 0.0,
        return_attn: bool = False,
        temperature: float = 1.0,
    ):
        # q, k, v의 shape: (B, num_heads, N, head_dim)
        # 먼저 q에 scaling을 적용합니다.
        _, _, N, _ = q.shape
        q = q * self.scale

        # attention score 계산: (B, num_heads, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(scores, dim=-1)
        # dropout 적용 (self.attn_drop.p에 따라)
        if dropout_p > 0.0:
            attn = self.attn_drop(attn)
        output = torch.matmul(attn, v)

        if return_attn:
            scores_1 = torch.matmul(q[..., 5:N//2, :], k[..., N//2+5:, :].transpose(-2, -1))
            # attn_1 = F.softmax(scores_1, dim=-1)
            attn_1 = F.softmax(scores_1 / temperature, dim=-1)
            # attn_1 = torch.matmul(q[..., 5:N//2, :], k[..., N//2+5:, :].transpose(-2, -1)) 

            scores_2 = torch.matmul(q[..., N//2+5:, :], k[..., 5:N//2, :].transpose(-2, -1))
            # attn_2 = F.softmax(scores_2, dim=-1)
            attn_2 = F.softmax(scores_2 / temperature, dim=-1)
            # attn_2 = torch.matmul(q[..., N//2+5:, :], k[..., 5:N//2, :].transpose(-2, -1))
            
            return output, torch.cat([attn_1, attn_2], dim=0)
        return output

    def forward(self, x: Tensor, pos=None, return_attn=False, temperature=1.0) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            # x = F.scaled_dot_product_attention(
            #     q,
            #     k,
            #     v,
            #     dropout_p=self.attn_drop.p if self.training else 0.0,
            # )
            if return_attn:
                x, attn = self.custom_scaled_dot_product_attention(
                    q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, return_attn=True, temperature=temperature
                )
            else:
                x = self.custom_scaled_dot_product_attention(
                    q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
                )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            # attn = attn.softmax(dim=-1)
            attn = (attn / temperature).softmax(dim=-1)
            # attn = self.attn_drop(attn)
            attn_drop = self.attn_drop(attn)
            x = attn_drop @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return x, attn

        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
