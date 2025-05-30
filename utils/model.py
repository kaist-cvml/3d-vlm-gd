import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, dim, bottleneck_dim):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim, bias=False)
        self.relu = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, dim, bias=False)

    def forward(self, x):
        return self.up(self.relu(self.down(x)))
    
class BlockWithAdapter(nn.Module):
    def __init__(self, block, adapter):
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, x):
        out = self.block(x)
        return out + self.adapter(out)    

class _LoRA_qkv(nn.Module):
    """
    In Dinov2 it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module = None,
            linear_b_v: nn.Module = None,
            linear_a_k: nn.Module = None,
            linear_b_k: nn.Module = None,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        
        qkv[:, :, : self.dim] += new_q

        if self.linear_a_v is not None and self.linear_b_v is not None:
            new_v = self.linear_b_v(self.linear_a_v(x))
            qkv[:, :, -self.dim:] += new_v

        if self.linear_a_k is not None and self.linear_b_k is not None:
            new_k = self.linear_b_k(self.linear_a_k(x))
            qkv[:, :, self.dim:2 * self.dim] += new_k

        return qkv

class _LoRA_Linear(nn.Module):
    def __init__(self, origin_linear: nn.Linear, linear_a: nn.Linear, linear_b: nn.Linear):
        super().__init__()
        self.origin_linear = origin_linear
        self.linear_a = linear_a
        self.linear_b = linear_b
        self.dim = origin_linear.in_features

    def forward(self, x):
        out = self.origin_linear(x)
        lora_out = self.linear_b(self.linear_a(x))
        out += lora_out
        return out


class DepthAwareFeatureFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, use_tanh=True):
        super().__init__()
        self.use_tanh = use_tanh
        
        self.depth_attention = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, features, depths=None):
        if depths is not None:
            # depths: (B, N) -> (B, N, 1)
            depth_embedding = depths.unsqueeze(-1)
            
            # Generate depth-based attention weights
            attention_weights = self.depth_attention(depth_embedding)
            
            # Apply depth-aware attention to features
            depth_modulated_features = features * attention_weights
            
            # Final prediction
            depth_diff = self.fusion_layer(depth_modulated_features)
        else:
            # If no depths provided, just use the features directly
            depth_diff = self.fusion_layer(features)
            
        if self.use_tanh:
            depth_diff = torch.tanh(depth_diff)
            
        return depth_diff.squeeze(-1)

