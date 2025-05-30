import json
import math
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

import os
import cv2
import matplotlib.cm as cm
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import tqdm

import timm
from PIL import Image
from visdom import Visdom
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.decomposition import PCA
from torch.nn.parameter import Parameter
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.transforms import functional

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.model import vis_attn_map

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from sklearn.decomposition import PCA

from typing import (
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from data_utils.dataset_mast3r_objaverse import AugmentedCustomObjaverseDataset, ObjaverseMASt3RDataset
from data_utils.dataset_mast3r_scannetpp import AugmentedCustomScanNetPPDataset, ScanNetPPMASt3RDataset

from utils.functions import fix_random_seeds, sigmoid, interpolate_features, \
    point_cloud_to_depth, extract_kp_depth, filter_kp_by_conf, post_process_depth, \
    get_masked_patch_cost, get_patch_mask_from_kp_tensor
from utils.model import _LoRA_qkv, DepthAwareFeatureFusion, Adapter, BlockWithAdapter
from utils.losses import kl_divergence_map, pairwise_logistic_ranking_loss

from copy import deepcopy
import warnings

warnings.filterwarnings(action='ignore')


model_configs = {
    'ViT-B-16': 'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k',
}

class FinetuneMASt3RTIMM(pl.LightningModule):
    def __init__(
            self, 
            r, 
            backbone_size, 
            datasets, 
            matcher_ckpt="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
            ap_loss_weight=1.0,
            depth_loss_weight=0.0,
            intra_depth_loss_weight=1.0,
            kl_loss_weight=1.0,
            init_temperature=1.0,
            final_temperature=0.5,
            ):
        super().__init__()

        # Save config as hparams
        self.save_hyperparameters()

        self.ap_loss_weight = ap_loss_weight
        self.depth_loss_weight = depth_loss_weight
        self.intra_depth_loss_weight = intra_depth_loss_weight
        self.kl_loss_weight = kl_loss_weight

        assert r > 0
        self.embedding_dim = 768

        self.backbone_name = model_configs[backbone_size]
        print(f"Loading {self.backbone_name}")
        model = timm.create_model(self.backbone_name, pretrained=True, dynamic_img_size=True).cuda().eval()

        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        
        self.datasets = datasets
        self.matcher = AsymmetricMASt3R.from_pretrained(matcher_ckpt)

        for param in self.matcher.parameters():
            param.requires_grad = False

        self.w_As = []
        self.w_Bs = []

        for param in model.parameters():
            param.requires_grad = False

        self.adapters = nn.ModuleList()
        for i, blk in enumerate(model.blocks[4:]):
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

            adapter = Adapter(dim=self.embedding_dim, bottleneck_dim=64)
            blk = BlockWithAdapter(blk, adapter)
            model.blocks[4 + i] = blk
            self.adapters.append(adapter)
        self.reset_parameters()

        self.model = model
        self.downsample_factor = 8

        self.refine_conv = nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=1, padding=1)

        self.thres3d_neg = 0.1
        self.patch_size = model.patch_embed.patch_size[0]
        self.target_res = 640
        self.min_conf_thr = 10
        self.input_transform = transforms.transforms[-1]
        
        # Initialize depth prediction heads based on selected mode
        self.depth_diff_head = DepthAwareFeatureFusion(input_dim=self.embedding_dim, use_tanh=True)

        self.use_hard_negative_mining = True
        self.hard_negative_ratio = 0.3
        self.use_gradient_consistency = True
        
        self.init_temperature = init_temperature
        self.final_temperature = final_temperature
        self.matcher.temperature = self.init_temperature

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
            
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        checkpoint['state_dict'] = {
            'refine_conv': self.refine_conv.state_dict(),
        }
        
        depth_diff_head = {
            'depth_diff_head': self.depth_diff_head.state_dict()
        }

        checkpoint.update(a_tensors)
        checkpoint.update(b_tensors)
        checkpoint.update(depth_diff_head)
        
        adapter_tensors = {f"adapter_{i:03d}": adapter.state_dict() for i, adapter in enumerate(self.adapters)}
        checkpoint.update(adapter_tensors)
        
        
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.refine_conv.load_state_dict(checkpoint['state_dict']['refine_conv'])
        
        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = checkpoint[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = checkpoint[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        self.depth_diff_head.load_state_dict(checkpoint['depth_diff_head'])
        
        for i, adapter in enumerate(self.adapters):
            saved_key = f"adapter_{i:03d}"
            adapter.load_state_dict(checkpoint[saved_key])

        self.loaded = True
            
    def update_temperature(self):
        total_epochs = self.trainer.max_epochs if hasattr(self.trainer, 'max_epochs') and self.trainer.max_epochs > 0 else 100
        current_epoch = self.current_epoch
        ratio = min(current_epoch / total_epochs, 1.0)
        
        new_temp = self.init_temperature * (1 - ratio) + self.final_temperature * ratio

        self.matcher.temperature = new_temp

    def on_train_batch_end(self, *args, **kwargs):
        self.update_temperature()

    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(42)
        return torch.utils.data.DataLoader(
            dataset=self.datasets,
            batch_size=1,
            shuffle=False,
            num_workers=10,
            pin_memory=True,
            worker_init_fn=lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id),
            generator=g,
        )

    def get_intermediate_feature(
        self,
        rgbs: torch.Tensor,
        pts=None,
        n=[0,1,2,3],
        reshape: bool = False,
        return_class_token: bool = False,
        normalize: bool = True,
    ):
        tgt_size = (int(rgbs.shape[-2] * self.target_res / rgbs.shape[-1]), self.target_res)
        if rgbs.shape[-2] > rgbs.shape[-1]:
            tgt_size = (self.target_res, int(rgbs.shape[-1] * self.target_res / rgbs.shape[-2]))
        
        patch_h, patch_w = tgt_size[0] // self.downsample_factor, tgt_size[1] // self.downsample_factor
        rgb_resized = functional.resize(rgbs, (patch_h * self.patch_size, patch_w * self.patch_size))
        
        resize_factor = [(patch_w * self.patch_size) / rgbs.shape[-1], (patch_h * self.patch_size) / rgbs.shape[-2]]
        
        pts = pts * torch.tensor(resize_factor).to(pts.device)
        
        outputs = self.model._intermediate_layers(self.input_transform(rgb_resized), n)
        if normalize:
            outputs = [self.model.norm(out) for out in outputs]
        if return_class_token:
            prefix_tokens = [out[:, 0] for out in outputs]

        outputs = [out[:, self.model.num_prefix_tokens :] for out in outputs]

        if reshape:
            results = []
            for out in outputs:
                res = out.reshape(rgb_resized.shape[0], patch_h, patch_w, -1).permute(0, 3, 1, 2).contiguous()
                res_kp = interpolate_features(res, pts, h=patch_h * self.patch_size, w=patch_w * self.patch_size, patch_size=self.patch_size, stride=self.patch_size, normalize=False).permute(0, 2, 1)
                results.append(res_kp)
            
            outputs = torch.stack(results, dim=0).mean(dim=0)

            if return_class_token:
                results_prefix = []
                for prefix_token in prefix_tokens:
                    results_prefix.append(prefix_token.unsqueeze(0).repeat(pts.size(1), 1))

                prefix_tokens = torch.stack(results_prefix, dim=0).mean(dim=0)

        if return_class_token:
            return outputs, prefix_tokens
        return outputs
    
    def get_feature(self, rgbs, pts, normalize=True, global_feature=False):
        tgt_size = (int(rgbs.shape[-2] * self.target_res / rgbs.shape[-1]), self.target_res)
        if rgbs.shape[-2] > rgbs.shape[-1]:
            tgt_size = (self.target_res, int(rgbs.shape[-1] * self.target_res / rgbs.shape[-2]))
        
        patch_h, patch_w = tgt_size[0] // self.downsample_factor, tgt_size[1] // self.downsample_factor
        rgb_resized = functional.resize(rgbs, (patch_h * self.patch_size, patch_w * self.patch_size))
        
        resize_factor = [(patch_w * self.patch_size) / rgbs.shape[-1], (patch_h * self.patch_size) / rgbs.shape[-2]]
        
        pts = pts * torch.tensor(resize_factor).to(pts.device)
        
        if global_feature:
            result = self.model.forward_features(self.input_transform(rgb_resized))
            global_feat, result = result[:, 0], result[:, 1:]
        else:    
            result = self.model.forward_features(self.input_transform(rgb_resized))[:, 1:]
        
        feature = result.reshape(rgb_resized.shape[0], patch_h, patch_w, -1).permute(0, 3, 1, 2)
        feature = self.refine_conv(feature)
            
        feature = interpolate_features(feature, pts, h=patch_h * self.patch_size, w=patch_w * self.patch_size, patch_size=self.patch_size, stride=self.patch_size, normalize=False).permute(0, 2, 1)
        if normalize:
            feature = F.normalize(feature, p=2, dim=-1)
        
        if global_feature:
            return feature, global_feat

        return feature
    

    def get_feature_cost(self, rgbs, normalize=True, resize=True):
        outputs = self.model._intermediate_layers(self.input_transform(rgbs), [4,5,6,7]) 
        if normalize:
            outputs = [self.model.norm(out) for out in outputs]

        outputs = [out[:, self.model.num_prefix_tokens :] for out in outputs]

        B, _, H, W = rgbs.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        results = []
        for out in outputs:
            res = out.reshape(rgbs.shape[0], patch_h, patch_w, -1).permute(0, 3, 1, 2).contiguous()
            results.append(res)
        
        feature = torch.stack(results, dim=0).mean(dim=0).permute(0, 2, 3, 1)

        if resize:
            feature = functional.resize(feature, (rgbs.shape[-2], rgbs.shape[-1])).permute(0, 2, 3, 1)

        return feature


    def extract_mast3r_features(self, rgb_mast3r_1, rgb_mast3r_2):
        """
        Extract features and 3D information from the MASt3R model.

        Args:
            rgb_mast3r_1: First RGB image tensor.
            rgb_mast3r_2: Second RGB image tensor.

        Returns:
            dict: A dictionary containing extracted MASt3R features and information.
        """
        rgbs_mast3r = [rgb_mast3r_1, rgb_mast3r_2]
        pairs = make_pairs(rgbs_mast3r, scene_graph='complete', prefilter=None, symmetrize=True)

        with torch.no_grad():
            mast3r_output = inference(pairs, self.matcher, self.device, verbose=False)

        mast3r_view_1, mast3r_pred_1 = mast3r_output['view1'], mast3r_output['pred1']
        mast3r_view_2, mast3r_pred_2 = mast3r_output['view2'], mast3r_output['pred2']

        mast3r_desc_1 = mast3r_pred_1['desc'][1].detach()
        mast3r_desc_2 = mast3r_pred_2['desc'][1].detach()
        mast3r_pts3d_1 = mast3r_pred_1['pts3d'][1].detach().to(self.device)
        mast3r_pts3d_2_from_1 = mast3r_pred_2['pts3d_in_other_view'][1].detach().to(self.device)
        mast3r_pts3d_2 = mast3r_pred_1['pts3d'][0].detach().to(self.device) # Note: This is pts3d from view1's perspective of view2 points
        mast3r_conf_1 = mast3r_pred_1['conf'][1].detach()
        mast3r_conf_2_from_1 = mast3r_pred_2['conf'][1].detach() # Confidence of pts3d_2_from_1
        mast3r_conf_2 = mast3r_pred_1['conf'][0].detach() # Confidence of pts3d_2

        mast3r_cost_1 = mast3r_pred_2['tgt_attn_map'][1].detach().to(self.device) # Attention from view 1 to view 2
        mast3r_cost_2 = mast3r_pred_2['tgt_attn_map'][0].detach().to(self.device) # Attention from view 2 to view 1 (seems reversed in original code, maintaining consistency)

        return {
            'view_1': mast3r_view_1,
            'view_2': mast3r_view_2,
            'desc_1': mast3r_desc_1,
            'desc_2': mast3r_desc_2,
            'pts3d_1': mast3r_pts3d_1,
            'pts3d_2_from_1': mast3r_pts3d_2_from_1,
            'pts3d_2': mast3r_pts3d_2,
            'conf_1': mast3r_conf_1,
            'conf_2': mast3r_conf_2, # Using conf_2 consistent with original kp filtering logic
            'cost_1': mast3r_cost_1,
            'cost_2': mast3r_cost_2
        }


    def filter_and_match_keypoints(self, mast3r_features, rgb_1, rgb_2):
        """
        Performs reciprocal nearest neighbor matching and filters keypoints based on 
        boundaries and confidence.

        Args:
            mast3r_features (dict): Output from extract_mast3r_features.
            rgb_1 (torch.Tensor): Original first RGB image.
            rgb_2 (torch.Tensor): Original second RGB image.

        Returns:
            tuple: kp_1, kp_2, rgb_1_resized, rgb_2_resized, w, h
                   Returns None for all if no valid keypoints are found.
        """
        mast3r_view_1 = mast3r_features['view_1']
        mast3r_view_2 = mast3r_features['view_2']
        mast3r_desc_1 = mast3r_features['desc_1']
        mast3r_desc_2 = mast3r_features['desc_2']
        mast3r_conf_1 = mast3r_features['conf_1']
        mast3r_conf_2 = mast3r_features['conf_2']

        # Use descriptors from the dictionary for NNs calculation
        mast3r_kp_1, mast3r_kp_2 = fast_reciprocal_NNs(
            mast3r_desc_1, mast3r_desc_2, subsample_or_initxy1=16, # subsample_or_initxy1=8,
            device=self.device, dist='dot', block_size=2**13
        )

        # --- Start Filtering --- 
        mh1, mw1 = mast3r_view_1['true_shape'][0]
        valid_mast3r_kp1 = (mast3r_kp_1[:, 0] >= 3) & (mast3r_kp_1[:, 0] < int(mw1) - 3) & (
            mast3r_kp_1[:, 1] >= 3) & (mast3r_kp_1[:, 1] < int(mh1) - 3)

        mh2, mw2 = mast3r_view_2['true_shape'][0]
        valid_mast3r_kp2 = (mast3r_kp_2[:, 0] >= 3) & (mast3r_kp_2[:, 0] < int(mw2) - 3) & (
            mast3r_kp_2[:, 1] >= 3) & (mast3r_kp_2[:, 1] < int(mh2) - 3)

        valid_mast3r_kp = valid_mast3r_kp1 & valid_mast3r_kp2

        kp_1_filtered = mast3r_kp_1[valid_mast3r_kp]
        kp_2_filtered = mast3r_kp_2[valid_mast3r_kp]

        # Resize images based on the shapes determined by MASt3R view info
        rgb_1_resized = functional.resize(rgb_1, (mh1, mw1))
        rgb_2_resized = functional.resize(rgb_2, (mh2, mw2))
        
        kp_1_tensor = torch.tensor(kp_1_filtered).float().unsqueeze(0).to(self.device)  # (1, N, 2)
        kp_2_tensor = torch.tensor(kp_2_filtered).float().unsqueeze(0).to(self.device)  # (1, N, 2)

        # Confidence filtering
        conf_vec_1 = torch.stack([mast3r_conf_1.reshape(-1)])
        conf_vec_2 = torch.stack([mast3r_conf_2.reshape(-1)])

        conf_sorted_1 = conf_vec_1.reshape(-1).sort()[0]
        conf_sorted_2 = conf_vec_2.reshape(-1).sort()[0]

        conf_thres_1 = conf_sorted_1[int(conf_sorted_1.shape[0] * float(self.min_conf_thr) * 0.01)]
        conf_thres_2 = conf_sorted_2[int(conf_sorted_2.shape[0] * float(self.min_conf_thr) * 0.01)]
        
        conf_mask_1 = (mast3r_conf_1.reshape(mh1, mw1) >= conf_thres_1).to(self.device)
        conf_mask_2 = (mast3r_conf_2.reshape(mh2, mw2) >= conf_thres_2).to(self.device)

        _, valid_kp_indices_1 = filter_kp_by_conf(kp_1_tensor, conf_mask_1)
        _, valid_kp_indices_2 = filter_kp_by_conf(kp_2_tensor, conf_mask_2)

        valid_kp_indices = torch.unique(torch.cat([valid_kp_indices_1, valid_kp_indices_2], dim=0))

        kp_1_final = kp_1_tensor[:, valid_kp_indices]
        kp_2_final = kp_2_tensor[:, valid_kp_indices]
        # --- End Filtering --- 

        # Get final width and height (assuming they are the same for both views after resize)
        w, h = int(mw1), int(mh1)

        # Return None if no keypoints are left after filtering
        if kp_1_final.size(1) == 0 or kp_2_final.size(1) == 0:
            return None, None, None, None, None, None

        return kp_1_final, kp_2_final, rgb_1_resized, rgb_2_resized, w, h


    def calculate_depth_loss(self, depth_pred_1, depth_pred_2, rgb_1_resized, rgb_2_resized, kp_1, kp_2, indices=[4,5,6,7]):
        """
        Generates depth maps, extracts features/depths at keypoints, 
        and calculates depth-related losses.
        """
        # Extract features for depth loss at keypoints
        kp_feat_1 = self.get_intermediate_feature(rgb_1_resized, n=indices, pts=kp_1, reshape=True, return_class_token=False, normalize=True)
        kp_feat_2 = self.get_intermediate_feature(rgb_2_resized, n=indices, pts=kp_2, reshape=True, return_class_token=False, normalize=True)

        # Extract keypoint depths from generated depth maps
        kp_depth_1 = extract_kp_depth(depth_pred_1, kp_1)
        kp_depth_2 = extract_kp_depth(depth_pred_2, kp_2)

        # Initialize loss values
        depth_loss = torch.tensor(0.0, device=self.device)
        intra_depth_loss = torch.tensor(0.0, device=self.device)

        # --- Relative depth loss calculation --- 
        kp_depth_diff = kp_depth_1 - kp_depth_2
        kp_feature_diff = kp_feat_1 - kp_feat_2
        pred_depth_diff = self.depth_diff_head(kp_feature_diff)
        
        depth_loss = F.l1_loss(pred_depth_diff, torch.tanh(kp_depth_diff).detach())
        
        # Intra-depth consistency loss
        pairwise_loss_1 = pairwise_logistic_ranking_loss(self.depth_diff_head, kp_feat_1, kp_depth_1, depth_threshold=0.05)
        pairwise_loss_2 = pairwise_logistic_ranking_loss(self.depth_diff_head, kp_feat_2, kp_depth_2, depth_threshold=0.05)
        intra_depth_loss = (pairwise_loss_1 + pairwise_loss_2) / 2

        return depth_loss, intra_depth_loss


    def calculate_cost_loss(self, rgb_1_resized, rgb_2_resized, kp_1, kp_2, mast3r_cost_1, mast3r_cost_2, batch_idx):
        """
        Calculates the KL divergence loss between MASt3R cost maps and feature cost maps.
        """
        feat_cost_1 = self.get_feature_cost(rgb_1_resized, normalize=False, resize=False)
        feat_cost_2 = self.get_feature_cost(rgb_2_resized, normalize=False, resize=False)

        B, _, H, W = rgb_1_resized.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        kp_xy_1 = kp_1[0]  # (N,2)  (x,y)
        mask_patch_1 = get_patch_mask_from_kp_tensor(kp_xy_1, H, W, self.patch_size)

        kp_xy_2 = kp_2[0]  # (N,2)
        mask_patch_2 = get_patch_mask_from_kp_tensor(kp_xy_2, H, W, self.patch_size)

        feat_cost_1 = feat_cost_1.view(1, patch_h * patch_w, -1)  # (B, H*W, C)
        feat_cost_2 = feat_cost_2.view(1, patch_h * patch_w, -1)  # (B, H*W, C)

        feat_cost_1 = F.normalize(feat_cost_1, p=2, dim=-1)
        feat_cost_2 = F.normalize(feat_cost_2, p=2, dim=-1)

        feat_cost_1_to_2 = torch.bmm(feat_cost_1, feat_cost_2.transpose(-1, -2))  # (B, H*W, H*W)
        feat_cost_2_to_1 = torch.bmm(feat_cost_2, feat_cost_1.transpose(-1, -2))  # (B, H*W, H*W)

        masked_mast3r_cost_1 = get_masked_patch_cost(mast3r_cost_1.unsqueeze(0), mask_patch_1, mask_patch_2=None)
        masked_mast3r_cost_2 = get_masked_patch_cost(mast3r_cost_2.unsqueeze(0), mask_patch_2, mask_patch_2=None)

        masked_feat_cost_1_to_2 = get_masked_patch_cost(feat_cost_1_to_2, mask_patch_1, mask_patch_2=None, use_softmax=True)
        masked_feat_cost_2_to_1 = get_masked_patch_cost(feat_cost_2_to_1, mask_patch_2, mask_patch_2=None, use_softmax=True)

        kl_loss_1 = kl_divergence_map(masked_mast3r_cost_1, masked_feat_cost_1_to_2)
        kl_loss_2 = kl_divergence_map(masked_mast3r_cost_2, masked_feat_cost_2_to_1)

        kl_loss = (kl_loss_1 + kl_loss_2) / 2
        return kl_loss


    def calculate_matching_loss(self, rgb_1_resized, rgb_2_resized, kp_1, kp_2, mast3r_pts3d_1, mast3r_pts3d_2_from_1):
        """
        Extracts features and 3D points at keypoints, then calculates the 
        Average Precision (AP) loss for matching.
        """
        # Extract features for matching loss
        desc_1 = self.get_feature(rgb_1_resized, kp_1, normalize=True)  # (B, N, C)
        desc_2 = self.get_feature(rgb_2_resized, kp_2, normalize=True)  # (B, N, C)

        # Extract 3D points for matching loss
        pts3d_1 = mast3r_pts3d_1[kp_1[...,1].long(), kp_1[...,0].long()]  # (B, N, 3)
        pts3d_2 = mast3r_pts3d_2_from_1[kp_2[...,1].long(), kp_2[...,0].long()] # (B, N, 3)

        # --- Original loss calculation ---
        pos_idxs = torch.stack([
            torch.zeros(desc_1.size(1), dtype=torch.long, device=self.device),
            torch.arange(desc_1.size(1), device=self.device),
            torch.arange(desc_2.size(1), device=self.device)
        ], dim=1)  # (N, 3)

        eye_mask = torch.eye(desc_1.size(1), device=self.device).bool().unsqueeze(0)  # (1, N, N)
        # Use pts3d_1 and pts3d_2 for distance calculation, consistent with original training_step
        neg_mask = (torch.cdist(pts3d_1, pts3d_2) > self.thres3d_neg) & ~eye_mask  # (B, N, N)

        sim = torch.bmm(desc_1, desc_2.transpose(-1, -2))  # (B, N, N)

        pos_sim = sim[pos_idxs[:,0], pos_idxs[:,1], pos_idxs[:,2]]  # (N)
        # rpos = sigmoid(1. - pos_sim, temp=0.01) + 1  # (N)
        rpos = sigmoid(pos_sim - 1., temp=0.01) + 1  # (N)
        rall = rpos + torch.sum(
            sigmoid(sim[pos_idxs[:,0], pos_idxs[:,1]] - 1., temp=0.01)  # sim[pos_idxs[:,0], pos_idxs[:,1]] shape: (N, N)
            * neg_mask[pos_idxs[:,0], pos_idxs[:,1]].float(),  # neg_mask[pos_idxs[:,0], pos_idxs[:,1]] shape: (N, N)
            dim=-1
        )
        ap1 = rpos / rall

        rpos = sigmoid(1. - pos_sim, temp=0.01) + 1
        rall = rpos + torch.sum(
            sigmoid(sim[pos_idxs[:,0], pos_idxs[:,1]] - pos_sim[:, None], temp=0.01) 
            * neg_mask[pos_idxs[:,0], pos_idxs[:,1]].float(),
            dim=-1
        )
        ap2 = rpos / rall

        ap = (ap1 + ap2) / 2
        ap_loss = torch.mean(1. - ap)
        return ap_loss


    def training_step(self, batch, batch_idx):
        rgb_1, rgb_mast3r_1 = batch['rgb_1'], batch['rgb_mast3r_1']
        rgb_2, rgb_mast3r_2 = batch['rgb_2'], batch['rgb_mast3r_2']
        intrinsic = batch['intrinsic'].detach().cpu().numpy() # Keep intrinsic here as it's needed for depth generation

        # 1. Extract MASt3R features
        mast3r_features = self.extract_mast3r_features(rgb_mast3r_1, rgb_mast3r_2)

        # 2. Match and filter keypoints
        kp_1, kp_2, rgb_1_resized, rgb_2_resized, w, h = self.filter_and_match_keypoints(mast3r_features, rgb_1, rgb_2)

        # Check if keypoints exist after filtering
        if kp_1 is None: # filter_and_match_keypoints returns None if no valid kps
            loss = torch.tensor(0., device=self.device, requires_grad=True)
            self.log('loss', loss, prog_bar=True)
            return loss
        
        # Retrieve necessary features for loss calculation from the dict
        mast3r_pts3d_1 = mast3r_features['pts3d_1']
        mast3r_pts3d_2_from_1 = mast3r_features['pts3d_2_from_1']
        mast3r_pts3d_2 = mast3r_features['pts3d_2'] # Needed for depth loss
        mast3r_cost_1 = mast3r_features['cost_1'] # Needed for cost loss
        mast3r_cost_2 = mast3r_features['cost_2'] # Needed for cost loss

        
        if 'depth_1' in batch and 'depth_2' in batch:
            # Use provided depth maps from the batch
            depth_pred_1 = batch['depth_1'].to(self.device)
            depth_pred_2 = batch['depth_2'].to(self.device)

            # Resize depth maps to match the resized RGB images
            depth_pred_1 = functional.resize(depth_pred_1, (h, w)).squeeze(0)
            depth_pred_2 = functional.resize(depth_pred_2, (h, w)).squeeze(0)
        # Generate depth maps from point clouds
        else:
            depth_pred_1 = point_cloud_to_depth(mast3r_pts3d_1.view(-1, 3), 
                            torch.tensor(intrinsic[0], device=self.device, dtype=torch.float32), w, h, self.device)
            depth_pred_1 = post_process_depth(depth_pred_1, kernel_size=3)

            depth_pred_2 = point_cloud_to_depth(mast3r_pts3d_2.view(-1, 3),
                            torch.tensor(intrinsic[0], device=self.device, dtype=torch.float32), w, h, self.device)
            depth_pred_2 = post_process_depth(depth_pred_2, kernel_size=3)

        # 3. Calculate depth loss
        depth_loss, intra_depth_loss = self.calculate_depth_loss(
            depth_pred_1, depth_pred_2, rgb_1_resized, rgb_2_resized, kp_1, kp_2
        )

        # 4. Calculate KL loss
        kl_loss = self.calculate_cost_loss(rgb_1_resized, rgb_2_resized, kp_1, kp_2, mast3r_cost_1, mast3r_cost_2, batch_idx)

        # 5. Calculate matching loss
        ap_loss = self.calculate_matching_loss(
            rgb_1_resized, rgb_2_resized, kp_1, kp_2, 
            mast3r_pts3d_1, mast3r_pts3d_2_from_1 # Pass necessary 3D points
        )

        # 6. Total loss
        loss = (self.ap_loss_weight * ap_loss + 
                self.depth_loss_weight * depth_loss + 
                self.intra_depth_loss_weight * intra_depth_loss + 
                self.kl_loss_weight * kl_loss)

        # visualize_matching_pairs(rgb_1_resized, rgb_2_resized, kp_1, kp_2, self.current_epoch, batch_idx, output_dir="visualization/debug_match_scannetpp_wo_line")


        # Logging
        self.log('loss', loss, prog_bar=True)
        self.log('depth_loss', depth_loss, prog_bar=True)
        self.log('intra_depth_loss', intra_depth_loss, prog_bar=True)
        self.log('kl_loss', kl_loss, prog_bar=True)
        self.log('ap_loss', ap_loss, prog_bar=True)
        
        if not hasattr(self, 'batch_metrics'):
            self.batch_metrics = {
                'depth_loss': [],
                'intra_depth_loss': [],
                'kl_loss': [],
                'ap_loss': [],
                'total_loss': []
            }
        
        self.batch_metrics['depth_loss'].append(depth_loss.item())
        self.batch_metrics['intra_depth_loss'].append(intra_depth_loss.item())
        self.batch_metrics['kl_loss'].append(kl_loss.item())
        self.batch_metrics['ap_loss'].append(ap_loss.item())
        self.batch_metrics['total_loss'].append(loss.item())
        
        return loss
        

    def configure_optimizers(self):
        return torch.optim.AdamW([layer.weight for layer in self.w_As]
                                 + [layer.weight for layer in self.w_Bs]
                                 + list(self.refine_conv.parameters())
                                 + list(self.depth_diff_head.parameters())
                                 +  list(self.adapters.parameters())
                                 , lr=1e-5, weight_decay=1e-4)

        return results

    def val_dataloader(self):
        # Use a small subset of the training data for validation
        # In a real scenario, you'd use a separate validation dataset
        n_samples = min(len(self.datasets), 100)  # Use at most 100 samples
        g = torch.Generator()
        g.manual_seed(42)  # Fixed seed for reproducibility
        
        # Create a subset for validation
        val_indices = torch.randperm(len(self.datasets), generator=g)[:n_samples].tolist()
        val_dataset = Subset(self.datasets, val_indices)
        
        return torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=5,
            pin_memory=True
        )

from pytorch_lightning.callbacks import ModelCheckpoint
import hydra

@hydra.main(config_path='./config', config_name='finetune_timm_mast3r', version_base='1.2')
def main(cfg):
    
    fix_random_seeds()
    
    if cfg.dataset == 'scannetpp' and cfg.matcher == 'mast3r':
        dataset_instance = AugmentedCustomScanNetPPDataset(ConcatDataset([ScanNetPPMASt3RDataset()]))
    elif cfg.dataset == 'objaverse' and cfg.matcher == 'mast3r':
        dataset_instance = AugmentedCustomObjaverseDataset(ConcatDataset([ObjaverseMASt3RDataset('data/objaverse_renderings', 10_000)]))
    else:
        raise ValueError(f"Not supported: {cfg.dataset}_{cfg.matcher}")
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    print(f"Output directory: {output_dir}")
    
    logger = TensorBoardLogger(output_dir, name=f"{cfg.backbone}_r{cfg.r}")
    
    loss_weights = cfg.loss_weights if hasattr(cfg, 'loss_weights') else {}
    
    pl_module = FinetuneMASt3RTIMM(
        r=cfg.r, 
        backbone_size=cfg.backbone, 
        datasets=dataset_instance,
        ap_loss_weight=loss_weights.get('ap_loss', 1.0),
        depth_loss_weight=loss_weights.get('depth_loss', 1.0),
        intra_depth_loss_weight=loss_weights.get('intra_depth_loss', 1.0),
        kl_loss_weight=loss_weights.get('kl_loss', 1.0),
        use_scale_invariant_loss=cfg.use_scale_invariant_loss,
    )
    
    # Trainer setup
    callbacks = [
        ModelCheckpoint(save_last=True, every_n_epochs=cfg.train.save_interval, save_top_k=-1, dirpath=output_dir),
    ]
    
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs, 
        accelerator='gpu',
        devices=cfg.train.devices,
        callbacks=callbacks,
        gradient_clip_val=cfg.train.grad_clip, 
        logger=logger,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches
    )
    print("Starting training...")
    trainer.fit(pl_module)
    print("Training finished.")

if __name__ == '__main__':
    main()
    pass