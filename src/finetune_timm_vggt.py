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

# --- VGGT Imports ---
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
# ---------------------

# Removed MASt3R/Dust3R imports

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

# --- Dataset Imports ---
from data_utils.dataset_vggt_objaverse import  AugmentedCustomObjaverseDataset, ObjaverseVGGTDataset
from data_utils.dataset_vggt_scannetpp import AugmentedCustomScanNetPPDataset, ScanNetPPVGGTDataset
# -----------------------

from utils.functions import fix_random_seeds, sigmoid, interpolate_features, \
    extract_kp_depth, get_masked_patch_cost, \
    get_patch_mask_from_kp_tensor, get_coview_masks, sample_keypoints_nms
from utils.model import _LoRA_qkv, Adapter, BlockWithAdapter, DepthAwareFeatureFusion
from utils.losses import kl_divergence_map, pairwise_logistic_ranking_loss
from utils.vis_utils import visualize_matching_pairs, visualize_depth_maps, vis_attn_map

from copy import deepcopy
import warnings

warnings.filterwarnings(action='ignore')


model_configs = {
    'ViT-B-16': 'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k',
}

# --- Dataset Dictionary Update ---
# dataset = {
#     'scannetpp': AugmentedCustomScanNetPPDataset(ConcatDataset([ScanNetPPVGGTDataset()])),
#     'objaverse': AugmentedCustomObjaverseDataset(ConcatDataset([ObjaverseVGGTDataset('data/objaverse_renderings', 10_000)])),
# }
# ---------------------------------

class FinetuneVGGTTIMM(pl.LightningModule):
    def __init__(self,
            r, 
            backbone_size, 
            datasets,
            ap_loss_weight=1.0,
            depth_loss_weight=1.0,
            intra_depth_loss_weight=1.0,
            kl_loss_weight=1.0,
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
        # --- VGGT Matcher Initialization ---
        print("Loading VGGT matcher...")
        self.matcher = VGGT.from_pretrained("facebook/VGGT-1B").eval() # Set to eval mode
        self.vggt_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.matcher = self.matcher.to(self.device)

        for param in self.matcher.parameters():
            param.requires_grad = False
        # -------------------------------------

        self.w_As = []
        self.w_Bs = []

        for param in model.parameters():
            param.requires_grad = False

        self.adapters = nn.ModuleList()
        # --- Adapter initialization for TIMM blocks (indices might need adjustment based on backbone) ---
        adapter_start_idx = 4 # Example: start adapting from the 5th block
        num_adapter_layers = len(model.blocks) - adapter_start_idx
        print(f"Applying LoRA and Adapters to last {num_adapter_layers} blocks starting from index {adapter_start_idx}.")

        for i in range(num_adapter_layers):
            blk_idx = adapter_start_idx + i
            if blk_idx >= len(model.blocks):
                print(f"Warning: Block index {blk_idx} out of range.")
                continue
            blk = model.blocks[blk_idx]

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
            blk_with_adapter = BlockWithAdapter(blk, adapter)
            model.blocks[blk_idx] = blk_with_adapter # Use modified block
            self.adapters.append(adapter)
        # -------------------------------------------------------------------------------------------
        self.reset_parameters()

        self.model = model
        # self.downsample_factor = model.patch_embed.patch_size[0] # Use backbone's downsample factor
        self.downsample_factor = 8

        self.refine_conv = nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=1, padding=1)

        # --- Thresholds (can be tuned) ---
        self.thres3d_neg = 0.1
        # --------------------------------

        self.patch_size = model.patch_embed.patch_size[0]
        self.target_res = 640 # Or derive from data_config if needed

        self.min_conf_thr = 10 # Confidence threshold for keypoint filtering (if using VGGT confidence)
        self.count = 0

        self.input_transform = transforms.transforms[-1]
        self.depth_diff_head = DepthAwareFeatureFusion(input_dim=self.embedding_dim, use_tanh=True)

        # --- VGGT Specific Parameters ---
        self.resize_patch_size = self.matcher.aggregator.patch_size
        self.init_temperature = 1.0
        self.final_temperature = 1.0
        self.matcher.aggregator.temperature = self.init_temperature
        # --------------------------------

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
        B, _, H, W = rgbs.shape
        patch_h = H // self.resize_patch_size
        patch_w = W // self.resize_patch_size

        rgbs_resized = functional.resize(rgbs, (patch_h * self.patch_size, patch_w * self.patch_size))

        outputs = self.model._intermediate_layers(self.input_transform(rgbs_resized), [7]) 
        if normalize:
            outputs = [self.model.norm(out) for out in outputs]

        outputs = [out[:, self.model.num_prefix_tokens :] for out in outputs]

        results = []
        for out in outputs:
            res = out.reshape(rgbs.shape[0], patch_h, patch_w, -1)
            results.append(res)
        
        feature = torch.stack(results, dim=0).mean(dim=0)

        return feature

    def extract_vggt_features(self, rgb_vggt, batch_idx=None):
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.vggt_dtype):
                images = rgb_vggt  # add batch dimension
                aggregated_tokens_list, ps_idx, attn = self.matcher.aggregator(images)

            # Predict Cameras
            pose_enc = self.matcher.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

            # Predict Depth Maps
            depth_map, depth_conf = self.matcher.depth_head(aggregated_tokens_list, images, ps_idx)

            # Predict Point Maps
            point_map, point_conf = self.matcher.point_head(aggregated_tokens_list, images, ps_idx)
                
            # Construct 3D Points from Depth Maps and Cameras
            # which usually leads to more accurate 3D points than point map branch
            point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                extrinsic.squeeze(0), 
                                                                intrinsic.squeeze(0))
            
            # point_map_view_1, point_map_view_2 = point_map[0, 0], point_map[0, 1]
            point_map_view_1 = torch.tensor(point_map_by_unprojection[0], dtype=torch.float32).to(self.device)
            point_map_view_2 = torch.tensor(point_map_by_unprojection[1], dtype=torch.float32).to(self.device)
            point_conf_view_1, point_conf_view_2 = point_conf[0, 0], point_conf[0, 1]
            extrinsic_1, extrinsic_2 = extrinsic[0, 0], extrinsic[0, 1]
            intrinsic_1, intrinsic_2 = intrinsic[0, 0], intrinsic[0, 1]
            depth_pred_1, depth_pred_2 = depth_map[0, 0].squeeze(-1), depth_map[0, 1].squeeze(-1)

            image_shape = tuple(rgb_vggt.shape[-2:])
            
            cost_1, cost_2 = attn.chunk(2, dim=0)
            cost_1 = cost_1.mean(dim=1)
            cost_2 = cost_2.mean(dim=1)
            
        return {
            'point_map_view_1': point_map_view_1,
            'point_map_view_2': point_map_view_2,
            'point_conf_view_1': point_conf_view_1, 
            'point_conf_view_2': point_conf_view_2,
            'extrinsic_1': extrinsic_1,
            'extrinsic_2': extrinsic_2,
            'intrinsic_1': intrinsic_1,
            'intrinsic_2': intrinsic_2,
            'depth_pred_1': depth_pred_1,
            'depth_pred_2': depth_pred_2,
            'image_shape': image_shape,
            'cost_1': cost_1,
            'cost_2': cost_2,
            'aggregated_tokens_list': aggregated_tokens_list,
            'images': images,
            'ps_idx': ps_idx
        }
    
    def sample_keypoints(self, vggt_features, num_keypoints=300, min_distance=5):
        point_map_view_1 = vggt_features['point_map_view_1']
        point_map_view_2 = vggt_features['point_map_view_2']
        point_conf_view_1 = vggt_features['point_conf_view_1']
        intrinsic_1 = vggt_features['intrinsic_1']
        extrinsic_1 = vggt_features['extrinsic_1']
        intrinsic_2 = vggt_features['intrinsic_2']
        extrinsic_2 = vggt_features['extrinsic_2']
        image_shape = vggt_features['image_shape']
        aggregated_tokens_list = vggt_features['aggregated_tokens_list']
        images = vggt_features['images']
        ps_idx = vggt_features['ps_idx']
        
        mask_1, mask_2 = get_coview_masks(point_map_view_1, point_map_view_2,
                                    intrinsic_1, extrinsic_1,
                                    intrinsic_2, extrinsic_2,
                                    image_shape)
        
        sampled_kp_1 = sample_keypoints_nms(mask_1, point_conf_view_1, N=num_keypoints, min_distance=min_distance, device=self.device)
        
        if sampled_kp_1 is None:
            print("No keypoints found in the first view.")
            return None, None, None, None, None

        sampled_kp_1 = sampled_kp_1[:, [1, 0]].int()  # (row, col) -> (x, y)
        sampled_kp_2, vis_score, conf_score = self.matcher.track_head(aggregated_tokens_list, images, ps_idx, query_points=sampled_kp_1[None])
        sampled_kp_2 = sampled_kp_2[-1][0][1].int()  # (x, y)
        
        mh, mw = image_shape
        valid_kp_1 = (sampled_kp_1[:, 0] >= 3) & (sampled_kp_1[:, 0] < int(mw) - 3) & (sampled_kp_1[:, 1] >= 3) & (sampled_kp_1[:, 1] < int(mh) - 3)
        valid_kp_2 = (sampled_kp_2[:, 0] >= 3) & (sampled_kp_2[:, 0] < int(mw) - 3) & (sampled_kp_2[:, 1] >= 3) & (sampled_kp_2[:, 1] < int(mh) - 3)
        valid_kp = valid_kp_1 & valid_kp_2
        
        kp_1 = sampled_kp_1[valid_kp].float().unsqueeze(0).to(self.device)
        kp_2 = sampled_kp_2[valid_kp].float().unsqueeze(0).to(self.device)
        
        return kp_1, kp_2, valid_kp, mask_1, mask_2
    
    def update_temperature(self):
        total_epochs = self.trainer.max_epochs if hasattr(self.trainer, 'max_epochs') and self.trainer.max_epochs > 0 else 100
        current_epoch = self.current_epoch
        ratio = min(current_epoch / total_epochs, 1.0)
        
        new_temp = self.init_temperature * (1 - ratio) + self.final_temperature * ratio
    
        self.matcher.aggregator.temperature = new_temp
    
    
    
    def on_train_batch_end(self, *args, **kwargs):
        self.update_temperature()

    def calculate_depth_loss(self, vggt_features, rgb_1_resized, rgb_2_resized, kp_1, kp_2, indices=[4,5,6,7]):
        depth_pred_1 = vggt_features['depth_pred_1']
        depth_pred_2 = vggt_features['depth_pred_2']
        
        kp_feat_1 = self.get_intermediate_feature(rgb_1_resized, pts=kp_1, n=indices, reshape=True, normalize=True)
        kp_feat_2 = self.get_intermediate_feature(rgb_2_resized, pts=kp_2, n=indices, reshape=True, normalize=True)

        kp_depth_1 = extract_kp_depth(depth_pred_1, kp_1)
        kp_depth_2 = extract_kp_depth(depth_pred_2, kp_2)
        
        kp_depth_diff = kp_depth_1 - kp_depth_2
        kp_feature_diff = kp_feat_1 - kp_feat_2
        pred_depth_diff = self.depth_diff_head(kp_feature_diff)
        
        depth_loss = F.l1_loss(pred_depth_diff, torch.tanh(kp_depth_diff).detach())
        
        pairwise_loss_1 = pairwise_logistic_ranking_loss(self.depth_diff_head, kp_feat_1, kp_depth_1, depth_threshold=0.05)
        pairwise_loss_2 = pairwise_logistic_ranking_loss(self.depth_diff_head, kp_feat_2, kp_depth_2, depth_threshold=0.05)
        intra_depth_loss = (pairwise_loss_1 + pairwise_loss_2) / 2
        
        return depth_loss, intra_depth_loss


    def calculate_cost_loss(self, rgb_1_resized, rgb_2_resized, vggt_cost_1, vggt_cost_2, kp_1=None, kp_2=None, mask_1=None, mask_2=None):
        B, C, H, W = rgb_1_resized.shape
        
        feat_cost_1 = self.get_feature_cost(rgb_1_resized, normalize=False, resize=True)
        feat_cost_2 = self.get_feature_cost(rgb_2_resized, normalize=False, resize=True)
        
        patch_h = H // self.resize_patch_size
        patch_w = W // self.resize_patch_size
        
        if kp_1 is not None and kp_2 is not None:
            kp_xy_1 = kp_1[0]  # (N, 2) (x,y)
            mask_patch_1 = get_patch_mask_from_kp_tensor(kp_xy_1, H, W, self.patch_size)
            
            kp_xy_2 = kp_2[0]  # (N, 2)
            mask_patch_2 = get_patch_mask_from_kp_tensor(kp_xy_2, H, W, self.patch_size)
        else:
            mask_patch_1 = F.interpolate(mask_1.unsqueeze(0).unsqueeze(0).float(), size=(patch_h, patch_w), mode='nearest').squeeze(0).bool()
            mask_patch_1 = mask_patch_1.view(-1)

            mask_patch_2 = F.interpolate(mask_2.unsqueeze(0).unsqueeze(0).float(), size=(patch_h, patch_w), mode='nearest').squeeze(0).bool()
            mask_patch_2 = mask_patch_2.view(-1)
        
        feat_cost_1 = feat_cost_1.view(1, patch_h * patch_w, -1)  # (B, H*W, C)
        feat_cost_2 = feat_cost_2.view(1, patch_h * patch_w, -1)  # (B, H*W, C)
        
        feat_cost_1 = F.normalize(feat_cost_1, p=2, dim=-1)
        feat_cost_2 = F.normalize(feat_cost_2, p=2, dim=-1)
        
        feat_cost_1_to_2 = torch.bmm(feat_cost_1, feat_cost_2.transpose(-1, -2))  # (B, H*W, H*W)
        feat_cost_2_to_1 = torch.bmm(feat_cost_2, feat_cost_1.transpose(-1, -2))  # (B, H*W, H*W)
        
        feat_cost_1_to_2 = torch.nn.functional.softmax(feat_cost_1_to_2, dim=-1)
        feat_cost_2_to_1 = torch.nn.functional.softmax(feat_cost_2_to_1, dim=-1)
        
        masked_cost_1 = get_masked_patch_cost(vggt_cost_1, mask_patch_1, mask_patch_2=None)
        masked_cost_2 = get_masked_patch_cost(vggt_cost_2, mask_patch_2, mask_patch_2=None)
        
        masked_feat_cost_1_to_2 = get_masked_patch_cost(feat_cost_1_to_2, mask_patch_1, mask_patch_2=None)
        masked_feat_cost_2_to_1 = get_masked_patch_cost(feat_cost_2_to_1, mask_patch_2, mask_patch_2=None)
        
        kl_loss_1 = kl_divergence_map(masked_cost_1, masked_feat_cost_1_to_2)
        kl_loss_2 = kl_divergence_map(masked_cost_2, masked_feat_cost_2_to_1)
        
        kl_loss = (kl_loss_1 + kl_loss_2) / 2
        
        return kl_loss


    def calculate_matching_loss(self, rgb_1_resized, rgb_2_resized, kp_1, kp_2, point_map_view_1, point_map_view_2):
        desc_1 = self.get_feature(rgb_1_resized, kp_1, normalize=True)  # (B, N, C)
        desc_2 = self.get_feature(rgb_2_resized, kp_2, normalize=True)  # (B, N, C)
        
        pts3d_1 = point_map_view_1[kp_1[...,1].long(), kp_1[...,0].long()]  # (B, N, 3)
        pts3d_2 = point_map_view_2[kp_2[...,1].long(), kp_2[...,0].long()]  # (B, N, 3)
        
        pos_idxs = torch.stack([
            torch.zeros(desc_1.size(1), dtype=torch.long, device=self.device),
            torch.arange(desc_1.size(1), device=self.device),
            torch.arange(desc_2.size(1), device=self.device)
        ], dim=1)  # (N, 3)
        
        eye_mask = torch.eye(desc_1.size(1), device=self.device).bool().unsqueeze(0)  # (1, N, N)
        neg_mask = (torch.cdist(pts3d_1, pts3d_2) > self.thres3d_neg) & ~eye_mask  # (B, N, N)
        
        sim = torch.bmm(desc_1, desc_2.transpose(-1, -2))  # (B, N, N)
        
        pos_sim = sim[pos_idxs[:,0], pos_idxs[:,1], pos_idxs[:,2]]  # (N)
        rpos = sigmoid(1. - pos_sim, temp=0.01) + 1  # (N)
        rall = rpos + torch.sum(
            sigmoid(sim[pos_idxs[:,0], pos_idxs[:,1]] - 1., temp=0.01)
            * neg_mask[pos_idxs[:,0], pos_idxs[:,1]].float(),
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
        rgb_1, rgb_vggt = batch['rgb_1'], batch['rgb_vggt']
        rgb_2 = batch['rgb_2']

        vggt_features = self.extract_vggt_features(rgb_vggt, batch_idx=batch_idx)
        
        kp_1, kp_2, valid_kp, mask_1, mask_2 = self.sample_keypoints(vggt_features, num_keypoints=300, min_distance=5)
        
        if kp_1 is None or kp_2 is None:
            loss = torch.tensor(0., device=self.device, requires_grad=True)
            self.log('loss', loss, prog_bar=True)
            return loss

        mh, mw = vggt_features['image_shape']
        rgb_1_resized = functional.resize(rgb_1, (mh, mw))
        rgb_2_resized = functional.resize(rgb_2, (mh, mw))

        if kp_1.size(1) == 0 or kp_2.size(1) == 0:
            loss = torch.tensor(0., device=self.device, requires_grad=True)
            self.log('loss', loss, prog_bar=True)
            return loss

        depth_loss, intra_depth_loss = self.calculate_depth_loss(
            vggt_features, rgb_1_resized, rgb_2_resized, kp_1, kp_2
        )

        kl_loss = self.calculate_cost_loss(
            rgb_1_resized, rgb_2_resized, vggt_features['cost_1'], vggt_features['cost_2'], 
            mask_1=mask_1, mask_2=mask_2
        )

        ap_loss = self.calculate_matching_loss(
            rgb_1_resized, rgb_2_resized, kp_1, kp_2, 
            vggt_features['point_map_view_1'], vggt_features['point_map_view_2']
        )

        loss = (self.ap_loss_weight * ap_loss + 
                self.depth_loss_weight * depth_loss + 
                self.intra_depth_loss_weight * intra_depth_loss + 
                self.kl_loss_weight * kl_loss)

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

from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
import yaml # Import yaml
import argparse # Import argparse

@hydra.main(config_path='./config', config_name='finetune_timm_vggt', version_base='1.2') # Updated config_name
def main(cfg):    
    fix_random_seeds()
    
    if cfg.dataset == 'scannetpp' and cfg.matcher == 'vggt':
        dataset_instance = AugmentedCustomScanNetPPDataset(ConcatDataset([ScanNetPPVGGTDataset()]))
    elif cfg.dataset == 'objaverse' and cfg.matcher == 'vggt':
        dataset_instance = AugmentedCustomObjaverseDataset(ConcatDataset([ObjaverseVGGTDataset('data/objaverse_renderings', 10_000)]))
    else:
        raise ValueError(f"Not supported: {cfg.dataset}_{cfg.matcher}")
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    print(f"Output directory: {output_dir}")

    logger = TensorBoardLogger(output_dir, name=f"{cfg.backbone}_r{cfg.r}")

    loss_weights = cfg.loss_weights if hasattr(cfg, 'loss_weights') else {}
    
    pl_module = FinetuneVGGTTIMM(
        r=cfg.r, 
        backbone_size=cfg.backbone, 
        datasets=dataset_instance,
        ap_loss_weight=loss_weights.get('ap_loss', 1.0),
        depth_loss_weight=loss_weights.get('depth_loss', 1.0),
        intra_depth_loss_weight=loss_weights.get('intra_depth_loss', 1.0),
        kl_loss_weight=loss_weights.get('kl_loss', 1.0),
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs, 
        accelerator='gpu',
        devices=cfg.train.devices,
        callbacks=[
            ModelCheckpoint(save_last=True, every_n_epochs=cfg.train.save_interval, save_top_k=-1, dirpath=output_dir)
        ],
        gradient_clip_val=cfg.train.grad_clip, 
        logger=logger,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches
    )
    print("Starting training...")
    trainer.fit(pl_module)
    print("Training finished.")

# --- Adjusted __main__ block ---
if __name__ == '__main__':
    # This allows running with Hydra or potentially directly with argparse (though Hydra is recommended)
    # The argparse part is removed as Hydra handles configuration.
    main()
# --------------------------------