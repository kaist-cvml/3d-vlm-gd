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

from utils.functions import fix_random_seeds, sigmoid, interpolate_features
from utils.losses import kl_divergence_map, pairwise_logistic_ranking_loss
from utils.vis_utils import visualize_matching_pairs, visualize_depth_maps

from copy import deepcopy
import warnings

warnings.filterwarnings(action='ignore')


model_configs = {
    'ViT-B-16': 'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k',
}

class VanillaTIMM(pl.LightningModule):
    def __init__(
            self, 
            r, 
            backbone_size, 
            datasets=None, 
            ):
        super().__init__()

        # Save config as hparams
        # self.save_hyperparameters()

        assert r > 0
        self.embedding_dim = 768

        self.backbone_name = model_configs[backbone_size]
        print(f"Loading {self.backbone_name}")
        model = timm.create_model(self.backbone_name, pretrained=True, dynamic_img_size=True).cuda().eval()

        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        self.model = model
        self.downsample_factor = 8

        self.patch_size = model.patch_embed.patch_size[0]
        self.target_res = 640
        
        # self.count = 0

        self.input_transform = transforms.transforms[-1]

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        pass

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


from pytorch_lightning.callbacks import ModelCheckpoint
import hydra

@hydra.main(config_path='./config', config_name='finetune_timm', version_base='1.2')
def main(cfg):
    pass

if __name__ == '__main__':
    main()
    pass