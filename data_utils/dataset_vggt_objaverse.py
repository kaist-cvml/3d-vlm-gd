import collections
import csv
import json
import os
import glob
import math
import random
import struct
from pathlib import Path

import albumentations as A
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from vggt.utils.load_fn import load_and_preprocess_images


class ObjaverseVGGTDataset(Dataset):
    def __init__(self, root, num) -> None:
    # def __init__(self, root) -> None:
        super().__init__()
        self.root = Path(root)

        scale_x = 512 / 512
        scale_y = 384 / 512

        self.intrinsic = np.array([
                    [16 * 512 * scale_x / 32., 0, 256 * scale_x],
                    [0, 16 * 512 * scale_y / 32., 256 * scale_y],
                    [0, 0, 1]
                ])

        with open('data/10k.txt', 'r') as file:
            txt_obj_names = [line.strip() for line in file.readlines()]

        self.obj_names = txt_obj_names[:num]
        self.num_objects = len(self.obj_names)
        self.obj_rgb_max_idx = {obj_name: self.get_rgb_max_idx(obj_name) for obj_name in self.obj_names}

    def get_rgb_max_idx(self, obj_name):
        regex_path = os.path.join(self.root, obj_name, 'color_*.png')
        max_idx = 0
        for path in glob.glob(regex_path):
            idx = int(path.split('_')[-1].split('.')[0])
            max_idx = max(max_idx, idx)
        return max_idx
    
    def get_item(self, index, suffix='', obj_name=None, i=None):
        if index >= len(self):
            raise IndexError('index out of range')
        if obj_name is None:
            while True:
                obj_name = np.random.choice(self.obj_names)
                if self.obj_rgb_max_idx[obj_name] > 1:
                    break
        if i is None:
            i = np.random.choice(self.obj_rgb_max_idx[obj_name])
        rgb_path = self.root / obj_name / f'color_{i:06d}.png'
        rgb = cv2.imread(str(rgb_path))[..., ::-1].copy()

        depth_path = self.root / obj_name / f'depth_{i:06d}.png'
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).copy()
        depth[depth == 0] = 5000
        depth[depth > 5000] = 5000   

        return {
            f'obj_name_{suffix}': obj_name,
            f'rgb_{suffix}': np.moveaxis((rgb / 255.).astype(np.float32), -1, 0),
            f'rgb_path_{suffix}': str(rgb_path),
            f'pose_idx_{suffix}': i,
            f'depth_{suffix}': (depth / 5000.).astype(np.float32),
            f'depth_path_{suffix}': str(depth_path),
        }
        
    def __getitem__(self, idx):
        try:
            res1 = self.get_item(idx, '1')
            obj_name_1 = res1['obj_name_1']
            pose_idx = res1[f'pose_idx_1']
            i = np.random.choice(self.obj_rgb_max_idx[obj_name_1])
            while i == pose_idx:
                i = np.random.choice(self.obj_rgb_max_idx[obj_name_1])
            res2 = self.get_item(idx, '2', obj_name_1, i)

            img1_path = res1['rgb_path_1']
            img2_path = res2['rgb_path_2']

            rgb_vggt = load_and_preprocess_images([str(img1_path), str(img2_path)])

            res = {**res1, **res2}
            
            res.update({'rgb_vggt': rgb_vggt})
            res.update({'intrinsic': self.intrinsic})

        except Exception as e:
            # print(e)
            res = self[(idx + 1) % len(self)]
        return res
    
    def __len__(self):
        # return len(self.obj_names)
        return 100



class AugmentedCustomObjaverseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """
        Args:
            dataset (Dataset): Instance of GoogleObjectsDataset.
            coco_root (str): Directory with all the images from COCO.
            coco_ann_file (str): Path to the JSON file with COCO annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset = dataset
        self.color_augs = A.Compose([
            A.GaussianBlur(blur_limit=(1, 3)),
            A.ISONoise(),
            A.GaussNoise(),
            A.CLAHE(),  # could probably be moved to the post-crop augmentations
            A.RandomBrightnessContrast(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        for img_idx in [1, 2]:
            obj_image = (np.moveaxis(data[f'rgb_{img_idx}'], 0, -1) * 255).astype(np.uint8)
            obj_image = self.color_augs(image=obj_image)['image']

            # Update the dataset entry
            data[f'rgb_{img_idx}'] = np.moveaxis((obj_image / 255.).astype(np.float32), -1, 0)

        return data

