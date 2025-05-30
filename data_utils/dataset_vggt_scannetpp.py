import collections
import csv
import json
import os
import glob
import random
import struct
from pathlib import Path

import albumentations as A
import cv2
import imageio
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from vggt.utils.load_fn import load_and_preprocess_images

# define a custom collate function, assume batch size is 1
def my_collate(batch):
    obj_name = batch[0]['obj_name']
    rgbs = torch.stack([torch.from_numpy(rgb) for rgb in batch[0]['rgbs']]).float() / 255.
    rgbs_mast3r = batch[0]['rgbs_mast3r']
    info = batch[0]['info']
    for key in info:
        info[key][0] = torch.nonzero(info[key][0])[:, 0].int()
    return {'obj_name': obj_name,'rgbs': rgbs, 'rgbs_mast3r': rgbs_mast3r, 'info': info}

class ScanNetPPVGGTDataset(Dataset):
    def __init__(self, 
                 root='data/scannetpp', 
                 sample_list='metadata/train_samples_all.txt',
                 pairs_file='metadata/train_image_pairs.npy',
                 img_size=512,
                 num=10000, # 1000
                #  load_image_pairs=False,
                load_image_pairs=True,
                 ):
        super().__init__()
        self.root = Path(root)
        self.img_size = img_size
        self.sample_list = os.path.join(root, sample_list)
        self.pairs_file = os.path.join(root, pairs_file)

        self.dist_thresh = 1.0 # 1.0
        self.angle_thresh = 90.0

        self.ids = np.loadtxt(self.sample_list, dtype=str)

        self.scene_to_imgs = collections.defaultdict(list)
        for img_id in self.ids:
            scene_name, img_name = img_id.split('_')
            self.scene_to_imgs[scene_name].append(img_name)

        if os.path.exists(self.pairs_file) and load_image_pairs:
            print(f"Loading image pairs from {self.pairs_file}")
            with open(self.pairs_file, 'rb') as f:
                self.image_pairs = pickle.load(f)
        else:
            print(f"Generating image pairs and saving to {self.pairs_file}")
            self.image_pairs = self.generate_image_pairs(num)
            with open(self.pairs_file, 'wb') as f:
                pickle.dump(self.image_pairs, f)

    def generate_image_pairs(self, desired_total):
        image_pairs = []
        scenes = list(self.scene_to_imgs.keys())
        n_scenes = len(scenes)
        pairs_per_scene = desired_total // n_scenes

        for scene_name in scenes:
            img_names = self.scene_to_imgs[scene_name]
            if len(img_names) < 2:
                continue

            transforms_json_path = self.root / 'scenes' / scene_name / 'transforms_train.json'
            with open(transforms_json_path, 'r') as f:
                transforms = json.load(f)

            scene_intrinsic = self.get_intrinsic(transforms)
            frames = { frame['file_path'].split('.')[0]: np.array(frame['transform_matrix']) 
                       for frame in transforms['frames'] }
            
            all_pairs = []
            for i in range(len(img_names)):
                mat_i = frames[img_names[i]]
                for j in range(i + 1, len(img_names)):
                    mat_j = frames[img_names[j]]
                    if not self.is_co_view_transform(mat_i, mat_j):
                        continue
                    all_pairs.append((scene_name, img_names[i], img_names[j], scene_intrinsic))
                    if len(all_pairs) >= pairs_per_scene:
                        break
            if len(all_pairs) > pairs_per_scene:
                sampled_pairs = random.sample(all_pairs, pairs_per_scene)
            else:
                sampled_pairs = all_pairs
            image_pairs.extend(sampled_pairs)
        return image_pairs

    def is_co_view_transform(self, matA, matB):
        centerA = matA[:3, 3]
        centerB = matB[:3, 3]
        if np.linalg.norm(centerA - centerB) > self.dist_thresh:
            return False

        forwardA = -matA[:3, 2]
        forwardB = -matB[:3, 2]

        dot_val = np.dot(forwardA, forwardB) / (np.linalg.norm(forwardA) * np.linalg.norm(forwardB) + 1e-8)
        angle_deg = np.degrees(np.arccos(np.clip(dot_val, -1.0, 1.0)))
        if angle_deg > self.angle_thresh:
            return False

        return True

    def get_intrinsic(self, transforms):
        flx, fly = transforms['fl_x'], transforms['fl_y']
        cx, cy = transforms['cx'], transforms['cy']
        w, h = transforms['w'], transforms['h']

        scale_x = 512 / w
        scale_y = 336 / h
        
        intrinsic = np.array([
            [flx  * scale_x, 0, cx * scale_x],
            [0, fly * scale_y, cy * scale_y],
            [0, 0, 1]
        ])
        
        return intrinsic

    def __len__(self):
        # return len(self.image_pairs)
        return 100

    def __getitem__(self, idx):
        idx = np.random.randint(len(self.image_pairs))
        scene_name, img1_name, img2_name, intrinsic = self.image_pairs[idx]
        
        img1_path = self.root/'scenes'/scene_name/'images'/(img1_name+'.JPG')
        img2_path = self.root/'scenes'/scene_name/'images'/(img2_name+'.JPG')

        img1 = self.process_image(img1_path)
        img2 = self.process_image(img2_path)
        
        rgb_vggt = load_and_preprocess_images([str(img1_path), str(img2_path)])

        return {
            'scene_name_1': scene_name,
            'scene_name_2': scene_name,
            'rgb_1': img1,
            'rgb_2': img2,
            'rgb_path_1': str(img1_path),
            'rgb_path_2': str(img2_path),
            'rgb_vggt': rgb_vggt,
            'intrinsic': intrinsic,
        }

    def process_image(self, img_path):
        img = Image.open(img_path)
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img).transpose(2,0,1) / 255.0
        img = img.astype(np.float32)
        return img



class AugmentedCustomScanNetPPDataset(Dataset):
    def __init__(self, base_dataset, augmentation=True):
        self.base_dataset = base_dataset
        
        self.color_aug = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.GaussianBlur(blur_limit=(3,7))
        ])

        self.augmentation = augmentation

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]

        for img_idx in [1, 2]:
            if self.augmentation:
                img = sample[f'rgb_{img_idx}'].transpose(1,2,0)*255
                augmented = self.color_aug(image=img.astype(np.uint8))
                sample[f'rgb_{img_idx}'] = augmented['image'].transpose(2,0,1)/255.0
            sample[f'rgb_{img_idx}'] = sample[f'rgb_{img_idx}'].astype(np.float32)

        return sample