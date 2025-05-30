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
from torchvision.transforms import functional
from utils.functions import img_coord_2_obj_coord

# define a custom collate function, assume batch size is 1
def my_collate(batch):
    obj_name = batch[0]['obj_name']
    poses = torch.from_numpy(batch[0]['poses']).float()
    rgbs = torch.stack([torch.from_numpy(rgb) for rgb in batch[0]['rgbs']]).float() / 255.
    depths = torch.stack([torch.from_numpy(depth) for depth in batch[0]['depths']]).float()
    masks = torch.stack([torch.from_numpy(mask > 0) for mask in batch[0]['masks']]).float()
    info = batch[0]['info']
    for key in info:
        if key == 'pts2d':
            continue
        info[key][0] = torch.nonzero(info[key][0])[:, 0].int()
    return {'obj_name': obj_name, 'poses': poses, 'rgbs': rgbs, 'depths': depths, 'masks': masks, 'info': info}


class ObjaverseCorrDataset(torch.utils.data.Dataset):
    def __init__(self, root, num) -> None:
    # def __init__(self, root) -> None:
        super().__init__()
        self.root = Path(root)
        self.poses = np.load('data/obj_poses.npy')
        self.intrinsic = np.array([[35 * 512 / 32., 0., 256],  # to assign with objaverse
                        [0., 35 * 512 / 32., 256],
                        [0., 0., 1.]])
        
        with open('data/10k.txt', 'r') as file:
            txt_obj_names = [line.strip() for line in file.readlines()]

        self.obj_names = txt_obj_names[:num]
        self.num_objects = len(self.obj_names)
    
    def get_item(self, index, suffix='', obj_name=None, i=None):
        if index >= len(self):
            raise IndexError('index out of range')
        if obj_name is None:
            obj_name = np.random.choice(self.obj_names)
            # obj_name = '000-151/607e232d7f14478082cfb13695ed7bd2'
            # obj_name = self.obj_names[index]
        if i is None:
            i = np.random.choice(self.poses.shape[0])
        rgb_path = self.root / obj_name / f'color_{i:06d}.png'
        rgb = cv2.imread(str(rgb_path))[..., ::-1].copy()
            
        depth_path = self.root / obj_name / f'depth_{i:06d}.png'
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH) / 1000.
            
        mask_path = self.root / obj_name / f'mask_{i:06d}.png'
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        keypoints2d = np.stack(np.where(mask > 0), -1)[:, ::-1]
        
        pose = self.poses[i]
        # extrinsic = np.linalg.inv(pose)
        extrinsic = pose
        chosen = np.random.choice(len(keypoints2d), 3000, replace=True if len(keypoints2d) < 3000 else False)
        keypoints2d = keypoints2d[chosen]
        keypoints3d = img_coord_2_obj_coord(keypoints2d, depth, self.intrinsic, extrinsic)
        return {
            f'obj_name_{suffix}': obj_name,
            f'rgb_{suffix}': np.moveaxis((rgb / 255.).astype(np.float32), -1, 0),
            f'mask_{suffix}': mask > 0,
            f'pts2d_{suffix}': keypoints2d.astype(np.float32),
            f'pts3d_{suffix}': keypoints3d.astype(np.float32),
            f'rot_{suffix}': extrinsic[:3, :3].astype(np.float32),
            f'pose_idx_{suffix}': i,
        }
        
    def __getitem__(self, idx):
        try:
            res1 = self.get_item(idx, '1')
            pose_idx = res1[f'pose_idx_1']
            i = np.random.choice(self.poses.shape[0])
            while i == pose_idx:
                i = np.random.choice(self.poses.shape[0])
            res = {**res1, **self.get_item(idx, '2', res1['obj_name_1'], i)}
        except Exception as e:
            # print(e)
            res = self[(idx + 1) % len(self)]
        return res
    
    def __len__(self):
        # return len(self.obj_names)
        return 100


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, geom_aug_prob=0.5):
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
        self.geom_augs = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=45, p=geom_aug_prob, border_mode=cv2.BORDER_CONSTANT, value=0),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        rot_1 = data['rot_1']
        rot_2 = data['rot_2']
        angle = np.rad2deg(np.arccos(np.clip((np.trace(rot_1 @ rot_2.T) - 1) / 2, -1., 1.)))
        if angle > 120:
            return self[(idx + 1) % len(self)]
        
        for img_idx in [1, 2]:
            obj_image = (np.moveaxis(data[f'rgb_{img_idx}'], 0, -1) * 255).astype(np.uint8)
            # Apply color augmentations
            if f'mask_{img_idx}' not in data:
                obj_image = self.color_augs(image=obj_image)['image']
                data[f'rgb_{img_idx}'] = np.moveaxis((obj_image / 255.).astype(np.float32), -1, 0)
                continue
            obj_mask = data[f'mask_{img_idx}']
            pts2d = data[f'pts2d_{img_idx}']
            pts3d = data[f'pts3d_{img_idx}']
            
            # Apply geometric augmentations, to both image and points
            aug = self.geom_augs(image=obj_image, keypoints=pts2d, mask=obj_mask.astype(np.uint8) * 255)
            obj_image = aug['image']
            
            keypoints2d = np.array(aug['keypoints'])
            valid = (keypoints2d[:, 0] >= 0) & (keypoints2d[:, 0] < obj_image.shape[1]) & (keypoints2d[:, 1] >= 0) & (keypoints2d[:, 1] < obj_image.shape[0])
            if not np.any(valid):
                data[f'rgb_{img_idx}'] = np.moveaxis((obj_image / 255.).astype(np.float32), -1, 0)
                continue
            
            obj_image = self.color_augs(image=obj_image)['image']
            obj_image[aug['mask'] == 0] = 0
            
            data[f'pts2d_{img_idx}'] = keypoints2d[valid]
            data[f'pts3d_{img_idx}'] = pts3d[valid]
            data[f'mask_{img_idx}'] = obj_mask = aug['mask'] > 0
            
            # Update the dataset entry
            data[f'rgb_{img_idx}'] = np.moveaxis((obj_image / 255.).astype(np.float32), -1, 0)

        return data

