import argparse
from functools import partial
import sys
import os
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import hydra
import math
import copy
import pickle
import types
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.modules.utils as nn_utils
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

import cv2
import json

project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

from utils.functions import query_pose_error, interpolate_features, preprocess_kps_pad, _fix_pos_enc
from utils.tracking_metrics import compute_tapvid_metrics_for_video
from utils.tracking_model import ModelInference, Tracker
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from finetune_timm_mast3r import FinetuneMASt3RTIMM
from finetune_timm_vggt import FinetuneVGGTTIMM
from finetune_timm_me import FinetuneTIMM
from vanilla_timm import VanillaTIMM


imagenet_norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

model_name = {
    'mast3r': FinetuneMASt3RTIMM,
    'vggt': FinetuneVGGTTIMM,
    'me': FinetuneTIMM,
}

def oneposepp(module, num_objs=None):
    stride = 16
    patch_size = 16

    # model = module.dinov2
    model = module

    root = 'data/lowtexture_test_data'
    sfm_dir = 'data/sfm_output/outputs_softmax_loftr_loftr'
    all_obj = [name for name in os.listdir(
        root) if os.path.isdir(os.path.join(root, name))]

    if num_objs is not None:
        all_obj = all_obj[:num_objs]

    threshold_1 = []
    threshold_3 = []
    threshold_5 = []

    for obj_name in all_obj:
        print(obj_name)
        anno_3d = np.load(f'{sfm_dir}/{obj_name}/anno/anno_3d_average.npz')
        keypoints3d = anno_3d['keypoints3d']

        templates = []
        all_json_fns = list(
            (Path(root) / obj_name / '{}-1'.format(obj_name.split('-')[1]) / 'anno_loftr').glob('*.json'))
        for json_fn in tqdm(all_json_fns):
            idx = json_fn.stem
            anno = json.load(open(json_fn))
            keypoints2d = np.array(anno['keypoints2d'])
            assign_matrix = np.array(anno['assign_matrix'])
            rgb = cv2.imread(str(Path(root) / obj_name / '{}-1'.format(
                obj_name.split('-')[1]) / 'color' / f'{idx}.png'))[..., ::-1].copy()
            intrinsic = np.loadtxt(str(Path(
                root) / obj_name / '{}-1'.format(obj_name.split('-')[1]) / 'intrin_ba' / f'{idx}.txt'))
            # pose = np.loadtxt(str(Path(root) / obj_name / '{}-1'.format(obj_name.split('-')[1]) / 'poses_ba' / f'{idx}.txt'))

            keypoints2d = keypoints2d[assign_matrix[0]]
            kp3ds_canon = keypoints3d[assign_matrix[1]]

            rgb_resized = cv2.resize(
                rgb / 255., (rgb.shape[1] // 8 * patch_size, rgb.shape[0] // 8 * patch_size))

            # desc = model.forward_features(imagenet_norm(torch.from_numpy(rgb_resized).cuda().float().permute(2, 0, 1)[None]))['x_norm_patchtokens']
            desc = model.model.forward_features(imagenet_norm(torch.from_numpy(
                rgb_resized).cuda().float().permute(2, 0, 1)[None]))[:, 1:]
            desc = desc.reshape(
                1, rgb.shape[0] // 8, rgb.shape[1] // 8, -1).permute(0, 3, 1, 2)

            # Only apply refine_conv if it exists in the module
            if hasattr(module, 'refine_conv'):
                desc = module.refine_conv(desc)
                
            desc_temp = interpolate_features(desc, torch.from_numpy(keypoints2d).float().cuda()[None] / 8 * patch_size,
                                             h=rgb_resized.shape[0], w=rgb_resized.shape[1], normalize=False, patch_size=patch_size, stride=stride).permute(0, 2, 1)[0]

            desc_temp /= (desc_temp.norm(dim=-1, keepdim=True) + 1e-9)
            kp_temp, kp3d_temp = keypoints2d, kp3ds_canon

            templates.append((kp_temp, desc_temp, kp3d_temp))

        all_descs_temp = torch.cat([t[1] for t in templates], 0).cuda()[::1]
        all_pts3d_temp = np.concatenate([t[2] for t in templates], 0)[::1]
        # print(all_descs_temp.shape, all_pts3d_temp.shape)

        # subsample if too many
        if len(all_descs_temp) > 120000:
            idx = np.random.choice(len(all_descs_temp), 120000, replace=False)
            all_descs_temp = all_descs_temp[idx]
            all_pts3d_temp = all_pts3d_temp[idx]

        R_errs = []
        t_errs = []
        pts3d_scale = 1000
        grid_stride = 4
        test_seq = '2'

        all_img_fns = list(sorted((Path(root) / obj_name / '{}-{}'.format(
            obj_name.split('-')[1], test_seq) / 'color').glob('*.png')))[::10]
        for i, img_fn in enumerate(tqdm(all_img_fns)):
            idx = img_fn.stem
            rgb = cv2.imread(str(Path(root) / obj_name / '{}-{}'.format(
                obj_name.split('-')[1], test_seq) / 'color' / f'{idx}.png'))[..., ::-1].copy()
            # mask = remove(rgb, only_mask=True) > 0
            intrinsic = np.loadtxt(str(Path(root) / obj_name / '{}-{}'.format(
                obj_name.split('-')[1], test_seq) / 'intrin_ba' / f'{idx}.txt'))
            pose_gt = np.loadtxt(str(Path(root) / obj_name / '{}-{}'.format(
                obj_name.split('-')[1], test_seq) / 'poses_ba' / f'{idx}.txt'))

            with torch.no_grad():
                if i == 0:
                    x_coords = np.arange(0, rgb.shape[1], grid_stride)
                    y_coords = np.arange(0, rgb.shape[0], grid_stride)

                    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
                    kp = np.column_stack(
                        (x_mesh.ravel(), y_mesh.ravel())).astype(float)

                rgb_resized = cv2.resize(
                    rgb / 255., (rgb.shape[1] // 8 * patch_size, rgb.shape[0] // 8 * patch_size))

                # desc = model.forward_features(imagenet_norm(torch.from_numpy(rgb_resized).cuda().float().permute(2, 0, 1)[None]))['x_norm_patchtokens']
                desc = model.model.forward_features(imagenet_norm(torch.from_numpy(
                    rgb_resized).cuda().float().permute(2, 0, 1)[None]))[:, 1:]
                desc = desc.reshape(
                    1, rgb.shape[0] // 8, rgb.shape[1] // 8, -1).permute(0, 3, 1, 2)

                # Only apply refine_conv if it exists in the module
                if hasattr(module, 'refine_conv'):
                    desc = module.refine_conv(desc)
                    
                desc = interpolate_features(desc, torch.from_numpy(kp).float().cuda()[None] / 8 * patch_size,
                                            h=rgb_resized.shape[0], w=rgb_resized.shape[1], normalize=False, patch_size=patch_size, stride=stride).permute(0, 2, 1)[0]
                desc /= (desc.norm(dim=-1, keepdim=True) + 1e-9)

            with torch.no_grad():
                nbr1 = []
                for d in torch.split(desc, (25000 * 10000 - 1) // all_descs_temp.shape[0] + 1):
                    sim = d @ all_descs_temp.T
                    nbr1.append(sim.argmax(-1))
                nbr1 = torch.cat(nbr1, 0)

                nbr2 = []
                for d in torch.split(all_descs_temp, (25000 * 10000 - 1) // desc.shape[0] + 1):
                    sim = d @ desc.T
                    nbr2.append(sim.argmax(-1))
                nbr2 = torch.cat(nbr2, 0)

            m_mask = nbr2[nbr1] == torch.arange(len(nbr1)).to(nbr1.device)
            # m_mask = m_mask.cpu().numpy()
            # nbr1 = nbr1.cpu().numpy()
            # nbr2 = nbr2.cpu().numpy()

            src_pts = kp[m_mask.cpu().numpy()].reshape(-1, 1, 2)
            dst_3dpts = all_pts3d_temp[nbr1[m_mask].cpu().numpy()]

            pose_pred = np.eye(4)
            if m_mask.sum() >= 4:
                _, R_exp, trans, pnp_inlier = cv2.solvePnPRansac(dst_3dpts * pts3d_scale,
                                                                 src_pts[:, 0],
                                                                 intrinsic,
                                                                 None,
                                                                 reprojectionError=8.0,
                                                                 iterationsCount=10000, flags=cv2.SOLVEPNP_EPNP)
                trans /= pts3d_scale
                if pnp_inlier is not None:
                    if len(pnp_inlier) > 5:
                        R, _ = cv2.Rodrigues(R_exp)
                        r_t = np.concatenate([R, trans], axis=-1)
                        pose_pred = np.concatenate(
                            (r_t, [[0, 0, 0, 1]]), axis=0)

            R_err, t_err = query_pose_error(pose_pred, pose_gt)
            R_errs.append(R_err)
            t_errs.append(t_err)
            # print(R_err, t_err, cnt, len(matches), len(templates[0][0]))
        print(f'object: {obj_name}')
        for pose_threshold in [1, 3, 5]:
            acc = np.mean(
                (np.array(R_errs) < pose_threshold) & (
                    np.array(t_errs) < pose_threshold)
            )
            print(f'pose_threshold: {pose_threshold}, acc: {acc}')

            if pose_threshold == 1:
                threshold_1.append(acc)
            elif pose_threshold == 3:
                threshold_3.append(acc)
            else:
                threshold_5.append(acc)

    result = {}
    result['threshold_1'] = threshold_1
    result['threshold_3'] = threshold_3
    result['threshold_5'] = threshold_5

    metrics_df = pd.DataFrame(result)
    metrics_df['objs'] = all_obj
    metrics_df.set_index(['objs'], inplace=True)

    return metrics_df


def tracking_single(video_id, module):
    patch_size = 16
    stride = patch_size // 2

    model = copy.deepcopy(module)

    h, w = 476, 854
    if h % patch_size != 0 or w % patch_size != 0:
        print(
            f'Warning: image size ({h}, {w}) is not divisible by patch size {patch_size}')
        h = h // patch_size * patch_size
        w = w // patch_size * patch_size
        print(f'New image size: {h}, {w}')

    # video_root = Path(f'data/tapvid-davis/{video_id}')
    video_root = Path(f'data/davis_480/{video_id}')

    images = []
    for img_fn in sorted((video_root / 'video').glob('*.jpg')):
        images.append(
            np.array(Image.open(img_fn).resize((w, h), Image.LANCZOS)))
    images = np.stack(images)
    images = torch.from_numpy(images).permute(
        0, 3, 1, 2).float().cuda() / 255.0

    features = []
    for image in tqdm(images):
        ph = 1 + (h - patch_size) // stride
        pw = 1 + (w - patch_size) // stride

        # fix the stride
        stride_pair = nn_utils._pair(stride)
        model.model.patch_embed.proj.stride = stride_pair
        # fix the positional encoding code
        model.model.interpolate_pos_encoding = types.MethodType(
            _fix_pos_enc(patch_size, stride_pair), model.model)

        feature = model.model.forward_features(
            imagenet_norm(image[None].cuda()))
        feature = feature[:, 1:]
        feature = feature.reshape(-1, ph, pw,
                                  feature.shape[-1]).permute(0, 3, 1, 2)

        # Only apply refine_conv if it exists in the module
        if hasattr(model, 'refine_conv'):
            feature = model.refine_conv(feature)
            
        features.append(feature)
    features = torch.cat(features)
    dino_tracker = Tracker(
        features, images, dino_patch_size=patch_size, stride=stride)

    anchor_cosine_similarity_threshold = 0.7
    cosine_similarity_threshold = 0.6
    model_inference = ModelInference(
        model=dino_tracker,
        range_normalizer=dino_tracker.range_normalizer,
        anchor_cosine_similarity_threshold=anchor_cosine_similarity_threshold,
        cosine_similarity_threshold=cosine_similarity_threshold,
    )

    rescale_sizes = [dino_tracker.video.shape[-1],
                     dino_tracker.video.shape[-2]]
    benchmark_config = pickle.load(
        open('data/tapvid_davis_data_strided.pkl', "rb"))
    
    for video_config in benchmark_config["videos"]:
        if video_config["video_idx"] == video_id:
            break
    rescale_factor_x = rescale_sizes[0] / video_config['w']
    rescale_factor_y = rescale_sizes[1] / video_config['h']
    query_points_dict = {}

    for frame_idx, q_pts_at_frame in video_config['query_points'].items():
        target_points = video_config['target_points'][frame_idx]
        query_points_at_frame = []
        for q_point in q_pts_at_frame:
            query_points_at_frame.append(
                [rescale_factor_x * q_point[0], rescale_factor_y * q_point[1], frame_idx])
        query_points_dict[frame_idx] = query_points_at_frame

    trajectories_dict = {}
    occlusions_dict = {}
    for frame_idx in tqdm(sorted(query_points_dict.keys()), desc="Predicting trajectories"):
        qpts_st_frame = torch.tensor(
            query_points_dict[frame_idx], dtype=torch.float32, device='cuda')  # N x 3, (x, y, t)
        trajectories_at_st_frame, occlusion_at_st_frame = model_inference.infer(
            query_points=qpts_st_frame, batch_size=None)  # N x T x 3, N x T

        trajectories = trajectories_at_st_frame[..., :2].cpu().detach().numpy()
        occlusions = occlusion_at_st_frame.cpu().detach().numpy()
        trajectories_dict[frame_idx] = trajectories
        occlusions_dict[frame_idx] = occlusions

    # only test video id 0 for now
    metrics = compute_tapvid_metrics_for_video(trajectories_dict=trajectories_dict,
                                               occlusions_dict=occlusions_dict,
                                               video_idx=video_id,
                                               benchmark_data=benchmark_config,
                                               pred_video_sizes=[w, h])
    metrics["video_idx"] = int(video_id)
    return metrics


def tracking(model, num_videos=1):
    metrics_list = []
    for id in range(num_videos):
        metrics = tracking_single(id, module=model)
        metrics_list.append(metrics)
        print(metrics)

    # print(f'summary:')
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.set_index(['video_idx'], inplace=True)
    return metrics_df


def resize(img, target_res, resize=True, to_pil=True, edge=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(
                    target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(
                    target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(
                    target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[
                         (top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(
                    target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[
                         (0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas


def load_pascal_data(path, size=256, category='cat', split='test', same_view=False):

    def get_points(point_coords_list, idx):
        X = np.fromstring(point_coords_list.iloc[idx, 0], sep=";")
        Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=";")
        Xpad = -np.ones(20)
        Xpad[: len(X)] = X
        Ypad = -np.ones(20)
        Ypad[: len(X)] = Y
        Zmask = np.zeros(20)
        Zmask[: len(X)] = 1
        point_coords = np.concatenate(
            (Xpad.reshape(1, 20), Ypad.reshape(1, 20), Zmask.reshape(1, 20)), axis=0
        )
        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords

    np.random.seed(42)
    files = []
    kps = []
    test_data = pd.read_csv('{}/{}_pairs_pf_{}_views.csv'.format(path,
                            split, 'same' if same_view else 'different'))
    cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    cls_ids = test_data.iloc[:, 2].values.astype("int") - 1
    cat_id = cls.index(category)
    subset_id = np.where(cls_ids == cat_id)[0]
    print(f'Number of SPairs for {category} = {len(subset_id)}')
    subset_pairs = test_data.iloc[subset_id, :]
    src_img_names = np.array(subset_pairs.iloc[:, 0])
    trg_img_names = np.array(subset_pairs.iloc[:, 1])
    # print(src_img_names.shape, trg_img_names.shape)
    point_A_coords = subset_pairs.iloc[:, 3:5]
    point_B_coords = subset_pairs.iloc[:, 5:]
    # print(point_A_coords.shape, point_B_coords.shape)
    for i in range(len(src_img_names)):
        point_coords_src = get_points(point_A_coords, i).transpose(1, 0)
        point_coords_trg = get_points(point_B_coords, i).transpose(1, 0)
        src_fn = f'{path}/../{src_img_names[i]}'
        trg_fn = f'{path}/../{trg_img_names[i]}'
        src_size = Image.open(src_fn).size
        trg_size = Image.open(trg_fn).size
        # print(src_size)
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(
            point_coords_src, src_size[0], src_size[1], size)
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(
            point_coords_trg, trg_size[0], trg_size[1], size)
        kps.append(source_kps)
        kps.append(target_kps)
        files.append(src_fn)
        files.append(trg_fn)

    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    print(f'Final number of used key points: {kps.size(1)}')
    return files, kps, None


def semantic_transfer(model, num_cats=None, same_view=False):
    img_size = 640
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    patch_size = 16
    stride = 16
    ph = 1 + (img_size - patch_size) // stride
    pw = 1 + (img_size - patch_size) // stride

    layer_name = 'x_norm_patchtokens'  # choose from x_prenorm, x_norm_patchtokens

    pcks = []
    pcks_05 = []
    pcks_01 = []

    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']  # for pascal

    if num_cats is not None:
        categories = categories[:num_cats]

    for cat in categories:
        files, kps, _ = load_pascal_data(
            'data/PF-dataset-PASCAL', size=img_size, category=cat, same_view=same_view)  # Use same_view parameter

        gt_correspondences = []
        pred_correspondences = []
        for pair_idx in tqdm(range(len(files) // 2)):
            # Load image 1
            img1 = Image.open(files[2*pair_idx]).convert('RGB')
            img1 = resize(img1, img_size, resize=True, to_pil=True, edge=False)
            img1_kps = kps[2*pair_idx]

            # # Get patch index for the keypoints
            img1_y, img1_x = img1_kps[:, 1].numpy(), img1_kps[:, 0].numpy()

            # Load image 2
            img2 = Image.open(files[2*pair_idx+1]).convert('RGB')
            img2 = resize(img2, img_size, resize=True, to_pil=True, edge=False)
            img2_kps = kps[2*pair_idx+1]

            # Get patch index for the keypoints
            img2_y, img2_x = img2_kps[:, 1].numpy(), img2_kps[:, 0].numpy()

            img1 = torch.from_numpy(
                np.array(img1) / 255.).cuda().float().permute(2, 0, 1)
            img2 = torch.from_numpy(
                np.array(img2) / 255.).cuda().float().permute(2, 0, 1)

            # img1_desc = model.dinov2.forward_features(imagenet_norm(img1[None]))[layer_name][:, (1 if 'prenorm' in layer_name else 0):]
            img1_desc = model.model.forward_features(
                imagenet_norm(img1[None]))[:, 1:]
            img1_desc = img1_desc.reshape(-1, ph, pw,
                                          img1_desc.shape[-1]).permute(0, 3, 1, 2)

            # img2_desc = model.dinov2.forward_features(imagenet_norm(img2[None]))[layer_name][:, (1 if 'prenorm' in layer_name else 0):]
            img2_desc = model.model.forward_features(
                imagenet_norm(img2[None]))[:, 1:]
            img2_desc = img2_desc.reshape(-1, ph, pw,
                                          img2_desc.shape[-1]).permute(0, 3, 1, 2)

            # Only apply refine_conv if it exists in the model
            if hasattr(model, 'refine_conv'):
                img1_desc = model.refine_conv(img1_desc)
                img2_desc = model.refine_conv(img2_desc)

            ds_size = ((img_size - patch_size) // stride) * stride + 1
            img2_desc = F.interpolate(img2_desc, size=(
                ds_size, ds_size), mode='bilinear', align_corners=True)
            img2_desc = VF.pad(img2_desc, (patch_size // 2, patch_size // 2,
                                           img_size -
                                           img2_desc.shape[2] -
                                           (patch_size // 2),
                                           img_size - img2_desc.shape[3] - (patch_size // 2)), padding_mode='edge')

            vis = img1_kps[:, 2] * img2_kps[:, 2] > 0
            img1_kp_desc = interpolate_features(img1_desc, img1_kps[None, :, :2].cuda(), h=img_size, w=img_size, normalize=True)  # N x F x K
            sim = torch.einsum('nfk,nif->nki', img1_kp_desc, img2_desc.permute(0, 2, 3, 1).reshape(1, img_size * img_size, -1))[0]
            nn_idx = torch.argmax(sim, dim=1)
            nn_x = nn_idx % img_size
            nn_y = nn_idx // img_size
            kps_1_to_2 = torch.stack([nn_x, nn_y]).permute(1, 0)
            
            gt_correspondences.append(img2_kps[vis][:, [1, 0]])
            pred_correspondences.append(kps_1_to_2[vis][:, [1, 0]])

        gt_correspondences = torch.cat(gt_correspondences, dim=0).cpu()
        pred_correspondences = torch.cat(pred_correspondences, dim=0).cpu()
        alpha = torch.tensor([0.1, 0.05, 0.15])
        correct = torch.zeros(len(alpha))

        err = (pred_correspondences - gt_correspondences).norm(dim=-1)
        err = err.unsqueeze(0).repeat(len(alpha), 1)
        threshold = alpha * img_size
        correct = err < threshold.unsqueeze(-1)
        correct = correct.sum(dim=-1) / len(gt_correspondences)

        alpha2pck = zip(alpha.tolist(), correct.tolist())
        print(' | '.join([f'PCK-Transfer@{alpha:.2f}: {pck_alpha * 100:.2f}%'
                          for alpha, pck_alpha in alpha2pck]))

        pck = correct

        pcks.append(pck[0])
        pcks_05.append(pck[1])
        pcks_01.append(pck[2])

    result = {}
    result['PCK0.05'] = [tensor.item() for tensor in pcks_05]
    result['PCK0.10'] = [tensor.item() for tensor in pcks]
    result['PCK0.15'] = [tensor.item() for tensor in pcks_01]

    metrics_df = pd.DataFrame(result)
    metrics_df['categories'] = categories[:num_cats]
    metrics_df.set_index(['categories'], inplace=True)

    weights = [15, 30, 10, 6, 8, 32, 19, 27, 13, 3,
               8, 24, 9, 27, 12, 7, 1, 13, 20, 15][:num_cats]

    metrics_df['Weighted PCK0.05'] = np.average(metrics_df['PCK0.05'], weights=weights)
    metrics_df['Weighted PCK0.10'] = np.average(metrics_df['PCK0.10'], weights=weights)
    metrics_df['Weighted PCK0.15'] = np.average(metrics_df['PCK0.15'], weights=weights)
    return metrics_df


class EvaluationCallback(pl.Callback):
    # def __init__(self, eval_every_n_epochs: int = 1, eval_methods: list = ['semantic_transfer'], same_view: bool = True):
    def __init__(self, cfg, eval_every_n_epochs: int = 1, eval_methods: list = ['semantic_transfer']):
        super().__init__()
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.eval_every_n_epochs = eval_every_n_epochs
        self.eval_methods = eval_methods
        self.cfg = cfg
        # self.output_dir = Path('evaluation_output/intermediate_results')
        self.output_dir = Path('evaluation_output') / cfg.model / cfg.matcher / cfg.backbone / cfg.dataset / start_time
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        print(f"------------------------------------------------------")
        print(f"(EvaluationCallback) Epoch {trainer.current_epoch + 1}")
        print(f"------------------------------------------------------")

        if hasattr(pl_module, 'batch_metrics'):
            metrics = {
                k: sum(v) / (len(v) + 1e-8) for k, v in pl_module.batch_metrics.items()
            }

            print(f"\n{'='*50}")
            print(f"Epoch {trainer.current_epoch + 1} end")
            print(f"{'='*50}")
            print(f"Average Total Loss: {metrics['total_loss']:.4f}")
            print(f"Average Depth Loss: {metrics['depth_loss']:.4f}")
            print(f"Average Intra Depth Loss: {metrics['intra_depth_loss']:.4f}")
            print(f"Average KL Loss: {metrics['kl_loss']:.4f}")
            print(f"Average AP Loss: {metrics['ap_loss']:.4f}")
            
            trainer.logger.experiment.add_scalar('epoch/total_loss', metrics['total_loss'], trainer.current_epoch)
            trainer.logger.experiment.add_scalar('epoch/depth_loss', metrics['depth_loss'], trainer.current_epoch)
            trainer.logger.experiment.add_scalar('epoch/intra_depth_loss', metrics['intra_depth_loss'], trainer.current_epoch)
            trainer.logger.experiment.add_scalar('epoch/kl_loss', metrics['kl_loss'], trainer.current_epoch)
            trainer.logger.experiment.add_scalar('epoch/ap_loss', metrics['ap_loss'], trainer.current_epoch)
            
            pl_module.batch_metrics = {
                'depth_loss': [],
                'intra_depth_loss': [],
                'kl_loss': [],
                'ap_loss': [],
                'total_loss': []
            }
            print(f"{'='*50}\n")

        if (trainer.current_epoch + 1) % self.eval_every_n_epochs == 0:
            print(f"\nEpoch {trainer.current_epoch + 1} evaluation start")

            pl_module.eval()
            with torch.no_grad():
                epoch_dir = self.output_dir / f"epoch_{trainer.current_epoch + 1}"
                epoch_dir.mkdir(exist_ok=True)

                # Create args object for visualization
                args = types.SimpleNamespace()
                args.exp_name = self.cfg.model
                args.matcher = self.cfg.matcher
                args.backbone = self.cfg.backbone
                args.dataset = self.cfg.dataset

                # Semantic transfer evaluation for both same and different views
                if 'semantic_transfer' in self.eval_methods:
                    print("\nPerforming semantic transfer evaluation...")

                    # --------------------
                    # Same view evaluation
                    # --------------------
                    print("\nSame view evaluation:")
                    metrics_transfer_same = semantic_transfer(pl_module, same_view=True)
                    metrics_transfer_same.to_csv(epoch_dir / 'semantic_transfer_same.csv')

                    mean_metrics_same = metrics_transfer_same.mean()
                    for metric_name, value in mean_metrics_same.items():
                        trainer.logger.experiment.add_scalar(
                            f'semantic_transfer/same_view/{metric_name}',
                            value,
                            trainer.current_epoch
                        )
                    print("\nSame view semantic transfer metrics:")
                    print(mean_metrics_same)

                    # -------------------------
                    # Different view evaluation
                    # -------------------------
                    print("\nDifferent view evaluation:")
                    metrics_transfer_diff = semantic_transfer(pl_module, same_view=False)
                    metrics_transfer_diff.to_csv(epoch_dir / 'semantic_transfer_diff.csv')

                    mean_metrics_diff = metrics_transfer_diff.mean()
                    for metric_name, value in mean_metrics_diff.items():
                        trainer.logger.experiment.add_scalar(
                            f'semantic_transfer/different_view/{metric_name}',
                            value,
                            trainer.current_epoch
                        )
                    print("\nDifferent view semantic transfer metrics:")
                    print(mean_metrics_diff)

                # Object pose estimation evaluation
                if 'pose' in self.eval_methods:
                    print("\nPerforming object pose estimation evaluation...")
                    
                    metrics_pose = oneposepp(pl_module)
                    metrics_pose.to_csv(epoch_dir / 'pose_estimation.csv')
                    
                    mean_metrics_pose = metrics_pose.mean()
                    for metric_name, value in mean_metrics_pose.items():
                        trainer.logger.experiment.add_scalar(
                            f'object_pose/{metric_name}',
                            value,
                            trainer.current_epoch
                        )
                    print("\nObject pose estimation metrics:")
                    print(mean_metrics_pose)

                # Video tracking evaluation
                if 'tracking' in self.eval_methods:
                    print("\nPerforming video tracking evaluation...")
                    
                    metrics_track = tracking(pl_module, num_videos=30)
                    metrics_track.to_csv(epoch_dir / 'tracking.csv')
                    
                    mean_metrics_track = metrics_track.iloc[:, 1:].mean()
                    for metric_name, value in mean_metrics_track.items():
                        trainer.logger.experiment.add_scalar(
                            f'tracking/{metric_name}',
                            value,
                            trainer.current_epoch
                        )
                    print("\nVideo tracking metrics:")
                    print(mean_metrics_track)

                print(f"\nAll results saved at: {epoch_dir}")

            pl_module.train()


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--exp_name', type=str, default=f'timm')
    parser.add_argument('--matcher', type=str, default='mast3r')
    parser.add_argument('--backbone', type=str, default='ViT-B-16')
    parser.add_argument('--pose', action='store_true')
    parser.add_argument('--tracking', action='store_true')
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--same_view', type=bool, default=False)
    args = parser.parse_args()

    out_dir = Path('evaluation_output') / args.exp_name / args.matcher / args.backbone / start_time
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.matcher == 'vanilla':
        model = VanillaTIMM(
            r=4, backbone_size=args.backbone
        )
    else:
        model = model_name[args.matcher].load_from_checkpoint(
            args.ckpt, r=4, backbone_size=args.backbone
        ).cuda().eval()

    if args.pose:
        metrics_pose = oneposepp(model)
        metrics_pose.to_csv(out_dir / 'pose_estimation.csv')
        print(metrics_pose.mean())

    if args.tracking:
        metrics_track = tracking(model, num_videos=30)
        metrics_track.to_csv(out_dir / 'tracking.csv')
        print(metrics_track.iloc[:, 1:].mean())

    if args.transfer:
        metrics_transfer = semantic_transfer(model, same_view=args.same_view)
        metrics_transfer.to_csv(out_dir / 'semantic_transfer.csv')
        print(metrics_transfer.mean())
