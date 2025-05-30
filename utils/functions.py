import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import functional
import math
import random
import yaml
import kornia
import kornia.filters as KF
import kornia.morphology as KM


def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid

    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y

def img_coord_2_obj_coord(kp2d, depth, k, pose_obj2cam):
    inv_k = np.linalg.inv(k[:3, :3])
    #pose_obj2cam = pose_obj2cam
    kp2d = kp2d[:, :2]
    kp2d = np.concatenate((kp2d, np.ones((kp2d.shape[0], 1))), 1)

    kp2d_int = np.round(kp2d).astype(int)[:, :2]
    kp_depth = depth[kp2d_int[:, 1], kp2d_int[:, 0]]  # num

    kp2d_cam = np.expand_dims(kp_depth, 1) * kp2d  # num, 3
    kp3d_cam = np.dot(inv_k, kp2d_cam.T).T  # num, 3

    kp3d_cam_pad1 = np.concatenate(
        (kp3d_cam, np.ones((kp2d_cam.shape[0], 1))), 1).T  # 4, num
    kp3d_obj = np.dot(np.linalg.inv(pose_obj2cam), kp3d_cam_pad1).T  # num, 4

    return kp3d_obj[:, :3]


# dino patch size is even, so the pixel corner is not really aligned, potential improvements here, borrowed from DINO-Tracker
def interpolate_features(descriptors, pts, h, w, normalize=True, patch_size=14, stride=14):
    last_coord_h = ( (h - patch_size) // stride ) * stride + (patch_size / 2)
    last_coord_w = ( (w - patch_size) // stride ) * stride + (patch_size / 2)
    ah = 2 / (last_coord_h - (patch_size / 2))
    aw = 2 / (last_coord_w - (patch_size / 2))
    bh = 1 - last_coord_h * 2 / ( last_coord_h - ( patch_size / 2 ))
    bw = 1 - last_coord_w * 2 / ( last_coord_w - ( patch_size / 2 ))
    
    a = torch.tensor([[aw, ah]]).to(pts).float()
    b = torch.tensor([[bw, bh]]).to(pts).float()
    keypoints = a * pts + b
    
    # Expand dimensions for grid sampling
    keypoints = keypoints.unsqueeze(-3)  # Shape becomes [batch_size, 1, num_keypoints, 2]
    
    # Interpolate using bilinear sampling
    interpolated_features = F.grid_sample(descriptors, keypoints, align_corners=True, padding_mode='border')
    
    # interpolated_features will have shape [batch_size, channels, 1, num_keypoints]
    interpolated_features = interpolated_features.squeeze(-2)
    
    return F.normalize(interpolated_features, dim=1) if normalize else interpolated_features

def resize_crop(img, padding=0.2, out_size=224, bbox=None):
    # return np.array(img), np.eye(3)
    img = Image.fromarray(img)
    if bbox is None:
        bbox = img.getbbox()
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    size = max(height, width) * (1 + padding)
    center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
    bbox_enlarged = center[0] - size / 2, center[1] - size / 2, \
        center[0] + size / 2, center[1] + size / 2
    img = functional.resize(functional.crop(img, bbox_enlarged[1], bbox_enlarged[0], size, size), (out_size, out_size))
    transform = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1.]])  \
        @ np.array([[size / out_size, 0, 0], [0, size / out_size, 0], [0, 0, 1]]) \
        @ np.array([[1, 0, -out_size / 2], [0, 1, -out_size / 2], [0, 0, 1.]])
    return np.array(img), transform


def parse_yaml(file_path):
    """
    Parses a YAML file and returns the data as a Python dictionary.

    Parameters:
    file_path (str): The path to the YAML file.

    Returns:
    dict: Parsed data from the YAML file.
    """
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("Error parsing YAML file:", exc)
    return data


def query_pose_error(pose_pred, pose_gt, unit='m'):
    """
    Input:
    -----------
    pose_pred: np.array 3*4 or 4*4
    pose_gt: np.array 3*4 or 4*4
    """
    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_gt.shape[0] == 4:
        pose_gt = pose_gt[:3]

    # Convert results' unit to cm
    if unit == 'm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) * 100
    elif unit == 'cm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3])
    elif unit == 'mm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) / 10
    else:
        raise NotImplementedError

    rotation_diff = np.dot(pose_pred[:, :3], pose_gt[:, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
    return angular_distance, translation_distance


def preprocess_kps_pad(kps, img_width, img_height, size):
    # Once an image has been pre-processed via border (or zero) padding,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    kps *= kps[:, 2:3].clone()  # zero-out any non-visible key points
    return kps, offset_x, offset_y, scale


def _fix_pos_enc(patch_size, stride_hw):
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        # compute number of tokens taking stride into account
        w0 = 1 + (w - patch_size) // stride_hw[1]
        h0 = 1 + (h - patch_size) // stride_hw[0]
        assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                        stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False, recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    return interpolate_pos_encoding


def filter_kp_by_conf(kp, conf_mask):
    kp_2d = kp[0]  # shape: (N, 2)
    indices_x = kp_2d[:, 0].round().long()  # shape: (N,)
    indices_y = kp_2d[:, 1].round().long()  # shape: (N,)
    valid = conf_mask[indices_y, indices_x]  # shape: (N,), boolean

    valid_idx = valid.nonzero(as_tuple=False).squeeze(1)  # (N_valid,)
    filtered_kp = kp[:, valid_idx, :]  # (1, N_valid, 2)
    return filtered_kp, valid_idx


def rotation_angle_from_matrix(R):
    trace = torch.trace(R)
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
    return angle


# Reference: https://github.com/luigifreda/pyslam/blob/master/utilities/utils_depth.py

def point_cloud_to_depth(points, K, w, h, device):
    """
    Parameters:
      points: (N, 3) tensor of points in camera coordinate system.
      K: (3, 3) tensor of camera intrinsics.
      w, h: target image width, height.
      device: torch.device
    Returns:
      depth_img: (1, 1, h, w) tensor, float32.
    """
    valid = points[:, 2] > 0
    valid_points = points[valid]
    if valid_points.shape[0] == 0:
        return torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)
    
    X = valid_points[:, 0]
    Y = valid_points[:, 1]
    Z = valid_points[:, 2]
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = torch.round((X / Z) * fx + cx).long()
    v = torch.round((Y / Z) * fy + cy).long()
    
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[mask]
    v = v[mask]
    Z = Z[mask]
    
    idx = v * w + u  # shape: (M,)

    unique_idxs, inv = torch.unique(idx, return_inverse=True)
    accum_Z = torch.bincount(inv, weights=Z)
    counts = torch.bincount(inv)
    avg_Z = accum_Z.float() / counts.float()
    
    depth_img = torch.zeros((h * w,), device=device, dtype=torch.float32)
    depth_img[unique_idxs] = avg_Z
    depth_img = depth_img.view(h, w)
    depth_img = depth_img.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
    return depth_img


def post_process_depth(
        depth_img, 
        kernel_size=5, # 3
        bilateral_d=3, 
        bilateral_sigma_color=0.1, # 0.1
        bilateral_sigma_space=1.0, 
        guided_r=8, 
        guided_eps=1e-2
    ):
    if depth_img.dim() == 2:
        depth_img = depth_img.unsqueeze(0).unsqueeze(0)
    elif depth_img.dim() == 3:
        depth_img = depth_img.unsqueeze(0)
    
    device = depth_img.device
    
    pad = kernel_size // 2
    dilated = F.max_pool2d(depth_img, kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-dilated, kernel_size, stride=1, padding=pad)
    
    empty_mask = (eroded < 1e-5).float()
    
    if empty_mask.sum() > 0:
        valid_mask = 1.0 - empty_mask

        large_kernel = torch.ones((1, 1, 5, 5), device=device)
        
        expanded_valid = F.conv2d(valid_mask, large_kernel, padding=2)
        expanded_valid = (expanded_valid > 0).float()
        
        dist_weight = F.conv2d(valid_mask, large_kernel, padding=2)
        
        value_prop = F.conv2d(eroded * valid_mask, large_kernel, padding=2)
        
        normalized_values = value_prop / (dist_weight + 1e-8)
        
        fill_mask = (expanded_valid - valid_mask).clamp(0, 1)
        eroded = eroded * valid_mask + normalized_values * fill_mask
        
        wider_kernel = torch.ones((1, 1, 7, 7), device=device)
        
        valid_mask_updated = (eroded > 0).float()
        expanded_valid = F.conv2d(valid_mask_updated, wider_kernel, padding=3)
        expanded_valid = (expanded_valid > 0).float()
        
        dist_weight = F.conv2d(valid_mask_updated, wider_kernel, padding=3)
        value_prop = F.conv2d(eroded * valid_mask_updated, wider_kernel, padding=3)
        normalized_values = value_prop / (dist_weight + 1e-8)
        
        fill_mask = (expanded_valid - valid_mask_updated).clamp(0, 1)
        eroded = eroded * valid_mask_updated + normalized_values * fill_mask
    
    depth_filled = eroded
    
    depth_median = KF.median_blur(depth_filled, kernel_size=(kernel_size, kernel_size))
    
    guide_img = depth_median.clone()
    depth_bilateral = KF.bilateral_blur(
        depth_median,
        kernel_size=(bilateral_d, bilateral_d),
        sigma_color=bilateral_sigma_color,
        sigma_space=(bilateral_sigma_space, bilateral_sigma_space)
    )
    
    depth_guided = KF.guided_blur(depth_bilateral, guide_img, guided_r, guided_eps)

    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device) / (kernel_size**2)
    local_mean = F.conv2d(depth_guided, kernel, padding=pad)
    local_var = F.conv2d(depth_guided**2, kernel, padding=pad) - local_mean**2
    local_std = torch.sqrt(local_var.clamp(min=1e-6))
    
    outlier_mask = (torch.abs(depth_guided - local_mean) > 3.0 * local_std).float()
    
    depth_filtered = depth_guided * (1.0 - outlier_mask) + depth_median * outlier_mask
    
    depth_final = KF.joint_bilateral_blur(
        depth_filtered,
        guide_img,
        kernel_size=(bilateral_d, bilateral_d),
        sigma_color=bilateral_sigma_color/2,
        sigma_space=(bilateral_sigma_space, bilateral_sigma_space)
    )
    
    return depth_final.squeeze()


def extract_kp_depth(depth_map, kp, window_size=3):
    B, N, _ = kp.shape
    
    if not torch.is_tensor(depth_map):
        depth_map = torch.tensor(depth_map, device=kp.device, dtype=torch.float)

    depth_map = depth_map.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)

    H, W = depth_map.shape[-2:]
    half = window_size // 2

    padded = F.pad(depth_map, (half, half, half, half), mode='replicate')  # shape: (1,1,H+2*half, W+2*half)
    
    patches = F.unfold(padded, kernel_size=window_size, stride=1)
    
    # patch_means = patches.mean(dim=1).squeeze(0)  # shape: (H*W)
    patch_means = patches.mean(dim=1) # shape: (B, H*W)
    
    # indices = kp[0, :, 1] * W + kp[0, :, 0]  # shape: (num_points,)
    indices = kp[..., 1] * W + kp[..., 0]  # shape: (B, num_points,)
    
    # avg_depths = patch_means[indices.long()]  # shape: (num_points)
    avg_depths = patch_means.gather(dim=1, index=indices.long())  # shape: (B, num_points)
    
    return avg_depths


def get_patch_mask_from_kp_tensor(kp_xy, H, W, patch_size, device=None):
    if device is None:
        device = kp_xy.device

    patch_h = H // patch_size
    patch_w = W // patch_size
    num_patches = patch_h * patch_w

    valid_mask = (kp_xy[:, 0] >= 0) & (kp_xy[:, 0] < W) \
            & (kp_xy[:, 1] >= 0) & (kp_xy[:, 1] < H)
    kp_xy_valid = kp_xy[valid_mask]  # shape: (M, 2), M <= N

    if kp_xy_valid.shape[0] == 0:
        return torch.zeros(num_patches, dtype=torch.bool, device=device)

    x_idx = kp_xy_valid[:, 0].long() // patch_size
    y_idx = kp_xy_valid[:, 1].long() // patch_size

    patch_idx = y_idx * patch_w + x_idx
    # shape: (M,)

    patch_mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
    patch_mask[patch_idx] = True

    return patch_mask


def get_masked_patch_cost(cost, mask_patch_1, mask_patch_2=None, eps=1e-8, use_softmax=False, temperature=1.0):
    B, hw, hw2 = cost.shape

    if mask_patch_2 is not None:
        mask_2d = mask_patch_1.unsqueeze(1) * mask_patch_2.unsqueeze(0)
    else:
        # do NOT mask on view 2
        mask_2d = mask_patch_1.unsqueeze(1) * torch.ones_like(mask_patch_1).unsqueeze(0)
    mask_2d = mask_2d.unsqueeze(0).expand(B, hw, hw2)

    masked_cost = cost.clone()
    masked_cost[~mask_2d] = 0.0

    if use_softmax:
        # masked_cost = torch.softmax(masked_cost, dim=-1)
        masked_cost = torch.softmax(masked_cost / temperature, dim=-1, dtype=torch.float32)
    else:
        row_sum = masked_cost.sum(dim=-1, keepdim=True).clamp_min(eps)
        masked_cost = masked_cost / row_sum

    return masked_cost


def compute_projection(P, points_3d):
    """    
    Args:
        P: (3,4) torch tensor, projection matrix.
        points_3d: (..., 3) tensor of 3D world points.
        
    Returns:
        proj_points: (..., 2) tensor of 2D pixel coordinates.
    """
    orig_shape = points_3d.shape[:-1]
    points_flat = points_3d.view(-1, 3)  # (N,3)
    ones = torch.ones((points_flat.shape[0], 1), dtype=points_flat.dtype, device=points_flat.device)
    points_h = torch.cat([points_flat, ones], dim=1)  # (N,4)
    
    proj_h = P @ points_h.T  # (3,N)
    proj_h = proj_h.T        # (N,3)
    proj_points = proj_h[:, :2] / (proj_h[:, 2:3] + 1e-8)
    return proj_points.view(*orig_shape, 2)


def get_coview_mask(point_map, P_other, image_shape):
    proj_points = compute_projection(P_other, point_map)
    u = proj_points[..., 0]
    v = proj_points[..., 1]
    H_img, W_img = image_shape
    mask = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
    return mask


def convert_camera_to_world(point_map, extrinsic):
    R = extrinsic[:, :3]  # (3,3)
    t = extrinsic[:, 3].unsqueeze(0)  # (1,3)
    R_inv = R.t()  # Inverse of R
    world_points = torch.matmul(point_map - t, R_inv)
    return world_points


def get_coview_masks(point_map_view1, point_map_view2, intrinsic1, extrinsic1, intrinsic2, extrinsic2, image_shape):
    world_points_view1 = convert_camera_to_world(point_map_view1, extrinsic1)
    world_points_view2 = convert_camera_to_world(point_map_view2, extrinsic1)
    
    P1 = intrinsic1 @ extrinsic1  # view1: world → view1 image
    P2 = intrinsic2 @ extrinsic2  # view2: world → view2 image
    
    mask1 = get_coview_mask(world_points_view1, P2, image_shape)
    mask2 = get_coview_mask(world_points_view2, P1, image_shape)
    
    return mask1, mask2


def sample_keypoints_nms(mask, conf, N, min_distance, device=None):
    if device is None:
        device = mask.device
    H, W = mask.shape

    score_map = torch.zeros_like(mask, dtype=torch.float32, device=device)
    score_map[mask] = conf[mask]
    
    kernel_size = int(min_distance) * 2 + 1
    pad = kernel_size // 2

    pooled = F.max_pool2d(score_map.unsqueeze(0).unsqueeze(0),
                          kernel_size=kernel_size,
                          stride=1,
                          padding=pad)
    pooled = pooled.squeeze()  # (H, W)

    eps = 1e-6
    nms_mask = (score_map - pooled).abs() < eps
    nms_mask = nms_mask & mask
    
    keypoints = torch.nonzero(nms_mask, as_tuple=False)  # (M, 2)
    
    M = keypoints.shape[0]
    if M == 0:
        return None

    if M > N:
        perm = torch.randperm(M, device=device)[:N]
        sampled_keypoints = keypoints[perm]
    else:
        sampled_keypoints = keypoints
    return sampled_keypoints

