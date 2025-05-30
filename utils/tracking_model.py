import gc
import math
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from torchvision import transforms as T

EPS = 1e-08


class RangeNormalizer(torch.nn.Module):
    """
    Scales dimensions to specific ranges.
    Will be used to normalize pixel coords. & time to destination ranges.
    For example: [0, H-1] x [0, W-1] x [0, T-1] -> [0,1] x [0,1] x [0,1]

    Args:
         shapes (tuple): represents the "boundaries"/maximal values for each input dimension.
            We assume that the dimensions range from 0 to max_value (as in pixels & frames).
    """
    def __init__(self, shapes: tuple, device='cuda'):
        super().__init__()

        normalizer = torch.tensor(shapes).float().to(device) - 1
        self.register_buffer("normalizer", normalizer)

    def forward(self, x, dst=(0, 1), dims=[0, 1, 2]):
        """
        Normalizes input to specific ranges.
        
            Args:       
                x (torch.tensor): input data
                dst (tuple, optional): range inputs where normalized to. Defaults to (0, 1).
                dims (list, optional): dimensions to normalize. Defaults to [0, 1, 2].
                
            Returns:
                normalized_x (torch.tensor): normalized input data
        """
        normalized_x = x.clone()
        normalized_x[:, dims] = x[:, dims] / self.normalizer[dims] # normalize to [0,1]
        normalized_x[:, dims] = (dst[1] - dst[0]) * normalized_x[:, dims] + dst[0] # shift range to dst

        return normalized_x
    
    def unnormalize(self, normalized_x:torch.tensor, src=(0, 1), dims=[0, 1, 2]):
        """Runs to reverse process of forward, unnormalizes input to original scale.

        Args:
            normalized_x (torch.tensor): input data
            src (tuple, optional): range inputs where normalized to. Defaults to (0, 1). unnormalizes from src to original scales.
            dims (list, optional): dimensions to normalize. Defaults to [0, 1, 2].

        Returns:
            x (torch.tensor): unnormalized input data
        """
        x = normalized_x.clone()
        x[:, dims] = (normalized_x[:, dims] - src[0]) / (src[1] - src[0]) # shift range to [0,1]
        x[:, dims] = x[:, dims] * self.normalizer[dims] # unnormalize to original ranges
        return x
    


def bilinear_interpolate_video(video:torch.tensor, points:torch.tensor, h:int, w:int, t:int, normalize_h=False, normalize_w=False, normalize_t=True):
    """
    Sample embeddings from an embeddings volume at specific points, using bilear interpolation per timestep.

    Args:
        video (torch.tensor): a volume of embeddings/features previously extracted from an image. shape: 1 x C x T x H' x W'
            Most likely used for DINO embeddings 1 x C x T x H' x W' (C=DINO_embeddings_dim, W'= W//8 & H'=H//8 of original image).
        points (torch.tensor): batch of B points (pixel cooridnates) (x,y,t) you wish to sample. shape: B x 3.
        h (int): True Height of images (as in the points) - H.
        w (int): Width of images (as in the points) - W.
        t (int): number of frames - T.

    Returns:
        sampled_embeddings: sampled embeddings at specific posiitons. shape: 1 x C x 1 x B x 1.
    """
    samples = points[None, None, :, None].detach().clone() # expand shape B x 3 TO (1 x 1 x B x 1 x 3), we clone to avoid altering the original points tensor.     
    if normalize_w:
        samples[:, :, :, :, 0] = samples[:, :, :, :, 0] / (w - 1)  # normalize to [0,1]
        samples[:, :, :, :, 0] = samples[:, :, :, :, 0] * 2 - 1  # normalize to [-1,1]
    if normalize_h:
        samples[:, :, :, :, 1] = samples[:, :, :, :, 1] / (h - 1)  # normalize to [0,1]
        samples[:, :, :, :, 1] = samples[:, :, :, :, 1] * 2 - 1  # normalize to [-1,1]
    if normalize_t:
        if t > 1:
            samples[:, :, :, :, 2] = (samples[:, :, :, :, 2]) / (t - 1)  # normalize to [0,1]
            samples[:, :, :, :, 2] = samples[:, :, :, :, 2] * 2 - 1  # normalize to [-1,1]
    return torch.nn.functional.grid_sample(video, samples, align_corners=True, padding_mode ='border') # points out-of bounds are padded with border values



# copied from OmniMotion
def gen_grid(h_start, w_start, h_end, w_end, step_h, step_w, device, normalize=False, homogeneous=False):
    """Generate a grid of coordinates in the image frame.
    Args:
        h, w: height and width of the grid.
        device: device to put the grid on.
        normalize: whether to normalize the grid coordinates to [-1, 1].
        homogeneous: whether to return the homogeneous coordinates. homogeneous coordinates are 3D coordinates.
    Returns:"""
    if normalize:
        lin_y = torch.linspace(-1., 1., steps=h_end, device=device)
        lin_x = torch.linspace(-1., 1., steps=w_end, device=device)
    else:
        lin_y = torch.arange(h_start, h_end, step=step_h, device=device)
        lin_x = torch.arange(w_start, w_end, step=step_w, device=device)
    grid_y, grid_x = torch.meshgrid((lin_y, lin_x))
    grid = torch.stack((grid_x, grid_y), -1)
    if homogeneous:
        grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
    return grid  # [h, w, 2 or 3]


class TrackerHead(nn.Module):
    def __init__(self,
                 in_channels=1,
                 hidden_channels=16,
                 out_channels=1,
                 kernel_size=3,
                 stride=1,
                 
                 patch_size=14,
                 step_h=14,
                 step_w=14,
                 argmax_radius=35,
                 video_h=480,
                 video_w=640):
        super(TrackerHead, self).__init__()
        
        padding = kernel_size // 2
        
        self.softmax = nn.Softmax(dim=2)
        self.argmax_radius = argmax_radius
        self.patch_size = patch_size
        self.step_h = step_h
        self.step_w = step_w
        self.video_h=video_h
        self.video_w=video_w
    
    def soft_argmax(self, heatmap, argmax_indices):
        """
        heatmap: shape (B, H, W)
        """
        h_start = self.patch_size // 2
        w_start = self.patch_size // 2
        h_end = ((self.video_h - 2 * h_start) // self.step_h) * self.step_h + h_start + math.ceil(self.step_h / 2)
        w_end = ((self.video_w - 2 * w_start) // self.step_w) * self.step_w + w_start + math.ceil(self.step_w / 2)
        grid = gen_grid(h_start=h_start, w_start=w_start, h_end=h_end, w_end=w_end, step_h=self.step_h, step_w=self.step_w,
                        device=heatmap.device, normalize=False, homogeneous=False) # shape (H, W, 2)
        grid = grid.unsqueeze(0).repeat(heatmap.shape[0], 1, 1, 1) # stack and repeat grid to match heatmap shape (B, H, W, 2)
        
        row, col = argmax_indices
        argmax_coord = torch.stack((col*self.step_w+w_start, row*self.step_h+h_start), dim=-1) # (x,y) coordinates, shape (B, 2)
        
        # generate a mask of a circle of radius radius around the argmax_coord (B, 2) in heatmap (B, H, W, 2)
        mask = torch.norm((grid - argmax_coord.unsqueeze(1).unsqueeze(2)).to(torch.float32), dim=-1) <= self.argmax_radius # shape (B, H, W)
        heatmap = heatmap * mask
        hm_sum = torch.sum(heatmap, dim=(1, 2)) # B
        hm_zero_indices = hm_sum < 1e-8
        
        # for numerical stability
        if sum(hm_zero_indices) > 0:
            uniform_w = 1 / mask[hm_zero_indices].sum(dim=(1,2))
            heatmap[hm_zero_indices] += uniform_w[:, None, None]
            heatmap[hm_zero_indices] = heatmap[hm_zero_indices] * mask[hm_zero_indices]
            hm_sum[hm_zero_indices] = torch.sum(heatmap[hm_zero_indices], dim=(1, 2))

        point = torch.sum(grid * heatmap.unsqueeze(-1), dim=(1, 2)) / hm_sum.unsqueeze(-1) # shape (B, 2)

        return point
    
    def softmax_heatmap(self, hm):
        b, c, h, w = hm.shape
        hm_sm = rearrange(hm, "b c h w -> b c (h w)") # shape (B, 1, H*W)
        hm_sm = self.softmax(hm_sm) # shape (B, 1, H*W)
        hm_sm = rearrange(hm_sm, "b c (h w) -> b c h w", h=h, w=w) # shape (B, 1, H, W)
        return hm_sm
    
    def forward(self, cost_volume):
        """
        cost_volume: shape (B, C, H, W)
        """
        
        range_normalizer = RangeNormalizer(shapes=(self.video_w, self.video_h)) # shapes are (W, H), correpsonding to (x, y) coordinates
        
        # crop heatmap around argmax point
        argmax_flat = torch.argmax(rearrange(cost_volume[:, 0], "b h w -> b (h w)"), dim=1)
        argmax_indices = (argmax_flat // cost_volume[:, 0].shape[-1], argmax_flat % cost_volume[:, 0].shape[-1])

        refined_heatmap = self.softmax_heatmap(cost_volume) # shape (B, 1, H, W)
        point = self.soft_argmax(refined_heatmap.squeeze(1),
                                 argmax_indices) # shape (B, 2), (x,y) coordinates
        return range_normalizer(point, dst=(-1,1), dims=[0, 1]) # shape (B, 2)
    
    
class Tracker(nn.Module):
    def __init__(
        self,
        dino_features,
        video=None,
        ckpt_path="",
        dino_patch_size=14,
        stride=7,
        device="cuda:0",
        ):
        super().__init__()

        self.stride = stride
        self.dino_patch_size = dino_patch_size
        self.device = device
        self.refined_features = None
        self.ckpt_path = ckpt_path
        
        self.video = video
        
        # DINO embed
        self.dino_embed_video = dino_features  # T x C x H x W

        # CNN-Refiner
        t, c, h, w = self.video.shape
        self.cmap_relu = nn.ReLU(inplace=True)
        self.tracker_head = TrackerHead(patch_size=dino_patch_size,
                                        step_h=stride,
                                        step_w=stride,
                                        video_h=h,
                                        video_w=w).to(device)
        self.range_normalizer = RangeNormalizer(shapes=(w, h, self.video.shape[0]))
  
    def get_dino_embed_video(self, frames_set_t):
        dino_emb = self.dino_embed_video[frames_set_t.to(self.dino_embed_video.device)] if frames_set_t.device != self.dino_embed_video.device else self.dino_embed_video[frames_set_t]
        return dino_emb
    
    def normalize_points_for_sampling(self, points):
        t, c, vid_h, vid_w = self.video.shape
        h = vid_h
        w = vid_w
        patch_size = self.dino_patch_size
        stride = self.stride
        
        last_coord_h =( (h - patch_size) // stride ) * stride + (patch_size / 2)
        last_coord_w =( (w - patch_size) // stride ) * stride + (patch_size / 2)
        ah = 2 / (last_coord_h - (patch_size / 2))
        aw = 2 / (last_coord_w - (patch_size / 2))
        bh = 1 - last_coord_h * 2 / ( last_coord_h - ( patch_size / 2 ))
        bw = 1 - last_coord_w * 2 / ( last_coord_w - ( patch_size / 2 ))
        
        a = torch.tensor([[aw, ah, 1]]).to(self.device)
        b = torch.tensor([[bw, bh, 0]]).to(self.device)
        normalized_points = a * points + b
        
        # normalized_points = points.clone()
        # h = vid_h // patch_size * patch_size
        # w = vid_w // patch_size * patch_size

        # # convert keypoint location to pixel center
        # normalized_points[..., 0] = ((normalized_points[..., 0] + 0.5) / w) * 2 - 1  # x coordinates
        # normalized_points[..., 1] = ((normalized_points[..., 1] + 0.5) / h) * 2 - 1  # y coordinates
        
        return normalized_points
    
    def sample_embeddings(self, embeddings, source_points):
        """embeddings: T x C x H x W. source_points: B x 3, where the last dimension is (x, y, t), x and y are in [-1, 1]"""
        t, c, h, w = embeddings.shape
        sampled_embeddings = bilinear_interpolate_video(video=rearrange(embeddings, "t c h w -> 1 c t h w"),
                                                               points=source_points,
                                                               h=h,
                                                               w=w,
                                                               t=t,
                                                               normalize_w=False,
                                                               normalize_h=False,
                                                               normalize_t=True)
        sampled_embeddings = sampled_embeddings.squeeze()
        if len(sampled_embeddings.shape) == 1:
            sampled_embeddings = sampled_embeddings.unsqueeze(1)
        sampled_embeddings = sampled_embeddings.permute(1,0)
        return sampled_embeddings
    
    def uncache_refined_embeddings(self, move_dino_to_gpu=False):
        self.refined_features = None
        torch.cuda.empty_cache()
        gc.collect()
        if move_dino_to_gpu:
            self.dino_embed_video = self.dino_embed_video.to("cuda")
    
    def get_corr_maps_for_frame_set(self, source_embeddings, frame_embeddings_set, target_frame_indices):
        corr_maps_set = torch.einsum("bc,nchw->bnhw", source_embeddings, frame_embeddings_set)
        corr_maps = corr_maps_set[torch.arange(source_embeddings.shape[0]), target_frame_indices.int(), :, :]
        
        embeddings_norm = frame_embeddings_set.norm(dim=1)
        target_embeddings_norm = embeddings_norm[target_frame_indices.int()]
        source_embeddings_norm = source_embeddings.norm(dim=1).unsqueeze(-1).unsqueeze(-1)
        corr_maps_norm = (source_embeddings_norm * target_embeddings_norm)
        corr_maps = corr_maps / torch.clamp(corr_maps_norm, min=EPS)
        corr_maps = rearrange(corr_maps, "b h w -> b 1 h w")
        
        return corr_maps
    
    def get_point_predictions_from_embeddings(self, source_embeddings, frame_embeddings_set, target_frame_indices):
        corr_maps = self.get_corr_maps_for_frame_set(source_embeddings, frame_embeddings_set, target_frame_indices)
        coords = self.tracker_head(self.cmap_relu(corr_maps))
        return coords
    
    def get_point_predictions(self, inp, frame_embeddings):
        source_points_unnormalized, source_frame_indices, target_frame_indices, _ = inp
        source_points = self.normalize_points_for_sampling(source_points_unnormalized)
        # print(frame_embeddings.device, source_points.device, source_frame_indices.device, target_frame_indices.device)
        source_embeddings = self.sample_embeddings(frame_embeddings, torch.cat([ source_points[:, :-1], source_frame_indices[:, None] ], dim=1)) # B x C
        return self.get_point_predictions_from_embeddings(source_embeddings, frame_embeddings, target_frame_indices)

    def forward(self, inp, use_raw_features=False):
        """
        inp: source_points_unnormalized, source_frame_indices, target_frame_indices, frames_set_t; where
        source_points_unnormalized: B x 3. ((x, y, t) in image scale - NOT normalized)
        source_frame_indices: the indices of frames of source points in frames_set_t
        target_frame_indices: the indices of target frames in frames_set_t
        frames_set_t: N, 0 to T-1 (NOT normalized)
        """
        frames_set_t = inp[-1]
        
        if use_raw_features:
            frame_embeddings = raw_embeddings = self.get_dino_embed_video(frames_set_t=frames_set_t)
        self.frame_embeddings = frame_embeddings
        self.raw_embeddings = raw_embeddings
        coords = self.get_point_predictions(inp, frame_embeddings)

        return coords


class Dust3rTracker(Tracker):
    def __init__(self, dust_model, video, ckpt_path="", dino_patch_size=14, stride=7, device="cuda:0"):
        super().__init__(None, video, ckpt_path, dino_patch_size, stride, device)
        self.dust_model = dust_model
        self.transforms = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.video_transformed = self.transforms(video)
        self.reset()
        
    def reset(self):
        self.cache = {}
        
    def forward(self, inp, use_raw_features=False):
        """
        inp: source_points_unnormalized, source_frame_indices, target_frame_indices, frames_set_t; where
        source_points_unnormalized: B x 3. ((x, y, t) in image scale - NOT normalized)
        source_frame_indices: the indices of frames of source points in frames_set_t
        target_frame_indices: the indices of target frames in frames_set_t
        frames_set_t: N, 0 to T-1 (NOT normalized)
        """
        source_points_unnormalized, source_frame_indices, target_frame_indices, frames_set_t = inp
        coords = []
        for i in range(source_frame_indices.shape[0]):
            src_idx, tgt_idx = frames_set_t[source_frame_indices[i]], frames_set_t[target_frame_indices[i]]
            if (src_idx.item(), tgt_idx.item()) not in self.cache:
                view1 = {'img': self.video_transformed[src_idx:src_idx+1], 'instance': [1]}
                view2 = {'img': self.video_transformed[tgt_idx:tgt_idx+1], 'instance': [2]}
                
                res1, res2 = self.dust_model(view1, view2)
                
                pts1 = res1['pts3d']
                pts2 = res2['pts3d_in_other_view']
                
                self.cache[(src_idx.item(), tgt_idx.item())] = (pts1, pts2)
            else:
                pts1, pts2 = self.cache[(src_idx.item(), tgt_idx.item())]
            
            src_pts3d = pts1[0][source_points_unnormalized[i, 1].int().item(), source_points_unnormalized[i, 0].int().item()]
            best_idx = torch.argmin(torch.norm(pts2.reshape(-1, 3) - src_pts3d, dim=-1))
            best_h = best_idx // (self.video.shape[-1])
            best_w = best_idx % (self.video.shape[-1])
            
            coords.append(torch.tensor([best_w, best_h], device=self.device))
        coords = self.range_normalizer(torch.stack(coords).float(), dst=(-1, 1), dims=[0, 1])
        return coords


# ---- Functions for generating trajectories ----
def generate_trajectory_input(query_point, video, start_t=None, end_t=None):
    """
    Receives a single point (x,y,t) and the video, and generates input for Tracker model.
    Args:
        query_point: shape 3. (x,y,t).
        video: shape T x H x W x 3.
    Returns:
        source_points, source_frame_indices, target_frame_indices, frames_set_t.
        source_points: query_point repeated rest times. shape rest x 3. (x,y,t).
        source_frame_indices: [0] repeated rest times. shape rest x 1.
        target_frame_indices: 0 to rest-1. shape rest.
        frames_set_t: [query_point[0, 2], start_t, ..., end_t]. shape rest + 1.
    """
    start_t = 0 if start_t is None else start_t
    end_t = video.shape[0] if end_t is None else end_t
    video_subset = video[start_t:end_t]
    rest = video_subset.shape[0]
    device = video.device
    
    source_points = query_point.unsqueeze(0).repeat(rest, 1) # rest x 3

    frames_set_t = torch.arange(start_t, end_t, dtype=torch.long, device=device) # rest
    frames_set_t = torch.cat([ torch.tensor([query_point[2]], device=device), frames_set_t ]).int() # rest + 1
    source_frame_indices = torch.tensor([0], device=device).repeat(end_t-start_t) # rest
    target_frame_indices = torch.arange(rest, dtype=torch.long, device=device) + 1 # T
    
    return source_points, source_frame_indices, target_frame_indices, frames_set_t


@torch.no_grad()
def generate_trajectory(query_point:torch.tensor, video:torch.tensor, model:torch.nn.Module, range_normalizer:RangeNormalizer, dst_range=(-1, 1), use_raw_features=False,
                               batch_size=None) -> torch.tensor:
    """
    Genrates trajectory using tracker predictions for all timesteps.
    Returns:
        trajectory_pred: rest x 3. (x,y,t) coordinates for each timestep.
    """
    batch_size = video.shape[0] if batch_size is None else batch_size
    
    trajectory_pred = []
    for start_t in range(0, video.shape[0], batch_size):
        end_t = min(start_t + batch_size, video.shape[0])
        trajectory_input = generate_trajectory_input(query_point, video, start_t=start_t, end_t=end_t)
        trajectory_coordinate_preds_normalized = model(trajectory_input, use_raw_features=use_raw_features)
        trajectory_coordinate_preds = range_normalizer.unnormalize(trajectory_coordinate_preds_normalized, dims=[0,1], src=dst_range)
        trajectory_timesteps = trajectory_input[-1][1:].to(dtype=torch.float32) # rest
        trajectory_pred_cur = torch.cat([trajectory_coordinate_preds, trajectory_timesteps.unsqueeze(dim=1)], dim=1)
        trajectory_pred.append(trajectory_pred_cur)
    trajectory_pred = torch.cat(trajectory_pred, dim=0)
    return trajectory_pred

@torch.no_grad()
def generate_trajectories(query_points:torch.tensor, video:torch.tensor, model:torch.nn.Module, range_normalizer:RangeNormalizer, dst_range=(-1, 1), use_raw_features=False,
                                 batch_size=None) -> torch.tensor:
    """
    Genrates trajectories using tracker predictions. wraps generate_trajectory function.
    Returns:
        trajectories: len(query_points) x rest x 3. (x,y,t) coordinates for each trajectory.
    """
    trajectories_list = []
    query_points = query_points.to(dtype=torch.float32) # just in case
    for query_point in query_points:
        trajectory_pred = generate_trajectory(query_point=query_point, video=video, model=model, range_normalizer=range_normalizer, dst_range=dst_range, use_raw_features=use_raw_features,
                                                     batch_size=batch_size)
        trajectories_list.append(trajectory_pred)
    trajectories = torch.stack(trajectories_list)
    return trajectories



class ModelInference(torch.nn.Module):
    def __init__(
        self,
        model: Tracker,
        range_normalizer: RangeNormalizer,
        anchor_cosine_similarity_threshold: float = 0.5,
        cosine_similarity_threshold: float = 0.5,
        ) -> None:
        super().__init__()


        self.model = model
        self.model.eval()

        self.range_normalizer = range_normalizer
        self.anchor_cosine_similarity_threshold = anchor_cosine_similarity_threshold
        self.cosine_similarity_threshold = cosine_similarity_threshold
    
    def compute_trajectories(self, query_points: torch.Tensor, batch_size=None,) -> torch.Tensor:
        trajecroies = generate_trajectories(
            query_points=query_points,
            model=self.model,
            video=self.model.video,
            range_normalizer=self.range_normalizer,
            dst_range=(-1,1),
            use_raw_features=True,
            batch_size=batch_size,
        )
        return trajecroies
    
    # ----------------- Cosine Similarity -----------------
    def compute_trajectory_cos_sims(self, trajectories, query_points) -> torch.Tensor:
        """Compute cosine similarities between trajectories and query points.
        Args:
            trajectories (torch.Tensor): Trajectories. N x T x 3. N is the number of trajectories. T is the number of time steps. (x, y, t).
            query_points (torch.Tensor): Query points. N x 3. used for retrieving corresponding query frames.
        Returns:
            trajectories_cosine_similarities (torch.Tensor): Cosine similarities between trajectories and query points. N x T."""
        # compute refined_features_at_trajectories
        N, T = trajectories.shape[:2]
        trajectories_normalized = self.model.normalize_points_for_sampling(trajectories) # N x T x 3
        features = self.model.get_dino_embed_video(frames_set_t=torch.arange(0, self.model.video.shape[0]))
        refined_features_at_trajectories = self.model.sample_embeddings(features, trajectories_normalized.view(-1, 3)) # (N*T) x C
        refined_features_at_trajectories = refined_features_at_trajectories.view(N, T, -1) # N x T x C
        
        query_frames = query_points[:, 2].long() # N
        refined_features_at_query_frames = refined_features_at_trajectories[torch.arange(N).to(self.model.device), query_frames] # N x C
        trajectories_cosine_similarities = torch.nn.functional.cosine_similarity(refined_features_at_query_frames.unsqueeze(1), refined_features_at_trajectories, dim=-1) # N x T
        return trajectories_cosine_similarities


    # ----------------- Anchor Trajectories -----------------
    def _get_model_preds_at_anchors(self, model, range_normalizer, preds, anchor_indices, batch_size=None):
        """ preds: N"""
        batch_size = batch_size if batch_size is not None else preds.shape[0]
        
        cycle_coords = []
        for vis_frame in anchor_indices:
            # iterate over frames_set_t in batches of size batch_size
            coords = []
            for i in range(0, preds.shape[0], batch_size):
                end_idx = min(i + batch_size, preds.shape[0])
                frames_set_t = torch.arange(i, end_idx, device=model.device)
                frames_set_t = torch.cat([ torch.tensor([vis_frame], device=model.device), frames_set_t ]).int()
                source_frame_indices = torch.arange(1, frames_set_t.shape[0], device=model.device)
                target_frame_indices = torch.tensor([0]*(frames_set_t.shape[0]-1), device=model.device)
                inp = preds[i:end_idx], source_frame_indices, target_frame_indices, frames_set_t
                batch_coords = model(inp, use_raw_features=True) # batch_size x 3
                batch_coords = range_normalizer.unnormalize(batch_coords, src=(-1, 1), dims=[0, 1])
                coords.append(batch_coords)
            coords = torch.cat(coords)
            
            cycle_coords.append(coords[:, :2]) # prediction of a target point to the top percentile
            
        cycle_coords = torch.stack(cycle_coords) # N_anchors x T x 2
        
        return cycle_coords
    
    def compute_anchor_trajectories(self, trajectories: torch.Tensor, cos_sims: torch.Tensor, batch_size=None) -> torch.Tensor:
        N, T = trajectories.shape[:2]
        eql_anchor_cyc_predictions = {}
            
        for qp_idx in tqdm(range(N), desc=f"Interating over query points"):
            preds = trajectories[qp_idx] # (T x 3)
            anchor_frames = torch.arange(T).to(self.model.device)[cos_sims[qp_idx] >= self.anchor_cosine_similarity_threshold] # T
            cycle_coords_eql_anchor = self._get_model_preds_at_anchors(self.model, self.range_normalizer, preds=preds, anchor_indices=anchor_frames, batch_size=batch_size)
            eql_anchor_cyc_predictions[qp_idx] = cycle_coords_eql_anchor
        return eql_anchor_cyc_predictions
    
    
    # ----------------- Occlusion -----------------
    def compute_occ_pred_for_qp(self, green_trajectories_qp: torch.tensor, source_trajectories_qp: torch.tensor, traj_cos_sim_qp: torch.tensor, anch_sim_th: float, cos_sim_th: float):
        visible_at_st_frame_qp = traj_cos_sim_qp >= anch_sim_th
        dists_from_source = torch.norm(green_trajectories_qp - source_trajectories_qp[visible_at_st_frame_qp, :].unsqueeze(1), dim=-1)  # dists_from_source (M x T), dists_from_source[anchor_t, source_t] = dist

        anchor_median_errors = torch.median(dists_from_source[:, visible_at_st_frame_qp], dim=0).values  # T_vis
        median_anchor_dist_th = anchor_median_errors.max()  # float
        dists_from_source_anchor_vis = dists_from_source  # (T_vis x T)
        median_dists_from_source_anchor_vis = torch.median(dists_from_source_anchor_vis, dim=0).values  # T
        return ((median_dists_from_source_anchor_vis > median_anchor_dist_th) | (traj_cos_sim_qp < cos_sim_th))

    def compute_occlusion(self, trajectories: torch.Tensor, trajs_cos_sims: torch.Tensor, anchor_trajectories: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Compute occlusion for trajectories.
        Args:
            trajectories (torch.Tensor): Trajectories. N x T x 3. N is the number of trajectories. T is the number of time steps. trajectory for qp_idx in query_points (N).
            trajs_cos_sims (torch.Tensor): Cosine similarities between trajectories and query points. N x T. traj_cos_sims[qp_idx, t] = cos_sim
            anchor_trajectories dict(torch.Tensor): Anchor trajectories. {qp_idx: T x T x 2}. N is the number of trajectories.
        Returns:
            occ_preds (torch.Tensor): Occlusion predictions. N x T. occ_preds[qp_idx, t] = 1 if occluded, 0 otherwise.
        """

        N = trajectories.shape[0]
        occ_preds_by_dist_th_anchor_frame_vis = []

        for qp_idx in range(N):
            source_trajectories_qp = trajectories[qp_idx, :, :2] # source_trajectories_qp (T x 2)
            traj_cos_sim_qp = trajs_cos_sims[qp_idx] # cos_sim_qp (T)
            green_trajectories_qp = anchor_trajectories[qp_idx] # (T x T x 2), green_trajectories_qp[achor_t, source_t] = [x, y], source_t = start_frame
            occ_preds_by_dist_th_anchor_frame_vis.append(self.compute_occ_pred_for_qp(green_trajectories_qp, source_trajectories_qp, traj_cos_sim_qp, self.anchor_cosine_similarity_threshold, self.cosine_similarity_threshold))

        occ_preds = torch.stack(occ_preds_by_dist_th_anchor_frame_vis) # (N x T)

        return occ_preds
    
    # ----------------- Inference -----------------
    @torch.no_grad()
    def infer(self, query_points: torch.Tensor, batch_size=None, output_occ=True) -> torch.Tensor:
        """Infer trajectory and occlusion for query points.
        Args:
            query_points (torch.Tensor): Query points. N x 3. N is the number of query points. (x, y, t).
            batch_size (int): Batch size for inference. if None, all frames are inferred at once.
        Returns:
            trajectories (torch.Tensor): Predicted trajectory. N x T x 2. T is the number of time steps.
            occlusion (torch.Tensor): Predicted occlusion. N x T. T is the number of time steps."""
        trajs = self.compute_trajectories(query_points, batch_size) # N x T x 3
        if output_occ:
            cos_sims = self.compute_trajectory_cos_sims(trajs, query_points)
            anchor_trajs = self.compute_anchor_trajectories(trajs, cos_sims, batch_size)
            occ = self.compute_occlusion(trajs, cos_sims, anchor_trajs)
        else:
            occ = torch.zeros(trajs.shape[0], trajs.shape[1], device=trajs.device)
        return trajs[..., :2], occ # N x T x 2, N x T

