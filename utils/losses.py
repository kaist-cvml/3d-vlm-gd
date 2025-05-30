import torch
import torch.nn.functional as F


def kl_divergence_map(mast3r_cost, feat_cost_sim, eps=1e-8):
    mast3r_cost_norm = mast3r_cost.clamp_min(eps)
    feat_cost_sim_norm = feat_cost_sim.clamp_min(eps)

    kl_map = mast3r_cost_norm * torch.log(mast3r_cost_norm / feat_cost_sim_norm)
    # shape: (B, H*W, H*W)

    kl_per_row = kl_map.sum(dim=-1)
    kl_loss = kl_per_row.mean()

    return kl_loss


def pairwise_logistic_ranking_loss(model, pred_scores, gt_depths, depth_threshold=0.0):
    B, N, D = pred_scores.shape
    
    pred_i = pred_scores.unsqueeze(2).expand(B, N, N, D)  # (B, N, N, D)
    pred_j = pred_scores.unsqueeze(1).expand(B, N, N, D)  # (B, N, N, D)

    depth_i = gt_depths.unsqueeze(2)   # (B, N, 1)
    depth_j = gt_depths.unsqueeze(1)   # (B, 1, N)

    sign_ij = torch.sign(depth_j - depth_i)

    valid_mask = (torch.abs(depth_j - depth_i) > depth_threshold)

    alpha_ij = sign_ij  # (B, N, N)
    # score_diff = (model(pred_j.contiguous().view(B, -1, D)) - model(pred_i.contiguous().view(B, -1, D))).view(B, N, N)  # (B, N, N)
    # score_diff = torch.tanh(score_diff)
    score_diff = model((pred_j - pred_i).view(B, -1, D)).view(B, N, N)  # (B, N, N)
    pairwise_loss = torch.log(1.0 + torch.exp(-alpha_ij * score_diff))

    valid_pairwise_loss = pairwise_loss[valid_mask]
    if valid_pairwise_loss.numel() == 0:
        return torch.tensor(0.0, device=pred_scores.device)
    loss = valid_pairwise_loss.mean()
    return loss


def intra_depth_loss(model, kp_feat, kp_depth, base_margin=0.05, depth_thresh=0.05):
    B, N, D = kp_feat.shape

    feat_i = kp_feat.unsqueeze(2).expand(B, N, N, D)
    feat_j = kp_feat.unsqueeze(1).expand(B, N, N, D)
    diff_feat = feat_i - feat_j  # (B, N, N, D)

    diff_feat_flat = diff_feat.view(B, -1, D)  # (B, N*N, D)
    pred_diff_flat = model(diff_feat_flat)  # (B, N*N)
    pred_diff = pred_diff_flat.view(B, N, N)  # (B, N, N)

    depth_i = kp_depth.unsqueeze(2).expand(B, N, N)
    depth_j = kp_depth.unsqueeze(1).expand(B, N, N)
    gt_diff = torch.tanh(depth_i - depth_j).detach()

    target = torch.sign(gt_diff)

    loss_matrix = F.relu(base_margin - target * pred_diff)

    valid_mask = (torch.abs(gt_diff) > depth_thresh)

    if valid_mask.sum() > 0:
        loss = loss_matrix[valid_mask].mean()
    else:
        loss = torch.tensor(0.0, device=kp_feat.device)
    return loss