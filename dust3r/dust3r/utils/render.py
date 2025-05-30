import torch
import numpy as np
from gsplat import rasterization
from dust3r.utils.geometry import inv, geotrf
import kornia
from kornia.geometry.quaternion import Quaternion
from kornia.geometry.conversions import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix


def decompose_pose(pose: torch.Tensor):
    r"""
    pose: shape (4,4)
      [R | t]
      [0 | 1]
    returns:
      quaternion: a `kornia.geometry.quaternion.Quaternion` object
      translation: shape (3,) tensor
    """
    if pose.shape != (4,4):
        raise ValueError(f"pose must be (4,4), got {pose.shape}")
    # 회전행렬 R, 평행이동 t
    R = pose[:3, :3]
    t = pose[:3, 3]
    # kornia Quaternion 객체 생성
    # 1) from_rotation_matrix expects shape (3,3)
    # 2) returns a `Quaternion`
    q = rotation_matrix_to_quaternion(R, eps=torch.finfo(R.dtype).eps)
    return q, t

def compose_pose(q: Quaternion, t: torch.Tensor) -> torch.Tensor:
    r"""
    q: kornia.geometry.quaternion.Quaternion
    t: (3,) translation
    returns pose: (4,4) tensor
    """
    # 회전행렬 (3,3)
    R = quaternion_to_rotation_matrix(q)
    # (4,4) 만들기
    pose = torch.eye(4, device=R.device, dtype=R.dtype)
    pose[:3, :3] = R
    pose[:3, 3]  = t
    return pose

def slerp(q1, q2, alpha):
    """
    두 quaternion q1, q2를 alpha 비율로 slerp한 quaternion 반환 (의사코드).
    """
    # normalize
    q1 = q1 / q1.norm()
    q2 = q2 / q2.norm()
    dot = (q1 * q2).sum()
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # linear interpolate
        result = q1 + alpha * (q2 - q1)
        return result / result.norm()
    # slerp
    theta_0 = torch.acos(dot)
    theta = theta_0 * alpha
    q3 = q2 - q1 * dot
    q3 = q3 / q3.norm()
    return q1 * torch.cos(theta) + q3 * torch.sin(theta)

def interpolate_poses(
    pose1: torch.Tensor,
    pose2: torch.Tensor,
    num_views: int=2,
):
    r"""
    pose1, pose2: (4,4) - 첫 번째 뷰와 두 번째 뷰의 카메라 pose
    num_views: int - 중간 포즈를 몇 개 만들지

    returns:
      list of (4,4) pose tensors
      e.g. num_views=2 => alpha=1/3,2/3 로 2개의 중간 pose
    """
    # 1) 4x4 -> (Quaternion, translation)
    q1, t1 = decompose_pose(pose1)
    q2, t2 = decompose_pose(pose2)

    out_poses = []
    for i in range(1, num_views+1):
        alpha = i/(num_views+1.0)  # ex) i=1 => 1/3, i=2 => 2/3
        # 2) Quaternion slerp
        q_slerp = slerp(q1, q2, alpha)  # => returns new Quaternion
        # 3) translation lerp
        t_lerp = t1 + alpha*(t2 - t1)
        # 4) (q_slerp, t_lerp) -> (4,4)
        pose_i = compose_pose(q_slerp, t_lerp)
        out_poses.append(pose_i)
    return out_poses


def render(
    intrinsics: torch.Tensor,
    pts3d: torch.Tensor,
    rgbs: torch.Tensor | None = None,
    scale: float = 0.004, # 0.01,
    opacity: float = 0.95,
    cam_poses: torch.Tensor | None = None,
):

    device = pts3d.device
    batch_size = len(intrinsics)
    img_size = pts3d.shape[1:3]
    pts3d = pts3d.reshape(batch_size, -1, 3)
    num_pts = pts3d.shape[1]
    quats = torch.randn((num_pts, 4), device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = scale * torch.ones((num_pts, 3), device=device)
    opacities = opacity * torch.ones((num_pts), device=device)
    if rgbs is not None:
        assert rgbs.shape[1] == 3
        rgbs = rgbs.reshape(batch_size, 3, -1).transpose(1, 2)
    else:
        rgbs = torch.ones_like(pts3d[:, :, :3])

    rendered_rgbs = []
    rendered_depths = []
    accs = []
    for i in range(batch_size):
        # rgbd, acc, _ = rasterization(
        #     pts3d[i],
        #     quats,
        #     scales,
        #     opacities,
        #     rgbs[i],
        #     torch.eye(4, device=device)[None],
        #     intrinsics[[i]],
        #     width=img_size[1],
        #     height=img_size[0],
        #     packed=False,
        #     render_mode="RGB+D",
        # )
        if cam_poses is None:
            # default identity or random
            view_matrix = torch.eye(4, device=device)[None]
        else:
            view_matrix = cam_poses[i][None]  # (1,4,4)

        rgb, acc, _ = rasterization(
            pts3d[i],
            quats,
            scales,
            opacities,
            rgbs[i],
            # torch.eye(4, device=device)[None],
            # change view matrix to randomized view transformation matrix (not identity)
            view_matrix,
            torch.tensor(intrinsics[[i]], device=device, dtype=torch.float32),
            width=img_size[1],
            height=img_size[0],
            packed=False,
            render_mode="RGB",
        )

        # rendered_depths.append(rgbd[..., 3])
        rendered_rgbs.append(rgb)

    # rendered_depths = torch.cat(rendered_depths, dim=0)
    rendered_rgbs = torch.cat(rendered_rgbs, dim=0)

    return rendered_rgbs, rendered_depths, accs


def get_render_results(gts, preds, self_view=False):
    device = preds[0]["pts3d_in_self_view"].device
    with torch.no_grad():
        depths = []
        gt_depths = []
        for i, (gt, pred) in enumerate(zip(gts, preds)):
            if self_view:
                camera = inv(gt["camera_pose"]).to(device)
                intrinsics = gt["camera_intrinsics"].to(device)
                pred = pred["pts3d_in_self_view"]
            else:
                camera = inv(gts[0]["camera_pose"]).to(device)
                intrinsics = gts[0]["camera_intrinsics"].to(device)
                pred = pred["pts3d_in_other_view"]
            gt_img = gt["img"].to(device)
            gt_pts3d = gt["pts3d"].to(device)

            _, depth, _ = render(intrinsics, pred, gt_img)
            _, gt_depth, _ = render(intrinsics, geotrf(camera, gt_pts3d), gt_img)
            depths.append(depth)
            gt_depths.append(gt_depth)
    return depths, gt_depths