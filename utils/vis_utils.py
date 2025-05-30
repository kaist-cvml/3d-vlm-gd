import os
import cv2 
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

def visualize_correspondences(img1, img2, kps1, kps2, pred_kps2, vis_mask, save_path):
    """
    img1, img2: PIL Image objects
    kps1, kps2: ground truth keypoints (N, 3) where last dim is [x, y, visibility]
    pred_kps2: predicted keypoints (N, 2) [x, y]
    vis_mask: visibility mask (N,)
    save_path: path to save visualization
    """
    # Convert PIL Images to numpy arrays
    img1_np = np.array(img1.permute(1, 2, 0).detach().cpu())
    img2_np = np.array(img2.permute(1, 2, 0).detach().cpu())
    # img1_np = np.array(img1)
    # img2_np = np.array(img2)
    
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Define colors for better visibility
    # Use more vibrant and distinguishable colors
    source_color = '#FF1493'  # Deep Pink
    target_gt_color = '#00FF00'  # Lime Green
    target_pred_color = '#00FFFF'  # Orange Red
    line_color = '#FFFB00'  # Dodger Blue
    
    # Define marker sizes and styles for better visibility
    source_marker_size = 150  # Increased from 30
    target_marker_size = 150  # Increased from 30
    pred_marker_size = 150  # Increased from 30
    line_width = 3.0  # Increased from 0.5
    source_marker = 'o'  # Circle
    target_marker = 's'  # Square
    pred_marker = 'x'  # X
    
    # 1. Save separate high-quality images for source and target with original aspect ratio
    
    # Create figure for source image (no axes, no grid)
    plt.figure(figsize=(img1_np.shape[1]/100, img1_np.shape[0]/100), dpi=300)
    plt.imshow(img1_np)
    plt.scatter(kps1[vis_mask, 0], kps1[vis_mask, 1], 
                c=source_color, s=source_marker_size, marker=source_marker, edgecolors='white', linewidths=1)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(os.path.join(save_dir, os.path.basename(save_path).replace('.png', '_source.png')), 
                bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Create figure for target image (no axes, no grid)
    plt.figure(figsize=(img2_np.shape[1]/100, img2_np.shape[0]/100), dpi=300)
    plt.imshow(img2_np)
    plt.scatter(kps2[vis_mask, 0], kps2[vis_mask, 1], 
                c=target_gt_color, s=target_marker_size, marker=target_marker, edgecolors='white', linewidths=1)
    plt.scatter(pred_kps2[vis_mask, 0], pred_kps2[vis_mask, 1], 
                c=target_pred_color, s=pred_marker_size, marker=pred_marker, linewidths=2)
    # Draw lines between corresponding points
    for i in range(len(kps1)):
        if vis_mask[i]:
            plt.plot([kps2[i, 0], pred_kps2[i, 0]], 
                    [kps2[i, 1], pred_kps2[i, 1]], 
                    color=line_color, linestyle='--', linewidth=line_width, alpha=0.7)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(os.path.join(save_dir, os.path.basename(save_path).replace('.png', '_target.png')), 
                bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 2. Create a combined visualization (side by side, no axes, no grid, no titles)
    # Calculate figure size based on actual image dimensions to maintain aspect ratio
    fig_width = (img1_np.shape[1] + img2_np.shape[1]) / 100
    fig_height = max(img1_np.shape[0], img2_np.shape[0]) / 100
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    
    # Calculate grid positions based on image dimensions
    grid_width = img1_np.shape[1] + img2_np.shape[1]
    grid = plt.GridSpec(1, grid_width)
    
    # Add source image to left side
    ax1 = fig.add_subplot(grid[0, :img1_np.shape[1]])
    ax1.imshow(img1_np)
    ax1.scatter(kps1[vis_mask, 0], kps1[vis_mask, 1], 
                c=source_color, s=source_marker_size, marker=source_marker, edgecolors='white', linewidths=1)
    ax1.axis('off')
    
    # Add target image to right side
    ax2 = fig.add_subplot(grid[0, img1_np.shape[1]:])
    ax2.imshow(img2_np)
    ax2.scatter(kps2[vis_mask, 0], kps2[vis_mask, 1], 
                c=target_gt_color, s=target_marker_size, marker=target_marker, edgecolors='white', linewidths=1)
    ax2.scatter(pred_kps2[vis_mask, 0], pred_kps2[vis_mask, 1], 
                c=target_pred_color, s=pred_marker_size, marker=pred_marker, linewidths=2)
    # Draw lines between corresponding points
    for i in range(len(kps1)):
        if vis_mask[i]:
            ax2.plot([kps2[i, 0], pred_kps2[i, 0]], 
                    [kps2[i, 1], pred_kps2[i, 1]], 
                    color=line_color, linestyle='--', linewidth=line_width, alpha=0.7)
    ax2.axis('off')
    
    # Remove spacing between subplots
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 3. Create a directly concatenated image using numpy (no matplotlib)
    # This ensures exact concatenation without any scaling or padding
    
    # Handle different heights by using the maximum height
    max_h = max(img1_np.shape[0], img2_np.shape[0])
    # Create a new canvas with the combined width and max height
    concat_img = np.ones((max_h, img1_np.shape[1] + img2_np.shape[1], 3))
    
    # Copy source image to left side
    concat_img[:img1_np.shape[0], :img1_np.shape[1]] = img1_np
    
    # Copy target image to right side
    concat_img[:img2_np.shape[0], img1_np.shape[1]:] = img2_np
    
    # Save the concatenated image directly
    plt.figure(figsize=((img1_np.shape[1] + img2_np.shape[1])/100, max_h/100), dpi=300)
    plt.imshow(concat_img)
    
    # Draw source keypoints
    plt.scatter(kps1[vis_mask, 0], kps1[vis_mask, 1], 
                c=source_color, s=source_marker_size, marker=source_marker, edgecolors='white', linewidths=1)
    
    # Draw target keypoints with offset
    offset_x = img1_np.shape[1]
    plt.scatter(kps2[vis_mask, 0] + offset_x, kps2[vis_mask, 1], 
                c=target_gt_color, s=target_marker_size, marker=target_marker, edgecolors='white', linewidths=1)
    plt.scatter(pred_kps2[vis_mask, 0] + offset_x, pred_kps2[vis_mask, 1], 
                c=target_pred_color, s=pred_marker_size, marker=pred_marker, linewidths=2)
    
    # Draw lines between corresponding points in target (with offset)
    for i in range(len(kps1)):
        if vis_mask[i]:
            plt.plot([kps2[i, 0] + offset_x, pred_kps2[i, 0] + offset_x], 
                    [kps2[i, 1], pred_kps2[i, 1]], 
                    color=line_color, linestyle='--', linewidth=line_width, alpha=0.7)
    
    # Add a legend to explain the markers
    legend_elements = [
        plt.Line2D([0], [0], marker=source_marker, color='w', markerfacecolor=source_color, 
                  markersize=10, label='Source Points'),
        plt.Line2D([0], [0], marker=target_marker, color='w', markerfacecolor=target_gt_color, 
                  markersize=10, label='Target GT'),
        plt.Line2D([0], [0], marker=pred_marker, color='w', markerfacecolor=target_pred_color, 
                  markersize=10, label='Target Pred')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(os.path.join(save_dir, os.path.basename(save_path).replace('.png', '_concat.png')), 
                bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_matching_pairs(image1, image2, kp1, kp2, epoch, batch_idx, output_dir="visualization/debug_match"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1) numpy array 로 변환
    image1_np = image1.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    image2_np = image2.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    # 2) 매칭 시각화 결합 이미지 생성
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), dpi=100)
    ax[0].imshow(image1_np); ax[0].axis("off")
    ax[1].imshow(image2_np); ax[1].axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)

    kp1_np = kp1.squeeze(0).cpu().numpy()
    kp2_np = kp2.squeeze(0).cpu().numpy()
    
    # 더 밝고 다양한 색상 사용
    colors = cm.rainbow(np.linspace(0, 1, kp1_np.shape[0]))
    
    # 포인트 시각화 매개변수 설정
    point_size = 50  # 크게 증가 (이전 10)
    marker_style = 'o'  # 원형 마커 사용
    edge_color = 'white'  # 흰색 테두리
    edge_width = 1.0  # 테두리 두께
    alpha = 0.9  # 약간의 투명도
    
    for (x1, y1), (x2, y2), c in zip(kp1_np, kp2_np, colors):
        # 왼쪽 이미지에 포인트 그리기
        ax[0].scatter(x1, y1, 
                     color=c, 
                     s=point_size, 
                     marker=marker_style, 
                     edgecolors=edge_color, 
                     linewidths=edge_width, 
                     alpha=alpha)
        
        # 오른쪽 이미지에 포인트 그리기
        ax[1].scatter(x2, y2, 
                     color=c, 
                     s=point_size, 
                     marker=marker_style, 
                     edgecolors=edge_color, 
                     linewidths=edge_width, 
                     alpha=alpha)
    
    combo_path = os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}_combo.png")
    plt.savefig(combo_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # 3) Pillow로 불러와 반으로 잘라 저장
    full = Image.open(combo_path)
    W, H = full.size
    left = full.crop((0, 0, W//2, H))
    right = full.crop((W//2, 0, W, H))
    
    left.save(os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}_left.png"))
    right.save(os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}_right.png"))


# def visualize_matching_pairs(image1, image2, kp1, kp2, epoch, batch_idx, output_dir="visualization/debug_match"):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     image1_np = image1.squeeze(0).cpu().numpy().transpose(1, 2, 0)
#     image2_np = image2.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
#     fig, ax = plt.subplots(1, 2, figsize=(15, 5))
#     ax[0].imshow(image1_np)
#     # ax[0].set_title("Image 1")
#     ax[0].axis("off")
    
#     ax[1].imshow(image2_np)
#     # ax[1].set_title("Image 2")
#     ax[1].axis("off")

#     kp1_np = kp1.squeeze(0).cpu().numpy()
#     kp2_np = kp2.squeeze(0).cpu().numpy()
#     num_matches = kp1_np.shape[0]
#     colors = cm.jet(np.linspace(0, 1, num_matches))
    
#     for (x1, y1), (x2, y2), color in zip(kp1_np, kp2_np, colors):
#         ax[0].scatter(x1, y1, color=color, s=10)
#         ax[1].scatter(x2, y2, color=color, s=10)
#         con = ConnectionPatch(xyA=(x1, y1), xyB=(x2, y2),
#                               coordsA="data", coordsB="data",
#                               axesA=ax[0], axesB=ax[1],
#                               color=color, lw=1)
#         fig.add_artist(con)
    
#     file_path = os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}.png")
#     plt.savefig(file_path)
#     plt.close(fig)


import os
from matplotlib import pyplot as plt
import numpy as np

def visualize_depth_maps(depth_pred_1, depth_pred_2, epoch, batch_idx, output_dir="visualization/debug_depth"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 결합된 시각화 (기존 코드)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    im0 = ax[0].imshow(depth_pred_1, cmap='plasma')
    ax[0].set_title("Depth Map 1")
    ax[0].axis("off")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    
    im1 = ax[1].imshow(depth_pred_2, cmap='plasma')
    ax[1].set_title("Depth Map 2")
    ax[1].axis("off")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    
    # 결합 이미지 저장
    file_path = os.path.join(output_dir, f"depth_epoch_{epoch}_batch_{batch_idx}.png")
    plt.savefig(file_path)
    plt.close(fig)
    
    # 각 뷰의 depth map을 개별적으로 저장 - 여백과 컬러바 없이
    # Depth Map 1 저장
    fig_single = plt.figure(frameon=False)
    ax_single = plt.Axes(fig_single, [0., 0., 1., 1.])
    ax_single.set_axis_off()
    fig_single.add_axes(ax_single)
    ax_single.imshow(depth_pred_1, cmap='plasma', aspect='auto')
    file_path_1 = os.path.join(output_dir, f"depth_epoch_{epoch}_batch_{batch_idx}_view1.png")
    plt.savefig(file_path_1, bbox_inches='tight', pad_inches=0)
    plt.close(fig_single)
    
    # Depth Map 2 저장
    fig_single = plt.figure(frameon=False)
    ax_single = plt.Axes(fig_single, [0., 0., 1., 1.])
    ax_single.set_axis_off()
    fig_single.add_axes(ax_single)
    ax_single.imshow(depth_pred_2, cmap='plasma', aspect='auto')
    file_path_2 = os.path.join(output_dir, f"depth_epoch_{epoch}_batch_{batch_idx}_view2.png")
    plt.savefig(file_path_2, bbox_inches='tight', pad_inches=0)
    plt.close(fig_single)


def vis_attn_map(attention_maps, img_target, img_source, count, p_size=14, save_path='./vis_ca_map'):
    
    ########################## VIS CROSS ATTN MAPS (START) ###############################
    
    b, _, H, W = img_target.shape 
    attn_maps = torch.stack(attention_maps, dim=1)  # b 12 196 196 (twelve layers of already head averaged attention maps)

    pH=H//p_size  # num patch H
    pW=W//p_size  # num patch W     

    t1=torch.rand(25).argsort()
    t2=torch.rand(37).argsort()
    rnd_pts=list(zip(t1,t2))
    vis_rnd_pts=[(i.item(), j.item()) for (i,j) in rnd_pts]
    num_vis=30

    for batch in range(b):  
        img_t = img_target[batch] # 3 224 224 
        img_s = img_source[batch] 
        attn_map = attn_maps[batch] # 12 196 196

        attn_map = attn_map.mean(dim=0) # average all layers of attention maps 

        np_img_s = (img_s-img_s.min()) / (img_s.max()-img_s.min()) * 255.0 # [0,255]
        np_img_t = (img_t-img_t.min())/(img_t.max()-img_t.min()) * 255.0   # [0,255]
        np_img_s = np_img_s.squeeze().permute(1,2,0).detach().cpu().numpy() # 224 224 3 
        np_img_t = np_img_t.squeeze().permute(1,2,0).detach().cpu().numpy()

        # List to store all visualizations
        all_vis_imgs = []
        
        for points in vis_rnd_pts[:num_vis]:
            idx_h=points[0]     # to vis idx_h
            idx_w=points[1]     # to vis idx_w
            idx_n=idx_h*pW+idx_w  # to vis token idx
            
            # plot white pixel to vis tkn location
            vis_np_img_s = np_img_s.copy()  # same as clone()
            vis_np_img_s[idx_h*p_size:(idx_h+1)*p_size, idx_w*p_size:(idx_w+1)*p_size,:]=255    # color with white pixel
            
            # breakpoint()
            # generate attn heat map
            attn_msk=attn_map[idx_n]  # hw=14*14=196
            # attn_msk[0]=0
            # attn_msk=attn_msk.softmax(dim=-1)
            attn_msk=attn_msk.view(1,1,pH,pW)
            attn_msk=F.interpolate(attn_msk, size=(H,W), mode='bilinear', align_corners=True)   # 224 224
            attn_msk=(attn_msk-attn_msk.min())/(attn_msk.max()-attn_msk.min())  # [0,1]
            attn_msk=attn_msk.squeeze().detach().cpu().numpy()*255  # [0,255]
            heat_mask=cv2.applyColorMap(attn_msk.astype(np.uint8), cv2.COLORMAP_JET)
            
            # overlap heat_mask to source image
            img_t_attn_msked = np_img_t[...,::-1] + heat_mask
            img_t_attn_msked = (img_t_attn_msked-img_t_attn_msked.min())/(img_t_attn_msked.max()-img_t_attn_msked.min())*255.0
            
            # Concatenate source and target images horizontally for this point
            combined_img = np.concatenate([vis_np_img_s[:,:,[2,1,0]], img_t_attn_msked], axis=1)
            all_vis_imgs.append(combined_img)
        
        # Stack all visualizations vertically
        final_vis = np.concatenate(all_vis_imgs, axis=0)
        
        # Save the combined visualization
        log_img_path = save_path
        if not os.path.exists(log_img_path):
            os.makedirs(log_img_path)
        cv2.imwrite(f'{log_img_path}/count{count}_batch{batch}_all_points.jpg', final_vis)


def visualize_tracking_results(images, trajectories_dict, occlusions_dict, save_dir, benchmark_config=None, video_idx=None):
    """
    Visualize tracking results by drawing trajectories on video frames
    
    Args:
        images: Tensor of video frames [T, C, H, W]
        trajectories_dict: Dictionary mapping frame_idx to trajectory arrays [N, T, 2]
        occlusions_dict: Dictionary mapping frame_idx to occlusion arrays [N, T]
        save_dir: Directory to save visualizations
        benchmark_config: Benchmark configuration containing ground truth data
        video_idx: Video index in benchmark_config
    """
    # Convert images to numpy for visualization
    images_np = images.permute(0, 2, 3, 1).numpy()
    num_frames = len(images_np)
    
    # Create colormap for trajectories
    cmap = plt.get_cmap('jet')
    
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create subdirectories for different visualizations
    pred_dir = os.path.join(save_dir, "predictions")
    gt_dir = os.path.join(save_dir, "ground_truth")
    comparison_dir = os.path.join(save_dir, "comparison")
    
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Extract ground truth data if available
    has_gt = benchmark_config is not None and video_idx is not None
    gt_trajectories_dict = {}
    gt_occlusions_dict = {}
    
    if has_gt:
        try:
            # Find the video config for the current video
            video_config = None
            for vc in benchmark_config["videos"]:
                if vc["video_idx"] == video_idx:
                    video_config = vc
                    break
            
            if video_config is None:
                print(f"Warning: Video with ID {video_idx} not found in benchmark_config")
                has_gt = False
            else:
                # Get image dimensions for rescaling
                h, w = images_np.shape[1], images_np.shape[2]
                rescale_factor_x = w / video_config['w']
                rescale_factor_y = h / video_config['h']
                
                # Extract ground truth trajectories
                for frame_idx, q_pts_at_frame in video_config['query_points'].items():
                    if 'target_points' not in video_config:
                        print(f"Warning: 'target_points' not found in video_config for video {video_idx}")
                        has_gt = False
                        break
                        
                    target_points = video_config['target_points'][frame_idx]
                    
                    # Check if visibility information is available
                    has_visibility = False
                    visibility_key = None
                    
                    # Try different possible keys for visibility information
                    for key in ['visibilities', 'visibility', 'occluded', 'occlusions', 'visible']:
                        if key in video_config and frame_idx in video_config[key]:
                            has_visibility = True
                            visibility_key = key
                            break
                    
                    # Convert to numpy array and rescale to match the image dimensions
                    gt_trajectories = []
                    gt_occlusions = []
                    
                    for i, traj in enumerate(target_points):
                        # Rescale trajectory points
                        scaled_traj = np.array(traj) * np.array([rescale_factor_x, rescale_factor_y])
                        gt_trajectories.append(scaled_traj)
                        
                        # Handle visibility/occlusion information
                        if has_visibility:
                            vis_info = video_config[visibility_key][frame_idx][i]
                            # Convert to occlusion format (1 = occluded, 0 = visible)
                            if visibility_key in ['visibilities', 'visibility', 'visible']:
                                # If it's visibility, invert it to get occlusion
                                occlusion = 1.0 - np.array(vis_info)
                            else:
                                # If it's already occlusion
                                occlusion = np.array(vis_info)
                        else:
                            # If no visibility info, assume all points are visible
                            occlusion = np.zeros(len(traj))
                            
                        gt_occlusions.append(occlusion)
                    
                    gt_trajectories_dict[frame_idx] = np.array(gt_trajectories)
                    gt_occlusions_dict[frame_idx] = np.array(gt_occlusions)
                    
        except Exception as e:
            print(f"Error extracting GT data: {e}")
            has_gt = False
    
    # First pass: draw all trajectories at once (overview) - PREDICTIONS
    plt.figure(figsize=(15, 8))
    
    # Add first frame as background
    plt.imshow(images_np[0])
    
    # For each starting frame
    for start_frame_idx, trajectories in trajectories_dict.items():
        occlusions = occlusions_dict[start_frame_idx]
        
        # For each point in the starting frame
        for point_idx, trajectory in enumerate(trajectories):
            occlusion = occlusions[point_idx]
            
            # Set color based on point index
            color = cmap(point_idx / max(1, len(trajectories) - 1))
            
            # Get valid trajectory points (not occluded)
            valid_frames = np.where(occlusion < 0.5)[0]
            valid_trajectory = trajectory[valid_frames]
            valid_frame_indices = valid_frames + start_frame_idx
            
            # Filter out points that go beyond video length
            valid_mask = valid_frame_indices < num_frames
            valid_trajectory = valid_trajectory[valid_mask]
            valid_frame_indices = valid_frame_indices[valid_mask]
            
            if len(valid_trajectory) > 0:
                plt.scatter(valid_trajectory[:, 0], valid_trajectory[:, 1], 
                           c=[color], s=20, alpha=0.7)
                plt.plot(valid_trajectory[:, 0], valid_trajectory[:, 1], 
                        c=color, linewidth=1, alpha=0.5)
    
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(pred_dir, "all_trajectories_overview.png"), 
               bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    # If GT is available, create GT overview visualization
    if has_gt:
        plt.figure(figsize=(15, 8))
        
        # Add first frame as background
        plt.imshow(images_np[0])
        
        # For each starting frame
        for start_frame_idx, trajectories in gt_trajectories_dict.items():
            occlusions = gt_occlusions_dict[start_frame_idx]
            
            # For each point in the starting frame
            for point_idx, trajectory in enumerate(trajectories):
                occlusion = occlusions[point_idx]
                
                # Set color based on point index
                color = cmap(point_idx / max(1, len(trajectories) - 1))
                
                # Get valid trajectory points (not occluded)
                valid_frames = np.where(occlusion < 0.5)[0]
                valid_trajectory = trajectory[valid_frames]
                valid_frame_indices = valid_frames + start_frame_idx
                
                # Filter out points that go beyond video length
                valid_mask = valid_frame_indices < num_frames
                valid_trajectory = valid_trajectory[valid_mask]
                valid_frame_indices = valid_frame_indices[valid_mask]
                
                if len(valid_trajectory) > 0:
                    plt.scatter(valid_trajectory[:, 0], valid_trajectory[:, 1], 
                               c=[color], s=20, alpha=0.7)
                    plt.plot(valid_trajectory[:, 0], valid_trajectory[:, 1], 
                            c=color, linewidth=1, alpha=0.5)
        
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(gt_dir, "all_trajectories_overview.png"), 
                   bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        # Create comparison visualization
        plt.figure(figsize=(15, 8))
        
        # Add first frame as background
        plt.imshow(images_np[0])
        
        # Draw GT trajectories
        for start_frame_idx, trajectories in gt_trajectories_dict.items():
            occlusions = gt_occlusions_dict[start_frame_idx]
            
            for point_idx, trajectory in enumerate(trajectories):
                occlusion = occlusions[point_idx]
                color = cmap(point_idx / max(1, len(trajectories) - 1))
                
                valid_frames = np.where(occlusion < 0.5)[0]
                valid_trajectory = trajectory[valid_frames]
                valid_frame_indices = valid_frames + start_frame_idx
                
                valid_mask = valid_frame_indices < num_frames
                valid_trajectory = valid_trajectory[valid_mask]
                
                if len(valid_trajectory) > 0:
                    plt.scatter(valid_trajectory[:, 0], valid_trajectory[:, 1], 
                              c=[color], s=20, alpha=0.7, marker='o')
                    plt.plot(valid_trajectory[:, 0], valid_trajectory[:, 1], 
                           c=color, linewidth=1, alpha=0.5, linestyle='-', label=f'GT {point_idx}' if point_idx < 5 else "")
        
        # Draw predicted trajectories with different marker/line style
        # Only draw predictions for points that have GT counterparts
        for start_frame_idx, trajectories in trajectories_dict.items():
            if start_frame_idx in gt_trajectories_dict:
                gt_traj = gt_trajectories_dict[start_frame_idx]
                occlusions = occlusions_dict[start_frame_idx]
                
                # Only visualize the same number of points as in GT
                max_points = min(len(trajectories), len(gt_traj))
                
                for point_idx in range(max_points):
                    trajectory = trajectories[point_idx]
                    occlusion = occlusions[point_idx]
                    color = cmap(point_idx / max(1, len(gt_traj) - 1))
                    
                    valid_frames = np.where(occlusion < 0.5)[0]
                    valid_trajectory = trajectory[valid_frames]
                    valid_frame_indices = valid_frames + start_frame_idx
                    
                    valid_mask = valid_frame_indices < num_frames
                    valid_trajectory = valid_trajectory[valid_mask]
                    
                    if len(valid_trajectory) > 0:
                        plt.scatter(valid_trajectory[:, 0], valid_trajectory[:, 1], 
                                  c=[color], s=20, alpha=0.7, marker='x')
                        plt.plot(valid_trajectory[:, 0], valid_trajectory[:, 1], 
                               c=color, linewidth=1, alpha=0.5, linestyle='--', label=f'Pred {point_idx}' if point_idx < 5 else "")
        
        # Add legend for the first few trajectories
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(comparison_dir, "comparison_overview.png"), 
                   bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
    
    # Second pass: draw trajectories on each frame - PREDICTIONS
    for frame_idx in range(num_frames):
        plt.figure(figsize=(10, 8))
        plt.imshow(images_np[frame_idx])
        
        # For each starting frame
        for start_frame_idx, trajectories in trajectories_dict.items():
            occlusions = occlusions_dict[start_frame_idx]
            
            # Skip if this frame is before the starting frame
            if frame_idx < start_frame_idx:
                continue
                
            # For each point in the starting frame
            for point_idx, trajectory in enumerate(trajectories):
                occlusion = occlusions[point_idx]
                
                # Calculate relative frame index
                rel_frame_idx = frame_idx - start_frame_idx
                
                # Skip if out of trajectory bounds
                if rel_frame_idx >= len(trajectory):
                    continue
                    
                # Skip if point is occluded at this frame
                if occlusion[rel_frame_idx] >= 0.5:
                    continue
                    
                # Draw point
                point = trajectory[rel_frame_idx]
                color = cmap(point_idx / max(1, len(trajectories) - 1))
                
                # Draw current point with larger marker
                plt.scatter(point[0], point[1], c=[color], s=30, alpha=1.0)
                
                # Draw trajectory up to current frame
                valid_frames = np.where(occlusion[:rel_frame_idx+1] < 0.5)[0]
                if len(valid_frames) > 1:
                    valid_traj = trajectory[valid_frames]
                    plt.plot(valid_traj[:, 0], valid_traj[:, 1], 
                            c=color, linewidth=1, alpha=0.5)
        
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(pred_dir, f"frame_{frame_idx:04d}.png"), 
                   bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        # If GT is available, create GT frame visualization
        if has_gt:
            # GT visualization
            plt.figure(figsize=(10, 8))
            plt.imshow(images_np[frame_idx])
            
            for start_frame_idx, trajectories in gt_trajectories_dict.items():
                occlusions = gt_occlusions_dict[start_frame_idx]
                
                if frame_idx < start_frame_idx:
                    continue
                    
                for point_idx, trajectory in enumerate(trajectories):
                    occlusion = occlusions[point_idx]
                    rel_frame_idx = frame_idx - start_frame_idx
                    
                    if rel_frame_idx >= len(trajectory):
                        continue
                        
                    if occlusion[rel_frame_idx] >= 0.5:
                        continue
                        
                    point = trajectory[rel_frame_idx]
                    color = cmap(point_idx / max(1, len(trajectories) - 1))
                    
                    plt.scatter(point[0], point[1], c=[color], s=30, alpha=1.0)
                    
                    valid_frames = np.where(occlusion[:rel_frame_idx+1] < 0.5)[0]
                    if len(valid_frames) > 1:
                        valid_traj = trajectory[valid_frames]
                        plt.plot(valid_traj[:, 0], valid_traj[:, 1], 
                                c=color, linewidth=1, alpha=0.5)
            
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(gt_dir, f"frame_{frame_idx:04d}.png"), 
                       bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
            
            # Comparison visualization
            plt.figure(figsize=(10, 8))
            plt.imshow(images_np[frame_idx])
            
            # Draw GT trajectories first
            for start_frame_idx, trajectories in gt_trajectories_dict.items():
                occlusions = gt_occlusions_dict[start_frame_idx]
                
                if frame_idx < start_frame_idx:
                    continue
                    
                for point_idx, trajectory in enumerate(trajectories):
                    occlusion = occlusions[point_idx]
                    rel_frame_idx = frame_idx - start_frame_idx
                    
                    if rel_frame_idx >= len(trajectory):
                        continue
                        
                    if occlusion[rel_frame_idx] >= 0.5:
                        continue
                        
                    point = trajectory[rel_frame_idx]
                    color = cmap(point_idx / max(1, len(trajectories) - 1))
                    
                    plt.scatter(point[0], point[1], c=[color], s=30, alpha=1.0, marker='o')
                    
                    valid_frames = np.where(occlusion[:rel_frame_idx+1] < 0.5)[0]
                    if len(valid_frames) > 1:
                        valid_traj = trajectory[valid_frames]
                        plt.plot(valid_traj[:, 0], valid_traj[:, 1], 
                                c=color, linewidth=1, alpha=0.5, linestyle='-')
            
            # Then draw predicted trajectories
            # Only draw predictions for points that have GT counterparts
            for start_frame_idx, trajectories in trajectories_dict.items():
                if start_frame_idx in gt_trajectories_dict and frame_idx >= start_frame_idx:
                    gt_traj = gt_trajectories_dict[start_frame_idx]
                    occlusions = occlusions_dict[start_frame_idx]
                    
                    # Only visualize the same number of points as in GT
                    max_points = min(len(trajectories), len(gt_traj))
                    
                    for point_idx in range(max_points):
                        trajectory = trajectories[point_idx]
                        occlusion = occlusions[point_idx]
                        rel_frame_idx = frame_idx - start_frame_idx
                        
                        if rel_frame_idx >= len(trajectory):
                            continue
                            
                        if occlusion[rel_frame_idx] >= 0.5:
                            continue
                            
                        point = trajectory[rel_frame_idx]
                        color = cmap(point_idx / max(1, len(gt_traj) - 1))
                        
                        plt.scatter(point[0], point[1], c=[color], s=30, alpha=1.0, marker='x')
                        
                        valid_frames = np.where(occlusion[:rel_frame_idx+1] < 0.5)[0]
                        if len(valid_frames) > 1:
                            valid_traj = trajectory[valid_frames]
                            plt.plot(valid_traj[:, 0], valid_traj[:, 1], 
                                    c=color, linewidth=1, alpha=0.5, linestyle='--')
            
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(comparison_dir, f"frame_{frame_idx:04d}.png"), 
                       bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
    
    # Third pass: create videos (optional - if ffmpeg is available)
    try:
        import subprocess
        framerate = 10  # Adjust as needed
        
        # Create video for predictions
        cmd = [
            'ffmpeg', '-y', '-framerate', str(framerate), 
            '-pattern_type', 'glob', '-i', f'{pred_dir}/frame_*.png',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            os.path.join(pred_dir, 'tracking_visualization.mp4')
        ]
        subprocess.call(cmd)
        print(f"Prediction video saved to {os.path.join(pred_dir, 'tracking_visualization.mp4')}")
        
        if has_gt:
            # Create video for GT
            cmd = [
                'ffmpeg', '-y', '-framerate', str(framerate), 
                '-pattern_type', 'glob', '-i', f'{gt_dir}/frame_*.png',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                os.path.join(gt_dir, 'tracking_visualization.mp4')
            ]
            subprocess.call(cmd)
            print(f"GT video saved to {os.path.join(gt_dir, 'tracking_visualization.mp4')}")
            
            # Create video for comparison
            cmd = [
                'ffmpeg', '-y', '-framerate', str(framerate), 
                '-pattern_type', 'glob', '-i', f'{comparison_dir}/frame_*.png',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                os.path.join(comparison_dir, 'tracking_visualization.mp4')
            ]
            subprocess.call(cmd)
            print(f"Comparison video saved to {os.path.join(comparison_dir, 'tracking_visualization.mp4')}")
    except Exception as e:
        print(f"Could not create video: {e}")
