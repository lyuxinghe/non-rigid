import os
import plotly.graph_objects as go
import numpy as np
import torch
from pytorch3d.transforms import Translate
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import Transform3d

from non_rigid.utils.augmentation_utils import maybe_apply_augmentations
from non_rigid.utils.pointcloud_utils import downsample_pcd, get_multi_anchor_scene
from non_rigid.utils.transform_utils import random_se3



def transform_pcd(pcd: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    if pcd.shape[1] != 4:
        ones = torch.ones((pcd.shape[0], 1), dtype=pcd.dtype, device=pcd.device)
        pcd = torch.cat([pcd, ones], dim=1)  # Add homogeneous coordinate

    pcd_new = (transform @ pcd.T)[:-1].T  # Apply transform and drop the last row
    return pcd_new

def load_rpdiff_dataset(path, 
                        sample_size_action=512, 
                        sample_size_anchor=512,
                        rotation_variance=0.0,
                        translation_variance=0.0,
                        scene_transform_type="identity",
                        ):
    demo = np.load(path, allow_pickle=True)
    
    # Access start and final PCDs for parent and child
    parent_start_pcd = demo['multi_obj_start_pcd'].item()['parent']
    child_start_pcd = demo['multi_obj_start_pcd'].item()['child']
    parent_final_pcd = demo['multi_obj_final_pcd'].item()['parent']
    child_final_pcd = demo['multi_obj_final_pcd'].item()['child']

    action_pc = torch.as_tensor(child_start_pcd).float()
    anchor_pc = torch.as_tensor(parent_start_pcd).float()
    goal_action_pc = torch.as_tensor(child_final_pcd).float()
    goal_anchor_pc = torch.as_tensor(parent_final_pcd).float()  # same as anchor_pc
    # Note: relative_trans is not defined in the original code, commenting out
    # relative_trans = torch.as_tensor(relative_trans).float()

    action_seg = torch.zeros_like(action_pc[:, 0]).int()
    anchor_seg = torch.ones_like(anchor_pc[:, 0]).int()

    # calculate the scale of action and anchor pcd, such that we can estimate reasonable params 
    # for 1. augmentations and 2. noise scale
    # TODO: this mighe be a bit inefficient ..., optimize it!
    action_point_dists = action_pc - action_pc.mean(dim=0, keepdim=True)
    action_point_scale = torch.linalg.norm(action_point_dists, dim=1, keepdim=True).max()
    anchor_point_dists = anchor_pc - anchor_pc.mean(dim=0, keepdim=True) 
    anchor_point_scale = torch.linalg.norm(anchor_point_dists, dim=1, keepdim=True).max()

    # Store indices for SVD computation after downsampling
    action_pc_indices = None
    
    # downsample action
    if sample_size_action > 0 and action_pc.shape[0] > sample_size_action:
        action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), sample_size_action, type="fps")
        action_pc = action_pc.squeeze(0)
        action_seg = action_seg[action_pc_indices.squeeze(0)]
        goal_action_pc = goal_action_pc[action_pc_indices.squeeze(0)]
        
    # downsample anchor
    if sample_size_anchor > 0 and anchor_pc.shape[0] > sample_size_anchor:
        anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), sample_size_anchor, type="fps")
        anchor_pc = anchor_pc.squeeze(0)
        anchor_seg = anchor_seg[anchor_pc_indices.squeeze(0)]

    # Compute SVD-based transformation from original action to goal (before augmentation)
    def compute_svd_transformation(source_points, target_points):
        """
        Compute rigid transformation from source to target using SVD (Kabsch algorithm)
        Returns 4x4 transformation matrix
        """
        # Center the point clouds
        source_centroid = source_points.mean(dim=0)
        target_centroid = target_points.mean(dim=0)
        
        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid
        
        # Compute cross-covariance matrix
        H = source_centered.T @ target_centered
        
        # SVD
        U, S, Vt = torch.linalg.svd(H)
        
        # Compute rotation matrix
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = target_centroid - R @ source_centroid
        
        # Construct 4x4 transformation matrix
        T_matrix = torch.eye(4)
        T_matrix[:3, :3] = R
        T_matrix[:3, 3] = t
        
        return T_matrix
    
    # Compute the ground truth transformation
    T_action2goal_raw = compute_svd_transformation(action_pc, goal_action_pc)

    # Apply scene-level augmentation.
    T = random_se3(
        N=1,
        rot_var=rotation_variance,
        trans_var=translation_variance,
        rot_sample_method=scene_transform_type,
    )
    action_pc = T.transform_points(action_pc)
    anchor_pc = T.transform_points(anchor_pc)
    goal_action_pc = T.transform_points(goal_action_pc)

    # Center point clouds in scene frame.
    scene_center = torch.cat([action_pc, anchor_pc], dim=0).mean(axis=0)
    goal_action_pc = goal_action_pc - scene_center
    anchor_pc = anchor_pc - scene_center
    action_pc = action_pc - scene_center

    # Update item.
    T_goal2world = Translate(scene_center.unsqueeze(0)).compose(T.inverse())
    T_action2world = Translate(scene_center.unsqueeze(0)).compose(T.inverse())

    T_action2goal = compute_svd_transformation(action_pc, goal_action_pc)

    '''
    T_translate = torch.eye(4)
    T_translate[:3, 3] = scene_center

    T_translate_inverse = torch.eye(4)
    T_translate_inverse[:3, 3] = -scene_center
    T_action2goal = torch.matmul(torch.matmul(T_translate_inverse, T_action2goal), T_translate)
    '''

    goal_flow = goal_action_pc - action_pc

    item = {}
    item["pc_action"] = action_pc # Action points in the action frame
    item["pc_anchor"] = anchor_pc # Anchor points in the scene frame
    item["seg"] = action_seg
    item["seg_anchor"] = anchor_seg
    item["T_goal2world"] = T_goal2world.get_matrix().squeeze(0) # Transform from goal action frame to world frame
    item["T_action2world"] = T_action2world.get_matrix().squeeze(0) # Transform from action frame to world frame
    item["T_action2goal"] = T_action2goal # SVD-computed transformation from raw action to goal
    item["T_action2goal_raw"] = T_action2goal_raw # SVD-computed transformation from raw action to goal
    breakpoint()
    # Training-specific labels.
    # TODO: eventually, rename this key to "point"
    item["pc"] = goal_action_pc # Ground-truth goal action points in the scene frame
    item["flow"] = goal_flow # Ground-truth flow (cross-frame) to action points

    return item

# NOTE:
# 1. We start from T_action2goal [4x4], that can be written as [[R, t], [0, 1]], where x = action and y = goal
# 2. For any frame transformation for x (c_x) or y (c_y), we should just update t by: t′=t + c_x @ R.T −c_y
# 3. Note that t= mean(y) - mean(x) @ R.T


def visualize(action_pcd, anchor_pcd, goal_action_pcd, goal_tf=None):
    fig = go.Figure()
    output_html = "/home/lyuxing/Desktop/tax3dv2/scripts/logs/test.html"

    # Only compute transformed pcd if both action_pcd and goal_tf exist
    trans_pcd = None
    if action_pcd is not None and goal_tf is not None:
        trans_pcd = transform_pcd(action_pcd, goal_tf)

    def color_with_intensity(base_hex, intensity):
        rgb = [int(base_hex[i : i + 2], 16) for i in (1, 3, 5)]
        return f"rgb({int(rgb[0]*intensity)}, {int(rgb[1]*intensity)}, {int(rgb[2]*intensity)})"

    def add_trace(pc, base_color, name):
        if pc is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=pc[:, 0],
                    y=pc[:, 1],
                    z=pc[:, 2],
                    mode="markers",
                    name=f"{name}",
                    marker=dict(size=2.5, color=color_with_intensity(base_color, 1)),
                )
            )

    # Add traces only if the data exists
    add_trace(goal_action_pcd, "#00FF00", "Goal")  # Green
    add_trace(action_pcd, "#FFD700", "Action")  # Yellow
    add_trace(anchor_pcd, "#1F77B4", "Anchor")  # Blue
    add_trace(trans_pcd, "#FF0000", "Transformed")  # Red

    fig.update_layout(
        legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01),
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    fig.write_html(output_html)
    print(f"Saved to {output_html}")


if __name__ == "__main__":
    
    item = load_rpdiff_dataset(path="/data/lyuxing/tax3d/rpdiff/data/task_demos/mug_rack_easy_single/task_name_mug_on_rack/demo_aug_100_9.npz")

    visualize(action_pcd=item["pc_action"], anchor_pcd=item["pc_anchor"], goal_action_pcd=item["pc"], goal_tf=item["T_action2goal"])

