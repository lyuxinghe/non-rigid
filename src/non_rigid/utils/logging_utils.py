import torch
import wandb

from non_rigid.utils.vis_utils import get_color
from typing import Optional


def viz_predicted_vs_gt(
    pc_pos_viz: torch.Tensor,
    pc_action_viz: torch.Tensor,
    pred_action_viz: torch.Tensor,
    pc_anchor_viz: Optional[torch.Tensor] = None,
) -> wandb.Object3D:
    """
    Generates a 3D visualization of the predicted point cloud and the ground truth point cloud.

    Args:
        pc_pos_viz: [n, 3] tensor of action point cloud in goal position.
        pc_action_viz: [n, 3] tensor of action point cloud in starting position.
        pred_action_viz: [n, 3] tensor of predicted action point cloud in goal position.
        pc_anchor_viz: [n, 3] tensor of anchor point cloud.

    Returns:
        wandb.Object3D: the 3D visualization.
    """
    if pc_anchor_viz is None:
        pc_viz = pc_action_viz
    else:
        pc_viz = torch.cat([pc_action_viz, pc_anchor_viz], dim=0)


    # pc_viz = torch.cat([pc_action_viz, pc_anchor_viz], dim=0)
    pc_viz_min = pc_viz.min(dim=0).values
    pc_viz_max = pc_viz.max(dim=0).values
    pc_viz_extent = pc_viz_max - pc_viz_min
    pred_action_viz = pred_action_viz[
        (pred_action_viz[:, 0] > pc_viz_min[0] - 0.5 * pc_viz_extent[0])
        & (pred_action_viz[:, 0] < pc_viz_max[0] + 0.5 * pc_viz_extent[0])
        & (pred_action_viz[:, 1] > pc_viz_min[1] - 0.5 * pc_viz_extent[1])
        & (pred_action_viz[:, 1] < pc_viz_max[1] + 0.5 * pc_viz_extent[1])
        & (pred_action_viz[:, 2] > pc_viz_min[2] - 0.5 * pc_viz_extent[2])
        & (pred_action_viz[:, 2] < pc_viz_max[2] + 0.5 * pc_viz_extent[2])
    ]

    if pc_anchor_viz is None:
        predicted_vs_gt_tensors = [
            pc_pos_viz,
            pred_action_viz,
            pc_action_viz,
        ]
        predicted_vs_gt_colors = ["green", "blue", "yellow"]
    else:
        predicted_vs_gt_tensors = [
            pc_pos_viz,
            pc_anchor_viz,
            pred_action_viz,
            pc_action_viz,
        ]
        predicted_vs_gt_colors = ["green", "red", "blue", "yellow"]

    predicted_vs_gt = get_color(
        tensor_list=predicted_vs_gt_tensors,
        color_list=predicted_vs_gt_colors,
    )
    return wandb.Object3D(predicted_vs_gt)
