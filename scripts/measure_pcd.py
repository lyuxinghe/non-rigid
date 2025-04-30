import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from tqdm import tqdm


def load_insertion(dir, num_demos=None):
    npz_files = [f for f in os.listdir(dir) if f.endswith('.npz')]
    loaded_anchor = []
    loaded_goal_action = []

    if num_demos:
        npz_files = npz_files[:num_demos]

    for npz_file in npz_files:
        file_path = os.path.join(dir, npz_file)
        demo = np.load(file_path, allow_pickle=True)

        # Extract point clouds
        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        points_action = points_raw[classes_raw == 0]
        points_anchor = points_raw[classes_raw == 1]

        loaded_anchor.append(points_anchor)
        loaded_goal_action.append(points_action)

    return loaded_anchor, loaded_goal_action


def compute_extent_and_scale_stats(pcd_list):
    """
    Compute statistics of point cloud extents and scaling factors.

    Args:
        pcd_list (List[np.ndarray or torch.Tensor]): 
            List of point clouds of shape (N_i, 3)

    Returns:
        dict: Dictionary with:
            - 'mean_extent': np.ndarray of shape (3,)
            - 'min_extent': np.ndarray of shape (3,)
            - 'max_extent': np.ndarray of shape (3,)
            - 'extent_25': np.ndarray of shape (3,)
            - 'extent_75': np.ndarray of shape (3,)
            - 'mean_scale': float
            - 'scale_25': float
            - 'scale_75': float
    """
    extents = []
    scales = []

    for pc in pcd_list:
        if isinstance(pc, torch.Tensor):
            pc = pc.detach().cpu().numpy()

        center = pc.mean(axis=0)
        pc_centered = pc - center

        min_xyz = pc_centered.min(axis=0)
        max_xyz = pc_centered.max(axis=0)
        extent = max_xyz - min_xyz
        avg_extent = extent.mean()
        scale = 1.0 / avg_extent

        extents.append(extent)
        scales.append(scale)

    extents = np.stack(extents, axis=0)
    scales = np.array(scales)

    return {
        "mean_extent": extents.mean(axis=0),
        "min_extent": extents.min(axis=0),
        "max_extent": extents.max(axis=0),
        "extent_25": np.percentile(extents, 25, axis=0),
        "extent_75": np.percentile(extents, 75, axis=0),
        "mean_scale": scales.mean(),
        "scale_25": np.percentile(scales, 25),
        "scale_75": np.percentile(scales, 75)
    }


def print_extent_stats(name, stats):
    """
    Nicely print extent and scale stats.

    Args:
        name (str): Label for the point cloud group
        stats (dict): Output from compute_extent_and_scale_stats
    """
    print(f"\n=== Stats for {name} ===")
    print(f"Mean Extent (x, y, z): {stats['mean_extent']}")
    print(f"Min Extent  (x, y, z): {stats['min_extent']}")
    print(f"Max Extent  (x, y, z): {stats['max_extent']}")
    print(f"25% Extent (x, y, z):  {stats['extent_25']}")
    print(f"75% Extent (x, y, z):  {stats['extent_75']}")
    print(f"Mean Scale:            {stats['mean_scale']:.5f}")
    print(f"25% Scale:             {stats['scale_25']:.5f}")
    print(f"75% Scale:             {stats['scale_75']:.5f}")

# Example usage
if __name__ == "__main__":
    # Create a list of example point clouds

    
    #rpdiff_action, rpdiff_anchor, rpdiff_goal_action = load_rpdiff(dir='/data/lyuxing/tax3d/rpdiff/data/task_demos/can_in_cabinet_stack/task_name_stack_can_in_cabinet/', num_demos=None)
    #rpdiff_action, rpdiff_anchor, rpdiff_goal_action = load_rpdiff(dir='/data/lyuxing/tax3d/rpdiff/data/task_demos/book_on_bookshelf_double_view_rnd_ori/task_name_book_in_bookshelf/preprocessed/', num_demos=None)
    #rpdiff_action, rpdiff_anchor, rpdiff_goal_action = load_rpdiff(dir='/data/lyuxing/tax3d/rpdiff/data/task_demos/mug_rack_easy_single//task_name_mug_on_rack/preprocessed/', num_demos=None)
    #rpdiff_action, rpdiff_anchor, rpdiff_goal_action = load_rpdiff(dir='/data/lyuxing/tax3d/rpdiff/data/task_demos/mug_rack_med_single/task_name_mug_on_rack/preprocessed/', num_demos=None)
    #rpdiff_action, rpdiff_anchor, rpdiff_goal_action = load_rpdiff(dir='/data/lyuxing/tax3d/rpdiff/data/task_demos/mug_on_rack_multi_large_proc_gen_demos/task_name_mug_on_rack_multi/preprocessed/', num_demos=None)
    #dedo_action, dedo_anchor = load_proccloth('/home/lyuxing/Desktop/tax3d_upgrade/datasets/ProcCloth/cloth=single-fixed anchor=single-random hole=single/train_tax3d')

    insertion_anchor, insertion_action = load_insertion('/data/lyuxing/tax3d/insertion/demonstrations_new/04-21-wp-2/learn_data/train')
    #insertion_anchor, insertion_action = load_insertion('/home/mfi/repos/rtc_vision_toolbox/data/tax3d-yk-creator/demonstrations/04-21-dsub-1/learn_data/train')

    test_scale = 50
    insertion_action = [pc * test_scale for pc in insertion_action]
    insertion_anchor = [pc * test_scale for pc in insertion_anchor]

    stats_action = compute_extent_and_scale_stats(insertion_action)
    stats_anchor = compute_extent_and_scale_stats(insertion_anchor)
    stats_combined = compute_extent_and_scale_stats(
        [np.concatenate((pc1, pc2), axis=0) for pc1, pc2 in zip(insertion_action, insertion_anchor)]
    )
    
    
    print_extent_stats("Action", stats_action)
    print_extent_stats("Anchor", stats_anchor)
    print_extent_stats("Combined", stats_combined)