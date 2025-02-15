import numpy as np
import os
from tqdm import tqdm

def load_proccloth(dir):
    npz_files = [f for f in os.listdir(dir) if f.endswith('.npz')]
    loaded_action = []
    loaded_anchor = []

    for npz_file in npz_files:
        file_path = os.path.join(dir, npz_file)
        demo = np.load(file_path, allow_pickle=True)
        loaded_action.append(demo["action_pc"])
        loaded_anchor.append(demo["anchor_pc"])
    
    return loaded_action, loaded_anchor

def load_ndf(dir):
    npz_files = [f for f in os.listdir(dir) if f.endswith('_teleport_obj_points.npz')]
    loaded_action = []
    loaded_anchor = []

    for npz_file in npz_files:
        file_path = os.path.join(dir, npz_file)
        demo = np.load(file_path, allow_pickle=True)

        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        points_action = points_raw[classes_raw == 0]
        points_anchor = points_raw[classes_raw == 1]

        loaded_action.append(points_action)
        loaded_anchor.append(points_anchor)

    return loaded_action, loaded_anchor

def load_rpdiff(dir):
    npz_files = [f for f in os.listdir(dir) if f.endswith('.npz')]
    loaded_action = []
    loaded_anchor = []

    for npz_file in npz_files:
        file_path = os.path.join(dir, npz_file)
        demo = np.load(file_path, allow_pickle=True)
        points_anchor = demo['multi_obj_start_pcd'].item()['parent']
        points_action = demo['multi_obj_start_pcd'].item()['child']
        loaded_action.append(points_action)
        loaded_anchor.append(points_anchor)

    return loaded_action, loaded_anchor

def calculate_mean_bounding_box_excluding_outliers(pcd_list, min_side_length=np.float32(1e-5), scaling_factor=1):
    bbox_sizes = []
    bbox_volumes = []

    # Calculate bounding box sizes and volumes for all point clouds
    for i, pcd in tqdm(enumerate(pcd_list), total=len(pcd_list), desc="Processing PCDs"):
        if pcd.shape[1] != 3:
            raise ValueError(f"Point cloud {i} must have shape [N, 3]. Got shape {pcd.shape}.")
        
        # Scale the point cloud
        scaled_pcd = pcd * scaling_factor

        # Compute axis-aligned bounding box (AABB)
        min_bound = np.min(scaled_pcd, axis=0)  # Minimum x, y, z values
        max_bound = np.max(scaled_pcd, axis=0)  # Maximum x, y, z values

        # Bounding box size and volume
        bbox_size = max_bound - min_bound

        # In case that one side of the box is too small
        adjusted_bbox_size = np.where(bbox_size < min_side_length, np.float32(1.0), bbox_size).astype(np.float32)
        bbox_volume = np.prod(adjusted_bbox_size)

        # Collect results
        bbox_sizes.append(bbox_size)
        bbox_volumes.append(bbox_volume)

    # Convert to numpy arrays for filtering
    bbox_sizes = np.array(bbox_sizes)  # Shape: [N, 3]
    bbox_volumes = np.array(bbox_volumes)  # Shape: [N]

    # Exclude outliers using the IQR method
    def exclude_outliers(data):
        Q1 = np.percentile(data, 25, axis=0)  # 25th percentile (lower quartile)
        Q3 = np.percentile(data, 75, axis=0)  # 75th percentile (upper quartile)
        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data >= lower_bound) & (data <= upper_bound)

    # Identify valid bounding box sizes and volumes
    valid_size_mask = exclude_outliers(bbox_sizes).all(axis=1)
    valid_volume_mask = exclude_outliers(bbox_volumes)

    # Combine masks
    valid_mask = valid_size_mask & valid_volume_mask

    # Filter valid sizes and volumes
    valid_bbox_sizes = bbox_sizes[valid_mask]
    valid_bbox_volumes = bbox_volumes[valid_mask]

    # Compute mean bounding box size and volume from valid data
    mean_bbox_size = np.mean(valid_bbox_sizes, axis=0)
    mean_bbox_volume = np.mean(valid_bbox_volumes)

    return mean_bbox_size, mean_bbox_volume, valid_mask

# Example usage
if __name__ == "__main__":
    # Create a list of example point clouds
    proccloth_action, proccloth_anchor = load_proccloth(dir='/data/lyuxing/tax3d/proccloth/cloth=multi-fixed anchor=single-random hole=single/train_tax3d/')

    mean_bbox_size, mean_bbox_volume, valid_mask = calculate_mean_bounding_box_excluding_outliers(proccloth_action, scaling_factor=1)
    print(f"ProcCloth Action Mean Bounding Box Size (Excluding Outliers): {mean_bbox_size}")
    print(f"ProcCloth Action Mean Bounding Box Volume (Excluding Outliers): {mean_bbox_volume:.2f}")

    mean_bbox_size, mean_bbox_volume, valid_mask = calculate_mean_bounding_box_excluding_outliers(proccloth_anchor, scaling_factor=1)
    print(f"ProcCloth Anchor Mean Bounding Box Size (Excluding Outliers): {mean_bbox_size}")
    print(f"ProcCloth Anchor Bounding Box Volume (Excluding Outliers): {mean_bbox_volume:.2f}")



    ndf_action, ndf_anchor = load_ndf(dir='/data/lyuxing/tax3d/ndf/mugplace/train_data/renders/')

    mean_bbox_size, mean_bbox_volume, valid_mask = calculate_mean_bounding_box_excluding_outliers(ndf_action, scaling_factor=15)
    print(f"NDF Action Mean Bounding Box Size (Excluding Outliers): {mean_bbox_size}")
    print(f"NDF Action Mean Bounding Box Volume (Excluding Outliers): {mean_bbox_volume:.2f}")

    mean_bbox_size, mean_bbox_volume, valid_mask = calculate_mean_bounding_box_excluding_outliers(ndf_anchor, scaling_factor=15)
    print(f"NDF Anchor Mean Bounding Box Size (Excluding Outliers): {mean_bbox_size}")
    print(f"NDF Anchor Bounding Box Volume (Excluding Outliers): {mean_bbox_volume:.2f}")



    rpdiff_action, rpdiff_anchor = load_rpdiff(dir='/data/lyuxing/tax3d/rpdiff/data/task_demos/mug_rack_easy_single/task_name_mug_on_rack')

    mean_bbox_size, mean_bbox_volume, valid_mask = calculate_mean_bounding_box_excluding_outliers(rpdiff_action, scaling_factor=1)
    print(f"RPDiff Action Mean Bounding Box Size (Excluding Outliers): {mean_bbox_size}")
    print(f"RPDiff Action Mean Bounding Box Volume (Excluding Outliers): {mean_bbox_volume:.2f}")

    mean_bbox_size, mean_bbox_volume, valid_mask = calculate_mean_bounding_box_excluding_outliers(rpdiff_anchor, scaling_factor=1)
    print(f"RPDiff Anchor Mean Bounding Box Size (Excluding Outliers): {mean_bbox_size}")
    print(f"RPDiff Anchor Bounding Box Volume (Excluding Outliers): {mean_bbox_volume:.2f}")