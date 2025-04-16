import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch
from non_rigid.utils.pointcloud_utils import downsample_pcd

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


def process_pointclouds(pointclouds, num_samples=1000):
    """
    Process pointclouds: center them, calculate average shape, and scale to unit cube
    using PyTorch with CUDA acceleration if available.
    
    Args:
        pointclouds: list of numpy arrays or torch tensors, each of shape (Ni, 3)
        num_samples: number of points to sample from each pointcloud using FPS
        
    Returns:
        avg_shape: torch tensor of shape (num_samples, 3) - average zero-meaned pointcloud
        scaling_factor: float - factor to scale to unit cube
        scaled_shape: torch tensor of shape (num_samples, 3) - scaled average pointcloud
    """
    # Set device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert pointclouds to PyTorch tensors if they're not already
    torch_pointclouds = []
    for pc in pointclouds:
        if isinstance(pc, np.ndarray):
            pc = torch.from_numpy(pc).float()
        torch_pointclouds.append(pc.to(device))
    
    # Center each pointcloud
    centered_pointclouds = []
    for pc in torch_pointclouds:
        centroid = torch.mean(pc, dim=0, keepdim=True)
        centered_pointclouds.append(pc - centroid)
    
    # Sample points using FPS (Farthest Point Sampling)
    sampled_pointclouds = []
    for pc in centered_pointclouds:

        sampled_pc, _ = downsample_pcd(pc.unsqueeze(0), num_samples, type="fps")
        sampled_pc = sampled_pc.squeeze(0)
        sampled_pointclouds.append(sampled_pc)
    
    # Stack and calculate the average shape
    all_sampled = torch.stack(sampled_pointclouds)
    avg_shape = torch.mean(all_sampled, dim=0)
    
    # Calculate scaling factor for unit cube
    min_coords, _ = torch.min(avg_shape, dim=0)
    max_coords, _ = torch.max(avg_shape, dim=0)
    dimensions = max_coords - min_coords
    avg_dimension = torch.mean(dimensions)
    scaling_factor = 1.0 / avg_dimension.item() if avg_dimension.item() > 0 else 1.0
    
    # Scale the average shape
    scaled_shape = avg_shape * scaling_factor
    
    # Print results
    print("Average Zero-Mean Pointcloud Dimensions:", dimensions.cpu().numpy())
    print("Scaling Factor:", scaling_factor)
    print("Scaled Pointcloud Dimensions:", (dimensions * scaling_factor).cpu().numpy())
    
    return avg_shape, scaling_factor, scaled_shape


# Example usage
if __name__ == "__main__":

    loaded_anchor, loaded_goal_action = load_insertion("/data/lyuxing/tax3d/insertion/demonstrations/12-15-ssd/learn_data/train")
    process_pointclouds(loaded_goal_action, num_samples=1024)