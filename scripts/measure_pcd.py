import os
import numpy as np
import torch
import open3d as o3d
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

def calculate_pcd_scale(points: np.ndarray) -> float:
    """
    Calculate the scale of a point cloud as the maximum distance from the centroid.
    
    Args:
        points: Point cloud as numpy array of shape (N, 3)
    
    Returns:
        Scale value as float
    """
    points_tensor = torch.as_tensor(points).float()
    point_dists = points_tensor - points_tensor.mean(dim=0, keepdim=True)
    point_scale = torch.linalg.norm(point_dists, dim=1, keepdim=True).max()
    return point_scale.item()

def process_demo_file(file_path: str) -> Dict[str, float]:
    """
    Process a single demo file and calculate scales for action and anchor point clouds.
    
    Args:
        file_path: Path to the .npz demo file
    
    Returns:
        Dictionary containing action and anchor scales, or None if processing fails
    """
    try:
        # Load the demo file
        demo = np.load(file_path, allow_pickle=True)
        
        # Extract point clouds (following dataloader pattern)
        points_action = demo.get('action_init_points') * 50.0
        points_anchor = demo.get('anchor_points') * 50.0
        goal_tf = demo.get('goal_tf')
        
        if points_action is None or points_anchor is None:
            print(f"Warning: Missing required keys in {file_path}")
            return None
        
        # Process action points (following dataloader transformation)
        points_action_pcd = o3d.geometry.PointCloud()
        points_action_pcd.points = o3d.utility.Vector3dVector(points_action)
        
        # Transform to goal frame if goal_tf is available
        if goal_tf is not None:
            points_action_goal_pcd = points_action_pcd.transform(np.linalg.inv(goal_tf))
            points_action_goal = np.asarray(points_action_goal_pcd.points)
        else:
            points_action_goal = points_action
            print(f"Warning: No goal_tf found in {file_path}, using original action points")
        
        # Calculate scales
        action_scale = calculate_pcd_scale(points_action)
        anchor_scale = calculate_pcd_scale(points_anchor)
        action_goal_scale = calculate_pcd_scale(points_action_goal)
        
        return {
            'action_scale': action_scale,
            'anchor_scale': anchor_scale,
            'action_goal_scale': action_goal_scale,
            'file_path': file_path
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def calculate_dataset_scales(data_folder: str) -> Dict[str, List[float]]:
    """
    Calculate scales for all .npz files in the given folder.
    
    Args:
        data_folder: Path to folder containing .npz demo files
    
    Returns:
        Dictionary containing lists of scales for each point cloud type
    """
    data_path = Path(data_folder)
    npz_files = list(data_path.glob('*.npz'))
    
    if not npz_files:
        print(f"No .npz files found in {data_folder}")
        return {}
    
    print(f"Found {len(npz_files)} .npz files")
    
    action_scales = []
    anchor_scales = []
    action_goal_scales = []
    processed_files = []
    
    for file_path in npz_files:
        #print(f"Processing: {file_path.name}")
        result = process_demo_file(str(file_path))
        
        if result is not None:
            action_scales.append(result['action_scale'])
            anchor_scales.append(result['anchor_scale'])
            action_goal_scales.append(result['action_goal_scale'])
            processed_files.append(result['file_path'])
    
    return {
        'action_scales': action_scales,
        'anchor_scales': anchor_scales,
        'action_goal_scales': action_goal_scales,
        'processed_files': processed_files
    }

def print_statistics(scales_dict: Dict[str, List[float]]):
    """Print detailed statistics for the calculated scales."""
    
    if not scales_dict:
        print("No data to analyze")
        return
    
    print(f"\n{'='*60}")
    print("POINT CLOUD SCALE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total files processed: {len(scales_dict['processed_files'])}")
    
    for scale_type in ['action_scales', 'anchor_scales', 'action_goal_scales']:
        if scale_type in scales_dict and scales_dict[scale_type]:
            scales = scales_dict[scale_type]
            scales_array = np.array(scales)
            
            print(f"\n{scale_type.replace('_', ' ').title()}:")
            print(f"  Count: {len(scales)}")
            print(f"  Mean:  {scales_array.mean():.6f}")
            print(f"  Std:   {scales_array.std():.6f}")
            print(f"  Min:   {scales_array.min():.6f}")
            print(f"  Max:   {scales_array.max():.6f}")
            print(f"  Median: {np.median(scales_array):.6f}")
            print(f"  25th percentile: {np.percentile(scales_array, 25):.6f}")
            print(f"  75th percentile: {np.percentile(scales_array, 75):.6f}")

def save_results(scales_dict: Dict[str, List[float]], output_file: str):
    """Save results to a numpy file for later analysis."""
    if scales_dict:
        np.savez(output_file, **scales_dict)
        print(f"\nResults saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Calculate point cloud scales for demo dataset')
    parser.add_argument('--data_folder', type=str, help='Path to folder containing .npz demo files')
    parser.add_argument('--output', '-o', type=str, default='pcd_scales_analysis.npz', 
                       help='Output file to save results (default: pcd_scales_analysis.npz)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Print detailed information for each file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_folder):
        print(f"Error: Data folder '{args.data_folder}' does not exist")
        return
    
    print(f"Analyzing point cloud scales in: {args.data_folder}")
    
    # Calculate scales for all files
    scales_dict = calculate_dataset_scales(args.data_folder)
    
    # Print statistics
    print_statistics(scales_dict)
    
    # Save results
    if scales_dict:
        #save_results(scales_dict, args.output)
        
        # Additional analysis
        if scales_dict['action_scales'] and scales_dict['anchor_scales']:
            action_scales = np.array(scales_dict['action_scales'])
            anchor_scales = np.array(scales_dict['anchor_scales'])
            
            print(f"\n{'='*60}")
            print("COMPARATIVE ANALYSIS")
            print(f"{'='*60}")
            print(f"Action vs Anchor scale ratio (mean): {(action_scales / anchor_scales).mean():.6f}")
            print(f"Action vs Anchor scale ratio (std):  {(action_scales / anchor_scales).std():.6f}")
            '''
            # Recommend scale factor
            max_scale = max(action_scales.max(), anchor_scales.max())
            recommended_scale = 1.0 / max_scale
            print(f"\nRecommended scale factor to normalize to [0,1]: {recommended_scale:.6f}")
            print(f"This would make the largest point cloud have scale = 1.0")
            '''
if __name__ == "__main__":
    main()