import numpy as np
import torch


def svd_based_registration(source: torch.Tensor, target: torch.Tensor):
    """
    Estimate translation and rotation using SVD-based Procrustes analysis.

    Args:
        source (torch.Tensor): Source point cloud, shape (N, 3).
        target (torch.Tensor): Target point cloud, shape (N, 3).

    Returns:
        estimated_translation (torch.Tensor): Estimated translation, shape (3,).
        estimated_rotation (torch.Tensor): Estimated rotation matrix, shape (3, 3).
    """
    # Compute centroids
    source_centroid = source.mean(dim=0)  # Shape (3,)
    target_centroid = target.mean(dim=0)  # Shape (3,)

    # Center the point clouds
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    # Compute the covariance matrix
    H = source_centered.T @ target_centered  # Shape (3, 3)

    # Singular Value Decomposition
    U, _, Vt = torch.linalg.svd(H)

    # Compute the rotation matrix
    estimated_rotation = Vt.T @ U.T

    # Ensure a proper rotation (det(R) = 1, not -1)
    if torch.det(estimated_rotation) < 0:
        Vt[-1, :] *= -1
        estimated_rotation = Vt.T @ U.T

    # Compute the translation in world-frame
    estimated_translation = target_centroid - source_centroid

    return estimated_translation, estimated_rotation


def calculate_errors(
    gt_translation: torch.Tensor,
    gt_rotation: torch.Tensor,
    estimated_translation: torch.Tensor,
    estimated_rotation: torch.Tensor,
):
    """
    Calculate translation and rotation errors.

    Args:
        gt_translation (torch.Tensor): Ground truth translation, shape (3,).
        gt_rotation (torch.Tensor): Ground truth rotation matrix, shape (3, 3).
        estimated_translation (torch.Tensor): Estimated translation, shape (3,).
        estimated_rotation (torch.Tensor): Estimated rotation matrix, shape (3, 3).

    Returns:
        translation_error (float): Translation error (Euclidean distance).
        rotation_error_deg (float): Rotation error in degrees.
    """
    # Translation error
    translation_error = torch.norm(gt_translation - estimated_translation).item()

    # Rotation error (angle between rotation matrices)
    R_diff = gt_rotation @ estimated_rotation.T
    trace = torch.trace(R_diff)  # Tr(R)
    rotation_error_rad = torch.arccos((trace - 1) / 2)
    rotation_error_deg = torch.rad2deg(rotation_error_rad).item()

    return translation_error, rotation_error_deg


# ---------------------------------------------------------------
#                    Toy Examples / Verification
# ---------------------------------------------------------------

if __name__ == "__main__":
    # Generate toy point clouds
    source_points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    
    # Compute centroid of the source point cloud
    source_centroid = source_points.mean(dim=0)

    # Apply ground truth rotation (object-centric) and translation (world-frame)
    gt_translation = torch.tensor([1.0, 2.0, 3.0])  # World-frame translation
    angle = np.pi / 4  # 45 degrees
    gt_rotation = torch.tensor(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    # Generate target points
    source_centered = source_points - source_centroid
    target_centered = (gt_rotation @ source_centered.T).T
    target_points = target_centered + source_centroid + gt_translation

    # Perform SVD-based registration
    est_translation, est_rotation = svd_based_registration(source_points, target_points)

    # Calculate errors
    trans_error, rot_error = calculate_errors(
        gt_translation, gt_rotation, est_translation, est_rotation
    )

    # Print results
    print("Ground Truth Translation:\n", gt_translation)
    print("Estimated Translation:\n", est_translation)
    print("Translation Error:", trans_error)

    print("\nGround Truth Rotation:\n", gt_rotation)
    print("Estimated Rotation:\n", est_rotation)
    print("Rotation Error (degrees):", rot_error)

    # Verify with another toy example
    source_points_2 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32
    )
    source_centroid_2 = source_points_2.mean(dim=0)

    gt_translation_2 = torch.tensor([2.0, -1.0, 1.0])  # World-frame translation
    angle_2 = np.pi / 3  # 60 degrees
    gt_rotation_2 = torch.tensor(
        [
            [np.cos(angle_2), -np.sin(angle_2), 0.0],
            [np.sin(angle_2), np.cos(angle_2), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    source_centered_2 = source_points_2 - source_centroid_2
    target_centered_2 = (gt_rotation_2 @ source_centered_2.T).T
    target_points_2 = target_centered_2 + source_centroid_2 + gt_translation_2

    # Perform SVD-based registration
    est_translation_2, est_rotation_2 = svd_based_registration(
        source_points_2, target_points_2
    )

    # Calculate errors
    trans_error_2, rot_error_2 = calculate_errors(
        gt_translation_2, gt_rotation_2, est_translation_2, est_rotation_2
    )

    # Print results
    print("\n--- Second Example ---")
    print("Ground Truth Translation:\n", gt_translation_2)
    print("Estimated Translation:\n", est_translation_2)
    print("Translation Error:", trans_error_2)

    print("\nGround Truth Rotation:\n", gt_rotation_2)
    print("Estimated Rotation:\n", est_rotation_2)
    print("Rotation Error (degrees):", rot_error_2)
