from typing import List, Tuple, cast

import numpy as np
import torch
from pytorch3d.common.datatypes import Optional
from pytorch3d.transforms import (
    Rotate,
    Transform3d,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    so3_rotation_angle,
)
from scipy.spatial.transform import Rotation as R


def random_se3(
    N: int = 1,
    rot_var: float = np.pi / 180 * 5,
    trans_var: float = 0.1,
    rot_sample_method: str = "axis_angle",
    device: Optional[str] = None,
) -> Transform3d:
    """
    Generates random transforms in SE(3) space.

    Args:
        N (int): Number of transforms to generate.
        rot_var (float): Maximum rotation in radians
        trans_var (float): Maximum translation
        rot_sample_method (str): Method to sample the rotation. Options are:
            - "axis_angle": Random axis angle sampling
            - "axis_angle_uniform_z": Random axis angle sampling with uniform z axis rotation
            - "quat_uniform": Uniform SE(3) sampling
            - "random_flat_upright": Random rotation around z axis and xy translation (no z translation)
            - "random_upright": Random rotation around z axis and xyz translation
        device: Device to put the transform on.

    Returns:
        (Transform3d): Random SE(3) transform(s).
    """

    assert rot_sample_method in [
        "axis_angle",
        "axis_angle_uniform_z",
        "quat_uniform",
        "random_flat_upright",
        "random_upright",
        "discrete_random_upright",
        "identity",
    ]

    if rot_sample_method == "axis_angle":
        # this is random axis angle sampling (rot_sample_method == "axis_angle")
        axis_angle_random = torch.randn(N, 3, device=device)
        rot_ratio = (
            torch.rand(1).item()
            * rot_var
            / torch.norm(axis_angle_random, dim=1).max().item()
        )
        constrained_axix_angle = rot_ratio * axis_angle_random  # max angle is rot_var
        R = axis_angle_to_matrix(constrained_axix_angle)
        random_translation = torch.randn(N, 3, device=device)
        translation_ratio = (
            trans_var / torch.norm(random_translation, dim=1).max().item()
        )
        t = torch.rand(1).item() * translation_ratio * random_translation
    elif rot_sample_method == "axis_angle_uniform_z":
        # this is random axis angle sampling
        axis_angle_random = torch.randn(N, 3, device=device)
        rot_ratio = (
            torch.rand(1).item()
            * rot_var
            / torch.norm(axis_angle_random, dim=1).max().item()
        )
        constrained_axix_angle = rot_ratio * axis_angle_random  # max angle is rot_var
        R_random = axis_angle_to_matrix(constrained_axix_angle)

        # this is uniform z axis rotation sampling
        theta = torch.rand(N, 1, device=device) * 2 * np.pi
        axis_angle_z = torch.cat([torch.zeros(N, 2, device=device), theta], dim=1)
        R_z = axis_angle_to_matrix(axis_angle_z)

        R = torch.bmm(R_z, R_random)
        random_translation = torch.randn(N, 3, device=device)
        translation_ratio = (
            trans_var / torch.norm(random_translation, dim=1).max().item()
        )
        t = torch.rand(1).item() * translation_ratio * random_translation
    elif rot_sample_method == "quat_uniform":
        # This true uniform SE(3) sampling tends to make it hard to train the models
        # In contrast, the axis angle sampling tends to leave the objects close to upright
        quat = torch.randn(N, 4, device=device)
        quat = quat / torch.linalg.norm(quat)
        R = quaternion_to_matrix(quat)
        random_translation = torch.randn(N, 3, device=device)
        translation_ratio = (
            trans_var / torch.norm(random_translation, dim=1).max().item()
        )
        t = torch.rand(1).item() * translation_ratio * random_translation
    elif rot_sample_method == "random_upright":
        # Random rotation around z axis and xy translation
        theta = torch.rand(N, 1, device=device) * 2 * np.pi
        axis_angle_z = torch.cat([torch.zeros(N, 2, device=device), theta], dim=1)
        R = axis_angle_to_matrix(axis_angle_z)

        random_translation = torch.randn(N, 3, device=device)
        translation_ratio = (
            trans_var / torch.norm(random_translation, dim=1).max().item()
        )
        t = torch.rand(1).item() * translation_ratio * random_translation
    elif rot_sample_method == "random_flat_upright":
        # Random rotation around z axis and xy translation (no z translation)
        theta = torch.rand(N, 1, device=device) * 2 * np.pi
        axis_angle_z = torch.cat([torch.zeros(N, 2, device=device), theta], dim=1)
        R = axis_angle_to_matrix(axis_angle_z)

        random_translation = torch.randn(N, 3, device=device)
        random_translation[:, 2] = 0
        translation_ratio = (
            trans_var / torch.norm(random_translation, dim=1).max().item()
        )
        t = torch.rand(1).item() * translation_ratio * random_translation
    elif rot_sample_method == "discrete_random_upright":
        # Discrete z-axis rotation with 0, 90, 180, 270 degrees
        angles = torch.tensor([0, np.pi/2], device=device)
        theta_idx = torch.randint(0, 2, (N,), device=device)
        theta = angles[theta_idx].unsqueeze(1)
        axis_angle_z = torch.cat([torch.zeros(N, 2, device=device), theta], dim=1)
        R = axis_angle_to_matrix(axis_angle_z)

        random_translation = torch.randn(N, 3, device=device)
        translation_ratio = (
            trans_var / torch.norm(random_translation, dim=1).max().item()
        )
        t = torch.rand(1).item() * translation_ratio * random_translation
    elif rot_sample_method == "identity":
        R = torch.eye(3, device=device).unsqueeze(0).repeat(N, 1, 1)
        t = torch.zeros(N, 3, device=device)
    else:
        raise ValueError(f"Unknown rot_sample_method: {rot_sample_method}")
    return Rotate(R, device=device).translate(t)


def symmetric_orthogonalization(M: torch.Tensor) -> torch.Tensor:
    """
    Maps arbitrary input matrices onto SO(3) via symmetric orthogonalization.
    (modified from https://github.com/amakadia/svd_for_pose)

    Args:
        M: should have size [batch_size, 3, 3]

    Returns:
        torch.Tensor: Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    U, _, Vh = torch.linalg.svd(M)
    det = torch.det(torch.bmm(U, Vh)).view(-1, 1, 1)
    Vh = torch.cat((Vh[:, :2, :], Vh[:, -1:, :] * det), 1)
    R = U @ Vh
    return cast(torch.Tensor, R)


def flow_to_tf(
    start_xyz: torch.Tensor,
    flow: torch.Tensor,
) -> Transform3d:
    """
    Converts point-wise flows into a rigid transform.

    Args:
        start_xyz: (B, N, 3) tensor of initial point positions
        flow: (B, N, 3) tensor of point-wise flows

    Returns:
        Transform3d: rigid transform
    """
    assert start_xyz.shape == flow.shape
    B, N, _ = start_xyz.shape

    start_xyz_mean = start_xyz.mean(dim=1, keepdim=True)
    start_xyz_demean = start_xyz - start_xyz_mean

    target_xyz = start_xyz + flow
    target_xyz_mean = target_xyz.mean(dim=1, keepdim=True)
    target_xyz_demean = target_xyz - target_xyz_mean

    X = torch.bmm(start_xyz_demean.transpose(-2, -1), target_xyz_demean)

    R = symmetric_orthogonalization(X)
    t = target_xyz_mean - torch.bmm(start_xyz_mean, R)

    return Rotate(R).translate(t.squeeze(1))


def get_degree_angle(T: Transform3d) -> Tuple[float, float, float]:
    """
    Get the maximum, minimum, and mean rotation angles in degrees from a Transform3d object.

    Args:
        T: Transform3d object

    Returns:
        Tuple[float, float, float]: Tuple of maximum, minimum, and mean rotation angles in degrees.
    """

    angle_rad_T = (
        so3_rotation_angle(T.get_matrix()[:, :3, :3], eps=1e-2) * 180 / np.pi
    )  # B

    max = torch.max(angle_rad_T).item()
    min = torch.min(angle_rad_T).item()
    mean = torch.mean(angle_rad_T).item()
    return max, min, mean


def get_translation(T: Transform3d) -> Tuple[float, float, float]:
    """
    Get the maximum, minimum, and mean translation magnitudes from a Transform3d object.

    Args:
        T: Transform3d object

    Returns:
        Tuple[float, float, float]: Tuple of maximum, minimum, and mean translation magnitudes.
    """
    t = T.get_matrix()[:, 3, :3]  # B,3
    t_norm = torch.norm(t, dim=1)  # B
    max = torch.max(t_norm).item()
    min = torch.min(t_norm).item()
    mean = torch.mean(t_norm).item()
    return max, min, mean


def get_transform_list_min_rotation_errors(
    T1: Transform3d, T_list: List[Transform3d]
) -> Tuple[float, float, float]:
    """
    Get the maximum, minimum, and mean rotation errors in degrees between a Transform3d object and a list of Transform3d objects.

    Args:
        T1: Transform3d object
        T_list: List of Transform3d objects

    Returns:
        Tuple[float, float, float]: Tuple of maximum, minimum, and mean rotation errors in degrees.
    """
    angle_rad_T1 = (
        so3_rotation_angle(T1.get_matrix()[:, :3, :3], eps=1e-2) * 180 / np.pi
    )  # B
    angle_rad_T_list = []
    for T in T_list:
        angle_rad_T = (
            so3_rotation_angle(T.get_matrix()[:, :3, :3], eps=1e-2) * 180 / np.pi
        )
        angle_rad_T_list.append(angle_rad_T)

    angle_rad_min = torch.min(angle_rad_T1, angle_rad_T_list[0])
    for angle_rad_T in angle_rad_T_list[1:]:
        angle_rad_min = torch.min(angle_rad_min, angle_rad_T)

    max = torch.max(angle_rad_min).item()
    min = torch.min(angle_rad_min).item()
    mean = torch.mean(angle_rad_min).item()
    return max, min, mean


def get_transform_list_min_translation_errors(
    T1: Transform3d, T_list: List[Transform3d]
) -> Tuple[float, float, float]:
    """
    Get the maximum, minimum, and mean translation errors between a Transform3d object and a list of Transform3d objects.

    Args:
        T1: Transform3d object
        T_list: List of Transform3d objects

    Returns:
        Tuple[float, float, float]: Tuple of maximum, minimum, and mean translation errors.
    """
    t1 = T1.get_matrix()[:, 3, :3]  # B,3
    t1_norm = torch.norm(t1, dim=1)  # B
    t_list = []
    for T in T_list:
        t = T.get_matrix()[:, 3, :3]  # B,3
        t_norm = torch.norm(t, dim=1)  # B
        t_list.append(t_norm)

    t_min = torch.min(t1_norm, t_list[0])
    for t in t_list[1:]:
        t_min = torch.min(t_min, t)

    max = torch.max(t_min).item()
    min = torch.min(t_min).item()
    mean = torch.mean(t_min).item()
    return max, min, mean


def matrix_from_list(pose_list: List[float]) -> np.ndarray:
    """
    Convert a list of pose parameters to a 4x4 matrix.

    Args:
        pose_list: List of pose parameters [tx, ty, tz, qx, qy, qz, qw]

    Returns:
        np.ndarray: 4x4 matrix
    """
    trans = pose_list[:3]
    quat = pose_list[3:]

    T = np.eye(4)
    T[:-1, :-1] = R.from_quat(quat).as_matrix()
    T[:-1, -1] = trans
    return T

def update_rt_transformation(
    R: torch.Tensor,
    t: torch.Tensor,
    c_A: torch.Tensor,
    c_B: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates a batch of rotations (R) and translations (t) for separate source and destination coordinate shifts.

    Args:
        R (torch.Tensor): The original batch of 3x3 rotation matrices.
                          Shape: (batch_size, 3, 3)
        t (torch.Tensor): The original batch of 3D translation vectors.
                          Shape: (batch_size, 3)
        c_A (torch.Tensor): The batch of 3D translation vectors for the source point clouds (A).
                            Shape: (batch_size, 3)
        c_B (torch.Tensor): The batch of 3D translation vectors for the destination point clouds (B).
                            Shape: (batch_size, 3)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - R_new (torch.Tensor): The updated batch of 3x3 rotation matrices (which is the same as R).
            - t_new (torch.Tensor): The updated batch of 3D translation vectors.
    """
    # Ensure all tensors are on the same device and have the same dtype
    device = R.device
    dtype = R.dtype
    t = t.to(device=device, dtype=dtype)
    c_A = c_A.to(device=device, dtype=dtype)
    c_B = c_B.to(device=device, dtype=dtype)

    # 1. The new rotation is the same as the original rotation
    R_new = R

    # 2. Calculate the new translation: t_new = R @ c_A + t - c_B
    # We need to reshape c_A for batched matrix-vector multiplication
    # (B, 3, 3) @ (B, 3, 1) -> (B, 3, 1) -> squeeze to (B, 3)
    rotated_c_A = torch.bmm(R, c_A.unsqueeze(-1)).squeeze(-1)
    t_new = rotated_c_A + t - c_B

    return R_new, t_new

# batch*n
def normalize_vector( v):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
    
# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out
        
#poses batch*6/batch*3*2
#poses
def compute_rotation_matrix_from_ortho6d(poses):
    #x_raw = poses[:,0:3]#batch*3
    #y_raw = poses[:,3:6]#batch*3

    x_raw = poses[:,:,0]#batch*3
    y_raw = poses[:,:,1]#batch*3

    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

def transform_pointcloud(pc: torch.Tensor, 
                         T: torch.Tensor = None, 
                         R: torch.Tensor = None, 
                         t: torch.Tensor = None) -> torch.Tensor:
    """
    Transforms a batched point cloud by a batched transformation matrix.

    Args:
        pc (torch.Tensor): The point cloud to transform. Shape: (bs, 3, N)
        T (torch.Tensor): The transformation matrix. Shape: (bs, 4, 4)
        R (torch.Tensor): The rotation matrix. Shape: (bs, 3, 3)
        t (torch.Tensor): The translation vector. Shape: (bs, 3)

    Returns:
        torch.Tensor: The transformed point cloud. Shape: (bs, 3, N)
    """
    # Decompose the transformation matrix
    assert (T is not None) or (R is not None and t is not None), \
        "Either T must be provided, or both R and t must be provided"
    
    if T is not None:
        R = T[:, :3, :3]  # Rotation matrix (bs, 3, 3)
        t = T[:, :3, 3]   # Translation vector (bs, 3)

    # Apply rotation
    # (bs, 3, 3) @ (bs, 3, N) -> (bs, 3, N)
    rotated_pc = torch.bmm(R, pc)

    # Apply translation using broadcasting
    # (bs, 3, N) + (bs, 3, 1) -> (bs, 3, N)
    transformed_pc = rotated_pc + t.unsqueeze(-1)
    
    return transformed_pc

def compute_transformed_pointcloud_from_diffusion(pred_r: torch.Tensor, 
                                                  pred_s: torch.Tensor,
                                                  pc_action: torch.Tensor) -> torch.Tensor:
    # Assert same batch dimensions
    assert pred_r.shape[0] == pred_s.shape[0] == pc_action.shape[0], \
        f"Batch dimensions must match: pred_r {pred_r.shape[0]}, pred_s {pred_s.shape[0]}, pc_action {pc_action.shape[0]}"
    
    # Assert pred_r has last two dimensions of (3, 1)
    assert pred_r.shape[-2:] == (3, 1), \
        f"pred_r must have shape (..., 3, 1), got {pred_r.shape}"

    # Assert pred_s has last two dimensions of (6, 1)
    '''
    assert pred_s.shape[-2:] == (6, 1), \
        f"pred_r must have shape (..., 6, 1), got {pred_s.shape}"
    '''
    assert pred_s.shape[-2:] == (3, 2), \
        f"pred_s must have shape (..., 3, 2), got {pred_s.shape}"

    t = pred_r.squeeze(-1)
    R = compute_rotation_matrix_from_ortho6d(pred_s)

    if pc_action.shape[-1] == 3:
        pc_action = pc_action.permute(0, 2, 1)

    transformed_pc = transform_pointcloud(pc=pc_action, R=R, t=t)

    return transformed_pc