import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import lightning as L
import numpy as np
import omegaconf
import torch
import torch.utils.data as data
from pytorch3d.transforms import Translate

from non_rigid.utils.augmentation_utils import maybe_apply_augmentations
from non_rigid.utils.pointcloud_utils import downsample_pcd, get_multi_anchor_scene
from non_rigid.utils.transform_utils import random_se3


@dataclass
class RigidDatasetCfg:
    name: str = "rigid"
    data_dir: str = "data/rigid"
    type: str = "train"
    # Misc. kwargs to pass to the dataset e.g. dataset specific parameters
    misc_kwargs: Dict = field(default_factory=dict)

    ###################################################
    # General Dataset Parameters
    ###################################################
    # Number of demos to load
    num_demos: Optional[int] = None
    # Length of the train dataset
    train_dataset_size: int = 256
    # Length of the validation dataset
    val_dataset_size: int = 16
    # Use default values for some parameters on the validation set
    val_use_defaults: bool = False

    ###################################################
    # Point Cloud Transformation Parameters
    ###################################################
    # [action, anchor, none], centers the point clouds w.r.t. the action, anchor, or no centering
    center_type: str = "anchor"
    # Number of points to downsample to
    sample_size_action: int = 1024
    sample_size_anchor: int = 1024
    # Method of downsampling the point cloud
    downsample_type: str = "fps"
    # Scale factor to apply to the point clouds
    pcd_scale_factor: float = 1.0

    # Demonstration transformation parameters
    action_transform_type: str = "quat_uniform"
    action_translation_variance: float = 0.0
    action_rotation_variance: float = 180

    # Demonstration transformation parameters
    anchor_transform_type: str = "quat_uniform"
    anchor_translation_variance: float = 0.0
    anchor_rotation_variance: float = 180

    ###################################################
    # Distractor parameters
    ###################################################
    # Number of distractor anchor point clouds to load
    distractor_anchor_pcds: int = 0
    # Transformation type to apply when generating distractor pcds
    distractor_transform_type: str = "random_flat_upright"
    # Translation and rotation variance for the distractor transformations
    distractor_translation_variance: float = 0.5
    distractor_rotation_variance: float = 180

    ###################################################
    # Data Augmentation Parameters
    ###################################################
    # Probability of applying plane occlusion
    action_plane_occlusion: float = 0.0
    anchor_plane_occlusion: float = 0.0
    # Standoff distance of the occluding plane from selected plane origin
    action_plane_standoff: float = 0.0
    anchor_plane_standoff: float = 0.0
    # Probability of applying ball occlusion
    action_ball_occlusion: float = 0.0
    anchor_ball_occlusion: float = 0.0
    # Radius of the occluding ball
    action_ball_radius: float = 0.0
    anchor_ball_radius: float = 0.0


class RigidPointDataset(data.Dataset):
    def __init__(
        self,
        root: Path,
        type: str = "train",
        dataset_cfg: RigidDatasetCfg = RigidDatasetCfg(),
    ):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg

        self.dataset_dir = self.root / self.type
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        self.demo_files = list(self.dataset_dir.glob("*_teleport_obj_points.npz"))
        if self.dataset_cfg.num_demos is not None and self.type == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.type} demos from {self.dataset_dir}")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_dataset_size
        elif self.type == "val":
            return self.dataset_cfg.val_dataset_size
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos])

        # Extract point clouds
        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        # TODO: Remove this hardcoded class selection
        points_action = points_raw[classes_raw == 0]
        points_anchor = points_raw[classes_raw == 1]

        # Center the point clouds
        if self.dataset_cfg.center_type == "action_center":
            center = points_action.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_center":
            center = points_anchor.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_random":
            center = points_anchor[np.random.choice(len(points_anchor))]
        elif self.dataset_cfg.center_type == "none":
            center = np.zeros(3)
        else:
            raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
        points_action -= center
        points_anchor -= center

        points_action = torch.as_tensor(points_action).float()
        points_anchor = torch.as_tensor(points_anchor).float()

        # Downsample the point clouds
        points_action, _ = downsample_pcd(
            points_action.unsqueeze(0),
            num_points=self.dataset_cfg.sample_size_action,
            type=self.dataset_cfg.downsample_type,
        )
        points_anchor, _ = downsample_pcd(
            points_anchor.unsqueeze(0),
            num_points=self.dataset_cfg.sample_size_anchor,
            type=self.dataset_cfg.downsample_type,
        )
        points_action = points_action.squeeze(0)
        points_anchor = points_anchor.squeeze(0)

        # Apply scale factor
        points_action *= self.dataset_cfg.pcd_scale_factor
        points_anchor *= self.dataset_cfg.pcd_scale_factor

        # Transform the point clouds
        # ransform the point clouds
        T0 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.action_rotation_variance,
            trans_var=self.dataset_cfg.action_translation_variance,
            rot_sample_method=self.dataset_cfg.action_transform_type,
        )
        T1 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.anchor_rotation_variance,
            trans_var=self.dataset_cfg.anchor_translation_variance,
            rot_sample_method=self.dataset_cfg.anchor_transform_type,
        )

        goal_points_action = T1.transform_points(points_action)
        goal_points_anchor = T1.transform_points(points_anchor)

        # Get starting action point cloud
        # Transform the action point cloud
        goal_points_action_mean = goal_points_action.mean(dim=0)
        points_action = goal_points_action - goal_points_action_mean
        points_action = T0.transform_points(points_action)

        T_action2goal = T0.inverse().compose(
            Translate(goal_points_action_mean.unsqueeze(0))
        )

        return {
            "pc": goal_points_action,  # Action points in goal position
            "pc_anchor": goal_points_anchor,  # Anchor points in goal position
            "pc_action": points_action,  # Action points for context
            "T0": T0.get_matrix().squeeze(0).T,
            "T1": T1.get_matrix().squeeze(0).T,
            "T_action2goal": T_action2goal.get_matrix().squeeze(0).T,
        }


class RigidFlowDataset(data.Dataset):
    def __init__(
        self,
        root: Path,
        type: str = "train",
        dataset_cfg: RigidDatasetCfg = RigidDatasetCfg(),
    ):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg

        self.dataset_dir = self.root / self.type
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        self.demo_files = list(self.dataset_dir.glob("*_teleport_obj_points.npz"))
        if self.dataset_cfg.num_demos is not None and self.type == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.type} demos from {self.dataset_dir}")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_dataset_size
        elif self.type == "val":
            return self.dataset_cfg.val_dataset_size
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos])

        # Extract point clouds
        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        # TODO: Remove this hardcoded class selection
        points_action = points_raw[classes_raw == 0]
        points_anchor = points_raw[classes_raw == 1]

        # Center the point clouds
        if self.dataset_cfg.center_type == "action_center":
            center = points_action.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_center":
            center = points_anchor.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_random":
            center = points_anchor[np.random.choice(len(points_anchor))]
        elif self.dataset_cfg.center_type == "none":
            center = np.zeros(3)
        else:
            raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
        goal_points_action = points_action - center
        goal_points_anchor = points_anchor - center

        goal_points_action = torch.as_tensor(goal_points_action).float()
        goal_points_anchor = torch.as_tensor(goal_points_anchor).float()

        # Downsample the point clouds
        goal_points_action, _ = downsample_pcd(
            goal_points_action.unsqueeze(0),
            num_points=self.dataset_cfg.sample_size_action,
            type=self.dataset_cfg.downsample_type,
        )
        goal_points_anchor, _ = downsample_pcd(
            goal_points_anchor.unsqueeze(0),
            num_points=self.dataset_cfg.sample_size_anchor,
            type=self.dataset_cfg.downsample_type,
        )
        goal_points_action = goal_points_action.squeeze(0)
        goal_points_anchor = goal_points_anchor.squeeze(0)

        # Apply scale factor
        goal_points_action *= self.dataset_cfg.pcd_scale_factor
        goal_points_anchor *= self.dataset_cfg.pcd_scale_factor

        # Transform the point clouds
        T0 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.action_rotation_variance,
            trans_var=self.dataset_cfg.action_translation_variance,
            rot_sample_method=self.dataset_cfg.action_transform_type,
        )
        T1 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.anchor_rotation_variance,
            trans_var=self.dataset_cfg.anchor_translation_variance,
            rot_sample_method=self.dataset_cfg.anchor_transform_type,
        )

        goal_points_action = T1.transform_points(goal_points_action)
        goal_points_anchor = T1.transform_points(goal_points_anchor)

        # Get starting action point cloud
        # Transform the action point cloud
        points_action = goal_points_action.clone() - goal_points_action.mean(dim=0)
        points_action = T0.transform_points(points_action)

        # Center the action point cloud
        points_action = points_action - points_action.mean(dim=0)

        # Calculate goal flow
        flow = goal_points_action - points_action

        return {
            "pc": points_action,  # Action points in starting position
            "pc_anchor": goal_points_anchor,  # Anchor points in goal position
            "pc_action": goal_points_action,  # Action points in goal position
            "flow": flow,
            "T0": T0.get_matrix().squeeze(0).T,
            "T1": T1.get_matrix().squeeze(0).T,
        }


class NDFDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split
        self.dataset_cfg = dataset_cfg

        # loading demo files
        dir_type = self.split if self.split == "train" else "test"
        self.dataset_dir = self.root / f"{dir_type}_data/renders"
        print(f"Loading NDF dataset from {self.dataset_dir}")
        self.demo_files = list(self.dataset_dir.glob("*_teleport_obj_points.npz"))  # try *_afterteleport_obj_points.npz
        self.num_demos = len(self.demo_files)
        if self.dataset_cfg.num_demos is not None and self.split == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.split} demos from {self.dataset_dir}")

        # setting dataset size
        if self.split == "train":
            self.size = self.dataset_cfg.train_dataset_size
        elif self.split == "val":
            self.size = self.dataset_cfg.val_dataset_size
        else:
            self.size = self.num_demos

        # setting sample sizes
        self.scene = self.dataset_cfg.scene
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor
        self.world_frame = self.dataset_cfg.world_frame

    def __len__(self):
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos])

        # Extract point clouds
        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        # TODO: Remove this hardcoded class selection
        points_action = points_raw[classes_raw == 0]
        points_anchor = points_raw[classes_raw == 1]

        points_action = torch.as_tensor(points_action).float()
        points_anchor = torch.as_tensor(points_anchor).float()

        # Apply scale factor
        points_action *= self.dataset_cfg.pcd_scale_factor
        points_anchor *= self.dataset_cfg.pcd_scale_factor

        # Add distractor anchor point clouds
        if self.dataset_cfg.distractor_anchor_pcds > 0:
            (
                _,
                points_action,
                points_anchor_base,
                distractor_anchor_pcd_list,
                T_distractor_list,
                debug,
            ) = get_multi_anchor_scene(
                points_gripper=None,
                points_action=points_action.unsqueeze(0),
                points_anchor_base=points_anchor.unsqueeze(0),
                rot_var=self.dataset_cfg.distractor_rotation_variance,
                trans_var=self.dataset_cfg.distractor_translation_variance,
                rot_sample_method=self.dataset_cfg.distractor_transform_type,
                num_anchors_to_add=self.dataset_cfg.distractor_anchor_pcds,
            )
            points_anchor = torch.cat(
                [points_anchor_base] + distractor_anchor_pcd_list, dim=1
            ).squeeze(0)
            points_action = points_action.squeeze(0)

        action_seg = torch.ones(points_action.shape[0], dtype=torch.bool)
        anchor_seg = torch.ones(points_anchor.shape[0], dtype=torch.bool)


        # Apply augmentations
        if self.split == "train" or (
            self.split == "val" and not self.dataset_cfg.val_use_defaults
        ):
            # Apply augmentations to the point clouds in their final positions
            points_action, points_action_indices = maybe_apply_augmentations(
                points_action,
                min_num_points=self.dataset_cfg.sample_size_action,
                ball_occlusion_param={
                    "ball_occlusion": self.dataset_cfg.action_ball_occlusion,
                    "ball_radius": self.dataset_cfg.action_ball_radius
                    * self.dataset_cfg.pcd_scale_factor,
                },
                plane_occlusion_param={
                    "plane_occlusion": self.dataset_cfg.action_plane_occlusion,
                    "plane_standoff": self.dataset_cfg.action_plane_standoff
                    * self.dataset_cfg.pcd_scale_factor,
                },
            )
            action_seg = action_seg[points_action_indices.squeeze(0)]

            points_anchor, points_anchor_indices = maybe_apply_augmentations(
                points_anchor,
                min_num_points=self.dataset_cfg.sample_size_anchor,
                ball_occlusion_param={
                    "ball_occlusion": self.dataset_cfg.anchor_ball_occlusion,
                    "ball_radius": self.dataset_cfg.anchor_ball_radius
                    * self.dataset_cfg.pcd_scale_factor,
                },
                plane_occlusion_param={
                    "plane_occlusion": self.dataset_cfg.anchor_plane_occlusion,
                    "plane_standoff": self.dataset_cfg.anchor_plane_standoff
                    * self.dataset_cfg.pcd_scale_factor,
                },
            )
            anchor_seg = anchor_seg[points_anchor_indices.squeeze(0)]

        # Set downsample types
        if self.split == "val" and self.dataset_cfg.val_use_defaults:
            downsample_type = "fps"
        else:
            downsample_type = self.dataset_cfg.downsample_type

        # downsample action
        if self.sample_size_action > 0 and points_action.shape[0] > self.sample_size_action:
            points_action, points_action_indices = downsample_pcd(points_action.unsqueeze(0), self.sample_size_action, type=downsample_type)
            points_action = points_action.squeeze(0)
            action_seg = action_seg[points_action_indices.squeeze(0)]

        # downsample anchor
        if self.sample_size_anchor > 0 and points_anchor.shape[0] > self.sample_size_anchor:
            points_anchor, points_anchor_indices = downsample_pcd(points_anchor.unsqueeze(0), self.sample_size_anchor, type=downsample_type)
            points_anchor = points_anchor.squeeze(0)
            anchor_seg = anchor_seg[points_anchor_indices.squeeze(0)]

        ##################### NEW FROM HERE #####################
        # Get initial transforms for the action point cloud
        # TODO: to collision check to avoid action_pc interscect with anchor_pc
        T_init = random_se3(
            N=1,
            rot_var=self.dataset_cfg.init_action_rotation_variance,
            trans_var=self.dataset_cfg.init_action_translation_variance,
            rot_sample_method=self.dataset_cfg.init_action_transform_type,
        )
        init_center = points_action.mean(dim=0)
        points_action_centered = points_action - init_center

        action_pc = T_init.transform_points(points_action_centered) + init_center
        goal_action_pc = points_action
        anchor_pc = points_anchor

        # Extract rotation and translation from T_init
        T_init_matrix = T_init.get_matrix().squeeze(0)

        R_init = T_init_matrix[:3, :3]  # Rotation matrix
        t_init = T_init_matrix[3, :3]   # Translation vector

        # Compute the rotation matrix for action_to_goal (R = R_init.T)
        R_action_to_goal = R_init.T

        # Compute the translation vector for action_to_goal (t = -R.T @ t_init)
        t_action_to_goal = -torch.matmul(R_action_to_goal, t_init)

        # scene-level dataset
        if self.scene:
            raise NotImplementedError('scene-level is not implemented yet!')
        
        # object-centric dataset
        else:
            # center the point clouds
            if self.dataset_cfg.center_type == "action_center":
                center = action_pc.mean(axis=0)
            elif self.dataset_cfg.center_type == "anchor_center":
                center = anchor_pc.mean(axis=0)
            elif self.dataset_cfg.center_type == "anchor_random":
                center = anchor_pc[np.random.choice(len(anchor_pc))]
            elif self.dataset_cfg.center_type == "none":
                center = torch.zeros(3, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
            
            if self.dataset_cfg.action_context_center_type == "center":
                action_center = action_pc.mean(axis=0)
            elif self.dataset_cfg.action_context_center_type == "random":
                action_center = action_pc[np.random.choice(len(action_pc))]
            elif self.dataset_cfg.action_context_center_type == "none":
                action_center = torch.zeros(3, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown action context center type: {self.dataset_cfg.action_context_center_type}")
            
            goal_action_pc = goal_action_pc - center
            anchor_pc = anchor_pc - center
            action_pc = action_pc - action_center

            # transform the point clouds
            T0 = random_se3(
                N=1,
                rot_var=self.dataset_cfg.action_rotation_variance,
                trans_var=self.dataset_cfg.action_translation_variance,
                rot_sample_method=self.dataset_cfg.action_transform_type,
            )
            T1 = random_se3(
                N=1,
                rot_var=self.dataset_cfg.anchor_rotation_variance,
                trans_var=self.dataset_cfg.anchor_translation_variance,
                rot_sample_method=self.dataset_cfg.anchor_transform_type,
            )

            goal_action_pc = T1.transform_points(goal_action_pc)
            anchor_pc = T1.transform_points(anchor_pc)
            action_pc = T0.transform_points(action_pc)

            T_goal2world = T1.inverse().compose(
                Translate(center.unsqueeze(0))
            )
            T_action2world = T0.inverse().compose(
                Translate(action_center.unsqueeze(0))
            )

            gt_flow = goal_action_pc - action_pc

            item = {}
            item["pc"] = goal_action_pc # Action points in goal position
            item["pc_action"] = action_pc # Action points in starting position for context
            item["pc_anchor"] = anchor_pc # Anchor points in goal position
            item["seg"] = action_seg
            item["seg_anchor"] = anchor_seg
            item["flow"] = gt_flow
            item["T_goal2world"] = T_goal2world.get_matrix().squeeze(0)
            item["T_action2world"] = T_action2world.get_matrix().squeeze(0)
            item["T_init"] = T_init.get_matrix().squeeze(0)
            item["R_action_to_goal"] = R_action_to_goal
            item["t_action_to_goal"] = t_action_to_goal
        return item



class NDFFeatureDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split
        self.dataset_cfg = dataset_cfg

        # loading demo files
        dir_type = self.split if self.split == "train" else "test"
        self.dataset_dir = self.root / f"{dir_type}_data/renders"
        print(f"Loading NDF dataset from {self.dataset_dir}")
        self.demo_files = list(self.dataset_dir.glob("*_teleport_obj_points.npz"))  # try *_afterteleport_obj_points.npz
        self.num_demos = len(self.demo_files)
        if self.dataset_cfg.num_demos is not None and self.split == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.split} demos from {self.dataset_dir}")

        # setting dataset size
        if self.split == "train":
            self.size = self.dataset_cfg.train_dataset_size
        elif self.split == "val":
            self.size = self.dataset_cfg.val_dataset_size
        else:
            self.size = self.num_demos

        # setting sample sizes
        self.scene = self.dataset_cfg.scene
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor
        self.world_frame = self.dataset_cfg.world_frame

    def __len__(self):
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos])

        # Extract point clouds
        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        # TODO: Remove this hardcoded class selection
        points_action = points_raw[classes_raw == 0]
        points_anchor = points_raw[classes_raw == 1]

        points_action = torch.as_tensor(points_action).float()
        points_anchor = torch.as_tensor(points_anchor).float()

        # Apply scale factor
        points_action *= self.dataset_cfg.pcd_scale_factor
        points_anchor *= self.dataset_cfg.pcd_scale_factor

        # Add distractor anchor point clouds
        if self.dataset_cfg.distractor_anchor_pcds > 0:
            (
                _,
                points_action,
                points_anchor_base,
                distractor_anchor_pcd_list,
                T_distractor_list,
                debug,
            ) = get_multi_anchor_scene(
                points_gripper=None,
                points_action=points_action.unsqueeze(0),
                points_anchor_base=points_anchor.unsqueeze(0),
                rot_var=self.dataset_cfg.distractor_rotation_variance,
                trans_var=self.dataset_cfg.distractor_translation_variance,
                rot_sample_method=self.dataset_cfg.distractor_transform_type,
                num_anchors_to_add=self.dataset_cfg.distractor_anchor_pcds,
            )
            points_anchor = torch.cat(
                [points_anchor_base] + distractor_anchor_pcd_list, dim=1
            ).squeeze(0)
            points_action = points_action.squeeze(0)

        action_seg = torch.ones(points_action.shape[0], dtype=torch.bool)
        anchor_seg = torch.ones(points_anchor.shape[0], dtype=torch.bool)


        # Apply augmentations
        if self.split == "train" or (
            self.split == "val" and not self.dataset_cfg.val_use_defaults
        ):
            # Apply augmentations to the point clouds in their final positions
            points_action, points_action_indices = maybe_apply_augmentations(
                points_action,
                min_num_points=self.dataset_cfg.sample_size_action,
                ball_occlusion_param={
                    "ball_occlusion": self.dataset_cfg.action_ball_occlusion,
                    "ball_radius": self.dataset_cfg.action_ball_radius
                    * self.dataset_cfg.pcd_scale_factor,
                },
                plane_occlusion_param={
                    "plane_occlusion": self.dataset_cfg.action_plane_occlusion,
                    "plane_standoff": self.dataset_cfg.action_plane_standoff
                    * self.dataset_cfg.pcd_scale_factor,
                },
            )
            action_seg = action_seg[points_action_indices.squeeze(0)]

            points_anchor, points_anchor_indices = maybe_apply_augmentations(
                points_anchor,
                min_num_points=self.dataset_cfg.sample_size_anchor,
                ball_occlusion_param={
                    "ball_occlusion": self.dataset_cfg.anchor_ball_occlusion,
                    "ball_radius": self.dataset_cfg.anchor_ball_radius
                    * self.dataset_cfg.pcd_scale_factor,
                },
                plane_occlusion_param={
                    "plane_occlusion": self.dataset_cfg.anchor_plane_occlusion,
                    "plane_standoff": self.dataset_cfg.anchor_plane_standoff
                    * self.dataset_cfg.pcd_scale_factor,
                },
            )
            anchor_seg = anchor_seg[points_anchor_indices.squeeze(0)]

        # Set downsample types
        if self.split == "val" and self.dataset_cfg.val_use_defaults:
            downsample_type = "fps"
        else:
            downsample_type = self.dataset_cfg.downsample_type

        # downsample action
        if self.sample_size_action > 0 and points_action.shape[0] > self.sample_size_action:
            points_action, points_action_indices = downsample_pcd(points_action.unsqueeze(0), self.sample_size_action, type=downsample_type)
            points_action = points_action.squeeze(0)
            action_seg = action_seg[points_action_indices.squeeze(0)]

        # downsample anchor
        if self.sample_size_anchor > 0 and points_anchor.shape[0] > self.sample_size_anchor:
            points_anchor, points_anchor_indices = downsample_pcd(points_anchor.unsqueeze(0), self.sample_size_anchor, type=downsample_type)
            points_anchor = points_anchor.squeeze(0)
            anchor_seg = anchor_seg[points_anchor_indices.squeeze(0)]

        ##################### NEW FROM HERE #####################
        # Get initial transforms for the action point cloud
        # TODO: to collision check to avoid action_pc interscect with anchor_pc
        T_init = random_se3(
            N=1,
            rot_var=self.dataset_cfg.init_action_rotation_variance,
            trans_var=self.dataset_cfg.init_action_translation_variance,
            rot_sample_method=self.dataset_cfg.init_action_transform_type,
        )
        init_center = points_action.mean(dim=0)
        points_action_centered = points_action - init_center

        action_pc = T_init.transform_points(points_action_centered) + init_center
        goal_action_pc = points_action
        anchor_pc = points_anchor

        # Extract rotation and translation from T_init
        T_init_matrix = T_init.get_matrix().squeeze(0)

        R_init = T_init_matrix[:3, :3]  # Rotation matrix
        t_init = T_init_matrix[3, :3]   # Translation vector

        # Compute the rotation matrix for action_to_goal (R = R_init.T)
        R_action_to_goal = R_init.T

        # Compute the translation vector for action_to_goal (t = -R.T @ t_init)
        t_action_to_goal = -torch.matmul(R_action_to_goal, t_init)

        ### Common Terms ###
        P_A = action_pc
        P_B = anchor_pc
        P_A_star = goal_action_pc

        # calculate reference centers
        y = P_B.mean(axis=0)
        y_action = P_A.mean(axis=0)

        # center anchor and action
        P_B_prime = P_B - y                 # P_B'
        P_A_prime = P_A - y_action          # P_A'
        P_A_star_prime = P_A_star - y       # P_A*'

        # generate random SE3 transform
        T0 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.action_rotation_variance,
            trans_var=self.dataset_cfg.action_translation_variance,
            rot_sample_method=self.dataset_cfg.action_transform_type,
        )
        T1 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.anchor_rotation_variance,
            trans_var=self.dataset_cfg.anchor_translation_variance,
            rot_sample_method=self.dataset_cfg.anchor_transform_type,
        )

        # record coordinate transformation matrices
        T_goal2world = T1.inverse().compose(
            Translate(y.unsqueeze(0))
        )
        T_action2world = T0.inverse().compose(
            Translate(y_action.unsqueeze(0))
        )
        
        # perform random SE3 transform 
        P_A_star_prime_ = T1.transform_points(P_A_star_prime)
        P_B_prime_ = T1.transform_points(P_B_prime)
        P_A_prime_ = T0.transform_points(P_A_prime)

        P_A_star_ = T1.transform_points(P_A_star)
        P_B_ = T1.transform_points(P_B)
        P_A_ = T0.transform_points(P_A)

        y_ = P_B_.mean(axis=0)
        y_action_ = P_A_.mean(axis=0) 
        
        # create one-hot encoding
        P_A_one_hot = torch.tensor([[1, 0]], dtype=torch.float).repeat(P_A_.shape[0], 1)  # Shape: [512, 2]
        P_B_one_hot = torch.tensor([[0, 1]], dtype=torch.float).repeat(P_B_.shape[0], 1)  # Shape: [512, 2]

        gt_flow = P_A_star_prime_ - P_A_prime_
        item = {}
        item["pc"] = P_A_star_prime_ # Action points in goal position
        item["pc_action"] = P_A_prime_ # Action points in starting position for context
        item["pc_anchor"] = P_B_prime_ # Anchor points in goal position
        item["seg"] = action_seg
        item["seg_anchor"] = anchor_seg
        item["flow"] = gt_flow
        item["T_goal2world"] = T_goal2world.get_matrix().squeeze(0)
        item["T_action2world"] = T_action2world.get_matrix().squeeze(0)
        item["P_A"] = P_A_
        item["P_B"] = P_B_
        item["P_A_star"] = P_A_star_
        item["y"] = y_
        item["y_action"] = y_action_
        item["P_A_one_hot"] = P_A_one_hot
        item["P_B_one_hot"] = P_B_one_hot
        item["T_init"] = T_init.get_matrix().squeeze(0)
        item["R_action_to_goal"] = R_action_to_goal
        item["t_action_to_goal"] = t_action_to_goal
        return item


class RPDiffPointDataset(data.Dataset):
    def __init__(
        self,
        root: Path,
        type: str = "train",
        dataset_cfg: RigidDatasetCfg = RigidDatasetCfg(),
    ):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg

        dir_type = self.type if self.type == "train" else "test"
        self.dataset_dir = self.root / f"{dir_type}_data/renders"
        print(f"Loading NDF dataset from {self.dataset_dir}")
        self.demo_files = list(self.dataset_dir.glob("*_teleport_obj_points.npz"))
        self.num_demos = len(self.demo_files)
        if self.dataset_cfg.num_demos is not None and self.type == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.type} demos from {self.dataset_dir}")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_dataset_size
        elif self.type == "val":
            return self.dataset_cfg.val_dataset_size
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        demo = np.load(self.demo_files[index % self.num_demos], allow_pickle=True)

        # Extract point clouds
        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        # TODO: Remove this hardcoded class selection
        points_action_raw = points_raw[classes_raw == 0]
        points_anchor_raw = points_raw[classes_raw == 1]

        points_action_raw = torch.as_tensor(points_action_raw).float()
        points_anchor_raw = torch.as_tensor(points_anchor_raw).float()

        # Apply scale factor
        points_action_scaled = points_action_raw * self.dataset_cfg.pcd_scale_factor
        points_anchor_scaled = points_anchor_raw * self.dataset_cfg.pcd_scale_factor

        # Add distractor anchor point clouds
        if self.dataset_cfg.distractor_anchor_pcds > 0:
            (
                _,
                points_action_scaled,
                points_anchor_scaled_base,
                distractor_anchor_pcd_list,
                T_distractor_list,
                debug,
            ) = get_multi_anchor_scene(
                points_gripper=None,
                points_action=points_action_scaled.unsqueeze(0),
                points_anchor_base=points_anchor_scaled.unsqueeze(0),
                rot_var=self.dataset_cfg.distractor_rotation_variance,
                trans_var=self.dataset_cfg.distractor_translation_variance,
                rot_sample_method=self.dataset_cfg.distractor_transform_type,
                num_anchors_to_add=self.dataset_cfg.distractor_anchor_pcds,
            )
            points_anchor_scaled = torch.cat(
                [points_anchor_scaled_base] + distractor_anchor_pcd_list, dim=1
            ).squeeze(0)
            points_action_scaled = points_action_scaled.squeeze(0)

        # Center the point clouds
        if self.dataset_cfg.center_type == "action_center":
            center = points_action_scaled.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_center":
            center = points_anchor_scaled.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_random":
            center = points_anchor_scaled[np.random.choice(len(points_anchor_scaled))]
        elif self.dataset_cfg.center_type == "none":
            center = np.zeros(3)
        else:
            raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
        points_action = points_action_scaled - center
        points_anchor = points_anchor_scaled - center

        if self.type == "train" or (
            self.type == "val" and not self.dataset_cfg.val_use_defaults
        ):
            # Apply augmentations to the point clouds in their final positions
            points_action = maybe_apply_augmentations(
                points_action,
                min_num_points=self.dataset_cfg.sample_size_action,
                ball_occlusion_param={
                    "ball_occlusion": self.dataset_cfg.action_ball_occlusion,
                    "ball_radius": self.dataset_cfg.action_ball_radius
                    * self.dataset_cfg.pcd_scale_factor,
                },
                plane_occlusion_param={
                    "plane_occlusion": self.dataset_cfg.action_plane_occlusion,
                    "plane_standoff": self.dataset_cfg.action_plane_standoff
                    * self.dataset_cfg.pcd_scale_factor,
                },
            )
            points_anchor = maybe_apply_augmentations(
                points_anchor,
                min_num_points=self.dataset_cfg.sample_size_anchor,
                ball_occlusion_param={
                    "ball_occlusion": self.dataset_cfg.anchor_ball_occlusion,
                    "ball_radius": self.dataset_cfg.anchor_ball_radius
                    * self.dataset_cfg.pcd_scale_factor,
                },
                plane_occlusion_param={
                    "plane_occlusion": self.dataset_cfg.anchor_plane_occlusion,
                    "plane_standoff": self.dataset_cfg.anchor_plane_standoff
                    * self.dataset_cfg.pcd_scale_factor,
                },
            )

        if self.type == "val" and self.dataset_cfg.val_use_defaults:
            # Downsample the point clouds
            points_action, _ = downsample_pcd(
                points_action.unsqueeze(0),
                num_points=self.dataset_cfg.sample_size_action,
                type="fps",
            )
            points_anchor, _ = downsample_pcd(
                points_anchor.unsqueeze(0),
                num_points=self.dataset_cfg.sample_size_anchor,
                type="fps",
            )
        else:
            # Downsample the point clouds
            points_action, _ = downsample_pcd(
                points_action.unsqueeze(0),
                num_points=self.dataset_cfg.sample_size_action,
                type=self.dataset_cfg.downsample_type,
            )
            points_anchor, _ = downsample_pcd(
                points_anchor.unsqueeze(0),
                num_points=self.dataset_cfg.sample_size_anchor,
                type=self.dataset_cfg.downsample_type,
            )
        points_action = points_action.squeeze(0)
        points_anchor = points_anchor.squeeze(0)

        # Get transforms for the action and anchor point clouds
        T0 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.action_rotation_variance,
            trans_var=self.dataset_cfg.action_translation_variance,
            rot_sample_method=self.dataset_cfg.action_transform_type,
        )
        T1 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.anchor_rotation_variance,
            trans_var=self.dataset_cfg.anchor_translation_variance,
            rot_sample_method=self.dataset_cfg.anchor_transform_type,
        )

        # Get the goal action and anchor point clouds
        goal_points_action = T1.transform_points(points_action)
        goal_points_anchor = T1.transform_points(points_anchor)

        # Get the 'context' action point cloud
        goal_points_action_mean = goal_points_action.mean(dim=0)
        points_action = goal_points_action - goal_points_action_mean
        points_action = T0.transform_points(points_action)

        # Get transforms for metrics calculation and visualizations
        T_action2goal = T0.inverse().compose(
            Translate(goal_points_action_mean.unsqueeze(0))
        )

        data = {
            "pc": goal_points_action,  # Action points in goal position
            "pc_anchor": goal_points_anchor,  # Anchor points in goal position
            "pc_action": points_action,  # Action points for context
            "T0": T0.get_matrix().squeeze(0).T,
            "T1": T1.get_matrix().squeeze(0).T,
            "T_action2goal": T_action2goal.get_matrix().squeeze(0).T,
        }

        # If we have distractor anchor point clouds, add their transforms
        if self.dataset_cfg.distractor_anchor_pcds > 0:
            T_aug_action2goal_list = []
            for T_distractor in T_distractor_list:
                T_aug_action2goal = T_action2goal.compose(
                    T1.inverse()
                    .compose(Translate(center.unsqueeze(0)))
                    .compose(T_distractor)
                    .compose(Translate(-center.unsqueeze(0)))
                    .compose(T1)
                )
                T_aug_action2goal_list.append(T_aug_action2goal)

            data["T_distractor_list"] = torch.stack(
                [T.get_matrix().squeeze(0).T for T in T_distractor_list]
            )
            data["T_action2distractor_list"] = torch.stack(
                [T.get_matrix().squeeze(0).T for T in T_aug_action2goal_list]
            )

        # Support for RPDiff precision WTA error
        if "multi_obj_mesh_file" in demo and "multi_obj_final_obj_pose" in demo:
            assert (
                "rpdiff_descriptions_path" in self.dataset_cfg.misc_kwargs
            ), "rpdiff_descriptions_path must be provided in the dataset config"
            data["goal_action_center"] = goal_points_action_mean
            data["scene_center"] = center

            parent_fname = demo["multi_obj_mesh_file"].item()["parent"][0]
            parent_obj_name = (
                parent_fname.split("/")[-1].replace(".obj", "").replace("_dec", "")
            )
            parent_full_fname = os.path.join(
                self.dataset_cfg.misc_kwargs["rpdiff_descriptions_path"],
                parent_fname.split(parent_obj_name)[0].split("descriptions/")[-1],
            )
            saved_poses_path = os.path.join(
                parent_full_fname,
                "open_slot_poses",
                parent_obj_name + "_open_slot_poses.txt",
            )

            data["rpdiff_obj_mesh_file"] = demo["multi_obj_mesh_file"].item()
            data["rpdiff_saved_poses_path"] = saved_poses_path
            data["rpdiff_obj_final_obj_pose"] = demo["multi_obj_final_obj_pose"].item()
            data["rpdiff_pcd_scale_factor"] = self.dataset_cfg.pcd_scale_factor

        return data

'''
DATASET_FN = {
    "rigid_point": RigidPointDataset,
    "rigid_flow": RigidFlowDataset,
    "ndf_point": NDFPointDataset,
    "rpdiff_point": RPDiffPointDataset,
}
'''
DATASET_FN = {
    "ndf": NDFDataset,
    #"rpdiff": RpdiffDataset,
}

F_DATASET_FN = {
    "ndf": NDFFeatureDataset,
    #"rpdiff": RpdiffDataset,
}

class RigidDataModule(L.LightningModule):
    def __init__(
        self,
        #root: Path,
        batch_size: int,
        val_batch_size: int,
        num_workers: int,
        dataset_cfg: omegaconf.DictConfig = None,
    ):
        super().__init__()
        #self.root = root
        data_dir = os.path.expanduser(dataset_cfg.data_dir)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dataset_cfg = dataset_cfg
        self.root = Path(data_dir)
    
    def prepare_data(self):
        pass

    def setup(self, stage: str) -> None:
        self.stage = stage

        # dataset sanity checks
        if self.dataset_cfg.scene and not self.dataset_cfg.world_frame:
            raise ValueError("Scene inputs require a world frame.")

        # if not in train mode, don't use rotation augmentations
        # TODO: Think about what to do here, since setting all these to 0 will cause action_pc == goal_action_pc !!!
        if self.stage != "fit":
            print("-------Turning off rotation augmentation for validation/inference.-------")
            self.dataset_cfg.action_transform_type = "identity"
            self.dataset_cfg.anchor_transform_type = "identity"
            self.dataset_cfg.action_translation_variance = 0
            self.dataset_cfg.action_rotation_variance = 0
            self.dataset_cfg.anchor_translation_variance = 0
            self.dataset_cfg.anchor_rotation_variance = 0

        # if world frame, don't mean-center the point clouds
        if self.dataset_cfg.world_frame:
            print("-------Turning off mean-centering for world frame predictions.-------")
            self.dataset_cfg.center_type = "none"
            self.dataset_cfg.action_context_center_type = "none"


        self.train_dataset = DATASET_FN[self.dataset_cfg.name](
            self.root, self.dataset_cfg, "train")
        
        self.val_dataset = DATASET_FN[self.dataset_cfg.name](
            self.root, self.dataset_cfg, "val")

        self.test_dataset = DATASET_FN[self.dataset_cfg.name](
            self.root, self.dataset_cfg, "test")

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True if self.stage == "fit" else False,
            drop_last=True,
        )

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

class RigidFeatureDataModule(L.LightningModule):
    def __init__(
        self,
        #root: Path,
        batch_size: int,
        val_batch_size: int,
        num_workers: int,
        dataset_cfg: omegaconf.DictConfig = None,
    ):
        super().__init__()
        #self.root = root
        data_dir = os.path.expanduser(dataset_cfg.data_dir)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dataset_cfg = dataset_cfg
        self.root = Path(data_dir)
    
    def prepare_data(self):
        pass

    def setup(self, stage: str) -> None:
        self.stage = stage

        # dataset sanity checks
        if self.dataset_cfg.scene and not self.dataset_cfg.world_frame:
            raise ValueError("Scene inputs require a world frame.")

        # if not in train mode, don't use rotation augmentations
        # TODO: Think about what to do here, since setting all these to 0 will cause action_pc == goal_action_pc !!!
        if self.stage != "fit":
            print("-------Turning off rotation augmentation for validation/inference.-------")
            self.dataset_cfg.action_transform_type = "identity"
            self.dataset_cfg.anchor_transform_type = "identity"
            self.dataset_cfg.action_translation_variance = 0
            self.dataset_cfg.action_rotation_variance = 0
            self.dataset_cfg.anchor_translation_variance = 0
            self.dataset_cfg.anchor_rotation_variance = 0

        # if world frame, don't mean-center the point clouds
        if self.dataset_cfg.world_frame:
            print("-------Turning off mean-centering for world frame predictions.-------")
            self.dataset_cfg.center_type = "none"
            self.dataset_cfg.action_context_center_type = "none"


        self.train_dataset = F_DATASET_FN[self.dataset_cfg.name](
            self.root, self.dataset_cfg, "train")
        
        self.val_dataset = F_DATASET_FN[self.dataset_cfg.name](
            self.root, self.dataset_cfg, "val")

        self.test_dataset = F_DATASET_FN[self.dataset_cfg.name](
            self.root, self.dataset_cfg, "test")

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True if self.stage == "fit" else False,
            drop_last=True,
        )

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
