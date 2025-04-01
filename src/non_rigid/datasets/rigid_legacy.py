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

        self.eval_mode = False if self.split == "train" else True

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
        if self.type == "train" or self.dataset_cfg.val_use_defaults:
            if not self.eval_mode:
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

    def set_eval_mode(self, eval_mode: bool):
        """ Toggle eval mode to enable/disable augmentation """
        self.eval_mode = eval_mode

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

        self.eval_mode = False if self.split == "train" else True

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
        if self.type == "train" or self.dataset_cfg.val_use_defaults:
            if not self.eval_mode:
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

    def set_eval_mode(self, eval_mode: bool):
        """ Toggle eval mode to enable/disable augmentation """
        self.eval_mode = eval_mode


class RPDiffDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, type):
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg
        self.rpdiff_task_name = dataset_cfg.rpdiff_task_name
        self.rpdiff_task_type = dataset_cfg.rpdiff_task_type
                
        self.eval_mode = False if self.type == "train" else True

        self.dataset_dir = self.root / self.rpdiff_task_name / self.rpdiff_task_type
        self.split_dir = self.dataset_dir / "split_info"
        self.split_file = f"{self.type}_split.txt" if self.type != "val" else "train_val_split.txt"

        # setting sample sizes
        self.scene = self.dataset_cfg.scene
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor
        self.world_frame = self.dataset_cfg.world_frame

        print(f"Loading RPDiff dataset from {self.dataset_dir}")

        with open(os.path.join(self.split_dir, self.split_file), "r") as file:
            self.demo_files = [f"{self.dataset_dir}/{line.strip()}" for line in file]

        self.num_demos = len(self.demo_files)
        if self.dataset_cfg.num_demos is not None and self.type == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.type} demos from {self.dataset_dir} with {self.split_file}, total of {self.__len__()} files")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_dataset_size if self.dataset_cfg.train_dataset_size is not None else len(self.demo_files)
        elif self.type == "val":
            return self.dataset_cfg.val_dataset_size if self.dataset_cfg.val_dataset_size is not None else len(self.demo_files)
        elif self.type == "test":
            return self.dataset_cfg.test_dataset_size if self.dataset_cfg.test_dataset_size is not None else len(self.demo_files)
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos], allow_pickle=True)
        
        # Access start and final PCDs for parent and child
        parent_start_pcd = demo['multi_obj_start_pcd'].item()['parent']
        child_start_pcd = demo['multi_obj_start_pcd'].item()['child']
        parent_final_pcd = demo['multi_obj_final_pcd'].item()['parent']
        child_final_pcd = demo['multi_obj_final_pcd'].item()['child']
        relative_trans = demo['relative_trans']

        action_pc = torch.as_tensor(child_start_pcd).float()
        anchor_pc = torch.as_tensor(parent_start_pcd).float()
        goal_action_pc = torch.as_tensor(child_final_pcd).float()
        goal_anchor_pc = torch.as_tensor(parent_final_pcd).float()  # same as anchor_pc
        relative_trans = torch.as_tensor(relative_trans).float()

        # TODO: apply scale factor to adjust relative_trans aw
        action_pc *= self.dataset_cfg.pcd_scale_factor
        anchor_pc *= self.dataset_cfg.pcd_scale_factor
        goal_action_pc *= self.dataset_cfg.pcd_scale_factor

        action_seg = torch.ones(action_pc.shape[0], dtype=torch.bool)
        anchor_seg = torch.ones(anchor_pc.shape[0], dtype=torch.bool)
        
        # Apply augmentations
        if self.type == "train" or self.dataset_cfg.val_use_defaults:
            if not self.eval_mode:
                # Apply augmentations to the point clouds in their final positions
                action_pc, action_pc_indices = maybe_apply_augmentations(
                    action_pc,
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
                action_seg = action_seg[action_pc_indices.squeeze(0)]
                goal_action_pc = goal_action_pc[action_pc_indices.squeeze(0)]
                if anchor_pc.shape[0] < self.dataset_cfg.sample_size_anchor:
                    print("Encounter action points of shape {} with less than min_num_points. No augmentations!".format(anchor_pc.shape[0]))
                    print(self.demo_files[index % self.num_demos])

                anchor_pc, anchor_pc_indices = maybe_apply_augmentations(
                    anchor_pc,
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

                if anchor_pc.shape[0] < self.dataset_cfg.sample_size_anchor:
                    print("Encounter anchor points of shape {} with less than min_num_points. No augmentations!".format(anchor_pc.shape[0]))
                    print(self.demo_files[index % self.num_demos])

                anchor_seg = anchor_seg[anchor_pc_indices.squeeze(0)]

        # Set downsample types
        if self.type == "val" and self.dataset_cfg.val_use_defaults:
            downsample_type = "fps"
        else:
            downsample_type = self.dataset_cfg.downsample_type

        # downsample action
        if self.sample_size_action > 0 and action_pc.shape[0] > self.sample_size_action:
            action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), self.sample_size_action, type=downsample_type)
            action_pc = action_pc.squeeze(0)
            action_seg = action_seg[action_pc_indices.squeeze(0)]
            goal_action_pc = goal_action_pc[action_pc_indices.squeeze(0)]

        # downsample anchor
        if self.sample_size_anchor > 0 and anchor_pc.shape[0] > self.sample_size_anchor:
            anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), self.sample_size_anchor, type=downsample_type)
            anchor_pc = anchor_pc.squeeze(0)
            anchor_seg = anchor_seg[anchor_pc_indices.squeeze(0)]


        # Extract rotation and translation from relative_trans
        T_init = relative_trans
        R_action_to_goal = relative_trans[:3, :3]  # Rotation matrix
        t_action_to_goal = relative_trans[:3, 3]   # Translation vector

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
            item["T_init"] = T_init
            item["R_action_to_goal"] = R_action_to_goal
            item["t_action_to_goal"] = t_action_to_goal
        return item

    def set_eval_mode(self, eval_mode: bool):
        """ Toggle eval mode to enable/disable augmentation """
        self.eval_mode = eval_mode

class RPDiffFeatureDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, type):
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg
        self.rpdiff_task_name = dataset_cfg.rpdiff_task_name
        self.rpdiff_task_type = dataset_cfg.rpdiff_task_type

        self.eval_mode = False if self.type == "train" else True

        self.dataset_dir = self.root / self.rpdiff_task_name / self.rpdiff_task_type
        self.split_dir = self.dataset_dir / "split_info"
        self.split_file = f"{self.type}_split.txt" if self.type != "val" else "train_val_split.txt"

        # setting sample sizes
        self.scene = self.dataset_cfg.scene
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor
        self.world_frame = self.dataset_cfg.world_frame

        print(f"Loading RPDiff dataset from {self.dataset_dir}")

        with open(os.path.join(self.split_dir, self.split_file), "r") as file:
            self.demo_files = [f"{self.dataset_dir}/{line.strip()}" for line in file]

        self.num_demos = len(self.demo_files)
        if self.dataset_cfg.num_demos is not None and self.type == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.type} demos from {self.dataset_dir} with {self.split_file}, total of {self.__len__()} files")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_dataset_size if self.dataset_cfg.train_dataset_size is not None else len(self.demo_files)
        elif self.type == "val":
            return self.dataset_cfg.val_dataset_size if self.dataset_cfg.val_dataset_size is not None else len(self.demo_files)
        elif self.type == "test":
            return self.dataset_cfg.test_dataset_size if self.dataset_cfg.test_dataset_size is not None else len(self.demo_files)
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos], allow_pickle=True)

        # Access start and final PCDs for parent and child
        parent_start_pcd = demo['multi_obj_start_pcd'].item()['parent']
        child_start_pcd = demo['multi_obj_start_pcd'].item()['child']
        parent_final_pcd = demo['multi_obj_final_pcd'].item()['parent']
        child_final_pcd = demo['multi_obj_final_pcd'].item()['child']
        relative_trans = demo['relative_trans']

        action_pc = torch.as_tensor(child_start_pcd).float()
        anchor_pc = torch.as_tensor(parent_start_pcd).float()
        goal_action_pc = torch.as_tensor(child_final_pcd).float()
        goal_anchor_pc = torch.as_tensor(parent_final_pcd).float()  # same as anchor_pc
        relative_trans = torch.as_tensor(relative_trans).float()

        # TODO: apply scale factor to adjust relative_trans aw
        action_pc *= self.dataset_cfg.pcd_scale_factor
        anchor_pc *= self.dataset_cfg.pcd_scale_factor
        goal_action_pc *= self.dataset_cfg.pcd_scale_factor

        action_seg = torch.ones(action_pc.shape[0], dtype=torch.bool)
        anchor_seg = torch.ones(anchor_pc.shape[0], dtype=torch.bool)

        # Apply augmentations
        if self.type == "train" or self.dataset_cfg.val_use_defaults:
            if not self.eval_mode:
                # Apply augmentations to the point clouds in their final positions
                action_pc, action_pc_indices = maybe_apply_augmentations(
                    action_pc,
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
                action_seg = action_seg[action_pc_indices.squeeze(0)]
                goal_action_pc = goal_action_pc[action_pc_indices.squeeze(0)]

                
                anchor_pc, anchor_pc_indices = maybe_apply_augmentations(
                    anchor_pc,
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

                if anchor_pc.shape[0] < self.dataset_cfg.sample_size_anchor:
                    print("Encounter points of shape {} with less than min_num_points. No augmentations!".format(anchor_pc.shape[0]))
                    print(self.demo_files[index % self.num_demos])

                anchor_seg = anchor_seg[anchor_pc_indices.squeeze(0)]

        # Set downsample types
        if self.type == "val" and self.dataset_cfg.val_use_defaults:
            downsample_type = "fps"
        else:
            downsample_type = self.dataset_cfg.downsample_type

        # downsample action
        if self.sample_size_action > 0 and action_pc.shape[0] > self.sample_size_action:
            action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), self.sample_size_action, type=downsample_type)
            action_pc = action_pc.squeeze(0)
            action_seg = action_seg[action_pc_indices.squeeze(0)]
            goal_action_pc = goal_action_pc[action_pc_indices.squeeze(0)]

        # downsample anchor
        if self.sample_size_anchor > 0 and anchor_pc.shape[0] > self.sample_size_anchor:
            anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), self.sample_size_anchor, type=downsample_type)
            anchor_pc = anchor_pc.squeeze(0)
            anchor_seg = anchor_seg[anchor_pc_indices.squeeze(0)]


        # Extract rotation and translation from relative_trans
        T_init = relative_trans
        R_action_to_goal = relative_trans[:3, :3]  # Rotation matrix
        t_action_to_goal = relative_trans[:3, 3]   # Translation vector

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
        item["T_init"] = T_init
        item["R_action_to_goal"] = R_action_to_goal
        item["t_action_to_goal"] = t_action_to_goal
        return item

    def set_eval_mode(self, eval_mode: bool):
        """ Toggle eval mode to enable/disable augmentation """
        self.eval_mode = eval_mode

class RPDiffNoisyGoalDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, type):
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg
        self.rpdiff_task_name = dataset_cfg.rpdiff_task_name
        self.rpdiff_task_type = dataset_cfg.rpdiff_task_type
                
        self.eval_mode = False if self.type == "train" else True

        self.dataset_dir = self.root / self.rpdiff_task_name / self.rpdiff_task_type
        self.split_dir = self.dataset_dir / "split_info"
        self.split_file = f"{self.type}_split.txt" if self.type != "val" else "train_val_split.txt"

        # setting sample sizes
        self.scene = self.dataset_cfg.scene
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor
        self.world_frame = self.dataset_cfg.world_frame

        # setting gmm error scale
        self.ref_error_scale = self.dataset_cfg.ref_error_scale

        print(f"Loading RPDiff dataset from {self.dataset_dir}")

        with open(os.path.join(self.split_dir, self.split_file), "r") as file:
            self.demo_files = [f"{self.dataset_dir}/{line.strip()}" for line in file]

        self.num_demos = len(self.demo_files)
        if self.dataset_cfg.num_demos is not None and self.type == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.type} demos from {self.dataset_dir} with {self.split_file}, total of {self.__len__()} files")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_dataset_size if self.dataset_cfg.train_dataset_size is not None else len(self.demo_files)
        elif self.type == "val":
            return self.dataset_cfg.val_dataset_size if self.dataset_cfg.val_dataset_size is not None else len(self.demo_files)
        elif self.type == "test":
            return self.dataset_cfg.test_dataset_size if self.dataset_cfg.test_dataset_size is not None else len(self.demo_files)
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos], allow_pickle=True)
        
        # Access start and final PCDs for parent and child
        parent_start_pcd = demo['multi_obj_start_pcd'].item()['parent']
        child_start_pcd = demo['multi_obj_start_pcd'].item()['child']
        parent_final_pcd = demo['multi_obj_final_pcd'].item()['parent']
        child_final_pcd = demo['multi_obj_final_pcd'].item()['child']
        relative_trans = demo['relative_trans']

        action_pc = torch.as_tensor(child_start_pcd).float()
        anchor_pc = torch.as_tensor(parent_start_pcd).float()
        goal_action_pc = torch.as_tensor(child_final_pcd).float()
        goal_anchor_pc = torch.as_tensor(parent_final_pcd).float()  # same as anchor_pc
        relative_trans = torch.as_tensor(relative_trans).float()

        # TODO: apply scale factor to adjust relative_trans aw
        action_pc *= self.dataset_cfg.pcd_scale_factor
        anchor_pc *= self.dataset_cfg.pcd_scale_factor
        goal_action_pc *= self.dataset_cfg.pcd_scale_factor

        action_seg = torch.ones(action_pc.shape[0], dtype=torch.bool)
        anchor_seg = torch.ones(anchor_pc.shape[0], dtype=torch.bool)
        
        # Apply augmentations
        if self.type == "train" or self.dataset_cfg.val_use_defaults:
            if not self.eval_mode:
                # Apply augmentations to the point clouds in their final positions
                action_pc, action_pc_indices = maybe_apply_augmentations(
                    action_pc,
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
                action_seg = action_seg[action_pc_indices.squeeze(0)]
                goal_action_pc = goal_action_pc[action_pc_indices.squeeze(0)]
                if anchor_pc.shape[0] < self.dataset_cfg.sample_size_anchor:
                    print("Encounter action points of shape {} with less than min_num_points. No augmentations!".format(anchor_pc.shape[0]))
                    print(self.demo_files[index % self.num_demos])

                anchor_pc, anchor_pc_indices = maybe_apply_augmentations(
                    anchor_pc,
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

                if anchor_pc.shape[0] < self.dataset_cfg.sample_size_anchor:
                    print("Encounter anchor points of shape {} with less than min_num_points. No augmentations!".format(anchor_pc.shape[0]))
                    print(self.demo_files[index % self.num_demos])

                anchor_seg = anchor_seg[anchor_pc_indices.squeeze(0)]

        # Set downsample types
        if self.type == "val" and self.dataset_cfg.val_use_defaults:
            downsample_type = "fps"
        else:
            downsample_type = self.dataset_cfg.downsample_type

        # downsample action
        if self.sample_size_action > 0 and action_pc.shape[0] > self.sample_size_action:
            action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), self.sample_size_action, type=downsample_type)
            action_pc = action_pc.squeeze(0)
            action_seg = action_seg[action_pc_indices.squeeze(0)]
            goal_action_pc = goal_action_pc[action_pc_indices.squeeze(0)]

        # downsample anchor
        if self.sample_size_anchor > 0 and anchor_pc.shape[0] > self.sample_size_anchor:
            anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), self.sample_size_anchor, type=downsample_type)
            anchor_pc = anchor_pc.squeeze(0)
            anchor_seg = anchor_seg[anchor_pc_indices.squeeze(0)]


        # Extract rotation and translation from relative_trans
        T_init = relative_trans
        R_action_to_goal = relative_trans[:3, :3]  # Rotation matrix
        t_action_to_goal = relative_trans[:3, 3]   # Translation vector

        center = goal_action_pc.mean(axis=0)
        noisy_center = center + self.ref_error_scale * torch.randn_like(center)
        action_center = action_pc.mean(axis=0)

        goal_action_pc = goal_action_pc - noisy_center
        anchor_pc = anchor_pc - noisy_center
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
            Translate(noisy_center.unsqueeze(0))
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
        item["T_init"] = T_init
        item["R_action_to_goal"] = R_action_to_goal
        item["t_action_to_goal"] = t_action_to_goal
        return item

    def set_eval_mode(self, eval_mode: bool):
        """ Toggle eval mode to enable/disable augmentation """
        self.eval_mode = eval_mode

DATASET_FN = {
    "ndf": NDFDataset,
    "ndf_fit": NDFDataset,
    "rpdiff": RPDiffDataset,
    "rpdiff_fit": RPDiffDataset,
}

F_DATASET_FN = {
    "ndf": NDFFeatureDataset,
    "ndf_fit": NDFFeatureDataset,
    "rpdiff": RPDiffFeatureDataset,
    "rpdiff_fit": RPDiffFeatureDataset,
}

NG_DATASET_FN = {
    #"ndf": NDFFeatureDataset,
    #"ndf_fit": NDFFeatureDataset,
    "rpdiff": RPDiffNoisyGoalDataset,
    "rpdiff_fit": RPDiffNoisyGoalDataset,
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
    
        print("Initializing RigidDataModule with dataset name {} and dataset class {}".format(dataset_cfg.name, DATASET_FN[dataset_cfg.name], ))

    def prepare_data(self):
        pass

    def setup(self, stage: str) -> None:
        self.stage = stage

        # dataset sanity checks
        if self.dataset_cfg.scene and not self.dataset_cfg.world_frame:
            raise ValueError("Scene inputs require a world frame.")

        # if not in train mode, don't use rotation augmentations
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

        self.val_ood_dataset = DATASET_FN[self.dataset_cfg.name](
            self.root, self.dataset_cfg, "test")

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True if self.stage == "fit" else False,
            drop_last=True,
        )

    def val_dataloader(self):
        val_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        val_ood_dataloader = data.DataLoader(
            self.val_ood_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return val_dataloader, val_ood_dataloader

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

        print("Initializing RigidFeatureDataModule with dataset name {} and dataset class {}".format(dataset_cfg.name, F_DATASET_FN[dataset_cfg.name], ))


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

        self.val_ood_dataset = F_DATASET_FN[self.dataset_cfg.name](
            self.root, self.dataset_cfg, "test")

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True if self.stage == "fit" else False,
            drop_last=True,
        )

    def val_dataloader(self):
        val_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        val_ood_dataloader = data.DataLoader(
            self.val_ood_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return val_dataloader, val_ood_dataloader


class RigidDataNoisyGoalModule(L.LightningModule):
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

        print("Initializing RigidDataNoisyGoalModule with dataset name {} and dataset class {}".format(dataset_cfg.name, NG_DATASET_FN[dataset_cfg.name], ))

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


        self.train_dataset = NG_DATASET_FN[self.dataset_cfg.name](
            self.root, self.dataset_cfg, "train")
        
        self.val_dataset = NG_DATASET_FN[self.dataset_cfg.name](
            self.root, self.dataset_cfg, "val")

        self.val_ood_dataset = NG_DATASET_FN[self.dataset_cfg.name](
            self.root, self.dataset_cfg, "test")

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True if self.stage == "fit" else False,
            drop_last=True,
        )

    def val_dataloader(self):
        val_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        val_ood_dataloader = data.DataLoader(
            self.val_ood_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return val_dataloader, val_ood_dataloader