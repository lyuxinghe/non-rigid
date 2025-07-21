import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import random

import lightning as L
import numpy as np
import omegaconf
import torch
import torch.utils.data as data
from pytorch3d.transforms import Translate
from scipy.spatial.transform import Rotation as R

from non_rigid.utils.augmentation_utils import maybe_apply_augmentations
from non_rigid.utils.pointcloud_utils import downsample_pcd, get_multi_anchor_scene
from non_rigid.utils.transform_utils import random_se3

from typing import List

def matrix_from_list(pose_list: List[float]) -> np.ndarray:
    trans = pose_list[:3]
    quat = pose_list[3:]

    T = np.eye(4)
    T[:-1, :-1] = R.from_quat(quat).as_matrix()
    T[:-1, -1] = trans
    return T

class RPDiffDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, type):
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg
        self.rpdiff_task_name = dataset_cfg.rpdiff_task_name
        self.rpdiff_task_type = dataset_cfg.rpdiff_task_type

        # additional processing flags
        self.augment_gt = False
        self.preprocess = False

        # if eval_mode = True, we turn off occlusion
        self.eval_mode = False if self.type == "train" else True
            
        # data loading
        self.dataset_dir = self.root / self.rpdiff_task_name / self.rpdiff_task_type
        self.split_dir = self.dataset_dir / "split_info"
        self.split_file = f"{self.type}_split.txt" if self.type != "val" else "train_val_split.txt"

        
        if 'preprocess' in dataset_cfg and dataset_cfg.preprocess:
            self.dataset_dir = self.dataset_dir / "preprocessed"
            self.preprocess = True
            print(f"Loading RPDiff Preprocessed Dataset from {self.dataset_dir}")
        else:
            print(f"Loading RPDiff Dataset from {self.dataset_dir}")

        if 'augment_gt' in dataset_cfg and dataset_cfg.augment_gt:
            self.augment_gt = True
            print(f"Augmenting RPDiff Dataset Ground Truths for Book/Bookshelf and Can/Cabinet")


        # setting sample sizes
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor

        with open(os.path.join(self.split_dir, self.split_file), "r") as file:
            self.demo_files = [f"{self.dataset_dir}/{line.strip()}" for line in file]

        self.num_demos = len(self.demo_files)
        if self.dataset_cfg.num_demos is not None and self.type == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.type} demos from {self.dataset_dir} with {self.split_file}, total of {self.__len__()} files")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_size if self.dataset_cfg.train_size is not None else len(self.demo_files)
        elif self.type == "val":
            return self.dataset_cfg.val_size if self.dataset_cfg.val_size is not None else len(self.demo_files)
        elif self.type == "test":
            return self.dataset_cfg.test_size if self.dataset_cfg.test_size is not None else len(self.demo_files)
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos], allow_pickle=True)
        
        # Access start and final PCDs for parent and child
        parent_start_pcd = demo['multi_obj_start_pcd'].item()['parent']
        child_start_pcd = demo['multi_obj_start_pcd'].item()['child']
        parent_final_pcd = demo['multi_obj_final_pcd'].item()['parent']
        child_final_pcd = demo['multi_obj_final_pcd'].item()['child']
        
        action_pc = torch.as_tensor(child_start_pcd).float()
        anchor_pc = torch.as_tensor(parent_start_pcd).float()
        goal_action_pc = torch.as_tensor(child_final_pcd).float()
        goal_anchor_pc = torch.as_tensor(parent_final_pcd).float()  # same as anchor_pc

        action_seg = torch.zeros_like(action_pc[:, 0]).int()
        anchor_seg = torch.ones_like(anchor_pc[:, 0]).int()

        # calculate the scale of action and anchor pcd, such that we can estimate reasonable params 
        # for 1. augmentations and 2. noise scale
        # TODO: this mighe be a bit inefficient ..., optimize it!
        action_point_dists = action_pc - action_pc.mean(dim=0, keepdim=True)
        action_point_scale = torch.linalg.norm(action_point_dists, dim=1, keepdim=True).max()
        anchor_point_dists = anchor_pc - anchor_pc.mean(dim=0, keepdim=True) 
        anchor_point_scale = torch.linalg.norm(anchor_point_dists, dim=1, keepdim=True).max()

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
                        * action_point_scale,
                    },
                    plane_occlusion_param={
                        "plane_occlusion": self.dataset_cfg.action_plane_occlusion,
                        "plane_standoff": self.dataset_cfg.action_plane_standoff
                        * action_point_scale,
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
                        * anchor_point_scale,
                    },
                    plane_occlusion_param={
                        "plane_occlusion": self.dataset_cfg.anchor_plane_occlusion,
                        "plane_standoff": self.dataset_cfg.anchor_plane_standoff
                        * anchor_point_scale,
                    },
                )

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

        # Apply task-specific augmentations for book/bookshelf and can/cabinet
        if self.augment_gt and ("bookshelf" in self.rpdiff_task_name or "cabinet" in self.rpdiff_task_name):
            final_parent_pose_list = demo['multi_obj_final_obj_pose'].item()['parent']
            final_child_pose_list = demo['multi_obj_final_obj_pose'].item()['child']
            final_parent_pose_mat = matrix_from_list(final_parent_pose_list)
            final_child_pose_mat = matrix_from_list(final_child_pose_list)
            parent_ori = final_parent_pose_mat[:3, :3]
            child_centroid = final_child_pose_mat[:3, 3]
            
            if "bookshelf" in self.rpdiff_task_name:
                rotations = [
                    R.from_euler('xyz', [0, 0, 0]).as_matrix(),
                    R.from_euler('xyz', [np.pi, 0, 0]).as_matrix(),
                    R.from_euler('xyz', [0, np.pi, 0]).as_matrix(),
                    R.from_euler('xyz', [np.pi, np.pi, 0]).as_matrix(),
                ]
            elif "cabinet" in self.rpdiff_task_name:
                rotations = [
                    R.from_euler('xyz', [0, 0, 0]).as_matrix(),
                    R.from_euler('xyz', [np.pi, 0, 0]).as_matrix(),
                ]
            else:
                raise ValueError(f"Unknown task: {self.rpdiff_task_name}")

            canonical_rot = random.choice(rotations)
            rot = parent_ori @ canonical_rot @ parent_ori.T

            tf = np.eye(4)
            tf[:3, :3] = rot

            tf_torch = torch.as_tensor(tf, dtype=goal_action_pc.dtype, device=goal_action_pc.device)
            child_centroid_torch = torch.as_tensor(child_centroid, dtype=goal_action_pc.dtype, device=goal_action_pc.device)

            # center the goal point cloud 
            child_centroid_torch = child_centroid_torch.unsqueeze(0)
            goal_action_pc_centered = goal_action_pc - child_centroid_torch  # [N, 3]

            # convert centered point cloud to homogeneous coordinates
            goal_action_pc_homogeneous = torch.cat([
                goal_action_pc_centered, 
                torch.ones(goal_action_pc_centered.shape[0], 1, dtype=goal_action_pc.dtype, device=goal_action_pc.device)
            ], dim=1) # [N, 3] -> [N, 4]

            # apply transformation and bring it back to world frame
            goal_action_pc_transformed = (tf_torch @ goal_action_pc_homogeneous.T).T
            goal_action_pc = goal_action_pc_transformed[:, :3] + child_centroid_torch

        # Apply scene-level augmentation.
        T = random_se3(
            N=1,
            rot_var=self.dataset_cfg.rotation_variance,
            trans_var=self.dataset_cfg.translation_variance,
            rot_sample_method=self.dataset_cfg.scene_transform_type,
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

        goal_flow = goal_action_pc - action_pc

        item = {}
        item["pc_action"] = action_pc # Action points in the action frame
        item["pc_anchor"] = anchor_pc # Anchor points in the scene frame
        item["seg"] = action_seg
        item["seg_anchor"] = anchor_seg
        item["T_goal2world"] = T_goal2world.get_matrix().squeeze(0) # Transform from goal action frame to world frame
        item["T_action2world"] = T_action2world.get_matrix().squeeze(0) # Transform from action frame to world frame
        
        # Training-specific labels.
        # TODO: eventually, rename this key to "point"
        item["pc"] = goal_action_pc # Ground-truth goal action points in the scene frame
        item["flow"] = goal_flow # Ground-truth flow (cross-frame) to action points
        
        if self.dataset_cfg.pred_frame == "noisy_goal":
            # "Simulate" the GMM prediction as noisy goal.
            goal_center = goal_action_pc.mean(axis=0)
            goal_noise = self.dataset_cfg.noisy_goal_scale * torch.normal(mean=torch.zeros(3), std=torch.ones(3))
            # New: we scale the noise added to the goal, such that the noise is roughly on the scale of object scale
            scaled_goal_noise = goal_noise * action_point_scale
            item["noisy_goal"] = goal_center + scaled_goal_noise

        return item

    def set_eval_mode(self, eval_mode: bool):
        """ Toggle eval mode to enable/disable augmentation """
        self.eval_mode = eval_mode

DATASET_FN = {
    "rpdiff": RPDiffDataset,
    "rpdiff_fit": RPDiffDataset,
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

        # if not in train mode, don't use rotation augmentations
        if self.stage != "fit":
            print("-------Turning off rotation augmentation for validation/inference.-------")
            self.dataset_cfg.scene_transform_type = "identity"
            self.dataset_cfg.rotation_variance = 0.0
            self.dataset_cfg.translation_variance = 0.0

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
            persistent_workers=True, 
            pin_memory=False
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