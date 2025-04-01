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

class RPDiffDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, type):
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg
        self.rpdiff_task_name = dataset_cfg.rpdiff_task_name
        self.rpdiff_task_type = dataset_cfg.rpdiff_task_type

        # if eval_mode = True, we turn off occlusion
        self.eval_mode = False if self.type == "train" else True

        # data loading
        self.dataset_dir = self.root / self.rpdiff_task_name / self.rpdiff_task_type
        self.split_dir = self.dataset_dir / "split_info"
        self.split_file = f"{self.type}_split.txt" if self.type != "val" else "train_val_split.txt"

        # setting sample sizes
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor

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
        # TODO: adjust them with the scaling factor
        T_init = relative_trans
        R_action_to_goal = relative_trans[:3, :3]  # Rotation matrix
        t_action_to_goal = relative_trans[:3, 3]   # Translation vector

        '''
        if self.centroid_frame:
            if self.centroid_frame == "anchor":
                center = anchor_pc.mean(axis=0)
            elif self.centroid_frame == "noisy_goal":
                center = goal_action_pc.mean(axis=0)
                goal_noise = self.ref_error_scale * torch.randn_like(center)
                center = center + goal_noise
            elif self.centroid_frame == "goal":
                center = goal_action_pc.mean(axis=0)
            elif self.centroid_frame == "scene":
                # TODO: implement scene centroid
                raise NotImplementedError(f"Not implemented: centroid_frame = scene")
            else:
                raise ValueError(f"Invalid centroid_frame: {self.centroid_frame}. Choose centroid frame from [anchor, noisy_goal, goal, scene]")
            action_center = action_pc.mean(axis=0)
        else:
            # TODO: implement world frame
            raise NotImplementedError(f"Not implemented: no centroid frame, i.e. world frame")
        '''

        # center scene (goal_action_pc + anchor_pc) on the centroid of the scene
        scene_pc = torch.cat([goal_action_pc, anchor_pc], dim=0)
        scene_center = scene_pc.mean(axis=0)
        s_goal_action_pc = goal_action_pc - scene_center
        s_anchor_pc = anchor_pc - scene_center

        # center action in its own frame
        action_center = action_pc.mean(axis=0)
        a_action_pc = action_pc - action_center

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

        s_goal_action_pc_ = T1.transform_points(s_goal_action_pc)
        s_anchor_pc_ = T1.transform_points(s_anchor_pc)
        a_action_pc_ = T0.transform_points(a_action_pc)

        T_goalUnAug = Translate(-scene_center.unsqueeze(0)).compose(T1.inverse().compose(Translate(scene_center.unsqueeze(0))))
        T_actionUnAug = Translate(-action_center.unsqueeze(0)).compose(T0.inverse().compose(Translate(action_center.unsqueeze(0))))

        goal_action_pc_ = s_goal_action_pc_ + scene_center
        anchor_pc_ = s_anchor_pc_ + scene_center
        action_pc_ = a_action_pc_ + action_center
        
        goal_flow_ = goal_action_pc_ - action_pc_

        item = {}
        # NOTE: below are all in world frame now, and T_goalUnAug and T_actionUnAug transform point clouds in world frame to world frame,
        #       with the purpose of removing augmentations
        
        # Training ONLY (we should never access them during inference, if not evaluating)
        item["pc"] = goal_action_pc_ # Goal action points, centered at the centroid frame
        item["flow"] = goal_flow_ # pc - action_pc
        # Training & Inference
        item["pc_action"] = action_pc_ # Action points in action centroid frame (if no world frame)
        item["pc_anchor"] = anchor_pc_ # Anchor points, centered at the centroid frame (if no world frame)
        item["seg"] = action_seg
        item["seg_anchor"] = anchor_seg
        item["T_goalUnAug"] = T_goalUnAug.get_matrix().squeeze(0)
        item["T_actionUnAug"] = T_actionUnAug.get_matrix().squeeze(0)
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
            self.dataset_cfg.action_transform_type = "identity"
            self.dataset_cfg.anchor_transform_type = "identity"
            self.dataset_cfg.action_translation_variance = 0
            self.dataset_cfg.action_rotation_variance = 0
            self.dataset_cfg.anchor_translation_variance = 0
            self.dataset_cfg.anchor_rotation_variance = 0

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