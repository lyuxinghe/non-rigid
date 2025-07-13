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
import open3d as o3d

from non_rigid.utils.augmentation_utils import maybe_apply_augmentations
from non_rigid.utils.pointcloud_utils import downsample_pcd, get_multi_anchor_scene
from non_rigid.utils.transform_utils import random_se3



class InsertionDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, type):
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg

        self.connector_type = dataset_cfg.connector_type
        self.data_type = dataset_cfg.data_type
        self.split = "train" if self.type == "train" else "test"

        # if eval_mode = True, we turn off occlusion
        self.eval_mode = False if self.type == "train" else True
            
        # data loading
        self.dataset_dir = self.root / self.connector_type / self.data_type / self.split

        if 'preprocess' in dataset_cfg and dataset_cfg.preprocess:
            self.dataset_dir = self.dataset_dir / "preprocessed"
            print(f"Loading Insertion Preprocessed Dataset from {self.dataset_dir}")
        else:
            print(f"Loading Insertion Dataset from {self.dataset_dir}")
        
        # setting sample sizes
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor

        self.demo_files = [
            os.path.join(self.dataset_dir, f)
            for f in os.listdir(self.dataset_dir)
            if f.endswith('.npz')
        ]

        self.num_demos = len(self.demo_files)
        if self.dataset_cfg.num_demos is not None and self.type == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.type} demos from {self.dataset_dir}, total of {self.__len__()} files")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_dataset_size if self.dataset_cfg.train_dataset_size is not None else len(self.demo_files)
        elif self.type == "val":
            return self.dataset_cfg.val_dataset_size if self.dataset_cfg.val_dataset_size is not None else len(self.demo_files)
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")
    '''
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos], allow_pickle=True)

        # Extract point clouds
        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        # TODO: Remove this hardcoded class selection
        points_action = points_raw[classes_raw == 0]
        points_anchor = points_raw[classes_raw == 1]

        action_pc = torch.as_tensor(points_action).float()
        anchor_pc = torch.as_tensor(points_anchor).float()

        # Apply scale factor
        action_pc *= self.dataset_cfg.pcd_scale_factor
        anchor_pc *= self.dataset_cfg.pcd_scale_factor

        action_seg = torch.zeros_like(action_pc[:, 0]).int()
        anchor_seg = torch.ones_like(anchor_pc[:, 0]).int()
        
        w_init_pc_action = action_pc.clone()
        w_init_pc_anchor = anchor_pc.clone()
        
        # Apply augmentations
        if self.type == "train" or self.dataset_cfg.val_use_defaults:
            if not self.eval_mode:
                # Apply augmentations to the point clouds in their final positions
                action_pc, action_pc_indices = maybe_apply_augmentations(
                    action_pc,
                    min_num_points=self.dataset_cfg.sample_size_action,
                    ball_occlusion_param={
                        "ball_occlusion": self.dataset_cfg.action_ball_occlusion,
                        #"ball_radius": self.dataset_cfg.action_ball_radius * self.dataset_cfg.pcd_scale_factor,
                        "ball_radius": self.dataset_cfg.action_ball_radius,
                    },
                    plane_occlusion_param={
                        "plane_occlusion": self.dataset_cfg.action_plane_occlusion,
                        #"plane_standoff": self.dataset_cfg.action_plane_standoff * self.dataset_cfg.pcd_scale_factor,
                        "plane_standoff": self.dataset_cfg.action_plane_standoff,
                    },
                )
                action_seg = action_seg[action_pc_indices.squeeze(0)]

                anchor_pc, anchor_pc_indices = maybe_apply_augmentations(
                    anchor_pc,
                    min_num_points=self.dataset_cfg.sample_size_anchor,
                    ball_occlusion_param={
                        "ball_occlusion": self.dataset_cfg.anchor_ball_occlusion,
                        #"ball_radius": self.dataset_cfg.anchor_ball_radius * self.dataset_cfg.pcd_scale_factor,
                        "ball_radius": self.dataset_cfg.action_ball_radius,
                    },
                    plane_occlusion_param={
                        "plane_occlusion": self.dataset_cfg.anchor_plane_occlusion,
                        #"plane_standoff": self.dataset_cfg.anchor_plane_standoff * self.dataset_cfg.pcd_scale_factor,
                        "plane_standoff": self.dataset_cfg.anchor_plane_standoff,
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

        # downsample anchor
        if self.sample_size_anchor > 0 and anchor_pc.shape[0] > self.sample_size_anchor:
            anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), self.sample_size_anchor, type=downsample_type)
            anchor_pc = anchor_pc.squeeze(0)
            anchor_seg = anchor_seg[anchor_pc_indices.squeeze(0)]


        # Apply scene-level augmentation.
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

        initial_action_pc = T0.transform_points(action_pc)
        goal_action_pc = T1.transform_points(action_pc)
        anchor_pc = T1.transform_points(anchor_pc)

        w_aug_pc_action = initial_action_pc.clone()
        w_aug_pc_anchor = anchor_pc.clone()
        w_aug_goal_action_pc = goal_action_pc.clone()

        # Center point clouds in scene frame.
        scene_center = torch.cat([initial_action_pc, anchor_pc], dim=0).mean(axis=0)
        goal_action_pc = goal_action_pc - scene_center
        anchor_pc = anchor_pc - scene_center
        initial_action_pc = initial_action_pc - scene_center

        # Update item.
        T_goal2world = Translate(scene_center.unsqueeze(0))
        T_action2world = Translate(scene_center.unsqueeze(0))
        T_action2goal = T0.inverse().compose(T1)

        goal_flow = goal_action_pc - initial_action_pc

        item = {}
        #item["w_init_pc_action"] = w_init_pc_action
        #item["w_init_pc_anchor"] = w_init_pc_anchor
        #item["w_aug_pc_action"] = w_aug_pc_action
        #item["w_aug_pc_anchor"] = w_aug_pc_anchor
        #item["w_aug_goal_action_pc"] = w_aug_goal_action_pc
        item["T0"] = T0.get_matrix().squeeze(0)
        item["T1"] = T1.get_matrix().squeeze(0)

        item["pc_action"] = initial_action_pc # Action points in the action frame
        item["pc_anchor"] = anchor_pc # Anchor points in the scene frame
        item["seg"] = action_seg
        item["seg_anchor"] = anchor_seg
        item["T_goal2world"] = T_goal2world.get_matrix().squeeze(0) # Transform from goal action frame to world frame
        item["T_action2world"] = T_action2world.get_matrix().squeeze(0) # Transform from action frame to world frame
        item["T_action2goal"] = T_action2goal.get_matrix().squeeze(0) # Transform from action frame to world frame

        # Training-specific labels.
        # TODO: eventually, rename this key to "point"
        item["pc"] = goal_action_pc # Ground-truth goal action points in the scene frame
        item["flow"] = goal_flow # Ground-truth flow (cross-frame) to action points
        
        if self.dataset_cfg.pred_frame == "noisy_goal":
            # "Simulate" the GMM prediction as noisy goal.
            goal_center = goal_action_pc.mean(axis=0)
            item["noisy_goal"] = goal_center + self.dataset_cfg.noisy_goal_scale * torch.normal(mean=torch.zeros(3), std=torch.ones(3))

        return item
    '''

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos], allow_pickle=True)

        # Extract point clouds
        points_action = demo['action_init_points']
        points_anchor = demo['anchor_points']
        #action_goal_points = demo['action_goal_points']        # we will not use the goal action pcd provided, since we don't have the same amount of points
        goal_tf = demo['goal_tf']                               # we instead use the provided gt transformation
                                                                # Note that: goal_pcd @ goal_tf = initial_action_pcd

        points_action_pcd = o3d.geometry.PointCloud()
        points_action_pcd.points = o3d.utility.Vector3dVector(points_action)
        points_action_goal_pcd = points_action_pcd.transform(np.linalg.inv(goal_tf))
        points_action_goal = np.asarray(points_action_goal_pcd.points)

        action_pc = torch.as_tensor(points_action).float()
        anchor_pc = torch.as_tensor(points_anchor).float()
        goal_action_pc = torch.as_tensor(points_action_goal).float()
        goal_tf = torch.as_tensor(goal_tf).float()

        action_seg = torch.zeros_like(action_pc[:, 0]).int()
        anchor_seg = torch.ones_like(anchor_pc[:, 0]).int()

        action_point_dists = action_pc - action_pc.mean(dim=0, keepdim=True)
        action_point_scale = torch.linalg.norm(action_point_dists, dim=1, keepdim=True).max()
        anchor_point_dists = anchor_pc - anchor_pc.mean(dim=0, keepdim=True) 
        anchor_point_scale = torch.linalg.norm(anchor_point_dists, dim=1, keepdim=True).max()

        
        # Put the action point cloud in the center of the anchor
        anchor_center = anchor_pc.mean(dim=0, keepdim=True)
        #goal_action_center = goal_action_pc.mean(dim=0, keepdim=True)
        #action_pc = goal_action_pc - goal_action_center + anchor_center
        action_center = action_pc.mean(dim=0, keepdim=True)
        action_pc = action_pc - action_center + anchor_center

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
        
        # Apply object-level augmentation.
        action_center = action_pc.mean(axis=0)
        action_pc_centered = action_pc - action_center

        T_obj = random_se3(
            N=1,
            rot_var=self.dataset_cfg.object_rotation_variance,
            trans_var=self.dataset_cfg.object_translation_variance * action_point_scale,
            rot_sample_method=self.dataset_cfg.object_transform_type,
        )

        action_pc_augmented = T_obj.transform_points(action_pc_centered)
        action_pc = action_pc_augmented + action_center
        
        # Apply scene-level augmentation.
        T_scene = random_se3(
            N=1,
            rot_var=self.dataset_cfg.scene_rotation_variance,
            trans_var=self.dataset_cfg.scene_translation_variance * anchor_point_scale,
            rot_sample_method=self.dataset_cfg.scene_transform_type,
        )
        action_pc = T_scene.transform_points(action_pc)
        anchor_pc = T_scene.transform_points(anchor_pc)
        goal_action_pc = T_scene.transform_points(goal_action_pc)

        # Center point clouds in scene frame.
        scene_center = torch.cat([action_pc, anchor_pc], dim=0).mean(axis=0)
        goal_action_pc = goal_action_pc - scene_center
        anchor_pc = anchor_pc - scene_center
        action_pc = action_pc - scene_center

        # Update item.
        T_goal2world = Translate(scene_center.unsqueeze(0)).compose(T_scene.inverse())
        #T_action2world = Translate(scene_center.unsqueeze(0)).compose(T_scene.inverse()).compose(Translate(-action_center.unsqueeze(0))).compose(T_obj.inverse()).compose(Translate(action_center.unsqueeze(0)))
        T_action2world = Translate(scene_center.unsqueeze(0)).compose(T_scene.inverse())

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
    "insertion": InsertionDataset
}


class RealWorldDataModule(L.LightningModule):
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
    
        print("Initializing RealWorldDataModule with dataset name {} and dataset class {}".format(dataset_cfg.name, DATASET_FN[dataset_cfg.name], ))

    def prepare_data(self):
        pass

    def setup(self, stage: str) -> None:
        self.stage = stage

        
        if self.stage != "fit":
            print("-------Turning off rotation augmentation for validation/inference.-------")
            self.dataset_cfg.scene_transform_type = "identity"
            self.dataset_cfg.scene_rotation_variance = 0.0
            self.dataset_cfg.scene_translation_variance = 0.0
            self.dataset_cfg.object_transform_type = "identity"
            self.dataset_cfg.object_rotation_variance = 0.0
            self.dataset_cfg.object_translation_variance = 0.0
        
        self.train_dataset = (
            DATASET_FN[self.dataset_cfg.name](self.root, self.dataset_cfg, "train")
        )
        
        self.val_dataset = (
            DATASET_FN[self.dataset_cfg.name](self.root, self.dataset_cfg, "val")
        )


    def train_dataloader(self) -> data.DataLoader:
        assert self.train_dataset is not None
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

        return val_dataloader