import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
import json
from omegaconf import OmegaConf
import torch
import wandb
import os

from non_rigid.utils.script_utils import (
    create_model,
    create_datamodule,
    load_checkpoint_config_from_wandb,
)
from non_rigid.metrics.flow_metrics import flow_rmse
from non_rigid.metrics.rigid_metrics import svd_estimation
from non_rigid.utils.pointcloud_utils import expand_pcd
from non_rigid.models.gmm_predictor import FrameGMMPredictor
from tqdm import tqdm
import numpy as np

import rpad.visualize_3d.plots as vpl
import plotly.graph_objects as go

# below are packages needed for loading data into batch
import open3d as o3d
from non_rigid.utils.pointcloud_utils import downsample_pcd
from pytorch3d.transforms import Translate
from non_rigid.utils.transform_utils import random_se3

def transform_pcd(pcd: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 transformation matrix to an Nx3 point cloud.

    Args:
        pcd (np.ndarray): Nx3 array representing the point cloud.
        transform (np.ndarray): 4x4 transformation matrix.

    Returns:
        np.ndarray: Transformed Nx3 point cloud.
    """
    assert pcd.shape[1] == 3, "Point cloud must be of shape Nx3"
    assert transform.shape == (4, 4), "Transformation matrix must be 4x4"

    # Convert to homogeneous coordinates
    pcd_h = np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1)  # Nx4

    # Apply transformation
    pcd_transformed_h = (transform @ pcd_h.T).T  # Nx4

    # Convert back to Cartesian coordinates
    return pcd_transformed_h[:, :3]

def visualize_prediction(pred_point_pcd, pred_T, goal_action_pcd, action_pcd, anchor_pcd, output_html: str):
    fig = go.Figure()

    # Only compute transformed pcd if both action_pcd and goal_tf exist
    trans_pcd = None
    if action_pcd is not None and pred_T is not None:
        trans_pcd = transform_pcd(action_pcd, pred_T)

    def color_with_intensity(base_hex, intensity):
        rgb = [int(base_hex[i:i+2], 16) for i in (1, 3, 5)]
        return f'rgb({int(rgb[0]*intensity)}, {int(rgb[1]*intensity)}, {int(rgb[2]*intensity)})'

    def add_trace(pc, base_color, name):
        if pc is not None:
            fig.add_trace(go.Scatter3d(
                x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
                mode='markers',
                name=f"{name}",
                marker=dict(size=2.5, color=color_with_intensity(base_color, 1)),
            ))

    # Add traces only if the data exists
    add_trace(goal_action_pcd, "#00FF00", "Goal")     # Green
    add_trace(action_pcd,      "#FFD700", "Action")   # Yellow
    add_trace(anchor_pcd,      "#1F77B4", "Anchor")   # Blue
    add_trace(trans_pcd,       "#FF0000", "Transformed")  # Red
    add_trace(pred_point_pcd,  "#E100FF", "Predicted")  # Purple

    fig.update_layout(
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01
        ),
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig.write_html(output_html)
    print(f"Saved to {output_html}")

def load_data_gt(path):
    demo = np.load(path, allow_pickle=True)
    points_action = demo['action_init_points']
    points_anchor = demo['anchor_points']
    action_goal_points = demo['action_goal_points']

    # !!! we need to put action into the anchor center, in order to 
    # !!! make sure that pred_T transform the action point cloud back correctly
    anchor_center = np.mean(points_anchor, axis=0, keepdims=True)
    action_center = np.mean(points_action, axis=0, keepdims=True)
    points_action = points_action - action_center + anchor_center

    item = {}
    item["pc_action_world"] = points_action
    item["pc_world"] = action_goal_points
    item["pc_anchor_world"] = points_anchor

    return item


def load_data_batch(path):
    demo = np.load(path, allow_pickle=True)

    # Extract point clouds
    points_action = demo['action_init_points']
    points_anchor = demo['anchor_points']
    #action_goal_points = demo['action_goal_points']        # we will not use the goal action pcd provided, since we don't have the same amount of points
    #goal_tf = demo['goal_tf']                               # we instead use the provided gt transformation
                                                            # Note that: goal_pcd @ goal_tf = initial_action_pcd

    points_action_pcd = o3d.geometry.PointCloud()
    points_action_pcd.points = o3d.utility.Vector3dVector(points_action)
    #points_action_goal_pcd = points_action_pcd.transform(np.linalg.inv(goal_tf))
    #points_action_goal = np.asarray(points_action_goal_pcd.points)

    action_pc = torch.as_tensor(points_action).float()
    anchor_pc = torch.as_tensor(points_anchor).float()
    #goal_action_pc = torch.as_tensor(points_action_goal).float()
    #goal_tf = torch.as_tensor(goal_tf).float()

    action_seg = torch.zeros_like(action_pc[:, 0]).int()
    anchor_seg = torch.ones_like(anchor_pc[:, 0]).int()

    action_point_dists = action_pc - action_pc.mean(dim=0, keepdim=True)
    action_point_scale = torch.linalg.norm(action_point_dists, dim=1, keepdim=True).max()
    anchor_point_dists = anchor_pc - anchor_pc.mean(dim=0, keepdim=True) 
    anchor_point_scale = torch.linalg.norm(anchor_point_dists, dim=1, keepdim=True).max()
    
    # Put the action point cloud in the center of the anchor
    anchor_center = anchor_pc.mean(dim=0, keepdim=True)
    action_center = action_pc.mean(dim=0, keepdim=True)
    action_pc = action_pc - action_center + anchor_center


    downsample_type = "fps"
    sample_size_action = 2048
    sample_size_anchor = 2048

    # downsample action
    if action_pc.shape[0] > sample_size_action:
        action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), sample_size_action, type=downsample_type)
        action_pc = action_pc.squeeze(0)
        action_seg = action_seg[action_pc_indices.squeeze(0)]
        #goal_action_pc = goal_action_pc[action_pc_indices.squeeze(0)]

    # downsample anchor
    if anchor_pc.shape[0] > sample_size_anchor:
        anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), sample_size_anchor, type=downsample_type)
        anchor_pc = anchor_pc.squeeze(0)
        anchor_seg = anchor_seg[anchor_pc_indices.squeeze(0)]

    # Apply scene-level augmentation.
    T_scene = random_se3(
        N=1,
        rot_var=0.0,
        trans_var=0.0,
        rot_sample_method="identity",
    )
    action_pc = T_scene.transform_points(action_pc)
    anchor_pc = T_scene.transform_points(anchor_pc)
    #goal_action_pc = T_scene.transform_points(goal_action_pc)

    # Center point clouds in scene frame.
    scene_center = torch.cat([action_pc, anchor_pc], dim=0).mean(axis=0)
    #goal_action_pc = goal_action_pc - scene_center
    anchor_pc = anchor_pc - scene_center
    action_pc = action_pc - scene_center

    # Update item.
    T_goal2world = Translate(scene_center.unsqueeze(0)).compose(T_scene.inverse())
    #T_action2world = Translate(scene_center.unsqueeze(0)).compose(T_scene.inverse()).compose(Translate(-action_center.unsqueeze(0))).compose(T_obj.inverse()).compose(Translate(action_center.unsqueeze(0)))
    T_action2world = Translate(scene_center.unsqueeze(0)).compose(T_scene.inverse())

    #goal_flow = goal_action_pc - action_pc

    item = {}
    item["pc_action"] = action_pc.unsqueeze(0) # Action points in the action frame
    item["pc_anchor"] = anchor_pc.unsqueeze(0) # Anchor points in the scene frame
    item["seg"] = action_seg.unsqueeze(0)
    item["seg_anchor"] = anchor_seg.unsqueeze(0)
    item["T_goal2world"] = T_goal2world.get_matrix() # Transform from goal action frame to world frame
    item["T_action2world"] = T_action2world.get_matrix() # Transform from action frame to world frame
    
    # Training-specific labels.
    # TODO: eventually, rename this key to "point"
    #item["pc"] = goal_action_pc # Ground-truth goal action points in the scene frame
    #item["flow"] = goal_flow # Ground-truth flow (cross-frame) to action points
    
    return item

@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval_insertion", version_base="1.3")
def main(cfg):
    task_overrides = HydraConfig.get().overrides.task
    cfg = load_checkpoint_config_from_wandb(
        cfg, 
        task_overrides, 
        cfg.wandb.entity, 
        cfg.wandb.project, 
        cfg.checkpoint.run_id
    )
    print(
        json.dumps(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
            sort_keys=True,
            indent=4,
        )
    )
    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("medium")

    # Global seed for reproducibility.
    L.seed_everything(42)

    device = f"cuda:{cfg.resources.gpus[0]}"

    ######################################################################
    # Manually setting eval-specific configs.
    ######################################################################
    '''
    # Using a custom cloth-specific batch size, to allow for simultaneous evaluation 
    # of RMSE, coverage, and precision.
    if cfg.dataset.hole == "single":
        bs = 1
    elif cfg.dataset.hole == "double":
        bs = 2
    else:
        raise ValueError(f"Unknown hole type: {cfg.dataset.hole}.")
    bs *= cfg.dataset.num_anchors

    cfg.inference.batch_size = bs
    cfg.inference.val_batch_size = bs
    '''
    # Since coverage and precision are can only be evaluated with RPDiff env, 
    # we are using default batch size here

    ######################################################################
    # Load the GMM frame predictor, if necessary.
    ######################################################################
    # Grab config file from saved run.
    if cfg.gmm.gmm is not None:
        # GMM can only be used with noisy goal models.
        if cfg.model.pred_frame != "noisy_goal":
            raise ValueError("GMM can only be used with noisy goal models.")

        # Create a deep copy of the cfg        
        cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

        gmm_epoch= cfg.gmm.gmm

        gmm_exp_path = os.path.join(
            os.path.expanduser(cfg_copy.gmm.gmm_log_dir),
            cfg_copy.gmm.task_name,
            cfg_copy.gmm.run_id,
        )
        if not os.path.exists(gmm_exp_path):
            raise ValueError(f"Experiment directory {gmm_exp_path} does not exist - train this model first.")
        
        gmm_saved_cfg = OmegaConf.load(os.path.join(gmm_exp_path, "config.yaml"))

        # Update config with run config.
        OmegaConf.update(cfg_copy, "dataset", OmegaConf.select(gmm_saved_cfg, "dataset"), merge=True, force_add=True)
        OmegaConf.update(cfg_copy, "model", OmegaConf.select(gmm_saved_cfg, "model"), merge=True, force_add=True)
        print(
            json.dumps(
                OmegaConf.to_container(cfg_copy, resolve=True, throw_on_missing=False),
                sort_keys=True,
                indent=4,
            )
        )

        cfg_copy, gmm_datamodule = create_datamodule(cfg_copy)
        gmm_network, _ = create_model(cfg_copy)
        gmm_model = FrameGMMPredictor(gmm_network, cfg_copy.model, device)
        gmm_model.eval()

        gmm_ckpt_path = os.path.join(gmm_exp_path, "checkpoints", f"epoch_{gmm_epoch}.pt")
        gmm_model.load_state_dict(torch.load(gmm_ckpt_path))
        print(f"Using GMM checkpoint: {gmm_ckpt_path}")

    else:
        gmm_model = None
        print(f"Using Noisy Oracle")
    
    ######################################################################
    # Create the datamodule. This is just to initialize the datasets - we are
    # not going to use the dataloaders, because we need to manually downsample 
    # and batch.
    ######################################################################
    cfg, datamodule = create_datamodule(cfg)

    ######################################################################
    # Create the network(s) which will be evaluated (same as training).
    # You might want to put this into a "create_network" function
    # somewhere so train and eval can be the same.
    #
    # We'll also load the weights.
    ######################################################################

    # Model architecture is dataset-dependent, so we have a helper
    # function to create the model (while separating out relevant vals).
    network, model = create_model(cfg)    

    # get checkpoint file (for now, this does not log a run)
    checkpoint_reference = cfg.checkpoint.reference
    if checkpoint_reference.startswith(cfg.wandb.entity):
        api = wandb.Api()
        artifact_dir = cfg.wandb.artifact_dir
        artifact = api.artifact(checkpoint_reference, type="model")

        if cfg.checkpoint.alias == "v0":
            model_name = "model.ckpt"
        elif cfg.checkpoint.alias == "monitor":
            # getting artifact names, and sanity checking monitor name
            artifact_file_names = [f.name for f in artifact.files()]
            monitor_name = cfg.checkpoint.monitor_name
            if not isinstance(monitor_name, str):
                raise ValueError(f"Invalid monitor name: {monitor_name}. Must be a string.")
            
            # searching for checkpoints with exact monitor name - should only be one for now.
            valid_artifact_file_names = [f for f in artifact_file_names if 
                                         f.split("-")[2].split("=")[0] == monitor_name]
            if len(valid_artifact_file_names) == 0:
                raise ValueError(f"Could not find any files with monitor name: {monitor_name}.")
            elif len(valid_artifact_file_names) > 1:
                raise ValueError(f"Found multiple files with monitor name: {monitor_name}.")
            else:
                model_name = valid_artifact_file_names[0]
        else:
            raise ValueError(f"Invalid checkpoint alias: {cfg.checkpoint.alias}.")
        ckpt_file = artifact.get_path(model_name).download(root=artifact_dir)
        print(f"Using Wandb checkpoint: {checkpoint_reference}, {model_name}")
    else:
        ckpt_file = checkpoint_reference
        print(f"Using local checkpoint: {ckpt_file}")
    # Load the network weights.
    ckpt = torch.load(ckpt_file, map_location=device)
    network.load_state_dict(
        {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items() if k.startswith("network.")}
    )
    # set model to eval mode
    network.eval()
    model.eval()
    model.to(device)

    # we load the data individually as if we are in actual deployment
    data_path = "/data/lyuxing/tax3d/insertion/demonstrations/0629-wp-mixed/learn_data/test/tax3d_training_input_16.npz"
    data_batch = load_data_batch(data_path)
    data_gt = load_data_gt(data_path)

    # run prediction here
    if gmm_model is not None:
        # Yucky hack for GMM; need to expand point clouds first for WTA GMM samples.
        pred_batch = model.update_batch_frames(data_batch, update_labels=False, gmm_model=gmm_model, num_gmm_trials=cfg.inference.num_gmm_trials)
        pred_dict = model.predict(pred_batch, cfg.inference.num_trials, progress=True, full_prediction=True)
    else:
        pred_batch = model.update_batch_frames(data_batch, update_labels=False)
        pred_dict = model.predict(pred_batch, cfg.inference.num_trials, progress=True, full_prediction=True)

    # pred = pred_dict[self.prediction_type]["pred"]
    pred_point_world = pred_dict["point"]["pred_world"]
    pred_T = pred_dict["pred_T"]
    

    
    for i in range(cfg.inference.num_gmm_trials):
        pred_point_pcd_i = pred_point_world[i].cpu().numpy()
        pred_T_i = pred_T[i].cpu().numpy()
        visualize_prediction(
            pred_point_pcd=pred_point_pcd_i,
            pred_T=pred_T_i,
            goal_action_pcd=data_gt["pc_world"],
            action_pcd=data_gt["pc_action_world"],
            anchor_pcd=data_gt["pc_anchor_world"],
            output_html=f"/home/lyuxing/Desktop/tax3dv2/scripts/logs/vis/insertion/vis{i}.html"
        )



if __name__ == "__main__":
    main()