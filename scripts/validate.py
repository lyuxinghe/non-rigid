import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
import json
import omegaconf
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
import open3d as o3d
from pytorch3d.transforms import Translate
from non_rigid.utils.pointcloud_utils import downsample_pcd
import plotly.graph_objects as go
import plotly.io as pio
from pytorch3d.transforms import Transform3d

'''
def save_demo_npz(path, action_pc, anchor_pc):
    """
    Save action and anchor point clouds to .npz format
    compatible with your demo loading code.
    
    Args:
        path (str): Path to save the .npz file.
        action_pc (np.ndarray): [Na, 3] array for action points (class 0).
        anchor_pc (np.ndarray): [Nb, 3] array for anchor points (class 1).
    """
    clouds = np.concatenate([action_pc, anchor_pc], axis=0)  # shape [N, 3]
    classes = np.concatenate([
        np.zeros(len(action_pc), dtype=int), 
        np.ones(len(anchor_pc), dtype=int)
    ])  # shape [N]

    np.savez(path, clouds=clouds, classes=classes)
'''

def save_demo_npz(input_path: str, action_pc: np.ndarray, anchor_pc: np.ndarray,
                  goal_pc: np.ndarray = None, goal_tf: np.ndarray = None):
    """
    Save a demo file in the expected format for the dataset loader.

    Args:
        input_path (str): Path to save the .npz file.
        action_pc (np.ndarray): [N, 3] initial action point cloud.
        anchor_pc (np.ndarray): [M, 3] anchor point cloud.
        goal_pc (np.ndarray, optional): [1, 3] or [N, 3] goal point cloud. Defaults to [[0, 0, 0]].
        goal_tf (np.ndarray, optional): [4, 4] transformation matrix. Defaults to identity.
    """
    if goal_pc is None:
        goal_pc = np.zeros((action_pc.shape[0], 3), dtype=np.float32)
    if goal_tf is None:
        goal_tf = np.eye(4, dtype=np.float32)

    np.savez(
        input_path,
        action_init_points=action_pc.astype(np.float32),
        anchor_points=anchor_pc.astype(np.float32),
        action_goal_points=goal_pc.astype(np.float32),
        goal_tf=goal_tf.astype(np.float32),
    )

def load_insertion_execution(dir):
    action_pcd_path = os.path.join(dir, 'tax3dv2_debug', 'action_points.npy')
    anchor_pcd_path = os.path.join(dir, 'tax3dv2_debug', 'anchor_points.npy')
    data_batch_path = os.path.join(dir, 'tax3dv2_debug', 'data_batch.npz')
    predicted_path = os.path.join(dir, 'tax3dv2_debug', 'pred_dict.npz')
    demo_path = os.path.join(dir, 'teleport_obj_points.npz')

    demo = np.load(demo_path, allow_pickle=True)

    points_raw = demo["clouds"]
    classes_raw = demo["classes"]

    goal_action_pcd = points_raw[classes_raw == 0]
    goal_anchor_pcd = points_raw[classes_raw == 1]

    action_pcd = np.load(action_pcd_path, allow_pickle=True)
    anchor_pcd = np.load(anchor_pcd_path, allow_pickle=True)
    #action_pcd = points_raw[classes_raw == 0]
    #anchor_pcd = points_raw[classes_raw == 1]
    data_batch = np.load(data_batch_path, allow_pickle=True)
    pred_dict = np.load(predicted_path, allow_pickle=True)

    return action_pcd, anchor_pcd, goal_action_pcd, goal_anchor_pcd, data_batch, pred_dict



def infer_tax3dv2(model, action_points, anchor_points, cfg):

    # flip z axis for model inference
    action_points[:,2] = action_points[:,2]
    anchor_points[:,2] = anchor_points[:,2]

    # infer placement pose
    anchor_pc = torch.from_numpy(anchor_points * cfg.dataset.pcd_scale_factor)
    action_pc = torch.from_numpy(action_points * cfg.dataset.pcd_scale_factor)
    
    # prob want to convert anchor/action_pc to tensor?
    sample_size_action = cfg.dataset.sample_size_action
    sample_size_anchor = cfg.dataset.sample_size_anchor
    num_samples = 1

    action_seg = torch.zeros_like(action_pc[:, 0]).int()
    anchor_seg = torch.ones_like(anchor_pc[:, 0]).int()

    # downsample action

    if action_pc.shape[0] > sample_size_action:
        action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), cfg.dataset.sample_size_action, type='fps')
        action_pc = action_pc.squeeze(0)
        action_seg = action_seg[action_pc_indices.squeeze(0)]

    # downsample anchor
    if anchor_pc.shape[0] > sample_size_anchor:
        anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), cfg.dataset.sample_size_anchor, type='fps')
        anchor_pc = anchor_pc.squeeze(0)
        anchor_seg = anchor_seg[anchor_pc_indices.squeeze(0)]

    action_pc = action_pc.to(torch.float32)
    anchor_pc = anchor_pc.to(torch.float32)

    scene_center = torch.cat([action_pc, anchor_pc], dim=0).mean(axis=0)
    anchor_pc = anchor_pc - scene_center
    action_pc = action_pc - scene_center

    T_goal2world = Translate(scene_center.unsqueeze(0))
    T_action2world = Translate(scene_center.unsqueeze(0))

    data_batch = {}

    data_batch["pc_action"] = action_pc.unsqueeze(0) # Action points in the action frame
    data_batch["pc_anchor"] = anchor_pc.unsqueeze(0) # Anchor points in the scene frame
    data_batch["seg"] = action_seg.unsqueeze(0)
    data_batch["seg_anchor"] = anchor_seg.unsqueeze(0)
    data_batch["T_goal2world"] = T_goal2world.get_matrix() # Transform from goal action frame to world frame
    data_batch["T_action2world"] = T_action2world.get_matrix() # Transform from action frame to world frame

    data_batch = model.update_batch_frames(data_batch, update_labels=False, gmm_model=None)
    pred_dict = model.predict(data_batch, num_samples, progress=True, full_prediction=True)

    # pred_point_world = pred_dict["point"]["pred_world"].squeeze(0) / tax3dv2_cfg.dataset.pcd_scale_factor               
    predicted_place_rel_transform = pred_dict['pred_T']
    predicted_tf = predicted_place_rel_transform.squeeze(0).detach().cpu().numpy()  
    '''
    T = np.eye(4)
    T[2,2] = -1
    predicted_tf = T @ predicted_tf @ T
    '''
    predicted_points = pred_dict["point"]["pred_world"]
    predicted_points = predicted_points.squeeze(0).detach().cpu().numpy() / cfg.dataset.pcd_scale_factor
    #predicted_points[:, 2] *= -1  # Flip z-axis
    
    return predicted_points, predicted_tf

import plotly.graph_objects as go
import numpy as np

def vis_pcd_toggle(
    aug_ground_truth: np.ndarray = None,
    aug_action_pc: np.ndarray = None,
    aug_anchor_pc: np.ndarray = None,
    aug_predictions: np.ndarray = None,
    exe_ground_truth: np.ndarray = None,
    exe_action_pc: np.ndarray = None,
    exe_anchor_pc: np.ndarray = None,
    exe_predictions: np.ndarray = None,
):
    fig = go.Figure()

    def add_trace(points, color, name):
        if points is not None and len(points) > 0:
            fig.add_trace(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                name=name,
                marker=dict(size=3, color=color),
            ))

    # Visualize all inputs with color scheme
    # Augmented
    add_trace(aug_action_pc,     "#FFD700", "Aug Action")
    add_trace(aug_anchor_pc,      "#19D3F3", "Aug Anchor")
    add_trace(aug_ground_truth,   "#98FB98", "Aug GT")
    add_trace(aug_predictions,    "#FF6F61", "Aug Prediction")

    # Executed
    add_trace(exe_action_pc,      "#FFA500", "Exe Action")
    add_trace(exe_anchor_pc,      "#1F77B4", "Exe Anchor")
    add_trace(exe_ground_truth,   "#2ECC71", "Exe GT")
    add_trace(exe_predictions,    "#AB63FA", "Exe Prediction")

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

    #fig.show()
    fig.write_html("/home/lyuxinghe/tax3d_realworld/scripts/output_vis_test.html")


import numpy as np

def svd_estimate_transform(source_points, target_points):
    """
    Estimate the rigid transformation T (4x4) that aligns source_points to target_points using SVD.

    Args:
        source_points (np.ndarray): (N, 3) source point cloud.
        target_points (np.ndarray): (N, 3) target point cloud.

    Returns:
        T (np.ndarray): (4, 4) homogeneous transformation matrix.
    """
    assert source_points.shape == target_points.shape, "Shape mismatch between source and target point clouds."

    # Step 1: Compute centroids
    centroid_src = source_points.mean(axis=0)
    centroid_tgt = target_points.mean(axis=0)

    # Step 2: Center the point clouds
    src_centered = source_points - centroid_src
    tgt_centered = target_points - centroid_tgt

    # Step 3: Compute covariance matrix
    H = src_centered.T @ tgt_centered

    # Step 4: Compute SVD
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure a right-handed coordinate system (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Step 5: Compute translation
    t = centroid_tgt - R @ centroid_src

    # Step 6: Form the homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def transform_point_cloud(pcd: np.ndarray, transform: np.ndarray) -> np.ndarray:
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

@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval_rigid", version_base="1.3")
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
            omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
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
    if cfg.gmm is not None:
        # GMM can only be used with noisy goal models.
        if cfg.model.pred_frame != "noisy_goal":
            raise ValueError("GMM can only be used with noisy goal models.")
        
        # Checking for GMM model directory.
        #gmm_exp_name = os.path.join(os.path.expanduser(cfg.gmm_log_dir), f"gmm_{cfg.dataset.name}_{cfg.gmm}")
        # Also adding pcd_scale
        gmm_exp_name = os.path.join(os.path.expanduser(cfg.gmm_log_dir), f"{cfg.gmm_pcd_scale}", f"gmm_{cfg.exp_name}_{cfg.gmm}")
        if not os.path.exists(gmm_exp_name):
            raise ValueError(f"GMM experiment directory {gmm_exp_name} does not exist - train this model first.")
        
        # Loading GMM frame predictor.
        gmm_model_cfg = omegaconf.OmegaConf.load(os.path.join(hydra.utils.get_original_cwd(), "../configs/model/df_cross.yaml"))
        gmm_model_cfg.rel_pos = True
        gmm_model = FrameGMMPredictor(gmm_model_cfg, device)
        gmm_model.load_state_dict(
            torch.load(
                os.path.join(gmm_exp_name, "checkpoints", f"epoch_{cfg.gmm}.pt")
            )
        )
    else:
        gmm_model = None

    ######################################################################
    # Create the datamodule. This is just to initialize the datasets - we are
    # not going to use the dataloaders, because we need to manually downsample 
    # and batch.
    ######################################################################
    #cfg, datamodule = create_datamodule(cfg)

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

    ########################################
    # Not Working
    
    #dir = /data/lyuxing/insertion/04-21-wp-2/execute_data/0428_164754/'
    #dir = '/data/lyuxing/insertion/04-21-wp-2/execute_data/0428_171839/'
    
    
    #dir = '/data/lyuxing/insertion//0428-wp-2/execute_data/0430_211732/'
    #action_pcd, anchor_pcd, goal_action_pcd, goal_anchor_pcd, data_batch, pred_dict = load_insertion_execution(dir)
    #z_offset = 0.1  # e.g., raise by 10 cm
    #action_pcd[:, 2] += z_offset

    
    path = '/data/lyuxing/insertion/0430-wp-1/learn_data/test/tax3d_training_input_86.npz'
    data = dict(np.load(path, allow_pickle=True))
    action_pcd = data['action_init_points']
    anchor_pcd = data['anchor_points']
    goal_action_pcd = data['action_goal_points']
    
    predicted_points, predicted_tf = infer_tax3dv2(model, action_pcd, anchor_pcd, cfg)

    trans_pcd = transform_point_cloud(action_pcd, predicted_tf)

    
    ########################################
    
    cfg.dataset.num_demos = 1
    cfg.inference.batch_size = 1
    cfg.inference.num_trials = 1
    cfg.inference.num_wta_trials = 1
    cfg.inference.val_batch_size = 1
    cfg.inference.batch_size = 1
    #cfg.dataset.scene_transform_type = "identity"
    #cfg.dataset.rotation_variance = 0.0
    #cfg.dataset.translation_variance = 0.0
    '''
    cfg.dataset.action_transform_type = "insertion_specific"
    cfg.dataset.anchor_transform_type = "identity"
    cfg.dataset.action_translation_variance = 0.1
    cfg.dataset.action_rotation_variance = 0.52
    cfg.dataset.anchor_translation_variance = 0.1
    cfg.dataset.anchor_rotation_variance = 0.52
    '''
    cfg, datamodule = create_datamodule(cfg)

    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.val_dataloader()

    train_dataloader.dataset.set_eval_mode(False)
    train_dataloader.dataset.demo_files = [None]


    dir = '/data/lyuxing/insertion/04-21-wp-2/execute_data/0428_174602/'
    #dir = '/data/lyuxing/insertion/04-21-wp-2/execute_data/0428_174347/'
    #dir = '/data/lyuxing/insertion/04-21-wp-2/execute_data/0428_174602/'
    #dir = '/data/lyuxing/insertion//0428-wp-2/execute_data/0430_211732/'
    #dir = '/data/lyuxing/insertion/0428-wp-2/execute_data/0430_171317/'
    exe_action_pcd, exe_anchor_pcd, exe_goal_action_pcd, goal_anchor_pcd, data_batch, pred_dict = load_insertion_execution(dir)

    save_path = os.path.join(dir, "init_demo.npz")

    save_demo_npz(save_path, action_pc=exe_action_pcd, anchor_pc=exe_anchor_pcd)
    
    #save_path = os.path.join(dir, "teleport_obj_points.npz")
    
    train_dataloader.dataset.demo_files = [str(save_path)]
    train_dataloader.dataset.set_eval_mode(False)
    assert len(train_dataloader.dataset) == 1

    with torch.no_grad():
        for batch in tqdm(train_dataloader):

            # Generate predictions.
            if gmm_model is not None:
               raise ValueError
            else:
                pred_batch = model.update_batch_frames(batch, update_labels=True)
                aug_pred_dict = model.predict(pred_batch, cfg.inference.num_trials, progress=True, full_prediction=True)
            break   # there should only be one file

    #aug_action_pc = pred_batch['pc_action'][0].clone().detach().cpu().numpy() / 50.0
    #aug_anchor_pc = pred_batch['pc_anchor'][0].clone().detach().cpu().numpy() / 50.0
    #aug_ground_truth = pred_batch['pc'][0].clone().detach().cpu().numpy() / 50.0

    aug_predicted_points = aug_pred_dict["point"]["pred_world"]
    aug_predicted_points = aug_predicted_points[0].detach().cpu().numpy() / cfg.dataset.pcd_scale_factor

    #w_init_pc_action = batch["w_init_pc_action"].squeeze().numpy() / cfg.dataset.pcd_scale_factor
    #w_init_pc_anchor = batch["w_init_pc_anchor"].squeeze().numpy() / cfg.dataset.pcd_scale_factor
    #w_aug_pc_action = batch["w_aug_pc_action"].squeeze().numpy() / cfg.dataset.pcd_scale_factor
    #w_aug_pc_anchor = batch["w_aug_pc_anchor"].squeeze().numpy() / cfg.dataset.pcd_scale_factor
    
    
    T = Transform3d(matrix=batch['T'])
    T_inv = T.inverse()
    
    aug_predicted_points = T_inv.transform_points(torch.from_numpy(aug_predicted_points)) 
    

    # Step 1: Convert to homogeneous [512, 4]
    #pcd_hom = np.hstack([action_pcd, np.ones((action_pcd.shape[0], 1))])  # [512, 4]

    # Step 2: Apply transform
    #pcd_transformed_hom = (predicted_tf @ pcd_hom.T).T  # [512, 4]

    # Step 3: Convert back to [512, 3]
    #pcd_transformed = pcd_transformed_hom[:, :3]
    
    #############################################
    vis_pcd_toggle(
        aug_ground_truth=goal_action_pcd, #        aug_ground_truth=w_init_pc_action,
        aug_action_pc=action_pcd,
        aug_anchor_pc=anchor_pcd,
        aug_predictions=predicted_points,
        exe_ground_truth=exe_goal_action_pcd,
        exe_action_pc=exe_action_pcd,
        exe_anchor_pc=exe_anchor_pcd,
        exe_predictions=aug_predicted_points,
    )    #plot_pointclouds_toggle(loaded_action, loaded_anchor, predicted_points)
    

def check_pred(pred_point=None, pred_T=None):
    dir = '/data/lyuxing/tax3d/insertion/demonstrations_new/04-21-wp-2/execute_data/0428_164754/'
    action_pcd, anchor_pcd, goal_action_pcd, goal_anchor_pcd, data_batch, pred_dict = load_insertion_execution(dir)
    
    if pred_point is None:
        pred_point = pred_dict['point'].item()['pred_world'][0]/50  # [512, 3]
    if pred_T is None:
        pred_T = pred_dict['pred_T'][0]

    action_in = data_batch['pc_action'][0]/50
    anchor_in = data_batch['pc_anchor'][0]/50

    # recover downsampled world frame initial action pcd
    pc_action = data_batch['pc_action'][0]  # (N, 3)
    action_context_frame = data_batch['action_context_frame'][0]  # (N, 3)
    points_local = pc_action + action_context_frame  # (N, 3)

    points_local_h = np.hstack([points_local, np.ones((points_local.shape[0], 1))])  # (N, 4)

    T_action2world = data_batch["T_action2world"][0]  # (4, 4)
    points_world_h = (T_action2world @ points_local_h.T).T  # (N, 4)
    action_recover = points_world_h[:, :3] / 50

    # compute transformed initial action pcd
    tramsformed_points = np.hstack([action_pcd, np.ones((action_pcd.shape[0], 1))])  # (512, 4)
    pcd_transformed_h = (pred_T @ tramsformed_points.T).T  # (512, 4)
    pcd_transformed = pcd_transformed_h[:, :3]  # (512, 3)
    
    '''
    relative_trans = svd_estimate_transform(action_in, pred_point.numpy())
    tramsformed_points = np.hstack([action_in, np.ones((action_in.shape[0], 1))])  # (512, 4)
    pcd_transformed_h = (relative_trans @ tramsformed_points.T).T  # (512, 4)
    pcd_transformed = pcd_transformed_h[:, :3]  # (512, 3)
    '''
    vis_pcd_toggle(
        ground_truth=goal_action_pcd,
        w_action_pc=action_pcd,
        w_anchor_pc=anchor_pcd,
        in_action_pc=action_in,
        in_anchor_pc=anchor_in,
        predictions=pred_point,
        predictions_trans=pcd_transformed,
        draw_correspondence=False
    )

if __name__ == "__main__":
    main()
    #check_pred()

# example usage: python validate.py resources.gpus=[0] checkpoint.run_id=p5htxw3u dataset.data_dir=/data/lyuxing/insertion/