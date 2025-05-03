import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
import json
import omegaconf
import torch
import wandb
import os, os.path as osp
import glob

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
import rpad.visualize_3d.plots as vpl
from non_rigid.utils.pointcloud_utils import expand_pcd, downsample_pcd

def visualize_point_cloud(vis_dict, save_dir, views_to_render=["default", "top", "side", "diagonal"]):
    """Processes and visualizes a single point cloud file."""
    
    # Create the subdirectory based on the input file name
    os.makedirs(save_dir, exist_ok=True)

    anchor_pc = vis_dict["anchor_pc"]
    action_pc = vis_dict["action_pc"]
    initial_action_pc = vis_dict["initial_action_pc"]
    gt_action_pc = vis_dict["gt_action_pc"]

    # Create Open3D point clouds
    action_cloud = o3d.geometry.PointCloud()
    anchor_cloud = o3d.geometry.PointCloud()
    initial_action_cloud = o3d.geometry.PointCloud()
    gt_action_cloud = o3d.geometry.PointCloud()

    action_cloud.points = o3d.utility.Vector3dVector(action_pc)
    anchor_cloud.points = o3d.utility.Vector3dVector(anchor_pc)
    initial_action_cloud.points = o3d.utility.Vector3dVector(initial_action_pc)
    gt_action_cloud.points = o3d.utility.Vector3dVector(gt_action_pc)

    action_cloud.paint_uniform_color([0, 0, 1])  # Blue
    anchor_cloud.paint_uniform_color([1, 0, 0])  # Red
    initial_action_cloud.paint_uniform_color([1, 1, 0])  # Yellow
    gt_action_cloud.paint_uniform_color([0, 1, 0])  # Green

    # Compute bounding box for dynamic camera positioning
    combined_cloud = action_cloud + anchor_cloud + initial_action_cloud + gt_action_cloud
    bounding_box = combined_cloud.get_axis_aligned_bounding_box()
    center = bounding_box.get_center()
    extent = bounding_box.get_extent()

    # Define camera positions for views
    camera_views = {
        "default": center + np.array([0, 0, max(extent) * 2.5]),
        "top": center + np.array([0, max(extent) * 2.5, 0]),
        "side": center + np.array([max(extent) * 2.5, 0, 0]),
        "diagonal": center + np.array([max(extent) * 2.5, max(extent) * 2.5, max(extent) * 2.5]),
    }


    # Render and save each view
    width, height = 1920, 1080  # Resolution
    for view in views_to_render:
        eye = camera_views[view]
        up = [0, 1, 0]

        # Create OffscreenRenderer
        scene = o3d.visualization.rendering.OffscreenRenderer(width, height)
        scene.scene.add_geometry("action", action_cloud, o3d.visualization.rendering.MaterialRecord())
        scene.scene.add_geometry("anchor", anchor_cloud, o3d.visualization.rendering.MaterialRecord())
        scene.scene.add_geometry("initial_action", initial_action_cloud, o3d.visualization.rendering.MaterialRecord())
        scene.scene.add_geometry("gt_action", gt_action_cloud, o3d.visualization.rendering.MaterialRecord())
        scene.scene.camera.look_at(center.tolist(), eye.tolist(), up)

        # Render the image and save it
        image = scene.render_to_image()
        save_path = os.path.join(save_dir, f"{view}_view.png")
        o3d.io.write_image(save_path, image)
        print(f"Saved {view} view visualization to {save_path}")

        # Explicitly clear the renderer to free up resources
        scene.scene.clear_geometry()
        del scene  # Ensure the renderer is destroyed

def move_to_cpu_nested(d):
    result = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.cpu()
        elif isinstance(v, dict):
            result[k] = move_to_cpu_nested(v)
        else:
            result[k] = v
    return result

def merge_and_move_to_cpu(batch, pred):
    return {
        "batch": move_to_cpu_nested(batch),
        "pred": move_to_cpu_nested(pred)
    }

@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval_cli", version_base="1.3")
def main(cfg):

    #####################################################################
    # Make sure we have in/out set before initializing inference
    #####################################################################

    data_in_path = cfg.data_in_path
    data_out_path = cfg.data_out_path
    assert data_in_path is not None
    assert data_out_path is not None

    vis_path = cfg.vis_path
    config_path = cfg.config_path

    #####################################################################
    # Fetch configs
    #####################################################################
    task_overrides = HydraConfig.get().overrides.task
    cfg = load_checkpoint_config_from_wandb(
        cfg, 
        task_overrides, 
        cfg.wandb.entity, 
        cfg.wandb.project, 
        cfg.checkpoint.run_id
    )

    ######################################################################
    # Manually setting eval-specific configs.
    ######################################################################
    cfg.dataset.num_demos = 1
    cfg.inference.batch_size = 1
    cfg.inference.num_wta_trials = 1
    cfg.inference.val_batch_size = 1
    cfg.inference.batch_size = 1

    
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
    # Load the GMM frame predictor, if necessary.
    ######################################################################
    if cfg.gmm is not None:
        # GMM can only be used with noisy goal models.
        if cfg.model.pred_frame != "noisy_goal":
            raise ValueError("GMM can only be used with noisy goal models.")

        # Construct the base search directory
        gmm_base_dir = os.path.join(os.path.expanduser(cfg.gmm_log_dir), f"{cfg.gmm_pcd_scale}")
        
        # Find subdirectory containing exp_name
        matched_dirs = [d for d in glob.glob(os.path.join(gmm_base_dir, "*")) if cfg.exp_name in os.path.basename(d)]
        if len(matched_dirs) == 0:
            raise ValueError(f"No GMM experiment directory containing '{cfg.exp_name}' found in {gmm_base_dir}.")
        elif len(matched_dirs) > 1:
            raise ValueError(f"Multiple matching directories found for '{cfg.exp_name}': {matched_dirs}")
        
        gmm_exp_name = matched_dirs[0]
        print(f"Found GMM experiment directory: {gmm_exp_name}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        gmm_yaml_path = os.path.join(script_dir, "..", "configs", "model", "df_cross.yaml")
        gmm_model_cfg = omegaconf.OmegaConf.load(os.path.abspath(gmm_yaml_path))

        # Load GMM model config and checkpoint
        #gmm_model_cfg = omegaconf.OmegaConf.load(os.path.join(hydra.utils.get_original_cwd(), "../configs/model/df_cross.yaml"))
        gmm_model_cfg.rel_pos = True
        gmm_model = FrameGMMPredictor(gmm_model_cfg, device)
        gmm_path = os.path.join(gmm_exp_name, "checkpoints", f"epoch_{cfg.gmm}.pt")
        gmm_model.load_state_dict(torch.load(gmm_path))
        print(f"Using GMM checkpoint: {gmm_path}")

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

    #################################################################################################################################

    # simple_eval(datamodule, model)
    # quit()

    #demo_fname = "/data/lyuxing/tax3d/rpdiff/data/task_demos/book_on_bookshelf_double_view_rnd_ori/task_name_book_in_bookshelf/preprocessed/demo_aug_19_0.npz"
    train_dataloader = datamodule.train_dataloader()
    val_dataloader, test_dataloader = datamodule.val_dataloader()

    test_dataloader.dataset.demo_files = [data_in_path]

    test_dataloader.dataset.set_eval_mode(True)
    val_dataloader.dataset.set_eval_mode(True)
    test_dataloader.dataset.set_eval_mode(True)

    num_samples = cfg.inference.num_wta_trials # // bs
    eval_keys = ["pc_action", "pc_anchor", "pc", "flow", "seg", "seg_anchor", "T_action2world", "T_goal2world"]
    if cfg.model.pred_frame == "noisy_goal":
        eval_keys.append("noisy_goal")

    assert len(test_dataloader.dataset) == 1
    for batch in tqdm(test_dataloader):
        # Generate predictions.
        if gmm_model is not None:
            # Yucky hack for GMM; need to expand point clouds first for WTA GMM samples.
            pred_batch = {key: expand_pcd(value, num_samples) for key, value in batch.items()}
            pred_batch = model.update_batch_frames(pred_batch, update_labels=True, gmm_model=gmm_model)
            batch = model.update_batch_frames(batch, update_labels=True)
            pred_dict = model.predict(pred_batch, 1, progress=False, full_prediction=True)
        else:
            pred_batch = model.update_batch_frames(batch, update_labels=True)
            pred_dict = model.predict(pred_batch, num_samples, progress=False, full_prediction=True)
        break   # there should only be one file

    #pred_dict = move_nested_to_cpu(pred_dict)
    #np.savez(data_out_path, **pred_dict)
    merged_dict = merge_and_move_to_cpu(pred_batch, pred_dict)
    np.savez(data_out_path, **merged_dict)

    if vis_path is not None:
        for i in range(batch["pc_anchor"].shape[0]):
            vis_dict = {
                "anchor_pc" : batch["pc_anchor"][i].to('cpu'),
                "action_pc" : pred_dict["point"]["pred"][i].to('cpu'),
                "initial_action_pc" : batch["pc_action"][i].to('cpu'),
                "gt_action_pc" : batch["pc"][i].to('cpu'),
            }
            visualize_point_cloud(vis_dict, vis_path)

    if config_path is not None:
        ful_hydra_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)  # resolves references
        full_hydra_fname = osp.join(config_path)
        json.dump(ful_hydra_dict, open(full_hydra_fname, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()









    '''
    ######################################################################    ######################################################################    ######################################################################
    ######################################################################
    # What Does Not Work #
    ######################################################################
    train_dataloader = datamodule.train_dataloader()
    val_dataloader, test_dataloader = datamodule.val_dataloader()

    dataset = train_dataloader.dataset
    #dataset.set_eval_mode(True)
    #demo_fname = "/home/lyuxing/Desktop/third_party/rpdiff/src/rpdiff/eval_data/eval_data/book_bookshelf/tax3dv2_fixed_kwcsrp5k_gmm1000_pfinetune_psf15.0/seed_0/data/demo_aug_0.npz"
    demo_fname = "/data/lyuxing/tax3d/rpdiff/data/task_demos/book_on_bookshelf_double_view_rnd_ori/task_name_book_in_bookshelf/preprocessed/demo_aug_19_0.npz"
    dataset.demo_files = [demo_fname]

    data_batch = dataset[0]
    data_batch = {k: v.unsqueeze(0) for k, v in data_batch.items()}
    num_samples = 1

    demo = np.load(demo_fname, allow_pickle=True)
    parent_start_pcd = demo['multi_obj_start_pcd'].item()['parent']
    child_start_pcd = demo['multi_obj_start_pcd'].item()['child']
    child_final_pcd = demo['multi_obj_final_pcd'].item()['child']
    
    if isinstance(child_final_pcd, np.ndarray):
        parent_start_pcd = torch.from_numpy(demo['multi_obj_start_pcd'].item()['parent'])
        child_start_pcd = torch.from_numpy(demo['multi_obj_start_pcd'].item()['child'])
        child_final_pcd = torch.from_numpy(demo['multi_obj_final_pcd'].item()['child'])
    
    parent_start_pcd, anchor_pc_indices = downsample_pcd(parent_start_pcd.unsqueeze(0), 1024, type='fps')
    child_start_pcd, action_pc_indices = downsample_pcd(child_start_pcd.unsqueeze(0), 512, type='fps')
    child_final_pcd, action_pc_indices = downsample_pcd(child_final_pcd.unsqueeze(0), 512, type='fps')

    if gmm_model is not None:
        # Yucky hack for GMM; need to expand point clouds first for WTA GMM samples.
        #gmm_batch = tax3dv2_model.update_batch_frames(data_batch, update_labels=False, gmm_model=gmm_model)
        gmm_batch = model.update_batch_frames(data_batch, update_labels=True, gmm_model=None)
        gmm_batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in gmm_batch.items()}    # prib dont need this?
        pred_dict = model.predict(gmm_batch, num_samples, progress=True, full_prediction=True)
    else:
        data_batch = model.update_batch_frames(data_batch, update_labels=False, gmm_model=None)
        data_batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in data_batch.items()}  # prob dont need this?
        pred_dict = model.predict(data_batch, num_samples, progress=True, full_prediction=True)

    # vis
    for i in range(data_batch["pc_anchor"].shape[0]):
        vis_dict = {
            "anchor_pc" : data_batch["pc_anchor"][i].to('cpu'),
            "action_pc" : pred_dict["point"]["pred"][i].to('cpu'),
            "initial_action_pc" : data_batch["pc_action"][i].to('cpu'),
            "gt_action_pc" : data_batch["pc"][i].to('cpu'),
        }
        visualize_point_cloud(vis_dict, i, save_dir=f"/home/lyuxing/Desktop/tax3d_upgrade/scripts/vis/bookbookshelf/toy")
    '''
    '''
    ######################################################################    ######################################################################    ######################################################################

    ######################################################################
    # What Works #
    ######################################################################
    def run_eval(dataloader, model):
        # for RPDiff tasks, we also need to take care of the scaling issues
        dataset = dataloader.dataset
        scaling_factor = dataset.dataset_cfg.pcd_scale_factor
        stage = dataset.type

        num_samples = cfg.inference.num_wta_trials # // bs
        eval_keys = ["pc_action", "pc_anchor", "pc", "flow", "seg", "seg_anchor", "T_action2world", "T_goal2world"]
        if cfg.model.pred_frame == "noisy_goal":
            eval_keys.append("noisy_goal")
            
        rmse_list = []
        rmse_wta_list = []
        t_err_list = []
        r_err_list = []
        t_err_wta_list = []
        r_err_wta_list = []

        # coverage and precision metrics are computed in rpdiff 
        #coverage = []
        #precision = []

        for batch in tqdm(dataloader):

            # Generate predictions.
            if gmm_model is not None:
                # Yucky hack for GMM; need to expand point clouds first for WTA GMM samples.
                gmm_batch = {key: expand_pcd(value, num_samples) for key, value in batch.items()}
                gmm_batch = model.update_batch_frames(gmm_batch, update_labels=True, gmm_model=gmm_model)
                batch = model.update_batch_frames(batch, update_labels=True)
                pred_dict = model.predict(gmm_batch, 1, progress=False, full_prediction=True)
            else:
                batch = model.update_batch_frames(batch, update_labels=True)
                pred_dict = model.predict(batch, num_samples, progress=False, full_prediction=True)

            seg = batch["seg"].to(device)
            ground_truth_point_world = batch["pc_world"].to(device)
            bs = ground_truth_point_world.shape[0]

            seg = expand_pcd(seg, num_samples)
            ground_truth_point_world = expand_pcd(ground_truth_point_world, num_samples)

            pred_point_world = pred_dict["point"]["pred_world"]
            pred_point_world_scaled = pred_point_world / scaling_factor
            ground_truth_point_world_scaled = ground_truth_point_world / scaling_factor

            # retrieve predicted translation and rotation (note that these are already properly scaled!)
            # not used here, simply for test usages
            pred_T = pred_dict["pred_T"]

            # computing error metrics
            seg = seg == 0

            rmse = flow_rmse(pred_point_world_scaled, ground_truth_point_world_scaled, mask=True, seg=seg).reshape(bs, num_samples)
            pred_point_world = pred_point_world.reshape(bs, num_samples, -1, 3)

            # computing winner-take-all metrics
            winner = torch.argmin(rmse, dim=-1)
            rmse_wta = rmse[torch.arange(bs), winner]

            translation_errs, rotation_errs = svd_estimation(source=pred_point_world_scaled.reshape(bs * num_samples, -1, 3), target=ground_truth_point_world_scaled, return_magnitude=True)

            translation_errs = translation_errs.reshape(bs, num_samples)
            rotation_errs = rotation_errs.reshape(bs, num_samples)

            trans_winner = torch.argmin(translation_errs, dim=-1)
            rot_winner = torch.argmin(rotation_errs, dim=-1)

            translation_err_wta = translation_errs[torch.arange(bs), trans_winner]
            rotation_err_wta = rotation_errs[torch.arange(bs), rot_winner]


            rmse_list.append(rmse.mean().item())
            rmse_wta_list.append(rmse_wta.mean().item())
            t_err_list.append(translation_errs.mean().item())
            t_err_wta_list.append(translation_err_wta.mean().item())
            r_err_list.append(rotation_errs.mean().item())
            r_err_wta_list.append(rotation_err_wta.mean().item())

            # vis
            for i in range(batch["pc_anchor"].shape[0]):
                fname = batch["finename"]
                print(f"visualizing file: {fname}")
                vis_dict = {
                    "anchor_pc" : batch["pc_anchor"][i],
                    "action_pc" : pred_dict["point"]["pred"][i].to('cpu'),
                    "initial_action_pc" : batch["pc_action"][i],
                    "gt_action_pc" : batch["pc"][i],
                }
                visualize_point_cloud(vis_dict, i, save_dir=f"/home/lyuxing/Desktop/tax3d_upgrade/scripts/vis/bookbookshelf/{stage}")

            break

        rmse_ = np.mean(rmse_list)
        rmse_wta_ = np.mean(rmse_wta_list)
        t_err_ = np.mean(t_err_list)
        t_err_wta_ = np.mean(t_err_wta_list)
        r_err_ = np.mean(r_err_list)
        r_err_wta_ = np.mean(r_err_wta_list)

        return rmse_, rmse_wta_, t_err_, t_err_wta_, r_err_, r_err_wta_
    

    ######################################################################
    # Run the model on the train/val/test sets.
    ######################################################################
    model.to(device)

    # simple_eval(datamodule, model)
    # quit()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader, test_dataloader = datamodule.val_dataloader()

    train_dataloader.dataset.set_eval_mode(True)
    val_dataloader.dataset.set_eval_mode(True)
    test_dataloader.dataset.set_eval_mode(True)


    train_rmse, train_rmse_wta, train_t_err, train_t_err_wta, train_r_err, train_r_err_wta = run_eval(train_dataloader, model)
    val_rmse, val_rmse_wta, val_t_err, val_t_err_wta, val_r_err, val_r_err_wta = run_eval(val_dataloader, model)
    test_rmse, test_rmse_wta, test_t_err, test_t_err_wta, test_r_err, test_r_err_wta = run_eval(test_dataloader, model)

    print(f"Train RMSE: {train_rmse}, Train RMSE_WTA: {train_rmse_wta}, Train T_err: {train_t_err}, Train T_err_WTA: {train_t_err_wta}, Train R_err: {train_r_err}, Train R_err_WTA: {train_r_err_wta}")
    print(f"Val RMSE: {val_rmse}, Val RMSE_WTA: {val_rmse_wta}, Val T_err: {val_t_err}, Val T_err_WTA: {val_t_err_wta}, Val R_err: {val_r_err}, Val R_err_WTA: {val_r_err_wta}")
    print(f"Test RMSE: {test_rmse}, Test RMSE_WTA: {test_rmse_wta}, Test T_err: {test_t_err}, Test T_err_WTA: {test_t_err_wta}, Test R_err: {test_r_err}, Test R_err_WTA: {test_r_err_wta}")
    '''