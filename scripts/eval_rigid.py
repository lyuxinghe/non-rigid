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

    ######################################################################
    # Helper function to run evals for a given dataset.
    ######################################################################
    def run_eval(dataloader, model):
        # for RPDiff tasks, we also need to take care of the scaling issues
        dataset = dataloader.dataset

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

        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Generate predictions.
                if gmm_model is not None:
                    # Yucky hack for GMM; need to expand point clouds first for WTA GMM samples.
                    pred_batch = model.update_batch_frames(batch, update_labels=True, gmm_model=gmm_model, num_gmm_trials=cfg.inference.num_gmm_trials)
                    pred_dict = model.predict(pred_batch, cfg.inference.num_wta_trials, progress=True, full_prediction=True)
                else:
                    pred_batch = model.update_batch_frames(batch, update_labels=True)
                    pred_dict = model.predict(pred_batch, cfg.inference.num_wta_trials, progress=True, full_prediction=True)

                seg = pred_batch["seg"].to(device)
                ground_truth_point_world = pred_batch["pc_world"].to(device)

                bs = ground_truth_point_world.shape[0]
                seg = expand_pcd(seg, num_samples)
                ground_truth_point_world = expand_pcd(ground_truth_point_world, num_samples)


                # pred = pred_dict[self.prediction_type]["pred"]
                pred_point_world = pred_dict["point"]["pred_world"]

                pred_point_world_clone = pred_point_world.clone()
                ground_truth_point_world_clone = ground_truth_point_world.clone()

                # computing error metrics
                seg = seg == 0
                breakpoint()
                rmse = flow_rmse(pred_point_world_clone, ground_truth_point_world_clone, mask=True, seg=seg).reshape(bs, num_samples)
                pred_point_world = pred_point_world.reshape(bs, num_samples, -1, 3)

                # computing winner-take-all metrics
                winner = torch.argmin(rmse, dim=-1)
                rmse_wta = rmse[torch.arange(bs), winner]

                translation_errs, rotation_errs = svd_estimation(source=pred_point_world_clone, target=ground_truth_point_world_clone, return_magnitude=True)

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

if __name__ == "__main__":
    main()