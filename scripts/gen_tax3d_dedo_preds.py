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
from non_rigid.utils.pointcloud_utils import expand_pcd, downsample_pcd
from non_rigid.models.gmm_predictor import FrameGMMPredictor
from tqdm import tqdm
import numpy as np

import rpad.visualize_3d.plots as vpl
from pytorch3d.transforms import Transform3d



@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval", version_base="1.3")
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
    # # Using a custom cloth-specific batch size, to allow for simultaneous evaluation 
    # # of RMSE, coverage, and precision.
    # if cfg.dataset.hole == "single":
    #     bs = 1
    # elif cfg.dataset.hole == "double":
    #     bs = 2
    # else:
    #     raise ValueError(f"Unknown hole type: {cfg.dataset.hole}.")
    # bs *= cfg.dataset.num_anchors

    # cfg.inference.batch_size = bs
    # cfg.inference.val_batch_size = bs
    # We're not computing evals, only generating and saving predictions, so 
    # one-by-one is fine.
    cfg.inference.batch_size = 1
    cfg.inference.val_batch_size = 1
    cfg.dataset.sample_size_action = -1

    ######################################################################
    # Load the GMM frame predictor, if necessary.
    ######################################################################
    if cfg.gmm is not None:
        # GMM can only be used with noisy goal models.
        if cfg.model.pred_frame != "noisy_goal":
            raise ValueError("GMM can only be used with noisy goal models.")
        
        # Checking for GMM model directory.
        gmm_exp_name = os.path.join(os.path.expanduser(cfg.gmm_log_dir), f"gmm_{cfg.dataset.name}_{cfg.gmm}")
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
        gmm_model.eval()
    else:
        gmm_model = None

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
    else:
        ckpt_file = checkpoint_reference
    # Load the network weights.
    ckpt = torch.load(ckpt_file, map_location=device)
    network.load_state_dict(
        {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items() if k.startswith("network.")}
    )
    # set model to eval mode
    network.eval()
    model.eval()

    ######################################################################
    # Helper function to sample and save predictions for goal-
    # conditioned policy evaluations.
    ######################################################################
    def generate_preds(dataset, save_path):
        # batch_keys = ["pc_action", "pc_anchor", "pc", "flow", "seg", "seg_anchor", "T_action2world", "T_goal2world"]
        batch_keys = ["pc_action", "pc_anchor", "pc", "T_action2world", "T_goal2world"]
        for i in tqdm(range(len(dataset))):
            item = dataset.__getitem__(i)
            batch = {key: item[key].unsqueeze(0) for key in batch_keys}

            # getting cloth-specific gripper points, and remaining action points
            deform_data = item["deform_data"]
            pc_action = batch["pc_action"]
            gripper_indices = [0, deform_data["deform_params"]["node_density"] - 1]
            gripper_points = pc_action[:, gripper_indices, :]
            non_gripper_points = torch.cat([
                pc_action[:, 1:gripper_indices[1], :],
                pc_action[:, gripper_indices[1]+1:, :]
            ], dim=1)

            # fps downsampling on non-gripper points
            non_gripper_points, non_gripper_indices = downsample_pcd(non_gripper_points, 510, type="fps")

            # concatenating gripper points with downsampled non-gripper points
            batch["pc_action"] = torch.cat([gripper_points, non_gripper_points], dim=1)

            # sampling model predictions
            if gmm_model is not None:
                gmm_batch = {key: expand_pcd(value, 10) for key, value in batch.items()}
                gmm_batch = model.update_batch_frames(gmm_batch, update_labels=False, gmm_model=gmm_model)
                pred_dict = model.predict(gmm_batch, 1, progress=False, full_prediction=True)
            else:
                batch = model.update_batch_frames(batch, update_labels=False)
                pred_dict = model.predict(batch, 10, progress=False, full_prediction=True)
            pred_point_world = pred_dict["point"]["pred_world"]
            results_world = pred_dict["results_world"]

            # updating indices for visualization
            non_gripper_indices += 1
            non_gripper_indices[non_gripper_indices >= gripper_indices[1]] += 1

            action_indices = np.concatenate(
                [np.array(gripper_indices), non_gripper_indices.squeeze().cpu().numpy()]
            )
            save_dict = {
                "pred_point_world": pred_point_world.cpu().numpy(),
                "results_world": torch.stack(results_world).cpu().numpy(),
                "action_indices": action_indices,
            }
            np.savez(
                save_path / f"pred_{i}.npz",
                **save_dict,
            )

    ######################################################################
    # Generate predictions on the train/val sets.
    ######################################################################
    # Creating directory for model predictions.
    pred_dir = datamodule.root / "tax3d_preds" / cfg.checkpoint.run_id
    if os.path.exists(pred_dir):
        print(f"Prediction directory {pred_dir} already exists. Delete first.")
        quit()
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(pred_dir / "train", exist_ok=True)
    os.makedirs(pred_dir / "val", exist_ok=True)

    model.to(device)
    generate_preds(datamodule.train_dataset, pred_dir / "train")
    generate_preds(datamodule.val_dataset, pred_dir / "val")

if __name__ == "__main__":
    main()