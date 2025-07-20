import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
import json
import omegaconf
import torch
import wandb
import os
from pytorch3d.transforms import Transform3d, Translate

from non_rigid.utils.script_utils import (
    create_model,
    create_datamodule,
    load_checkpoint_config_from_wandb,
)

from non_rigid.utils.pointcloud_utils import expand_pcd
from non_rigid.models.gmm_predictor import FrameGMMPredictor
from non_rigid.utils.vis_utils import visualize_sampled_predictions, visualize_diffusion_timelapse, visualize_multimodality
from tqdm import tqdm
import numpy as np

import rpad.visualize_3d.plots as vpl
from plotly import graph_objects as go

@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval_deform", version_base="1.3")
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

    ######################################################################
    # Load the reference frame predictor, if necessary.
    ######################################################################
    if cfg.gmm is not None:
        # GMM can only be used with noisy goal models.
        if cfg.model.pred_frame != "noisy_goal":
            raise ValueError("GMM can only be used with noisy goal models.")
        
        # Grab config file from saved run.
        exp_name = os.path.join(
            os.path.expanduser(cfg.gmm_log_dir),
            cfg.gmm,
        )
        if not os.path.exists(exp_name):
            raise ValueError(f"Experiment directory {exp_name} does not exist - train this model first.")
        gmm_cfg = omegaconf.OmegaConf.load(os.path.join(exp_name, "config.yaml"))
        
        # Creating GMM network.
        network, _ = create_model(gmm_cfg)
        gmm_model = FrameGMMPredictor(network, gmm_cfg.model, device)
        gmm_model.load_state_dict(
            torch.load(
                os.path.join(exp_name, "checkpoints", f"epoch_{gmm_cfg.epochs}.pt")
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
    # Helper function to run evals for a given dataset.
    ######################################################################
    def run_vis(dataset, model, indices):
        num_samples = cfg.inference.num_wta_trials
        eval_keys = ["pc_action", "pc_anchor", "pc", "flow", "seg", "seg_anchor", "T_action2world", "T_goal2world"]
        if cfg.model.pred_frame == "noisy_goal":
            eval_keys.append("noisy_goal")

        for i in tqdm(indices):
            # Index and batchify item.
            item = dataset[i]
            batch = [{key: item[key] for key in eval_keys}]
            batch = {key: torch.stack([item[key] for item in batch]) for key in eval_keys}

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

            #batch = model.update_batch_frames(batch, update_labels=True)
            #pred_dict = model.predict(batch, num_samples, progress=False, full_prediction=True)

            # for j in range(101):
            #     frame = pred_dict["results_r"][j][4]
            #     shape = pred_dict["results_s"][j][4]
            #     print(torch.norm(frame), torch.norm(torch.mean(shape, dim=1)))
            #     # print(frame)
            # breakpoint()
            batch = {key: value.to(device) for key, value in batch.items()}

            viz_args = model.get_viz_args(batch, 0)

            # Get point clouds in world coordinates.
            pred_pc_world = pred_dict["point"]["pred_world"].cpu().numpy()
            gt_pc_world = viz_args["pc_pos_viz"].cpu().numpy()
            action_pc_world = viz_args["pc_action_viz"].cpu().numpy()
            anchor_pc_world = viz_args["pc_anchor_viz"].cpu().numpy()

            # Another hack; if scene-as-anchor, revert back to just anchor.
            if cfg.model.scene_anchor:
                anchor_pc_world = anchor_pc_world[:anchor_pc_world.shape[0] // 2, :]

            # Create GMM viz args, if needed.
            if gmm_model is not None:
                gmm_viz = {
                    "gmm_probs": gmm_batch["gmm_probs"].cpu().numpy(),
                    "gmm_means": gmm_batch["gmm_means"].cpu().numpy(),
                    "sampled_idxs": gmm_batch["sampled_idxs"].cpu().numpy(),
                }
            else:
                gmm_viz = None

            # visualize sampled predictions
            context = {
                "Action": action_pc_world,
                "Anchor": anchor_pc_world,
            }
            fig = visualize_sampled_predictions(
                ground_truth = gt_pc_world,
                context = context,
                predictions = pred_pc_world,
                gmm_viz = gmm_viz,
            )
            # fig.show()
            # break

            # Visualize diffusion timelapse.
            VIZ_IDX = 4 # 0
            results = [res[VIZ_IDX].cpu().numpy() for res in pred_dict["results_world"]]
            # For TAX3Dv2, also grab logit/residual predictions.
            # if cfg.model.tax3dv2:
            #     extras = [ext[VIZ_IDX].cpu().numpy() for ext in pred_dict["extras"]]
                # ref_frame_results = [ref_frame_res[VIZ_IDX].cpu().numpy() for ref_frame_res in pred_dict["ref_frame_results_world"]]
            # else:
            extras = None
            ref_frame_results = [ref_res[VIZ_IDX].cpu().numpy() for ref_res in pred_dict["results_r_world"]]
            fig = visualize_diffusion_timelapse(
                context = {
                    "Action": action_pc_world,
                    "Anchor": anchor_pc_world,
                },
                results = results,
                extras = extras,
                ref_frame_results = ref_frame_results,
            )
            fig.show()
            break

            # Create multi-modal gif.
            gif_results = [res.cpu().numpy() for res in pred_dict["results_world"]]
            fig = visualize_multimodality(
                context = {
                    "Anchor": anchor_pc_world,
                },
                predictions = pred_pc_world,
                results = gif_results,
                # indices = [0, 1],
                # indices = [0, 3, 9, 13], # 8, double-hole
                # indices = [0, 1, 3, 9], # 13, double-hole
                # indices = [0, 1, 2, 5], # 0, double-hole
                # indices = [0, 6, 8, 19], # 64, double-hole
                # indices = [0, 1, 3, 7], # 0, single-hole
                # indices = [0, 1, 2, 5], # 3, single-hole
                # indices = [0, 1, 3, 5], # 6, single-hole
                indices = [0, 1, 2, 4], # 33, single-hole
                gif_path = os.path.expanduser("~/data/multimodal_viz"),
            )
            # fig.show()


    ######################################################################
    # Run the model on the train/val/test sets.
    ######################################################################
    train_indices = []
    val_indices = [33] # single-hole: 3, 0, 6, 33
    val_ood_indices = []
    model.to(device)
    run_vis(datamodule.train_dataset, model, train_indices)
    run_vis(datamodule.val_dataset, model, val_indices)
    run_vis(datamodule.val_ood_dataset, model, val_ood_indices)


if __name__ == "__main__":
    main()