import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
import json
import omegaconf
import torch
import wandb
from pytorch3d.transforms import Transform3d, Translate

from non_rigid.utils.script_utils import (
    create_model,
    create_datamodule,
    load_checkpoint_config_from_wandb,
)

from non_rigid.nets.dgcnn import DGCNN
from non_rigid.metrics.flow_metrics import flow_rmse
from non_rigid.utils.pointcloud_utils import expand_pcd
from non_rigid.utils.vis_utils import visualize_sampled_predictions, visualize_diffusion_timelapse
from tqdm import tqdm
import numpy as np

import rpad.visualize_3d.plots as vpl
from plotly import graph_objects as go

# def visualize_batched_point_clouds(point_clouds):
#     """
#     Helper function to visualize a list of batched point clouds. This is meant to be used
#     when visualizing action/anchor/prediction point clouds, without having to add

#     point_clouds: list of point clouds, each of shape (B, N, 3)
#     """
#     pcs = [pc.cpu().flatten(0, 1) for pc in point_clouds]
#     segs = []
#     for i, pc in enumerate(pcs):
#         segs.append(torch.ones(pc.shape[0]).int() * i)

#     return vpl.segmentation_fig(
#         torch.cat(pcs),
#         torch.cat(segs),
#     )

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

    ######################################################################
    # Load the reference frame predictor, if necessary.
    ######################################################################


    if cfg.use_gmm:
        # gmm can only be used with oracle models
        if not cfg.model.oracle and not cfg.model.tax3dv2:
            raise ValueError("GMM can only be used with oracle models or TAX3Dv2 models.")

        # cannot predict and diffuse reference frame together
        if cfg.model.diffuse_ref_frame and not cfg.model.tax3dv2:
            raise ValueError("Can only predict and diffuse reference frame together for TAX3Dv2.")

        import torch_geometric.data as tgd
        import os

        # ref_frame_predictor = FramePredictorDGCNN(5)
        ref_frame_predictor = None
        checkpoint_dir = os.path.expanduser("~/non-rigid-robot/notebooks/gmm_init/checkpoints/")
        gmm_ckpt = torch.load(checkpoint_dir + "model_1000.pt", map_location=device)

        ref_frame_predictor.load_state_dict(gmm_ckpt)
        ref_frame_predictor.eval()
        ref_frame_predictor.to(device)

        # update the dataset_cfg if needed
        cfg.dataset.oracle = False
        cfg.model.oracle = False

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
            batch = [{key: item[key].to(device) for key in eval_keys}]
            batch = {key: torch.stack([item[key] for item in batch]) for key in eval_keys}

            # Generate predictions.
            if cfg.use_gmm:
                raise NotImplementedError("GMM not implemented yet.")
                # expand action and anchor point clouds
                gmm_pc = expand_pcd(batch["pc"], num_samples)
                gmm_action = expand_pcd(batch["pc_action"], num_samples)
                gmm_anchor = expand_pcd(batch["pc_anchor"], num_samples)
                gmm_batch = tgd.Batch.from_data_list([
                    tgd.Data(pos=gmm_anchor[i]) for i in range(num_samples)
                ]).to(device)

                # sample reference frames
                gmm_pred = ref_frame_predictor(gmm_batch)
                gmm_probs, gmm_means = gmm_pred["probs"], gmm_pred["means"]
                idxs = torch.multinomial(gmm_probs.squeeze(-1), 1).squeeze()
                sampled_ref_frames = gmm_means[torch.arange(num_samples), idxs].unsqueeze(-2)
                sampled_ref_frames = sampled_ref_frames.cpu()                
            else:
                pass

            batch = model.update_batch_frames(batch, update_labels=True)
            pred_dict = model.predict(batch, num_samples, progress=False, full_prediction=True)
            viz_args = model.get_viz_args(batch, 0)

            # Get point clouds in world coordinates.
            pred_pc_world = pred_dict["point"]["pred_world"].cpu().numpy()
            gt_pc_world = viz_args["pc_pos_viz"].cpu().numpy()
            action_pc_world = viz_args["pc_action_viz"].cpu().numpy()
            anchor_pc_world = viz_args["pc_anchor_viz"].cpu().numpy()

            # visualize sampled predictions
            context = {
                "Action": action_pc_world,
                "Anchor": anchor_pc_world,
            }
            # # For TAX3Dv2, also visualize reference frame predictions.
            # if cfg.model.tax3dv2:
            #     context["Ref Frame"] = pred_dict["ref_frame_world"].squeeze().cpu().numpy()
            fig = visualize_sampled_predictions(
                ground_truth = gt_pc_world,
                context = context,
                predictions = pred_pc_world,
            )
            fig.show()

            # visualize diffusion timelapse
            VIZ_IDX = 4 # 0
            results = [res[VIZ_IDX].cpu().numpy() for res in pred_dict["results_world"]]
            # For TAX3Dv2, also grab logit/residual predictions.
            # if cfg.model.tax3dv2:
            #     extras = [ext[VIZ_IDX].cpu().numpy() for ext in pred_dict["extras"]]
                # ref_frame_results = [ref_frame_res[VIZ_IDX].cpu().numpy() for ref_frame_res in pred_dict["ref_frame_results_world"]]
            # else:
            extras = None
            ref_frame_results = None
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

    ######################################################################
    # Run the model on the train/val/test sets.
    ######################################################################
    train_indices = []
    val_indices = [0, 2, 6]
    val_ood_indices = []
    model.to(device)
    run_vis(datamodule.train_dataset, model, train_indices)
    run_vis(datamodule.val_dataset, model, val_indices)
    run_vis(datamodule.val_ood_dataset, model, val_ood_indices)


if __name__ == "__main__":
    main()