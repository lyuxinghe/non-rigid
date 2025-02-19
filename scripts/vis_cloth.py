import hydra
import lightning as L
import json
import omegaconf
import torch
import torch.utils._pytree as pytree
import wandb

from pathlib import Path
import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from non_rigid.datasets.proc_cloth_flow import ProcClothFlowDataset, ProcClothFlowDataModule
from non_rigid.models.df_base import DiffusionFlowBase, FlowPredictionInferenceModule, PointPredictionInferenceModule
from non_rigid.utils.vis_utils import FlowNetAnimation
from non_rigid.utils.script_utils import (
    PROJECT_ROOT,
    LogPredictionSamplesCallback,
    create_model,
    create_datamodule,
    match_fn,
    flatten_outputs
)

from non_rigid.metrics.flow_metrics import flow_cos_sim, flow_rmse
from non_rigid.models.dit.diffusion import create_diffusion
from non_rigid.utils.pointcloud_utils import expand_pcd
from tqdm import tqdm
import numpy as np

from pytorch3d.transforms import Transform3d, Rotate, Translate, euler_angles_to_matrix
import rpad.visualize_3d.plots as vpl



def visualize_batched_point_clouds(point_clouds):
    """
    Helper function to visualize a list of batched point clouds. This is meant to be used 
    when visualizing action/anchor/prediction point clouds, without having to add 

    point_clouds: list of point clouds, each of shape (B, N, 3)
    """
    pcs = [pc.cpu().flatten(0, 1) for pc in point_clouds]
    segs = []
    for i, pc in enumerate(pcs):
        segs.append(torch.ones(pc.shape[0]).int() * i)

    return vpl.segmentation_fig(
        torch.cat(pcs),
        torch.cat(segs),
    )




@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval", version_base="1.3")
def main(cfg):
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
    # Create the datamodule.
    # Should be the same one as in training, but we're gonna use val+test
    # dataloaders.
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
        ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    else:
        ckpt_file = checkpoint_reference
    # Load the network weights.
    ckpt = torch.load(ckpt_file, map_location=device)
    network.load_state_dict(
        {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
    )
    # set model to eval mode
    network.eval()
    model.eval()

    if cfg.model.name == "df_base":
        raise NotImplementedError("DF Base model is not supported for visualization.")


    VISUALIZE_DEMOS = False
    VISUALIZE_PREDS = False
    VISUALIZE_SINGLE = False
    VISUALIZE_PULL = False
    VISUALIZE_DATASET = False
    VISUALIZE_CUSTOM = True
    ######################################################################
    # Run the model on the train/val/test sets.
    # This outputs a list of dictionaries, one for each batch. This
    # is annoying to work with, so later we'll flatten.
    #
    # If a downstream eval, you can swap it out with whatever the eval
    # function is.
    ######################################################################


    # TODO: Remove viz from config

    if VISUALIZE_DATASET:
        # dataset = "train"
        world_frame = True
        model.to(device)
        dataloader = torch.utils.data.DataLoader(
            datamodule.train_dataset, batch_size=1, shuffle=False
        )

        goal_pc = [] # list of goal point clouds
        anchor_pc = [] # list of anchor point clouds
        action_pc = [] # list of action point clouds
        goal_seg = [] # list of goal segmentations
        for i, batch in enumerate(dataloader):
            if i >= datamodule.train_dataset.num_demos:
                break
            # don't model predict yet
            goal = batch["pc"]
            anchor = batch["pc_anchor"]
            action = batch["pc_action"]
            print(goal.shape, anchor.shape, batch["rot"])

            rot = batch['rot']
            trans = batch['trans']
            # reverting transforms
            T_world2origin = Translate(trans).inverse().compose(
                Rotate(euler_angles_to_matrix(rot, 'XYZ'))
            )
            T_goal2world = Transform3d(
                matrix=batch["T_goal2world"]
            )
            T_action2world = Transform3d(
                matrix=batch["T_action2world"]
            )
            goal = T_goal2world.transform_points(goal)
            anchor = T_goal2world.transform_points(anchor)
            action = T_action2world.transform_points(action)

            if not world_frame:
                goal = T_world2origin.transform_points(goal)
                anchor = T_world2origin.transform_points(anchor)

            # TODO: if not world frame, revert transforms
            goal_pc.append(goal.squeeze(0))
            anchor_pc.append(anchor.squeeze(0))
            action_pc.append(action.squeeze(0))
            goal_seg.append(torch.ones(goal.shape[1]).int() * (i + 1))

        goal_pc = torch.cat(goal_pc, dim=0).cpu().numpy()
        anchor_pc = torch.cat(anchor_pc, dim=0).cpu().numpy()
        action_pc = torch.cat(action_pc, dim=0).cpu().numpy()
        print(goal_pc.shape, anchor_pc.shape, action_pc.shape)

        # goal_seg = np.ones(goal_pc.shape[0]).astype(int)
        goal_seg = torch.cat(goal_seg, dim=0).cpu().numpy()
        anchor_seg = np.zeros(anchor_pc.shape[0]).astype(int)
        # action_seg = np.zeros(action_pc.shape[0]).astype(int)
        action_seg = goal_seg.copy()

        vpl.segmentation_fig(
            np.concatenate((goal_pc, anchor_pc, action_pc)),
            np.concatenate((goal_seg, anchor_seg, action_seg)),
        ).show()

    if VISUALIZE_PREDS:
        num_preds = 5
        num_samples = 1
        model.to(device)
        dataloader = torch.utils.data.DataLoader(
            datamodule.val_dataset, batch_size=1, shuffle=False
        )
        iterator = iter(dataloader)

        for _ in range(num_preds):
            batch = next(iterator)
            pred_dict = model.predict(batch, num_samples)
            # extracting anchor point cloud depending on model type
            goal_pc = batch["pc"].flatten(0, 1).cpu().numpy()
            anchor_pc = batch["pc_anchor"].flatten(0, 1).cpu().numpy()

            # pred_action = pred_dict["pred_action"][[8]] # 0,8
            pred_action = pred_dict["pred_action"]
            pred_action_size = pred_action.shape[1]
            pred_action = pred_action.flatten(0, 1).cpu().numpy()
            # color-coded segmentations
            anchor_seg = np.zeros(anchor_pc.shape[0], dtype=np.int64)
            # if cfg.model.type == "flow":
            #     pred_action_size = cfg.dataset.sample_size_action + cfg.dataset.sample_size_anchor
            # else:
            #     pred_action_size = cfg.dataset.sample_size_action
            goal_seg = np.ones(goal_pc.shape[0], dtype=np.int64)
            pred_action_seg = np.array([np.arange(2, 2 + num_samples)] * pred_action_size).T.flatten()
            # visualize
            fig = vpl.segmentation_fig(
                np.concatenate((anchor_pc, pred_action, goal_pc)),
                np.concatenate((anchor_seg, pred_action_seg, goal_seg)),
            )
            fig.show()
        

    if VISUALIZE_DEMOS:
        model.to(device)
        bs = 12
        train_dataloader = torch.utils.data.DataLoader(
            datamodule.train_dataset, batch_size=400, shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            datamodule.val_dataset, batch_size=40, shuffle=True
        )
        val_ood_loader = torch.utils.data.DataLoader(
            datamodule.val_ood_dataset, batch_size=40, shuffle=True
        )

        train_batch = next(iter(train_dataloader))
        val_batch = next(iter(val_dataloader))
        val_ood_batch = next(iter(val_ood_loader))


        # train_dict = model.predict_wta(train_batch, 'train')
        # val_dict = model.predict_wta(val_batch, 'val')
        # val_ood_dict = model.predict_wta(val_ood_batch, 'val_ood')

        # val_errors = val_dict['rmse']
        # val_ood_errors = val_ood_dict['rmse']


        cdw_errs = np.load('/home/eycai/datasets/nrp/cd-w.npz')
        cd_errs = np.load('/home/eycai/datasets/nrp/tax3dcd.npz')

        vem_cdw = cdw_errs['vem']
        voem_cdw = cdw_errs['voem']
        vem_cd = cd_errs['vem']
        voem_cd = cd_errs['voem']



        val_errors = np.random.rand(40)
        val_ood_errors = np.random.rand(40) * 4

        train_pc = train_batch["pc_anchor"]
        val_pc = val_batch["pc_anchor"]
        val_ood_pc = val_ood_batch["pc_anchor"]


        train_locs = torch.mean(train_pc, dim=1)
        val_locs = torch.mean(val_pc, dim=1)
        val_ood_locs = torch.mean(val_ood_pc, dim=1)
        # plotly go to scatter plot locs
        fig = go.Figure()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            row_heights=[0.5, 0.5],
                            subplot_titles=("CD-W", "TAX3D-CP (Ours)"))

        # ----------- PLOTTING X VS Y ----------------
        fig.add_trace(go.Scatter(
            x=train_locs[:, 0].cpu(),
            y=train_locs[:, 1].cpu(),
            mode='markers',
            marker_symbol='x-thin',
            marker=dict(
                size=20,
                color='rgb(38,133,249)',
                line=dict(
                    width=4,
                    color='rgb(38,133,249)'
                ),
            ),
            name='Train',
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=val_locs[:, 0].cpu(),
            y=val_locs[:, 1].cpu(),
            mode='markers',
            marker_symbol='square',
            marker=dict(
                size=20,
                color=vem_cdw,
                coloraxis='coloraxis',
                line=dict(
                    width=2,
                    color='Black'
                ),
            ),
            name='Unseen',
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=val_ood_locs[:, 0].cpu(),
            y=val_ood_locs[:, 1].cpu(),
            mode='markers',
            marker_symbol='diamond',
            marker=dict(
                size=20,
                color=voem_cdw,
                coloraxis='coloraxis',
                line=dict(
                    width=2,
                    color='Black'
                ),
            ),
            name='Unseen (OOD)',
        ), row=1, col=1)


        # ----------- PLOTTING X VS Z ----------------
        fig.add_trace(go.Scatter(
            x=train_locs[:, 0].cpu(),
            y=train_locs[:, 1].cpu(),
            mode='markers',
            marker_symbol='x-thin',
            marker=dict(
                size=20,
                color='rgb(38,133,249)',
                line=dict(
                    width=4,
                    color='rgb(38,133,249)'
                ),
            ),
            name='Train',
            showlegend=False,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=val_locs[:, 0].cpu(),
            y=val_locs[:, 1].cpu(),
            mode='markers',
            marker_symbol='square',
            marker=dict(
                size=20,
                color=vem_cd,
                coloraxis='coloraxis',
                line=dict(
                    width=2,
                    color='Black'
                ),
            ),
            name='Unseen',
            showlegend=False,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=val_ood_locs[:, 0].cpu(),
            y=val_ood_locs[:, 1].cpu(),
            mode='markers',
            marker_symbol='diamond',
            marker=dict(
                size=20,
                color=voem_cd,
                coloraxis='coloraxis',
                line=dict(
                    width=2,
                    color='Black'
                ),
            ),
            name='Unseen (OOD)',
            showlegend=False,
        ), row=2, col=1)


        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'font': dict(
                family="Arial",
                size=52,
                color="Black"
            ),
            })
        fig.update_annotations(font_size=72)
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(title_text="Y", row=1, col=1)
        fig.update_yaxes(title_text="Y", row=2, col=1)
        fig.update_xaxes(title_text="X", row=2, col=1)

        fig.update_layout(legend=dict(
            # yanchor="top",
            # y=0.65,
            xanchor="left",
            x=0.68,
            orientation="h",
        ))
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="RMSE",
            ),
            coloraxis=dict(
                colorscale=['green', 'red'],
                #cmin=0,
                #cmax=4,
            )
        )

        fig.show()
        quit()
        visualize_batched_point_clouds([train_pc, val_pc, val_ood_pc]).show()

        pass

    # if VISUALIZE_PREDS:
    #     model.to(device)
    #     dataloader = torch.utils.data.DataLoader(
    #         datamodule.val_dataset, batch_size=1, shuffle=False
    #     )
    #     iterator = iter(dataloader)
    #     for _ in range(32):
    #         batch = next(iterator)
    #     pred_dict = model.predict(batch, 50)
    #     # extracting anchor point cloud depending on model type
    #     if cfg.model.type == "flow":
    #         scene_pc = batch["pc"].flatten(0, 1).cpu().numpy()
    #         seg = batch["seg"].flatten(0, 1).cpu().numpy()
    #         anchor_pc = scene_pc[~seg.astype(bool)]
    #     else:
    #         anchor_pc = batch["pc_anchor"].flatten(0, 1).cpu().numpy()

    #     # pred_action = pred_dict["pred_action"][[8]] # 0,8
    #     pred_action = pred_dict["pred_action"]
    #     pred_action_size = pred_action.shape[1]
    #     pred_action = pred_action.flatten(0, 1).cpu().numpy()
    #     # color-coded segmentations
    #     anchor_seg = np.zeros(anchor_pc.shape[0], dtype=np.int64)
    #     # if cfg.model.type == "flow":
    #     #     pred_action_size = cfg.dataset.sample_size_action + cfg.dataset.sample_size_anchor
    #     # else:
    #     #     pred_action_size = cfg.dataset.sample_size_action
    #     pred_action_seg = np.array([np.arange(1, 11)] * pred_action_size).T.flatten()
    #     # visualize
    #     fig = vpl.segmentation_fig(
    #         np.concatenate((anchor_pc, pred_action)),
    #         np.concatenate((anchor_seg, pred_action_seg)),
    #     )
    #     fig.show()

    if VISUALIZE_PULL:
        model.to(device)
        dataloader = torch.utils.data.DataLoader(
            datamodule.val_dataset, batch_size=1, shuffle=False
        )
        iterator = iter(dataloader)
        for _ in range(11):
            batch = next(iterator)
        pred_dict = model.predict(batch, 1)
        results = pred_dict["results"]
        action_pc = batch["pc_action"].flatten(0, 1).cpu()
        # pred_action = .cpu()
        if cfg.model.type == "flow":
            # pcd = batch["pc_action"].flatten(0, 1).cpu()
            pcd = torch.cat([
                batch["pc_action"].flatten(0, 1),
                pred_dict["pred_action"].flatten(0, 1).cpu(),
            ]).cpu()
        elif cfg.model.type == "flow_cross":
            pcd = torch.cat([
                batch["pc_anchor"].flatten(0, 1),
                batch["pc_action"].flatten(0, 1),
                # pred_dict['pred_action'].flatten(0, 1).cpu(),
            ], dim=0).cpu()
        elif cfg.model.type == "point_cross":
            pcd = torch.cat([
                batch["pc_anchor"].flatten(0, 1),
                pred_dict["pred_action"].flatten(0, 1).cpu()
            ], dim=0).cpu()    
        
        # visualize
        animation = FlowNetAnimation()
        for noise_step in tqdm(results):
            pred_step = noise_step[0].permute(1, 0).cpu()
            if cfg.model.type == "point_cross":
                flows = torch.zeros_like(pred_step)
                animation.add_trace(
                    pcd,
                    [flows],
                    [pred_step],
                    "red",
                )
            else:
                animation.add_trace(
                    pcd,
                    [action_pc],# if cfg.model.type == "flow_cross" else pcd],
                    [pred_step],
                    "red",
                )
        fig = animation.animate()
        fig.show()

    if VISUALIZE_SINGLE:
        model.to(device)
        dataloader = torch.utils.data.DataLoader(
            datamodule.val_dataset, batch_size=1, shuffle=False
        )
        batch = next(iter(dataloader))
        pred_dict = model.predict(batch, 1)

        results = pred_dict["results"]
        action_pc = batch["pc_action"].flatten(0, 1).cpu()
        # pred_action = .cpu()
        if cfg.model.type == "flow":
            # pcd = batch["pc_action"].flatten(0, 1).cpu()
            pcd = torch.cat([
                batch["pc_action"].flatten(0, 1),
                pred_dict["pred_action"].flatten(0, 1).cpu(),
            ]).cpu()
        elif cfg.model.type == "flow_cross":
            pcd = torch.cat([
                batch["pc_anchor"].flatten(0, 1),
                batch["pc_action"].flatten(0, 1),
                # pred_dict['pred_action'].flatten(0, 1).cpu(),
            ], dim=0).cpu()
        elif cfg.model.type == "point_cross":
            pcd = torch.cat([
                batch["pc_anchor"].flatten(0, 1),
                pred_dict["pred_action"].flatten(0, 1).cpu()
            ], dim=0).cpu()    
        
        # visualize
        animation = FlowNetAnimation()
        for noise_step in tqdm(results):
            pred_step = noise_step[0].permute(1, 0).cpu()
            if cfg.model.type == "point_cross":
                flows = torch.zeros_like(pred_step)
                animation.add_trace(
                    pcd,
                    [flows],
                    [pred_step],
                    "red",
                )
            else:
                animation.add_trace(
                    pcd,
                    [action_pc],# if cfg.model.type == "flow_cross" else pcd],
                    [pred_step],
                    "red",
                )
        fig = animation.animate()
        fig.show()

    if VISUALIZE_CUSTOM:
        model.to(device)
        dataloader = torch.utils.data.DataLoader(
            datamodule.val_dataset, batch_size=1, shuffle=False
        )
        batch = next(iter(dataloader))
        pred_dict = model.predict(batch, 1)

        results = pred_dict["results_world"]

        # Ensure the directory exists before saving
        save_dir = "/home/lyuxing/Desktop/tax3d_upgrade/scripts/vis/noise_viz_cloth/"
        os.makedirs(save_dir, exist_ok=True)

        # Visualization
        idx = 0


        T_goal2world = Transform3d(
            matrix=batch["T_goal2world"]
        )
        T_action2world = Transform3d(
            matrix=batch["T_action2world"]
        )

        goal = T_goal2world.transform_points(batch["pc"])
        anchor = T_goal2world.transform_points(batch["pc_anchor"])
        pred_action = pred_dict["point"]["pred_world"]

        goal_pc = goal.flatten(0, 1).cpu().numpy()
        anchor_pc = anchor.flatten(0, 1).cpu().numpy()
        pred_action_size = pred_action.shape[1]
        pred_action_pc = pred_action.flatten(0, 1).cpu().numpy()
        # color-coded segmentations
        anchor_seg = np.zeros(anchor_pc.shape[0], dtype=np.int64)
        # if cfg.model.type == "flow":
        #     pred_action_size = cfg.dataset.sample_size_action + cfg.dataset.sample_size_anchor
        # else:
        #     pred_action_size = cfg.dataset.sample_size_action
        goal_seg = np.ones(goal_pc.shape[0], dtype=np.int64)
        pred_action_seg = np.array([np.arange(2, 2 + 1)] * pred_action_size).T.flatten()
        fig = vpl.segmentation_fig(
            np.concatenate((anchor_pc, pred_action_pc, goal_pc)),
            np.concatenate((anchor_seg, pred_action_seg, goal_seg)),
        )
        fig.write_image(f"{save_dir}/pred.png")

        for noise_step in tqdm(results):
            if idx % 10 == 0:  # Visualize every 10th step

                pred_action = noise_step
                pred_action_size = pred_action.shape[1]
                pred_action = pred_action.flatten(0, 1).cpu().numpy()

                # Compute Centroids (Mean of each PCD)
                goal_centroid = goal_pc.mean(axis=0)  # Shape: (3,)
                anchor_centroid = anchor_pc.mean(axis=0)  # Shape: (3,)
                pred_centroid = pred_action.mean(axis=0)  # Shape: (3,)

                # Print centroid coordinates
                print(f"Idx: {idx}")
                print(f"  - Goal Centroid: {goal_centroid}")
                print(f"  - Anchor Centroid: {anchor_centroid}")
                print(f"  - Prediction Centroid: {pred_centroid}")

                # Color-coded segmentations
                anchor_seg = np.zeros(anchor_pc.shape[0], dtype=np.int64)
                goal_seg = np.ones(goal_pc.shape[0], dtype=np.int64)
                pred_action_seg = np.array([np.arange(2, 2 + 1)] * pred_action_size).T.flatten()

                # Visualize
                fig = vpl.segmentation_fig(
                    np.concatenate((anchor_pc, pred_action, goal_pc)),
                    np.concatenate((anchor_seg, pred_action_seg, goal_seg)),
                )
                fig.write_image(f"{save_dir}/{idx}_noise.png")

            idx += 1  # Increment counter
        

if __name__ == "__main__":
    main()