from typing import Any, Dict

import lightning as L
import numpy as np
import omegaconf
import plotly.express as px
import rpad.pyg.nets.dgcnn as dgcnn
import rpad.visualize_3d.plots as vpl
import torch
import torch.nn.functional as F
import torch_geometric.data as tgd
import torch_geometric.transforms as tgt
import torchvision as tv
import wandb
from diffusers import get_cosine_schedule_with_warmup
from pytorch3d.transforms import Transform3d
from torch import nn, optim
from torch_geometric.nn import fps

from non_rigid.metrics.error_metrics import get_pred_pcd_rigid_errors
from non_rigid.metrics.flow_metrics import flow_cos_sim, flow_rmse, pc_nn
from non_rigid.metrics.rigid_metrics import svd_estimation, translation_err, rotation_err
from non_rigid.models.dit.diffusion import create_diffusion_mu
from non_rigid.models.dit.models import Mu_DiT_Take1, Mu_DiT_Take2, Mu_DiT_Take3, Mu_DiT_Take4
from non_rigid.utils.logging_utils import viz_predicted_vs_gt
from non_rigid.utils.pointcloud_utils import expand_pcd


def Mu_DiT_Take1_xS(use_rotary, **kwargs):
    # hidden size divisible by 3 for rotary embedding, and divisible by num_heads for multi-head attention
    print("MuFrameDiffusionTransformerNetwork: Mu_DiT_Take1_xS")
    hidden_size = 132 if use_rotary else 128
    return Mu_DiT_Take1(depth=5, hidden_size=hidden_size, num_heads=4, **kwargs)

def Mu_DiT_Take2_xS(use_rotary, **kwargs):
    # hidden size divisible by 3 for rotary embedding, and divisible by num_heads for multi-head attention
    print("MuFrameDiffusionTransformerNetwork: Mu_DiT_Take2_xS")
    hidden_size = 132 if use_rotary else 128
    return Mu_DiT_Take2(depth=5, hidden_size=hidden_size, num_heads=4, **kwargs)

def Mu_DiT_Take3_xS(use_rotary, **kwargs):
    # hidden size divisible by 3 for rotary embedding, and divisible by num_heads for multi-head attention
    print("MuFrameDiffusionTransformerNetwork: Mu_DiT_Take3_xS")
    hidden_size = 132 if use_rotary else 128
    return Mu_DiT_Take3(depth=5, hidden_size=hidden_size, num_heads=4, **kwargs)

def Mu_DiT_Take4_xS(use_rotary, **kwargs):
    # hidden size divisible by 3 for rotary embedding, and divisible by num_heads for multi-head attention
    print("MuFrameDiffusionTransformerNetwork: Mu_DiT_Take4_xS")
    hidden_size = 132 if use_rotary else 128
    return Mu_DiT_Take4(depth=5, hidden_size=hidden_size, num_heads=4, **kwargs)

# TODO: clean up all unused functions
DiT_models = {
    "Mu_DiT_Take1_xS": Mu_DiT_Take1_xS,
    "Mu_DiT_Take2_xS": Mu_DiT_Take2_xS,
    "Mu_DiT_Take3_xS": Mu_DiT_Take3_xS,
    "Mu_DiT_Take4_xS": Mu_DiT_Take4_xS,
}


def get_model(model_cfg):
    #rotary = "Rel3D_" if model_cfg.rotary else ""
    if model_cfg.model_take == 1:
        model_take = "Take1"
    elif model_cfg.model_take == 2:
        model_take = "Take2"
    elif model_cfg.model_take == 3:
        model_take = "Take3"
    elif model_cfg.model_take == 4:
        model_take = "Take4"
    else:
        raise ValueError(f"Invalid model take: {model_cfg.model_take}")
    # model_name = f"{rotary}DiT_pcu_{cross}{model_cfg.size}"
    model_name = f"Mu_DiT_{model_take}_{model_cfg.size}"
    return DiT_models[model_name]


class MuFrameDiffusionTransformerNetwork(nn.Module):
    """
    Network containing the specified Diffusion Transformer architecture.
    """
    def __init__(self, model_cfg=None):
        super().__init__()
        self.dit = get_model(model_cfg)(
            use_rotary=model_cfg.rotary,
            in_channels=model_cfg.in_channels,
            learn_sigma=model_cfg.learn_sigma,
            model_cfg=model_cfg,
        )
    
    def forward(self, xr_t, xs_t, t, **kwargs):
        return self.dit(xr_t, xs_t, t, **kwargs)
    


class MuFrameDenseDisplacementDiffusionModule(L.LightningModule):
    """
    Generalized Dense Displacement Diffusion (DDD) module that handles model training, inference, 
    evaluation, and visualization. This module is inherited and overriden by scene-level and 
    object-centric modules.
    """
    def __init__(self, network, cfg) -> None:
        super().__init__()
        self.network = network
        self.model_cfg = cfg.model
        self.dataset_cfg = cfg.dataset
        self.wandb_cfg = cfg.wandb
        self.prediction_type = self.model_cfg.type # flow or point
        self.mode = cfg.mode # train or eval
        self.pred_ref_frame = self.model_cfg.pred_ref_frame

        # prediction type-specific processing
        # TODO: eventually, this should be removed by updating dataset to use "point" instead of "pc"
        if self.prediction_type == "flow":
            self.label_key = "flow"
        elif self.prediction_type == "point":
            self.label_key = "pc"
        else:
            raise ValueError(f"Invalid prediction type: {self.prediction_type}")
        
        if self.pred_ref_frame == "noisy_goal":
            assert self.model_cfg.ref_error_scale != 0.0
            self.ref_error_scale = self.model_cfg.ref_error_scale
        elif self.pred_ref_frame == "anchor":
            self.ref_error_scale = 0.0
        else:
            raise ValueError(f"Invalid prediction type: {self.pred_ref_frame}")

        # mode-specific processing
        if self.mode == "train":
            self.run_cfg = cfg.training
            # training-specific params
            self.lr = self.run_cfg.lr
            self.weight_decay = self.run_cfg.weight_decay
            self.num_training_steps = self.run_cfg.num_training_steps
            self.lr_warmup_steps = self.run_cfg.lr_warmup_steps
            self.additional_train_logging_period = self.run_cfg.additional_train_logging_period
        elif self.mode == "eval":
            self.run_cfg = cfg.inference
            # inference-specific params
            self.num_trials = self.run_cfg.num_trials
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # data params
        self.batch_size = self.run_cfg.batch_size
        self.val_batch_size = self.run_cfg.val_batch_size
        # TODO: it is debatable if the module needs to know about the sample size
        self.sample_size = self.run_cfg.sample_size
        self.sample_size_anchor = self.run_cfg.sample_size_anchor

        # diffusion params
        # self.noise_schedule = model_cfg.diff_noise_schedule
        # self.noise_scale = model_cfg.diff_noise_scale
        self.diff_steps = self.model_cfg.diff_train_steps # TODO: rename to diff_steps
        self.num_wta_trials = self.run_cfg.num_wta_trials
        self.diff_translation_noise_scale = self.model_cfg.diff_translation_noise_scale
        self.diff_rotation_noise_scale = self.model_cfg.diff_rotation_noise_scale
        #self.time_based_weighting = self.model_cfg.time_based_weighting

        self.diffusion = create_diffusion_mu(
            timestep_respacing=None,
            diffusion_steps=self.diff_steps,
            # noise_schedule=self.noise_schedule,
            time_based_weighting="even",
        )

        # TODO: make this a info helper method
        print("Initializing Mu-Frame Diffusion Transformer Network")
        #print(f"### Translation Noise Scale: {self.diff_translation_noise_scale}")
        #print(f"### Rotation Noise Scale: {self.diff_rotation_noise_scale}")
        print("Initializing Mu-Frame Dense Displacement Diffusion Module")
        print(f"### Prediction Reference Frame: {self.pred_ref_frame}")
        print(f"### Referemce Noise Scale: {self.ref_error_scale}")

    def configure_optimizers(self):
        assert self.mode == "train", "Can only configure optimizers in training mode."
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'}}

    def get_model_kwargs(self, batch, nun_samples=None):
        """
        Get the model kwargs for the forward pass.
        """
        raise NotImplementedError("This should be implemented in the derived class.")
    
    def get_world_preds(self, batch, num_samples, pc_action, pred_dict):
        """
        Get world frame predictions from the given batch and predictions.
        """
        raise NotImplementedError("This should be implemented in the derived class.")
    
    def update_prediction_frame_batch(self, batch, stage):
        """
        Convert data batch from world frame to prediction frame.
        """
        raise NotImplementedError("This should be implemented in the derived class.")
    
    def get_viz_args(self, batch, viz_idx):
        """
        Get visualization arguments for wandb logging.
        """
        raise NotImplementedError("This should be implemented in the derived class.")

    def forward(self, batch, t):
        """
        Forward pass to compute diffusion training loss.
        """
        assert "pred_ref" in batch.keys(), "Please run update_prediction_frame_batch to update the data batch!"
        pred_ref = batch["pred_ref"]
        
        ground_truth = batch[self.label_key].permute(0, 2, 1) # channel first
        model_kwargs = self.get_model_kwargs(batch)

        # run diffusion
        # noise = torch.randn_like(ground_truth) * self.noise_scale
        loss_dict = self.diffusion.training_losses(
            model=self.network,
            # TODO: set x_start as tuple, and dynamically noise 1 or 2 targets in diffusion code
            x_start=ground_truth,
            t=t,
            translation_noise_scale=self.diff_translation_noise_scale,
            rotation_noise_scale=self.diff_rotation_noise_scale,
            model_kwargs=model_kwargs,
            # noise=noise,
        )
        
        loss_r = loss_dict["loss_r"].mean()
        loss_s = loss_dict["loss_s"].mean()
        
        loss = loss_dict["loss"].mean()

        return None, loss, loss_r, loss_s

    @torch.no_grad()
    def predict(self, batch, num_samples, unflatten=False, progress=True, full_prediction=True):
        """
        Compute prediction for a given batch.

        Args:
            batch: the input batch
            num_samples: the number of samples to generate
            progress: whether to show progress bar
            full_prediction: whether to return full prediction (flow and point, goal and world frame)
        """
        # TODO: replace bs with batch_size?
        assert "pred_ref" in batch.keys(), "Please run update_prediction_frame_batch to update the data batch!"
        pred_ref = batch["pred_ref"].clone()
        pred_ref = expand_pcd(pred_ref, num_samples)
        
        bs, sample_size = batch["pc_action"].shape[:2]
        model_kwargs = self.get_model_kwargs(batch, num_samples)

        # generating latents and running diffusion
        z_s = torch.randn(bs * num_samples, 3, sample_size, device=self.device)
        z_r = pred_ref.to(self.device).permute(0,2,1)
        #z_r = torch.zeros(bs * num_samples, 3, 1, device=self.device)

        # test: in inference, if we trained with translation noise with scale=t, we sample noise from N(0, 1+t)
        #trans_noise_scale = torch.tensor(0.6, device=self.device)
        #z = torch.randn(bs * num_samples, 3, sample_size, device=self.device) * torch.sqrt(1 + trans_noise_scale)

        # TODO: p_sample for z_r is probably wrong
        final_dict, results = self.diffusion.p_sample_loop(
            self.network,
            z_r.shape,
            z_s.shape,
            z_r,
            z_s,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=progress,
            device=self.device,
        )
        pred_r = final_dict["sample_r"]
        pred_s = final_dict["sample_s"]
        pred = pred_s + pred_r        
        results = [res["sample_r"] + res["sample_s"] for res in results]
        pred = pred.permute(0, 2, 1)

        if not full_prediction:
            # only return the prediction type in the goal frame
            return {self.prediction_type: {"pred": pred}}
        else:
            # return full prediction (flow and point, goal and world frame)
            pc_action = model_kwargs["x0"].permute(0, 2, 1)

            # computing flow and point predictions
            if self.prediction_type == "flow":
                pred_flow = pred
                pred_point = pc_action + pred_flow
                # for flow predictions, convert results to point predictions
                results = [
                    pc_action + res.permute(0, 2, 1) for res in results
                ]
            elif self.prediction_type == "point":
                pred_point = pred
                pred_flow = pred_point - pc_action
                results = [
                    res.permute(0, 2, 1) for res in results
                ]

            pred_dict = {
                "flow": {
                    "pred": pred_flow,
                },
                "point": {
                    "pred": pred_point,
                },
                "results": results,
            }

            # compute world frame predictions
            pred_flow_world, pred_point_world, results_world = self.get_world_preds(
                batch, num_samples, pc_action, pred_dict)
            
            pred_dict["flow"]["pred_world"] = pred_flow_world
            pred_dict["point"]["pred_world"] = pred_point_world
            pred_dict["results_world"] = results_world
            return pred_dict

    def predict_wta(self, batch, num_samples):
        """
        Predict WTA (winner-take-all) samples, and compute WTA metrics. Unlike predict, this 
        function assumes the ground truth is available.

        Args:
            batch: the input batch
            num_samples: the number of samples to generate
        """
        assert "pred_ref" in batch.keys(), "Please run update_prediction_frame_batch to update the data batch!"
        pred_ref = batch["pred_ref"]
        
        ground_truth = batch[self.label_key].to(self.device)
        seg = batch["seg"].to(self.device)
        goal_pc = batch["pc"].to(self.device)
        scaling_factor = self.dataset_cfg.pcd_scale_factor

        # re-shaping and expanding for winner-take-all
        bs = ground_truth.shape[0]
        ground_truth = expand_pcd(ground_truth, num_samples)
        seg = expand_pcd(seg, num_samples)
        goal_pc = expand_pcd(goal_pc, num_samples)

        # generating diffusion predictions
        # TODO: this should probably specific full_prediction=False
        pred_dict = self.predict(
            batch, num_samples, unflatten=False, progress=True
        )
        pred = pred_dict[self.prediction_type]["pred"]

        # scale the pcd back to original size for error calculation
        pred_scaled = pred / scaling_factor
        ground_truth_scaled = ground_truth / scaling_factor

        # computing error metrics
        rmse = flow_rmse(pred_scaled, ground_truth_scaled, mask=True, seg=seg).reshape(bs, num_samples)
        pred = pred.reshape(bs, num_samples, -1, 3)

        # computing winner-take-all metrics
        winner = torch.argmin(rmse, dim=-1)
        rmse_wta = rmse[torch.arange(bs), winner]
        pred_wta = pred[torch.arange(bs), winner]

        if self.dataset_cfg.material == "rigid":
            T_goalUnAug = Transform3d(
                matrix=expand_pcd(batch["T_goalUnAug"].to(self.device), num_samples)
            )
            pred_ref = expand_pcd(pred_ref, num_samples)
            
            pred_point_world = T_goalUnAug.transform_points(pred_dict["point"]["pred"])
            goal_point_world = T_goalUnAug.transform_points(goal_pc)
            
            pred_point_world = pred_point_world / scaling_factor
            goal_point_world = goal_point_world / scaling_factor

            translation_errs, rotation_errs = svd_estimation(pred_point_world, goal_point_world, return_magnitude=True)

            translation_errs = translation_errs.reshape(bs, num_samples)
            rotation_errs = rotation_errs.reshape(bs, num_samples)

            trans_winner = torch.argmin(translation_errs, dim=-1)
            rot_winner = torch.argmin(rotation_errs, dim=-1)

            translation_err_wta = translation_errs[torch.arange(bs), trans_winner]
            rotation_err_wta = rotation_errs[torch.arange(bs), rot_winner]
            
            return {
                "pred": pred,
                "pred_wta": pred_wta,
                "rmse": rmse,
                "rmse_wta": rmse_wta,
                "trans": translation_errs,
                "trans_wta": translation_err_wta,
                "rot": rotation_errs,
                "rot_wta": rotation_err_wta,
            }
        
        else:
            return {
                "pred": pred,
                "pred_wta": pred_wta,
                "rmse": rmse,
                "rmse_wta": rmse_wta,
            }



    def log_viz_to_wandb(self, batch, pred_wta_dict, tag):
        """
        Log visualizations to wandb.

        Args:
            batch: the input batch
            pred_wta_dict: the prediction dictionary
            tag: the tag to use for logging
        """
        # pick a random sample in the batch to visualize
        viz_idx = np.random.randint(0, batch["pc"].shape[0])
        pred_viz = pred_wta_dict["pred"][viz_idx, 0, :, :3]
        pred_wta_viz = pred_wta_dict["pred_wta"][viz_idx, :, :3]
        viz_args = self.get_viz_args(batch, viz_idx)

        # getting predicted action point cloud
        if self.prediction_type == "flow":
            pred_action_viz = viz_args["pc_action_viz"] + pred_viz
            pred_action_wta_viz = viz_args["pc_action_viz"] + pred_wta_viz
        elif self.prediction_type == "point":
            pred_action_viz = pred_viz
            pred_action_wta_viz = pred_wta_viz

        # logging predicted vs ground truth point cloud
        viz_args["pred_action_viz"] = pred_action_viz
        predicted_vs_gt = viz_predicted_vs_gt(**viz_args)
        wandb.log({f"{tag}/predicted_vs_gt": predicted_vs_gt})

        # logging predicted vs ground truth point cloud (wta)
        viz_args["pred_action_viz"] = pred_action_wta_viz
        predicted_vs_gt_wta = viz_predicted_vs_gt(**viz_args)
        wandb.log({f"{tag}/predicted_vs_gt_wta": predicted_vs_gt_wta})

    def training_step(self, batch):
        """
        Training step for the module. Logs training metrics and visualizations to wandb.
        """
        self.train()
        t = torch.randint(
            0, self.diff_steps, (self.batch_size,), device=self.device
        ).long()
        
        batch = self.update_prediction_frame_batch(batch, stage="train")
                
        _, loss, loss_r, loss_s = self(batch, t)
        #########################################################
        # logging training metrics
        #########################################################
        self.log_dict(
            {"train/loss": loss,
            "train/loss_r": loss_r,
            "train/loss_s": loss_s},
            add_dataloader_idx=False,
            prog_bar=True,
        )

        # determine if additional logging should be done
        do_additional_logging = (
            self.global_step % self.additional_train_logging_period == 0
        )

        # additional logging
        if do_additional_logging:
            # TODO: VERIFY WHETHER THIS SHOULD BE TURNED INTO EVAL & NO GRAD!
            self.eval()
            with torch.no_grad():
                # winner-take-all predictions
                train_dataloader = self.trainer.datamodule.train_dataloader()
                train_dataloader.dataset.set_eval_mode(True)

                batch = next(iter(train_dataloader))  # Fetch new batch with eval_mode=True
                batch = self.update_prediction_frame_batch(batch, stage="inference")

                # TODO: Debug why without this line, it will rasie gpu/cpu different device error
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                pred_wta_dict = self.predict_wta(batch, self.num_wta_trials)

                ####################################################
                # logging training wta metrics
                ####################################################
                self.log_dict(
                    {
                        "train/rmse": pred_wta_dict["rmse"].mean(),
                        "train/rmse_wta": pred_wta_dict["rmse_wta"].mean(),
                    },
                    add_dataloader_idx=False,
                    prog_bar=True,
                )

                if self.dataset_cfg.material == "rigid":
                    ####################################################
                    # logging rigid metrics (translation & rotation)
                    ####################################################
                    self.log_dict(
                        {
                            "train/trans": pred_wta_dict["trans"].mean(),
                            "train/trans_wta": pred_wta_dict["trans_wta"].mean(),
                            "train/rot": pred_wta_dict["rot"].mean(),
                            "train/rot_wta": pred_wta_dict["rot_wta"].mean(),
                        },
                        add_dataloader_idx=False,
                        prog_bar=True,
                    )

                ####################################################
                # logging visualizations
                ####################################################
                self.log_viz_to_wandb(batch, pred_wta_dict, "train")

                train_dataloader.dataset.set_eval_mode(False)


        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step for the module. Logs validation metrics and visualizations to wandb.
        """
        self.eval()
        with torch.no_grad():
            # winner-take-all predictions
            batch = self.update_prediction_frame_batch(batch, stage="inference")
            pred_wta_dict = self.predict_wta(batch, self.num_wta_trials)
        
        ####################################################
        # logging validation wta metrics
        ####################################################
        self.log_dict(
            {
                f"val_rmse_{dataloader_idx}": pred_wta_dict["rmse"].mean(),
                f"val_rmse_wta_{dataloader_idx}": pred_wta_dict["rmse_wta"].mean(),
            },
            add_dataloader_idx=False,
            prog_bar=True,
        )

        if self.dataset_cfg.material == "rigid":
            ####################################################
            # logging rigid metrics (translation & rotation)
            ####################################################
            self.log_dict(
                {
                    f"val_trans_{dataloader_idx}": pred_wta_dict["trans"].mean(),
                    f"val_trans_wta_{dataloader_idx}": pred_wta_dict["trans_wta"].mean(),
                    f"val_rot_{dataloader_idx}": pred_wta_dict["rot"].mean(),
                    f"val_rot_wta_{dataloader_idx}": pred_wta_dict["rot_wta"].mean(),
                },
                add_dataloader_idx=False,
                prog_bar=True,
            )

        ####################################################
        # logging visualizations
        ####################################################
        self.log_viz_to_wandb(batch, pred_wta_dict, f"val_{dataloader_idx}")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step for model evaluation. Computes winner-take-all metrics.
        """
        if self.dataset_cfg.material == "rigid":
            # winner-take-all predictions
            batch = self.update_prediction_frame_batch(batch, stage="inference")
            pred_wta_dict = self.predict_wta(batch, self.num_wta_trials)
            return {
                "rmse": pred_wta_dict["rmse"],
                "rmse_wta": pred_wta_dict["rmse_wta"],
                "trans": pred_wta_dict["trans"].mean(),
                "trans_wta": pred_wta_dict["trans_wta"].mean(),
                "rot": pred_wta_dict["rot"].mean(),
                "rot_wta": pred_wta_dict["rot_wta"].mean(),
            }
        else:
            # winner-take-all predictions
            batch = self.update_prediction_frame_batch(batch, stage="inference")
            pred_wta_dict = self.predict_wta(batch, self.num_wta_trials)
            return {
                "rmse": pred_wta_dict["rmse"],
                "rmse_wta": pred_wta_dict["rmse_wta"],
            }
    
class MuFrameCrossDisplacementModule(MuFrameDenseDisplacementDiffusionModule):
    """
    Object-centric DDD module. Applies cross attention between action and anchor objects.
    """
    def __init__(self, network, cfg) -> None:
        super().__init__(network, cfg)

    def get_model_kwargs(self, batch, num_samples=None):
        pc_action = batch["pc_action"].to(self.device)
        pc_anchor = batch["pc_anchor"].to(self.device)

        if num_samples is not None:
            # expand point clouds if num_samples is provided; used for WTA predictions
            pc_action = expand_pcd(pc_action, num_samples)
            pc_anchor = expand_pcd(pc_anchor, num_samples)
        
        pc_action = pc_action.permute(0, 2, 1) # channel first
        pc_anchor = pc_anchor.permute(0, 2, 1) # channel first
        model_kwargs = dict(x0=pc_action, y=pc_anchor)
        return model_kwargs
    
    def get_world_preds(self, batch, num_samples, pc_action, pred_dict):
        """
        Get world-frame predictions from the given batch and predictions.
        """
        T_actionUnAug = Transform3d(
            matrix=expand_pcd(batch["T_actionUnAug"].to(self.device), num_samples)
        )
        T_goalUnAug = Transform3d(
            matrix=expand_pcd(batch["T_goalUnAug"].to(self.device), num_samples)
        )

        pred_ref = expand_pcd(batch["pred_ref"].to(self.device), num_samples)
        action_ref = expand_pcd(batch["action_ref"].to(self.device), num_samples)

        pred_point_world = T_goalUnAug.transform_points(pred_dict["point"]["pred"])
        pc_action_world = T_actionUnAug.transform_points(pc_action + action_ref)
        pred_flow_world = pred_point_world - pc_action_world
        results_world = [
            T_goalUnAug.transform_points(res) for res in pred_dict["results"]
        ]
        return pred_flow_world, pred_point_world, results_world

    def update_prediction_frame_batch(self, batch, stage):
        # TODO: for now, since gmm is not available, we will use noisy goal to simulate it
        if self.pred_ref_frame == "anchor":
            raise NotImplementedError("Mu-frame not implemented for anchor centroid prediction!")
        elif self.pred_ref_frame == "noisy_goal":
            if stage == "train":
                # NOTE: during training, we will not have access to prediction frame until t=100,
                #       since pred_ref will be gt_center + sqrt(alpha) * gt_error.
                B, _, C = batch["pc"].shape
                pred_ref = torch.zeros((B, 1, C)).to(self.device)
            elif stage == "inference":
                # During inference,
                # for now, we are using noisy goal to simulate gmm
                pred_ref = batch["pc"].mean(axis=1, keepdim=True)
                pred_ref = pred_ref + self.ref_error_scale * torch.randn_like(pred_ref)
            else:
                raise ValueError(f"Invalid stage: {stage}")
        else:
            raise ValueError(f"Invalid prediction reference frame: {self.pred_ref_frame}")

        action_mean = batch["pc_action"].mean(axis=1, keepdim=True)
        anchor_mean = pred_ref
                
        batch["pc_action"] = batch["pc_action"] - action_mean
        #batch["pc_anchor"] = batch["pc_anchor"] - anchor_mean
        
        #batch["pc"] = batch["pc"] - anchor_mean
        #batch["flow"] = batch["flow"] - anchor_mean + action_mean
        batch["flow"] = batch["flow"] + action_mean

        batch["pred_ref"] = pred_ref
        batch["action_ref"] = action_mean

        return batch
        
    def get_viz_args(self, batch, viz_idx):
        """
        Get visualization arguments for wandb logging.
        """
        pc_pos_viz = batch["pc"][viz_idx, :, :3]
        pc_action_viz = batch["pc_action"][viz_idx, :, :3]
        pc_anchor_viz = batch["pc_anchor"][viz_idx, :, :3]
        viz_args = {
            "pc_pos_viz": pc_pos_viz,
            "pc_action_viz": pc_action_viz,
            "pc_anchor_viz": pc_anchor_viz,
        }
        return viz_args

