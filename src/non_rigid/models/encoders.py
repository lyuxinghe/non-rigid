import torch
import torch.nn as nn
import torch_geometric.data as tgd
import numpy as np

from non_rigid.nets.pn2 import PN2Dense, PN2DenseParams

from functools import partial

#################################################################################
#                               Point Cloud Encoders                            #
#################################################################################

def mlp_encoder(in_channels, out_channels):
    """
    MLP encoder for point clouds.
    """
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
    )

def pn2_encoder(in_channels, out_channels, model_cfg):
    """
    PointNet++ encoder for point clouds.
    """
    pn2_params = PN2DenseParams()
    if model_cfg.object_scale is not None or model_cfg.scene_scale is not None:
        pn2_params.sa1.r = 0.5 * model_cfg.pcd_scale
        pn2_params.sa2.r = 1.0 * model_cfg.pcd_scale
    else:
        pn2_params.sa1.r = 0.2 * model_cfg.pcd_scale
        pn2_params.sa2.r = 0.4 * model_cfg.pcd_scale

    class PN2DenseWrapper(nn.Module):
        def __init__(self, in_channels, out_channels, p):
            super().__init__()
            self.pn2dense = PN2Dense(
                in_channels=in_channels - 3,
                out_channels=out_channels,
                p=p,
            )

        def forward(self, x):
            batch_size, num_channels = x.shape[0], x.shape[1]
            batch_indices = torch.arange(
                batch_size, device=x.device
            ).repeat_interleave(x.shape[2])

            if num_channels == 3:
                input_batch = tgd.Batch(
                    pos=x.permute(0, 2, 1).reshape(-1, 3), batch=batch_indices
                )
            elif num_channels > 3:
                input_batch = tgd.Batch(
                    pos=x[:, :3, :].permute(0, 2, 1).reshape(-1, 3),
                    x=x[:, 3:, :].permute(0, 2, 1).reshape(-1, num_channels - 3),
                    batch=batch_indices,
                )
            else:
                raise ValueError(f"Invalid number of input channels: {num_channels}")
            
            output = self.pn2dense(input_batch)
            output = output.reshape(batch_size, -1, output.shape[-1]).permute(0, 2, 1)
            return output
    
    return PN2DenseWrapper(in_channels=in_channels, out_channels=out_channels, p=pn2_params)

#################################################################################
#                                 Feature Encoders                              #
#################################################################################

class DisjointFeatureEncoder(nn.Module):
    """
    TODO: fill this out
    """
    def __init__(self, in_channels, hidden_size, model_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.model_cfg = model_cfg

        # Initializing point cloud encoder wrapper.
        if self.model_cfg.point_encoder == "mlp":
            encoder_fn = partial(mlp_encoder, in_channels=self.in_channels)
        elif self.model_cfg.point_encoder == "pn2":
            encoder_fn = partial(pn2_encoder, in_channels=self.in_channels, model_cfg=self.model_cfg)
        else:
            raise ValueError(f"Invalid point_encoder: {self.model_cfg.point_encoder}")

        # Creating base encoders - action (x0), anchor (y), and noised prediction (x).
        self.x_encoder = encoder_fn(out_channels=hidden_size)
        self.x0_encoder = encoder_fn(out_channels=hidden_size)
        self.y_encoder = encoder_fn(out_channels=hidden_size)

        # Creating extra feature encoders, if necessary.
        if self.model_cfg.feature:
            self.shape_encoder = encoder_fn(out_channels=hidden_size)
            self.flow_zeromean_encoder = encoder_fn(out_channels=hidden_size)
            self.x_corr_encoder = encoder_fn(out_channels=hidden_size)
            self.action_mixer = mlp_encoder(5 * hidden_size, hidden_size)
        else:
            self.action_mixer = mlp_encoder(2 * hidden_size, hidden_size)
    
    def forward(self, x, y, x0):
        """
        TODO: fill this out
        """
        if self.model_cfg.type == "flow":
            x_flow = x
            x_recon = x + x0
        else:
            x_flow = x - x0
            x_recon = x
        
        # Encode base features - action (x0), anchor (y), and noised prediction (x).
        x_enc = self.x_encoder(x)
        x0_enc = self.x0_encoder(x0)
        y_enc = self.y_encoder(y).permute(0, 2, 1)

        # Encode extra features, if necessary.
        if self.model_cfg.feature:
            shape_enc = self.shape_encoder(
                x_recon - torch.mean(x_recon, dim=2, keepdim=True)
            )
            flow_zeromean_enc = self.flow_zeromean_encoder(
                x_flow - torch.mean(x_flow, dim=2, keepdim=True)
            )
            x_corr_enc = self.x_corr_encoder(
                x_recon if self.model_cfg.type == "flow" else x_flow
            )
            action_features = [x_enc, x0_enc, shape_enc, flow_zeromean_enc, x_corr_enc]
        else:
            action_features = [x_enc, x0_enc]

        # Compress action features to hidden size through action mixer.
        x_enc = torch.cat(action_features, dim=1)
        x_enc = self.action_mixer(x_enc).permute(0, 2, 1)

        return x_enc, y_enc

class JointFeatureEncoder(nn.Module):
    """
    TODO: fill this out
    """
    def __init__(self, in_channels, hidden_size, model_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.model_cfg = model_cfg

        # Initializing point cloud encoder wrapper.
        if self.model_cfg.point_encoder == "mlp":
            encoder_fn = partial(mlp_encoder, in_channels=self.in_channels)
        elif self.model_cfg.point_encoder == "pn2":
            encoder_fn = partial(pn2_encoder, in_channels=self.in_channels, model_cfg=self.model_cfg)
        else:
            raise ValueError(f"Invalid point_encoder: {self.model_cfg.point_encoder}")

        # Creating base encoders - action-frame, and prediction-frame.
        self.action_encoder = encoder_fn(out_channels=hidden_size)
        self.pred_encoder = encoder_fn(out_channels=hidden_size)

        # Creating extra feature encoders, if necessary.
        if self.model_cfg.feature:
            self.feature_encoder = encoder_fn(in_channels=9, out_channels=hidden_size)
            self.action_mixer = mlp_encoder(3 * hidden_size, hidden_size)
        else:
            self.action_mixer = mlp_encoder(2 * hidden_size, hidden_size)
    
    def forward(self, x, y, x0):
        """
        TODO: fill this out
        """
        if self.model_cfg.type == "flow":
            x_flow = x
            x_recon = x + x0
        else:
            x_flow = x - x0
            x_recon = x
        
        # Encode base features - action-frame, and prediction frame.
        action_size = x0.shape[-1]
        action_enc = self.action_encoder(x0)
        pred_enc = self.pred_encoder(torch.cat([x_recon, y], dim=-1))
        action_pred_enc, anchor_pred_enc = pred_enc[:, :, :action_size], pred_enc[:, :, action_size:]
        anchor_pred_enc = anchor_pred_enc.permute(0, 2, 1)

        # Encode extra features, if necessary.
        if self.model_cfg.feature:
            shape = x_recon - torch.mean(x_recon, dim=2, keepdim=True)
            flow_zeromean = x_flow - torch.mean(x_flow, dim=2, keepdim=True)
            feature_enc = self.feature_encoder(
                torch.cat([shape, x_flow, flow_zeromean], dim=1)
            )
            action_features = [action_enc, action_pred_enc, feature_enc]
        else:
            action_features = [action_enc, action_pred_enc]
        
        # Compress action features to hidden size through action mixer.
        x_enc = torch.cat(action_features, dim=1)
        x_enc = self.action_mixer(x_enc).permute(0, 2, 1)

        return x_enc, anchor_pred_enc