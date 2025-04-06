# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import omegaconf
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from typing import Optional

from non_rigid.nets.dgcnn import DGCNN
from non_rigid.nets.pn2 import PN2Dense, PN2DenseParams
from non_rigid.models.dit.relative_encoding import RotaryPositionEncoding3D, MultiheadRelativeAttentionWrapper

#import rpad.pyg.nets.pointnet2 as pnp_original
from torch_geometric.data import Data
import torch_geometric.data as tgd
from functools import partial


torch.set_printoptions(precision=8, sci_mode=True)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

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
#                                 Custom Attention Layers                       #
#################################################################################


class CrossAttention(nn.Module):
    """
    Cross attention layer adapted from
    https://github.com/pprp/timm/blob/e9aac412de82310e6905992e802b1ee4dc52b5d1/timm/models/crossvit.py#L132
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert dim_x % num_heads == 0, "dim x must be divisible by num_heads"
        head_dim = dim_x // num_heads
        self.scale = head_dim**-0.5

        self.wq = nn.Linear(dim_x, dim_x, bias=qkv_bias)
        self.wk = nn.Linear(dim_y, dim_x, bias=qkv_bias)
        self.wv = nn.Linear(dim_y, dim_x, bias=qkv_bias)
        self.proj = nn.Linear(dim_x, dim_x)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, Cx = x.shape
        # _, _, Cy = y.shape
        _, Ny, Cy = y.shape
        q = (
            self.wq(x)
            .reshape(B, N, self.num_heads, Cx // self.num_heads)
            .transpose(1, 2)
        )
        k = (
            self.wk(y)
            .reshape(B, Ny, self.num_heads, Cx // self.num_heads)
            .transpose(1, 2)
        )
        v = (
            self.wv(y)
            .reshape(B, Ny, self.num_heads, Cx // self.num_heads)
            .transpose(1, 2)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, Cx)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#                                 Core DiT Layers                               #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class DiTRelativeBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning 
    and 3D relative self attention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiheadRelativeAttentionWrapper(
            embed_dim=hidden_size, 
            num_heads=num_heads,
            dropout=0.0,
            bias=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, x_pos=None):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            query=x, key=x, value=x, rotary_pe=(x_pos, x_pos)
        )[0] # [0] is the attention output

        x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x)
        return x

class DiTCrossBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning and scene cross attention.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(
            dim_x=hidden_size,
            dim_y=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs,
        )
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, y, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_x,
            scale_x,
            gate_x,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(
            modulate(self.norm2(x), shift_mca, scale_mca), y
        )
        x = x + gate_x.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_x, scale_x)
        )
        return x

class DiTRefBlock(nn.Module):
    """
    A simplified DiT block for the reference branch.
    This block processes a single reference token by attending to scene tokens,
    using only cross attention. Self-attention is omitted because with one token it
    doesn't provide additional interactions.
    
    The block still uses layer normalization and an MLP for refinement, and it accepts
    a conditioning vector (e.g. t_emb) to modulate its operations.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(
            dim_x=hidden_size,
            dim_y=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **kwargs,
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=0,
        )
        # We can include a simple modulation block if desired.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )

    def forward(self, ref_token: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref_token: Tensor of shape (B, 1, hidden_size) representing the reference token.
            y: Scene (anchor) tokens of shape (B, N, hidden_size).
            c: Conditioning vector (e.g. t_emb) of shape (B, hidden_size).
        Returns:
            Updated reference token of shape (B, 1, hidden_size).
        """
        # Compute modulation parameters from the conditioning vector.
        # Here we split c into three parts: shift, scale, and gate.
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=1)
        
        # Normalize the reference token and modulate it.
        ref_norm = self.norm1(ref_token)
        modulated_ref = modulate(ref_norm, shift, scale)
        
        # Let the reference token attend to the scene tokens via cross attention.
        attn_out = self.cross_attn(modulated_ref, y)
        x = ref_token + gate.unsqueeze(1) * attn_out
        
        # Further refine with an MLP.
        x = x + self.mlp(self.norm2(x))
        return x

class DiTRelativeCrossBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning 
    and 3D relative self attention + 3D relative scene cross attention.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiheadRelativeAttentionWrapper(
            embed_dim=hidden_size, 
            num_heads=num_heads,
            dropout=0.0,
            bias=True
        )
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = MultiheadRelativeAttentionWrapper(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            bias=True
        )
        
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, y, c, x_pos=None, y_pos=None):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_x,
            scale_x,
            gate_x,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)
        
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            query=x, key=x, value=x, rotary_pe=(x_pos, x_pos)
        )[0] # [0] is the attention output
        
        x = modulate(self.norm2(x), shift_mca, scale_mca)
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(
            query=x, key=y, value=y, rotary_pe=(x_pos, y_pos)
        )[0] # [0] is the attention output
        
        x = x + gate_x.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_x, scale_x)
        )
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class FinalLayer_s(nn.Module):
    """
    Final layer to predict shape (xs) component
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class FinalLayer_r(nn.Module):
    """
    Final layer to predict mean (xr) component
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # x shape: [8, 512, 128]
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        
        x = x.mean(dim=1, keepdim=True)  # [8, 1, 128]
        
        x = self.linear(x)  # [8, 1, 6]
        return x

#################################################################################
#                                 Core DiT Models                               #
#################################################################################

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class LinearRegressionModel(nn.Module):
    """
    Linear regression baseline, with attention - no diffusion.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            model_cfg=None,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.out_channels = 3

        # initializing embedder for action point cloud
        self.x_embedder = nn.Conv1d(
            in_channels,
            hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # initializing embedder for anchor point cloud
        self.y_embedder = nn.Conv1d(
            in_channels,
            hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        class LinearCrossBlock(nn.Module):
            """
            Cross attention block for linear regression model.
            """
            def __init__(self, hidden_size, num_heads, mlp_ratio):
                super().__init__()
                self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
                self.cross_attn = CrossAttention(
                    dim_x=hidden_size,
                    dim_y=hidden_size,
                    num_heads=num_heads,
                    qkv_bias=True,
                )
                mlp_hidden_dim = int(hidden_size * mlp_ratio)
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = Mlp(
                    in_features=hidden_size,
                    hidden_features=mlp_hidden_dim,
                    act_layer=approx_gelu,
                    drop=0,
                )

            def forward(self, x, y):
                x = self.cross_attn(self.norm(x), y)
                x = self.mlp(x)
                return x
            
        class LinearFinalLayer(nn.Module):
            """
            Final layer of the linear regression model.
            """
            def __init__(self, hidden_size, out_channels):
                super().__init__()
                self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
                self.linear = nn.Linear(hidden_size, out_channels, bias=True)

            def forward(self, x):
                x = self.norm_final(x)
                x = self.linear(x)
                return x


        # TODO: DUPLICATE THE CROSS ATTENTION LAYER BASED ON DEPTH
        self.blocks = nn.ModuleList(
            [
                LinearCrossBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.final_layer = LinearFinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of linear regression model.
        """

        if self.model_cfg.center_noise:
            raise NotImplementedError("Center noise not implemented for linear regression model.")
        if self.model_cfg.rotary:
            raise NotImplementedError("Rotary not implemented for linear regression model.")
        
        # encode x and y
        x = torch.transpose(self.x_embedder(x), -1, -2)
        y = torch.transpose(self.y_embedder(y), -1, -2)
        # forward pass through cross attention blocks
        for block in self.blocks:
            x = block(x, y)
        
        # final layer
        x = self.final_layer(x)
        x = torch.transpose(x, -1, -2)
        return x
    

class DiT_PointCloud_Cross(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        x_encoder_hidden_dims = hidden_size
        if self.model_cfg.x_encoder is not None and self.model_cfg.x0_encoder is not None:
            # We are concatenating x and x0 features so we halve the hidden size
            x_encoder_hidden_dims = hidden_size // 2

        # Encoder for current timestep x features       
        if self.model_cfg.x_encoder == "mlp":
            # x_embedder is conv1d layer instead of 2d patch embedder
            self.x_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")
        
        # Encoder for y features
        if self.model_cfg.y_encoder == "mlp":
            self.y_embedder = nn.Conv1d(
                in_channels,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.y_encoder == "dgcnn":
            self.y_embedder = DGCNN(
                input_dims=in_channels, emb_dims=hidden_size
            )
        else:
            raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

        # Encoder for x0 features
        if self.model_cfg.x0_encoder == "mlp":
            self.x0_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.x0_encoder == "dgcnn":
            self.x0_embedder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif self.model_cfg.x0_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")
        
        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTRelativeCrossBlock if self.model_cfg.rotary else DiTCrossBlock
        self.blocks = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        # rotary position embedding, if enabled
        if self.model_cfg.rotary:
            x_pos = self.rotary_pos_enc(x.permute(0, 2, 1))
            y_pos = self.rotary_pos_enc(y.permute(0, 2, 1))

        # encode x, y, x0 features
        x_emb = self.x_embedder(x)

        if self.model_cfg.x0_encoder is not None:
            assert x0 is not None, "x0 features must be provided if x0_encoder is not None"
            x0_emb = self.x0_embedder(x0)
            x_emb = torch.cat([x_emb, x0_emb], dim=1)

        if self.model_cfg.y_encoder is not None:
            y_emb = self.y_embedder(y)
            y_emb = y_emb.permute(0, 2, 1)

        x = x_emb.permute(0, 2, 1)

        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks:
            if self.model_cfg.rotary:
                x = block(x, y_emb, t_emb, x_pos, y_pos)
            else:
                x = block(x, y_emb, t_emb)

        # final layer
        x = self.final_layer(x, t_emb)
        x = x.permute(0, 2, 1)
        return x



#################################################################################
#                                DDRD DiT Models                                #
#################################################################################

class Joint_DiT_Deformation_Reference_Cross_Feature(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        x_encoder_hidden_dims = hidden_size
        if self.model_cfg.x_encoder is not None and self.model_cfg.x0_encoder is not None:
            # We are concatenating x and x0 features so we halve the hidden size
            x_encoder_hidden_dims = hidden_size // 2

        # Encoder for current timestep x features       
        if self.model_cfg.x_encoder == "mlp":
            # x_embedder is conv1d layer instead of 2d patch embedder
            self.x_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")
        
        # Encoder for y features
        if self.model_cfg.y_encoder == "mlp":
            self.y_embedder = nn.Conv1d(
                in_channels,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.y_encoder == "dgcnn":
            self.y_embedder = DGCNN(
                input_dims=in_channels, emb_dims=hidden_size
            )
        else:
            raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

        # Encoder for x0 features
        if self.model_cfg.x0_encoder == "mlp":
            self.x0_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.x0_encoder == "dgcnn":
            self.x0_embedder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif self.model_cfg.x0_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")
        
        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTRelativeCrossBlock if self.model_cfg.rotary else DiTCrossBlock
        self.blocks = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        # noise-centering, if enabled
        if self.model_cfg.type == "point":
            delta_center = torch.mean(x, dim=2, keepdim=True)
            x = x - delta_center
            y = y - delta_center
            x0 = x0 - delta_center
        elif self.model_cfg.type == "flow":
            reconstruction = x + x0
            delta_center = torch.mean(reconstruction, dim=2, keepdim=True)
            x = x - delta_center
            y = y - delta_center
            x0 = x0 - delta_center


        # rotary position embedding, if enabled
        if self.model_cfg.rotary:
            x_pos = self.rotary_pos_enc(x.permute(0, 2, 1))
            y_pos = self.rotary_pos_enc(y.permute(0, 2, 1))

        # encode x, y, x0 features
        x_emb = self.x_embedder(x)

        if self.model_cfg.x0_encoder is not None:
            assert x0 is not None, "x0 features must be provided if x0_encoder is not None"
            x0_emb = self.x0_embedder(x0)
            x_emb = torch.cat([x_emb, x0_emb], dim=1)

        if self.model_cfg.y_encoder is not None:
            y_emb = self.y_embedder(y)
            y_emb = y_emb.permute(0, 2, 1)

        x = x_emb.permute(0, 2, 1)

        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks:
            if self.model_cfg.rotary:
                x = block(x, y_emb, t_emb, x_pos, y_pos)
            else:
                x = block(x, y_emb, t_emb)

        # final layer
        # (8,512, 128)
        x = self.final_layer(x, t_emb)
        # (8,512, 6)

        x = x.permute(0, 2, 1)

        return x, delta_center


class Separate_DiT_Deformation_Reference_Cross_Feature(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        x_encoder_hidden_dims = hidden_size
        if self.model_cfg.x_encoder is not None and self.model_cfg.x0_encoder is not None:
            # We are concatenating x and x0 features so we halve the hidden size
            x_encoder_hidden_dims = hidden_size // 2
        
        # Encoder for current timestep x features       
        if self.model_cfg.x_encoder == "mlp":
            # x_embedder is conv1d layer instead of 2d patch embedder
            self.x_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims//2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")
        
        self.xr_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims//2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
        )

        self.xs_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims//2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
        )

        # Encoder for y features
        if self.model_cfg.y_encoder == "mlp":
            self.y_embedder = nn.Conv1d(
                in_channels,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.y_encoder == "dgcnn":
            self.y_embedder = DGCNN(
                input_dims=in_channels, emb_dims=hidden_size
            )
        else:
            raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

        # Encoder for x0 features
        if self.model_cfg.x0_encoder == "mlp":
            self.x0_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.x0_encoder == "dgcnn":
            self.x0_embedder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif self.model_cfg.x0_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")
        
        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTRelativeCrossBlock if self.model_cfg.rotary else DiTCrossBlock
        self.blocks_r = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.blocks_s = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        # functionally setting patch size to 1 for a point cloud
        self.final_layer_r = FinalLayer_r(hidden_size, 1, self.out_channels)
        self.final_layer_s = FinalLayer_s(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        wr = self.xr_embedder.weight.data
        nn.init.xavier_uniform_(wr.view([wr.shape[0], -1]))
        nn.init.constant_(self.xr_embedder.bias, 0)
        ws = self.xs_embedder.weight.data
        nn.init.xavier_uniform_(ws.view([ws.shape[0], -1]))
        nn.init.constant_(self.xs_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks_r:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.blocks_s:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_r.linear.weight, 0)
        nn.init.constant_(self.final_layer_r.linear.bias, 0)

        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_s.linear.weight, 0)
        nn.init.constant_(self.final_layer_s.linear.bias, 0)

    def forward(
            self,
            xr_t: torch.Tensor,
            xs_t: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        # noise-centering, if enabled
        
        #if self.model_cfg.type == "point":
        #    delta_center = torch.mean(x, dim=2, keepdim=True)
        #    x = x - delta_center
        #    y = y - delta_center
        #    x0 = x0 - delta_center
        #elif self.model_cfg.type == "flow":
        #    reconstruction = x + x0
        #    delta_center = torch.mean(reconstruction, dim=2, keepdim=True)
        #    x = x - delta_center
        #    y = y - delta_center
        #    x0 = x0 - delta_center
        

        # rotary position embedding, if enabled
        if self.model_cfg.rotary:
            x_pos = self.rotary_pos_enc(x.permute(0, 2, 1))
            y_pos = self.rotary_pos_enc(y.permute(0, 2, 1))

        # encode x, y, x0 features
        
        x = xr_t + xs_t
        recon_emb = self.x_embedder(x)
        
        xr_emb = self.xr_embedder(xr_t)
        xr_emb_exp = xr_emb.expand(-1, -1, recon_emb.size(-1))  # [B, hidden_half, N]
        xr_emb = torch.cat([xr_emb_exp, recon_emb], dim=1)

        xs_emb = self.xs_embedder(xs_t)
        xs_emb = torch.cat([xs_emb, recon_emb], dim=1)

        if self.model_cfg.x0_encoder is not None:
            assert x0 is not None, "x0 features must be provided if x0_encoder is not None"
            x0_emb = self.x0_embedder(x0)
            xr_emb = torch.cat([xr_emb, x0_emb], dim=1)
            xs_emb = torch.cat([xs_emb, x0_emb], dim=1)

        if self.model_cfg.y_encoder is not None:
            y_emb = self.y_embedder(y)
            y_emb = y_emb.permute(0, 2, 1)

        xr = xr_emb.permute(0, 2, 1)
        xs = xs_emb.permute(0, 2, 1)

        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks_r:
            if self.model_cfg.rotary:
                xr = block(xr, y_emb, t_emb, x_pos, y_pos)
            else:
                xr = block(xr, y_emb, t_emb)
        for block in self.blocks_s:
            if self.model_cfg.rotary:
                xs = block(xs, y_emb, t_emb, x_pos, y_pos)
            else:
                xs = block(xs, y_emb, t_emb)
        # (8,512, 128)

        # final layer
        xs = self.final_layer_s(xs, t_emb) # (8,512, 6)

        xr = self.final_layer_r(xr, t_emb) # (8,1, 6)

        xs = xs.permute(0, 2, 1)
        xr = xr.permute(0, 2, 1)

        return xr, xs
    
'''


class Separate_DiT_Deformation_Reference_Cross_Feature(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        x_encoder_hidden_dims = hidden_size
        if self.model_cfg.x_encoder is not None and self.model_cfg.x0_encoder is not None:
            # We are concatenating x and x0 features so we halve the hidden size
            x_encoder_hidden_dims = hidden_size // 2
        
        # Encoder for current timestep x features       
        if self.model_cfg.x_encoder == "mlp":
            # x_embedder is conv1d layer instead of 2d patch embedder
            self.x_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims // 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")

        self.xs_embedder = nn.Conv1d(
            in_channels,
            x_encoder_hidden_dims // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Encoder for y features
        if self.model_cfg.y_encoder == "mlp":
            self.y_embedder = nn.Conv1d(
                in_channels,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.y_encoder == "dgcnn":
            self.y_embedder = DGCNN(
                input_dims=in_channels, emb_dims=hidden_size
            )
        else:
            raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

        # Encoder for x0 features
        if self.model_cfg.x0_encoder == "mlp":
            self.x0_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.x0_encoder == "dgcnn":
            self.x0_embedder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif self.model_cfg.x0_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")
        
        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTRelativeCrossBlock if self.model_cfg.rotary else DiTCrossBlock
        self.blocks = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # functionally setting patch size to 1 for a point cloud
        self.final_layer_r = FinalLayer(hidden_size, 1, self.out_channels)
        self.final_layer_s = FinalLayer(hidden_size, 1, self.out_channels)
        
        self.ref_frame_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        ws = self.xs_embedder.weight.data
        nn.init.xavier_uniform_(ws.view([w.shape[0], -1]))
        nn.init.constant_(self.xs_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_r.linear.weight, 0)
        nn.init.constant_(self.final_layer_r.linear.bias, 0)

        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_s.linear.weight, 0)
        nn.init.constant_(self.final_layer_s.linear.bias, 0)

    def forward(
            self,
            xr_t: torch.Tensor,
            xs_t: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        # noise-centering, if enabled
        
        #if self.model_cfg.type == "point":
        #    delta_center = torch.mean(x, dim=2, keepdim=True)
        #    x = x - delta_center
        #    y = y - delta_center
        #    x0 = x0 - delta_center
        #elif self.model_cfg.type == "flow":
        #    reconstruction = x + x0
        #    delta_center = torch.mean(reconstruction, dim=2, keepdim=True)
        #    x = x - delta_center
        #    y = y - delta_center
        #    x0 = x0 - delta_center
        

        # encode x, y, x0 features
        x = xr_t + xs_t        
        x_emb = self.x_embedder(x)
        shape_emb = self.xs_embedder(xs_t)
        x_emb = torch.cat([x_emb, shape_emb], dim=1)

        if self.model_cfg.x0_encoder is not None:
            assert x0 is not None, "x0 features must be provided if x0_encoder is not None"
            x0_emb = self.x0_embedder(x0)
            x_emb = torch.cat([x_emb, x0_emb], dim=1)

        if self.model_cfg.y_encoder is not None:
            y_emb = self.y_embedder(y)
            y_emb = y_emb.permute(0, 2, 1)

        x = x_emb.permute(0, 2, 1)

        token = self.ref_frame_token.expand(x.size(0), 1, self.ref_frame_token.size(-1))
        x = torch.cat([x, token], dim=1)

        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks:
            x = block(x, y_emb, t_emb)
        # (8,512, 128)
        
        xr_token = x[:, -1:, :]
        xs_token = x[:, :-1, :]

        # final layer
        xs = self.final_layer_s(xs_token, t_emb) # (8,512, 6)

        xr = self.final_layer_r(xr_token, t_emb) # (8,1, 6)

        xs = xs.permute(0, 2, 1)
        xr = xr.permute(0, 2, 1)

        return xr, xs
'''
#################################################################################
#                               Mu-Frame Model                                  #
#################################################################################
class Mu_DiT_Take1(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        x_encoder_hidden_dims = hidden_size
        if self.model_cfg.x_encoder is not None and self.model_cfg.x0_encoder is not None:
            # We are concatenating x and x0 features so we halve the hidden size
            x_encoder_hidden_dims = hidden_size // 2
        
        # Encoder for current timestep x features       
        if self.model_cfg.x_encoder == "mlp":
            # x_embedder is conv1d layer instead of 2d patch embedder
            self.x_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims//2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")
        
        self.xr_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims//2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
        )

        self.xs_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims//2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
        )

        # Encoder for y features
        if self.model_cfg.y_encoder == "mlp":
            self.y_embedder = nn.Conv1d(
                in_channels,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.y_encoder == "dgcnn":
            self.y_embedder = DGCNN(
                input_dims=in_channels, emb_dims=hidden_size
            )
        else:
            raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

        # Encoder for x0 features
        if self.model_cfg.x0_encoder == "mlp":
            self.x0_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.x0_encoder == "dgcnn":
            self.x0_embedder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif self.model_cfg.x0_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")
        
        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTRelativeCrossBlock if self.model_cfg.rotary else DiTCrossBlock
        self.blocks_r = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.blocks_s = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        # functionally setting patch size to 1 for a point cloud
        self.final_layer_r = FinalLayer_r(hidden_size, 1, self.out_channels)
        self.final_layer_s = FinalLayer_s(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        '''
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)
        '''
        wr = self.xr_embedder.weight.data
        nn.init.xavier_uniform_(wr.view([wr.shape[0], -1]))
        nn.init.constant_(self.xr_embedder.bias, 0)
        ws = self.xs_embedder.weight.data
        nn.init.xavier_uniform_(ws.view([ws.shape[0], -1]))
        nn.init.constant_(self.xs_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks_r:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.blocks_s:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_r.linear.weight, 0)
        nn.init.constant_(self.final_layer_r.linear.bias, 0)

        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_s.linear.weight, 0)
        nn.init.constant_(self.final_layer_s.linear.bias, 0)

    def forward(
            self,
            xr_t: torch.Tensor,
            xs_t: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        # noise-centering, if enabled
        '''
        if self.model_cfg.type == "point":
            delta_center = torch.mean(x, dim=2, keepdim=True)
            x = x - delta_center
            y = y - delta_center
            x0 = x0 - delta_center
        elif self.model_cfg.type == "flow":
            reconstruction = x + x0
            delta_center = torch.mean(reconstruction, dim=2, keepdim=True)
            x = x - delta_center
            y = y - delta_center
            x0 = x0 - delta_center
        '''

        # rotary position embedding, if enabled
        if self.model_cfg.rotary:
            x_pos = self.rotary_pos_enc(x.permute(0, 2, 1))
            y_pos = self.rotary_pos_enc(y.permute(0, 2, 1))

        y = y - xr_t

        # encode x, y, x0 features        
        x = xr_t + xs_t
        recon_emb = self.x_embedder(x)
        
        xr_emb = self.xr_embedder(xr_t)
        xr_emb_exp = xr_emb.expand(-1, -1, recon_emb.size(-1))  # [B, hidden_half, N]
        xr_emb = torch.cat([xr_emb_exp, recon_emb], dim=1)

        xs_emb = self.xs_embedder(xs_t)
        xs_emb = torch.cat([xs_emb, recon_emb], dim=1)

        if self.model_cfg.x0_encoder is not None:
            assert x0 is not None, "x0 features must be provided if x0_encoder is not None"
            x0_emb = self.x0_embedder(x0)
            xr_emb = torch.cat([xr_emb, x0_emb], dim=1)
            xs_emb = torch.cat([xs_emb, x0_emb], dim=1)

        if self.model_cfg.y_encoder is not None:
            y_emb = self.y_embedder(y)
            y_emb = y_emb.permute(0, 2, 1)

        xr = xr_emb.permute(0, 2, 1)
        xs = xs_emb.permute(0, 2, 1)

        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks_r:
            if self.model_cfg.rotary:
                xr = block(xr, y_emb, t_emb, x_pos, y_pos)
            else:
                xr = block(xr, y_emb, t_emb)
        for block in self.blocks_s:
            if self.model_cfg.rotary:
                xs = block(xs, y_emb, t_emb, x_pos, y_pos)
            else:
                xs = block(xs, y_emb, t_emb)
        # (8,512, 128)

        # final layer
        xs = self.final_layer_s(xs, t_emb) # (8,512, 6)

        xr = self.final_layer_r(xr, t_emb) # (8,1, 6)

        xs = xs.permute(0, 2, 1)
        xr = xr.permute(0, 2, 1)

        return xr, xs

class Mu_DiT_Take2(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        x_encoder_hidden_dims = hidden_size
        if self.model_cfg.x_encoder is not None and self.model_cfg.x0_encoder is not None:
            # We are concatenating x and x0 features so we halve the hidden size
            x_encoder_hidden_dims = hidden_size // 2
        
        # Encoder for current timestep x features       
        if self.model_cfg.x_encoder == "mlp":
            # x_embedder is conv1d layer instead of 2d patch embedder
            self.x_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")
        
 
        # Encoder for y features
        if self.model_cfg.y_encoder == "mlp":
            self.y_embedder = nn.Conv1d(
                in_channels,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.y_encoder == "dgcnn":
            self.y_embedder = DGCNN(
                input_dims=in_channels, emb_dims=hidden_size
            )
        else:
            raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

        # Encoder for x0 features
        if self.model_cfg.x0_encoder == "mlp":
            self.x0_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.x0_encoder == "dgcnn":
            self.x0_embedder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif self.model_cfg.x0_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")
        
        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTRelativeCrossBlock if self.model_cfg.rotary else DiTCrossBlock
        self.blocks = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # functionally setting patch size to 1 for a point cloud
        self.final_layer_r = FinalLayer(hidden_size, 1, self.out_channels)
        self.final_layer_s = FinalLayer(hidden_size, 1, self.out_channels)
        
        self.ref_frame_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        '''
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)
        '''
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_r.linear.weight, 0)
        nn.init.constant_(self.final_layer_r.linear.bias, 0)

        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_s.linear.weight, 0)
        nn.init.constant_(self.final_layer_s.linear.bias, 0)

    def forward(
            self,
            xr_t: torch.Tensor,
            xs_t: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        # noise-centering, if enabled
        '''
        if self.model_cfg.type == "point":
            delta_center = torch.mean(x, dim=2, keepdim=True)
            x = x - delta_center
            y = y - delta_center
            x0 = x0 - delta_center
        elif self.model_cfg.type == "flow":
            reconstruction = x + x0
            delta_center = torch.mean(reconstruction, dim=2, keepdim=True)
            x = x - delta_center
            y = y - delta_center
            x0 = x0 - delta_center
        '''

        # rotary position embedding, if enabled
        if self.model_cfg.rotary:
            x_pos = self.rotary_pos_enc(x.permute(0, 2, 1))
            y_pos = self.rotary_pos_enc(y.permute(0, 2, 1))

        y = y - xr_t

        # encode x, y, x0 features        
        xs_emb = self.x_embedder(xs_t)

        if self.model_cfg.x0_encoder is not None:
            assert x0 is not None, "x0 features must be provided if x0_encoder is not None"
            x0_emb = self.x0_embedder(x0)
            xs_emb = torch.cat([xs_emb, x0_emb], dim=1)

        if self.model_cfg.y_encoder is not None:
            y_emb = self.y_embedder(y)
            y_emb = y_emb.permute(0, 2, 1)

        xs = xs_emb.permute(0, 2, 1)

        token = self.ref_frame_token.expand(xs.size(0), 1, self.ref_frame_token.size(-1))
        xs = torch.cat([xs, token], dim=1)

        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks:
            if self.model_cfg.rotary:
                xs = block(xs, y_emb, t_emb, x_pos, y_pos)
            else:
                xs = block(xs, y_emb, t_emb)
        # (8,512, 128)
        
        xr_token = xs[:, -1:, :]
        xs_token = xs[:, :-1, :]

        # final layer
        xs = self.final_layer_s(xs_token, t_emb) # (8,512, 6)

        xr = self.final_layer_r(xr_token, t_emb) # (8,1, 6)

        xs = xs.permute(0, 2, 1)
        xr = xr.permute(0, 2, 1)

        return xr, xs

class Mu_DiT_Take3(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        x_encoder_hidden_dims = hidden_size
        if self.model_cfg.x_encoder is not None and self.model_cfg.x0_encoder is not None:
            # We are concatenating x and x0 features so we halve the hidden size
            x_encoder_hidden_dims = hidden_size // 2
        
        # Encoder for current timestep x features       
        if self.model_cfg.x_encoder == "mlp":
            # x_embedder is conv1d layer instead of 2d patch embedder
            self.x_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")
        
        self.recon_embedder = nn.Conv1d(
            in_channels,
            x_encoder_hidden_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )  
 
        # Encoder for y features
        if self.model_cfg.y_encoder == "mlp":
            self.y_embedder = nn.Conv1d(
                in_channels,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.y_encoder == "dgcnn":
            self.y_embedder = DGCNN(
                input_dims=in_channels, emb_dims=hidden_size
            )
        else:
            raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

        self.y_embedder_local = nn.Conv1d(
            in_channels,
            hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Encoder for x0 features
        if self.model_cfg.x0_encoder == "mlp":
            self.x0_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.x0_encoder == "dgcnn":
            self.x0_embedder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif self.model_cfg.x0_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")
        
        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTRelativeCrossBlock if self.model_cfg.rotary else DiTCrossBlock
        self.blocks_r = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.blocks_s = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        # functionally setting patch size to 1 for a point cloud
        self.final_layer_r = FinalLayer(hidden_size, 1, self.out_channels)
        self.final_layer_s = FinalLayer(hidden_size, 1, self.out_channels)
        
        self.ref_frame_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)
        w_recon = self.recon_embedder.weight.data
        nn.init.xavier_uniform_(w_recon.view([w_recon.shape[0], -1]))
        nn.init.constant_(self.recon_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks_r:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.blocks_s:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_r.linear.weight, 0)
        nn.init.constant_(self.final_layer_r.linear.bias, 0)

        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_s.linear.weight, 0)
        nn.init.constant_(self.final_layer_s.linear.bias, 0)

    def forward(
            self,
            xr_t: torch.Tensor,
            xs_t: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        # noise-centering, if enabled
        '''
        if self.model_cfg.type == "point":
            delta_center = torch.mean(x, dim=2, keepdim=True)
            x = x - delta_center
            y = y - delta_center
            x0 = x0 - delta_center
        elif self.model_cfg.type == "flow":
            reconstruction = x + x0
            delta_center = torch.mean(reconstruction, dim=2, keepdim=True)
            x = x - delta_center
            y = y - delta_center
            x0 = x0 - delta_center
        '''

        recon = xr_t + xs_t
        # encode x, y, x0 features        
        xs_emb = self.x_embedder(xs_t)

        if self.model_cfg.x0_encoder is not None:
            assert x0 is not None, "x0 features must be provided if x0_encoder is not None"
            x0_emb = self.x0_embedder(x0)
            s_emb = torch.cat([xs_emb, x0_emb], dim=1)
        s_emb = s_emb.permute(0, 2, 1)

        if self.model_cfg.y_encoder is not None:
            y_local = y - xr_t
            y_emb_local = self.y_embedder_local(y_local)
            y_emb_local = y_emb_local.permute(0, 2, 1)

            y_emb_global = self.y_embedder(y)
            y_emb_global = y_emb_global.permute(0, 2, 1)

        #recon_emb = self.recon_embedder(xs_t)
        recon_emb = self.recon_embedder(recon)
        recon_emb = torch.cat([recon_emb, x0_emb], dim=1)
        recon_emb = recon_emb.permute(0, 2, 1)
        token = self.ref_frame_token.expand(recon_emb.size(0), 1, self.ref_frame_token.size(-1))
        r_emb = torch.cat([recon_emb, token], dim=1)

        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks_r:
            xr = block(r_emb, y_emb_global, t_emb)
        for block in self.blocks_s:
            xs = block(s_emb, y_emb_local, t_emb)
        # (8,512, 128)
        
        xr_token = xr[:, -1:, :]
        xs_token = xs

        # final layer
        xs = self.final_layer_s(xs_token, t_emb) # (8,512, 6)

        xr = self.final_layer_r(xr_token, t_emb) # (8,1, 6)

        xs = xs.permute(0, 2, 1)
        xr = xr.permute(0, 2, 1)

        return xr, xs

class Mu_DiT_Take4(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        x_encoder_hidden_dims = hidden_size
        if self.model_cfg.x_encoder is not None and self.model_cfg.x0_encoder is not None:
            # We are concatenating x and x0 features so we halve the hidden size
            x_encoder_hidden_dims = hidden_size // 2
        
        # Encoder for current timestep x features       
        if self.model_cfg.x_encoder == "mlp":
            # x_embedder is conv1d layer instead of 2d patch embedder
            self.x_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")
        
 
        # Encoder for y features
        if self.model_cfg.y_encoder == "mlp":
            self.y_embedder = nn.Conv1d(
                in_channels,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.y_encoder == "dgcnn":
            self.y_embedder = DGCNN(
                input_dims=in_channels, emb_dims=hidden_size
            )
        else:
            raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

        # Encoder for x0 features
        if self.model_cfg.x0_encoder == "mlp":
            self.x0_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.x0_encoder == "dgcnn":
            self.x0_embedder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif self.model_cfg.x0_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")
        
        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn_s = DiTRelativeCrossBlock if self.model_cfg.rotary else DiTCrossBlock
        block_fn_r = DiTRefBlock
        self.blocks_r = nn.ModuleList(
            [
                block_fn_r(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.blocks_s = nn.ModuleList(
            [
                block_fn_s(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        # functionally setting patch size to 1 for a point cloud
        self.final_layer_r = FinalLayer(hidden_size, 1, self.out_channels)
        self.final_layer_s = FinalLayer(hidden_size, 1, self.out_channels)
        
        self.ref_frame_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks_s:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_r.linear.weight, 0)
        nn.init.constant_(self.final_layer_r.linear.bias, 0)

        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_s.linear.weight, 0)
        nn.init.constant_(self.final_layer_s.linear.bias, 0)

    def forward(
            self,
            xr_t: torch.Tensor,
            xs_t: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        # noise-centering, if enabled
        '''
        if self.model_cfg.type == "point":
            delta_center = torch.mean(x, dim=2, keepdim=True)
            x = x - delta_center
            y = y - delta_center
            x0 = x0 - delta_center
        elif self.model_cfg.type == "flow":
            reconstruction = x + x0
            delta_center = torch.mean(reconstruction, dim=2, keepdim=True)
            x = x - delta_center
            y = y - delta_center
            x0 = x0 - delta_center
        '''
   
        xs_emb = self.x_embedder(xs_t)

        if self.model_cfg.x0_encoder is not None:
            assert x0 is not None, "x0 features must be provided if x0_encoder is not None"
            x0_emb = self.x0_embedder(x0)
            s_emb = torch.cat([xs_emb, x0_emb], dim=1)
        s_emb = s_emb.permute(0, 2, 1)

        if self.model_cfg.y_encoder is not None:
            y = y - xr_t
            y_emb = self.y_embedder(y)
            y_emb = y_emb.permute(0, 2, 1)
        
        #recon_emb = self.recon_embedder(xs_t)
        token = self.ref_frame_token.expand(s_emb.size(0), 1, self.ref_frame_token.size(-1))

        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks_r:
            xr = block(token, y_emb, t_emb)
        for block in self.blocks_s:
            xs = block(s_emb, y_emb, t_emb)
        # (8,512, 128)

        # final layer
        xs = self.final_layer_s(xs, t_emb) # (8,512, 6)

        xr = self.final_layer_r(xr, t_emb) # (8,1, 6)

        xs = xs.permute(0, 2, 1)
        xr = xr.permute(0, 2, 1)

        return xr, xs

#################################################################################
#                                TAX3Dv2 Models                                 #
#################################################################################
class TAX3Dv2_MuFrame_DiT(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention, 
    and joint-feature encoding.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Initializing point cloud encoder wrapper.
        if self.model_cfg.point_encoder == "mlp":
            encoder_fn = partial(mlp_encoder, in_channels=self.in_channels)
        elif self.model_cfg.point_encoder == "pn2":
            encoder_fn = partial(pn2_encoder, in_channels=self.in_channels, model_cfg=self.model_cfg)
        else:
            raise ValueError(f"Invalid point_encoder: {self.model_cfg.point_encoder}")
        
        # Creating base encoders - action-frame, and prediction frame.
        self.action_encoder = encoder_fn(out_channels=hidden_size)
        self.pred_encoder = encoder_fn(out_channels=hidden_size)

        # Creating extra feature encoders, if necessary.
        if self.model_cfg.feature:
            self.feature_encoder = encoder_fn(in_channels=9, out_channels=hidden_size)
            self.action_mixer = mlp_encoder(3 * hidden_size, hidden_size)
        else:
            self.action_mixer = mlp_encoder(2 * hidden_size, hidden_size)

        # Timestamp embedding.
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks.
        self.blocks = nn.ModuleList(
            [
                DiTCrossBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # Final layer; functionally setting patch size to 1 for a point cloud.
        self.final_layer_r = FinalLayer(hidden_size, 1, self.out_channels)
        self.final_layer_s = FinalLayer(hidden_size, 1, self.out_channels)
        
        self.ref_frame_token = nn.Parameter(torch.randn(1, 1, hidden_size))
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_r.linear.weight, 0)
        nn.init.constant_(self.final_layer_r.linear.bias, 0)

        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_s.linear.weight, 0)
        nn.init.constant_(self.final_layer_s.linear.bias, 0)

    def forward(
            self,
            xr_t: torch.Tensor,
            xs_t: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (torch.Tensor): (B, D, N) tensor of un-noised x (e.g. action) features
        """

        if self.model_cfg.type == "flow":
            x_flow = xs_t
            x_recon = xs_t + x0
        else:
            x_flow = xs_t - x0
            x_recon = xs_t

        # Dynamically center anchor
        y = y - xr_t

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
        token = self.ref_frame_token.expand(x_enc.size(0), 1, self.ref_frame_token.size(-1))
        x_enc = torch.cat([x_enc, token], dim=1)

        # Timestep embedding.
        t_emb = self.t_embedder(t)

        # Forward pass through DiT blocks.
        for block in self.blocks:
            x_enc = block(x_enc, anchor_pred_enc, t_emb)

        xr_token = x_enc[:, -1:, :]
        xs_token = x_enc[:, :-1, :]

        # Final layer
        xs = self.final_layer_s(xs_token, t_emb) # (8, 512, 6)

        xr = self.final_layer_r(xr_token, t_emb) # (8, 1, 6)

        xs = xs.permute(0, 2, 1)
        xr = xr.permute(0, 2, 1)

        return xr, xs

class TAX3Dv2_FixedFrame_Token_DiT(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention, 
    and joint-feature encoding.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Initializing point cloud encoder wrapper.
        if self.model_cfg.point_encoder == "mlp":
            encoder_fn = partial(mlp_encoder, in_channels=self.in_channels)
        elif self.model_cfg.point_encoder == "pn2":
            encoder_fn = partial(pn2_encoder, in_channels=self.in_channels, model_cfg=self.model_cfg)
        else:
            raise ValueError(f"Invalid point_encoder: {self.model_cfg.point_encoder}")
        
        # Creating base encoders - action-frame, and prediction frame.
        self.action_encoder = encoder_fn(out_channels=hidden_size)
        self.pred_encoder = encoder_fn(out_channels=hidden_size)

        # Creating extra feature encoders, if necessary.
        if self.model_cfg.feature:
            self.feature_encoder = encoder_fn(in_channels=9, out_channels=hidden_size)
            self.action_mixer = mlp_encoder(3 * hidden_size, hidden_size)
        else:
            self.action_mixer = mlp_encoder(2 * hidden_size, hidden_size)

        # Timestamp embedding.
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks.
        self.blocks = nn.ModuleList(
            [
                DiTCrossBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # Final layer; functionally setting patch size to 1 for a point cloud.
        self.final_layer_r = FinalLayer(hidden_size, 1, self.out_channels)
        self.final_layer_s = FinalLayer(hidden_size, 1, self.out_channels)
        
        self.ref_frame_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            xr_t: torch.Tensor,
            xs_t: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (torch.Tensor): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        x = xr_t + xs_t
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
        token = self.ref_frame_token.expand(x_enc.size(0), 1, self.ref_frame_token.size(-1))
        x_enc = torch.cat([x_enc, token], dim=1)

        # Timestep embedding.
        t_emb = self.t_embedder(t)

        # Forward pass through DiT blocks.
        for block in self.blocks:
            x_enc = block(x_enc, anchor_pred_enc, t_emb)

        xr_token = x_enc[:, -1:, :]
        xs_token = x_enc[:, :-1, :]

        # Final layer
        xs = self.final_layer_s(xs_token, t_emb) # (8, 512, 6)

        xr = self.final_layer_r(xr_token, t_emb) # (8, 1, 6)

        xs = xs.permute(0, 2, 1)
        xr = xr.permute(0, 2, 1)

        return xr, xs

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}



##############################################################################################
###################################### OLD MODELS ############################################
##############################################################################################
### CREATING NEW DiT (unconditional) FOR POINT CLOUD INPUTS ###
class DiT_PointCloud_Unc(nn.Module):
    """
    Diffusion model with a Transformer backbone - point cloud, unconditional.
    """
    def __init__(
        self,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        # x_embedder is conv1d layer instead of 2d patch embedder
        self.x_embedder = nn.Conv1d(in_channels, hidden_size, kernel_size=1, stride=1, padding=0, bias=True)
        # no pos_embed, or y_embedder
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
            x0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of DiT.
        x: (N, L, 3) tensor of spatial inputs (point clouds)
        t: (N,) tensor of diffusion timesteps
        """
        # concat x and pos
        x = torch.cat((x, x0), dim=1)
        x = torch.transpose(self.x_embedder(x), -1, -2)
        c = self.t_embedder(t)

        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = torch.transpose(x, -1, -2)
        return x


class DiT_PointCloud(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses scene-level self-attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        # Encoder for current timestep x features
        self.x_embedder = nn.Conv1d(
            in_channels,
            hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTRelativeBlock if self.model_cfg.rotary else DiTBlock
        self.blocks = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            x0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of DiT.
        x: (N, L, 3) tensor of spatial inputs (point clouds)
        t: (N,) tensor of diffusion timesteps
        x0: (N, L, 3) tensor of un-noised x (e.g. scene) features
        """
        # noise-centering, if enabled
        if self.model_cfg.center_noise:
            relative_center = torch.mean(x, dim=2, keepdim=True)
            x = x - relative_center
            x0 = x0 - relative_center

        # rotary position embedding, if enabled
        if self.model_cfg.rotary:
            x_pos = self.rotary_pos_enc(x.permute(0, 2, 1))

        # encode x, x0 features
        x = torch.cat((x, x0), dim=1)
        x = torch.transpose(self.x_embedder(x), -1, -2)

        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks:
            if self.model_cfg.rotary:
                x = block(x, t_emb, x_pos)
            else:
                x = block(x, t_emb)

        # final layer
        x = self.final_layer(x, t_emb)
        x = torch.transpose(x, -1, -2)
        return x


class PN2_DiT_PointCloud(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        x_encoder_hidden_dims = hidden_size
        if self.model_cfg.x_pre_encoder is not None:
            # We are concatenating x and x0 features so we halve the hidden size
            x_encoder_hidden_dims = hidden_size // 2

        # Encoder for current timestep x features       
        if self.model_cfg.x_pre_encoder == "mlp":
            print("Using mlp for pre-encoding x")
            self.x_pre_encoder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif not self.model_cfg.x_pre_encoder:
            self.x_pre_encoder = None
            print("No pre-encoding for x")
        else:
            raise ValueError(f"Invalid x_pre_encoder: {self.model_cfg.x_pre_encoder}")
        
        # Encoder for y features
        if self.model_cfg.y_pre_encoder == "mlp":
            print("Using mlp for pre-encoding y")
            self.y_pre_encoder = nn.Conv1d(
                in_channels,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.y_pre_encoder == "dgcnn":
            self.y_pre_encoder = DGCNN(
                input_dims=in_channels, emb_dims=hidden_size
            )
        elif not self.model_cfg.y_pre_encoder:
            self.y_pre_encoder = None
            print("No pre-encoding for y")
        else:
            raise ValueError(f"Invalid y_pre_encoder: {self.model_cfg.y_pre_encoder}")            

        # Encoder for x0 features
        if self.model_cfg.x0_pre_encoder == "mlp":
            print("Using mlp for pre-encoding x0")
            self.x0_pre_encoder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.x0_pre_encoder == "dgcnn":
            self.x0_pre_encoder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif not self.model_cfg.x0_pre_encoder:
            self.x0_pre_encoder = None
            print("No pre-encoding for x0")
        elif self.model_cfg.x0_pre_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_pre_encoder: {self.model_cfg.x0_pre_encoder}")

        # x embedding (pcd features)
        self.x_embedder = pn2_pyg.PN2Dense(
            in_channels=3,  # additional 3 channels for x0 feature
            out_channels=hidden_size,
            p=pn2_pyg.PN2DenseParams(),
        )

        # y embedding (pcd features)
        self.y_embedder = pn2_pyg.PN2Dense(
            in_channels=0,  # no additional feature
            out_channels=hidden_size,
            p=pn2_pyg.PN2DenseParams(),
        )

        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTBlock
        self.blocks = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        
        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        if self.x_pre_encoder is not None:
            w = self.x_pre_encoder.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_pre_encoder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        # noise-centering, if enabled
        if self.model_cfg.center_noise:
            relative_center = torch.mean(x, dim=2, keepdim=True)
            x = x - relative_center
            y = y - relative_center
        
        # rotary position embedding, if enabled
        if self.model_cfg.rotary:
            x_pos = self.rotary_pos_enc(x.permute(0, 2, 1))
            y_pos = self.rotary_pos_enc(y.permute(0, 2, 1))

        # pre-encode x, y, x0 features if required
        if self.x_pre_encoder is not None:
            x = self.x_pre_encoder(x)

        if self.x0_pre_encoder is not None:
            assert x0 is not None, "x0 features must be provided if x0_encoder is not None"
            x0 = self.x0_pre_encoder(x0)

        if self.y_pre_encoder is not None:
            y = self.y_pre_encoder(y)


        # encode x and y features using Pointnet++ 
        assert self.x_embedder is not None
        assert self.y_embedder is not None
        '''
        context.x = (
            torch.flatten(x, start_dim=2, end_dim=3).permute(0, 2, 1).reshape(-1, 3)
        )
        encoded_pcd = self.x_embedder(context.cuda())
        '''
        '''
        ### VARIANT 1 ###
        B, D, N = y.shape
        
        # construct features
        f_x = x.permute(0, 2, 1).reshape(-1, D)  # Shape: (B * N, D*2), 2 for 2 features
        f_y = None

        # construct pos
        pos_x = x0.permute(0, 2, 1).reshape(-1, D)  # using original action as positional grouping for PN2
        pos_y = y.permute(0, 2, 1).reshape(-1, D)   # using original anchor as positional grouping for PN2

        # construct batch
        batch_idx_x = torch.arange(B).repeat_interleave(N)  # Shape: (B * N,)
        batch_idx_y = torch.arange(B).repeat_interleave(N)  # Shape: (B * N,)

        data_x = Data(x=f_x, pos=pos_x, batch=batch_idx_x)
        data_y = Data(x=f_y, pos=pos_y, batch=batch_idx_y)

        # PN2 embedding
        x = self.x_embedder(data_x.cuda())
        y_emb = self.y_embedder(data_y.cuda())

        x = x.reshape(B, N, -1)             # reshape back to (B, N, d_embed)
        y_emb = y_emb.reshape(B, N, -1)     # reshape back to (B, N, d_embed)
        
        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # final layer
        x = self.final_layer(x, t_emb)
        x = x.permute(0, 2, 1)
        return x
        '''

        ### VARIANT 2 ###
        B, D, N = y.shape






        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # final layer
        x = self.final_layer(x, t_emb)
        x = x.permute(0, 2, 1)
        return x


class PN2_DiT_PointCloud_Cross(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        x_encoder_hidden_dims = hidden_size
        if self.model_cfg.x_pre_encoder is not None:
            # We are concatenating x and x0 features so we halve the hidden size
            x_encoder_hidden_dims = hidden_size // 2

        # Encoder for current timestep x features       
        if self.model_cfg.x_pre_encoder == "mlp":
            print("Using mlp for pre-encoding x")
            self.x_pre_encoder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif not self.model_cfg.x_pre_encoder:
            self.x_pre_encoder = None
            print("No pre-encoding for x")
        else:
            raise ValueError(f"Invalid x_pre_encoder: {self.model_cfg.x_pre_encoder}")
        
        # Encoder for y features
        if self.model_cfg.y_pre_encoder == "mlp":
            print("Using mlp for pre-encoding y")
            self.y_pre_encoder = nn.Conv1d(
                in_channels,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.y_pre_encoder == "dgcnn":
            self.y_pre_encoder = DGCNN(
                input_dims=in_channels, emb_dims=hidden_size
            )
        elif not self.model_cfg.y_pre_encoder:
            self.y_pre_encoder = None
            print("No pre-encoding for y")
        else:
            raise ValueError(f"Invalid y_pre_encoder: {self.model_cfg.y_pre_encoder}")            

        # Encoder for x0 features
        if self.model_cfg.x0_pre_encoder == "mlp":
            print("Using mlp for pre-encoding x0")
            self.x0_pre_encoder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.x0_pre_encoder == "dgcnn":
            self.x0_pre_encoder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif not self.model_cfg.x0_pre_encoder:
            self.x0_pre_encoder = None
            print("No pre-encoding for x0")
        elif self.model_cfg.x0_pre_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_pre_encoder: {self.model_cfg.x0_pre_encoder}")
        
        ######################################################
        #TODO: Assuming seperating encoding for x and y
        #TODO: Scene level PN2 encoding (i.e. combine x and y)
        ######################################################

        '''
        ### VARIANT 1.1.1 ###
        # We'll use pn2_pyg.PN2SAGA code bade
        # We'll use PN2 as encoder for x0+x and y seperatelt. 
        # 1. For the PN2 for action object, noise x will be used as feature, and x0 will be used for positional encoding. 
        # 2. For the PN2 for anchor object, there will be no feature, and y will be used for positional encoding.

        # x embedding (pcd features)
        self.x_embedder = pn2_pyg.PN2SAGA(
            in_channels=3,  # additional 3 channels for x0 feature
            out_channels=hidden_size,
            p=pn2_pyg.PN2SAGAParams(),
        )

        # y embedding (pcd features)
        self.y_embedder = pn2_pyg.PN2SAGA(
            in_channels=0,  # no additional feature
            out_channels=hidden_size,
            p=pn2_pyg.PN2SAGAParams(),
        )
        '''
        
        ### VARIANT 1.1.2 ###
        # We'll use pn2_pyg.PN2Dense code bade
        # We'll use PN2 as encoder for x0+x and y seperately. 
        # 1. For the PN2 for action object, noise x will be used as feature, and x0 will be used for positional encoding. 
        # 2. For the PN2 for anchor object, there will be no feature, and y will be used for positional encoding.

        # x embedding (pcd features)
        self.x_embedder = pn2_pyg.PN2Dense(
            in_channels=3,  # additional 3 channels for x0 feature
            out_channels=hidden_size,
            p=pn2_pyg.PN2DenseParams(),
        )

        # y embedding (pcd features)
        self.y_embedder = pn2_pyg.PN2Dense(
            in_channels=0,  # no additional feature
            out_channels=hidden_size,
            p=pn2_pyg.PN2DenseParams(),
        )
        


        '''
        ### VARIANT 3 ###
        # We'll use PN2 as encoder for x0 and y only. The noise x will be seperately encoded using MLP,
        # and it will be concatinated with encoded x (i.e. MLP(x) + PN2(x0)) and pass it to DiT.

        # x embedding (pcd features)
        self.x_embedder = pnp_original.PN2Dense(
            in_channels=0,  # additional 3 channels for x0 feature
            out_channels=x_encoder_hidden_dims,
            p=pnp_original.PN2DenseParams(),
        )

        # y embedding (pcd features)
        self.y_embedder = pnp_original.PN2Dense(
            in_channels=0,  # no additional feature
            out_channels=hidden_size,
            p=pnp_original.PN2DenseParams(),
        )
        '''

        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTRelativeCrossBlock if self.model_cfg.rotary else DiTCrossBlock
        self.blocks = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        
        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        if self.x_pre_encoder is not None:
            w = self.x_pre_encoder.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_pre_encoder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        # noise-centering, if enabled
        if self.model_cfg.center_noise:
            relative_center = torch.mean(x, dim=2, keepdim=True)
            x = x - relative_center
            y = y - relative_center
        
        # rotary position embedding, if enabled
        if self.model_cfg.rotary:
            x_pos = self.rotary_pos_enc(x.permute(0, 2, 1))
            y_pos = self.rotary_pos_enc(y.permute(0, 2, 1))

        # pre-encode x, y, x0 features if required
        if self.x_pre_encoder is not None:
            x = self.x_pre_encoder(x)

        if self.x0_pre_encoder is not None:
            assert x0 is not None, "x0 features must be provided if x0_encoder is not None"
            x0 = self.x0_pre_encoder(x0)

        if self.y_pre_encoder is not None:
            y = self.y_pre_encoder(y)


        # encode x and y features using Pointnet++ 
        assert self.x_embedder is not None
        assert self.y_embedder is not None

        
        ### VARIANT 1 ###
        ### PN2(xyz=x0, f=x)
        B, D, N = y.shape
        
        # construct features
        f_x = x.permute(0, 2, 1).reshape(-1, D)  # Shape: (B * N, D)
        f_y = None

        # construct pos
        pos_x = x0.permute(0, 2, 1).reshape(-1, D)  # using original action as positional grouping for PN2
        pos_y = y.permute(0, 2, 1).reshape(-1, D)   # using original anchor as positional grouping for PN2

        # construct batch
        batch_idx_x = torch.arange(B).repeat_interleave(N)  # Shape: (B * N,)
        batch_idx_y = torch.arange(B).repeat_interleave(N)  # Shape: (B * N,)

        data_x = Data(x=f_x, pos=pos_x, batch=batch_idx_x)
        data_y = Data(x=f_y, pos=pos_y, batch=batch_idx_y)

        # PN2 embedding
        #breakpoint()
        x = self.x_embedder(data_x.cuda())
        #breakpoint()
        y_emb = self.y_embedder(data_y.cuda())

        x = x.reshape(B, N, -1)             # reshape back to (B, N, d_embed)
        y_emb = y_emb.reshape(B, N, -1)     # reshape back to (B, N, d_embed)
        

        '''
        ### VARIANT 2 ###
        ### PN2(xyz=x0, f=None)
        B, D, N = y.shape
        
        # construct features
        f_x = None
        f_y = None

        # construct pos
        pos_x = x0.permute(0, 2, 1).reshape(-1, D)  # using original action as positional grouping for PN2
        pos_y = y.permute(0, 2, 1).reshape(-1, D)   # using original anchor as positional grouping for PN2

        # construct batch
        batch_idx_x = torch.arange(B).repeat_interleave(N)  # Shape: (B * N,)
        batch_idx_y = torch.arange(B).repeat_interleave(N)  # Shape: (B * N,)

        data_x = Data(x=f_x, pos=pos_x, batch=batch_idx_x)
        data_y = Data(x=f_y, pos=pos_y, batch=batch_idx_y)

        # PN2 embedding
        x0_emb = self.x_embedder(data_x.cuda())
        y_emb = self.y_embedder(data_y.cuda())

        x0_emb = x0_emb.reshape(B, N, -1)             # reshape back to (B, N, d_embed)
        y_emb = y_emb.reshape(B, N, -1)     # reshape back to (B, N, d_embed)
        
        # encode x features
        x = x.permute(0, 2, 1)
        x = torch.cat([x, x0_emb], dim=-1)
        '''
        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks:
            if self.model_cfg.rotary:
                x = block(x, y_emb, t_emb, x_pos, y_pos)
            else:
                x = block(x, y_emb, t_emb)

        # final layer
        x = self.final_layer(x, t_emb)
        #breakpoint()
        x = x.permute(0, 2, 1)
        #breakpoint()
        return x

class DiT_PointCloud_Cross_Point_Feature(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg
        self.num_features = 1
        
        # define possible feature context types and normalization types
        feature_context_types = ["anchor_mean", "action_mean", "all"]
        feature_normalize_types = ["unnorm", "unit", "clip", "zeromean", "all"]

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        ################################
        # For now, do not use y and x0 #
        ################################
        print("Building Upgrade Architecture Option 1")

        if self.model_cfg.x_encoder is not None:
            if self.model_cfg.onehot_encoder not in [None, "off"]:
                self.num_features += 1
                print("Using One-Hot Encoding Feature!")

            if self.model_cfg.context_encoder not in [None, "off"]:
                if self.model_cfg.feature_context_type == "all":
                    self.num_features += len(feature_context_types) -1
                    print("Using All Contextual Features!")
                elif self.model_cfg.feature_context_type in feature_context_types:
                    self.num_features += 1
                    print("Using {} Context Feature".format(self.model_cfg.feature_context_type))

            if self.model_cfg.flow_encoder not in [None, "off"]:
                if self.model_cfg.feature_normalize_type == "all":
                    self.num_features += len(feature_normalize_types) -1
                    print("Using All Flow Features!")
                elif self.model_cfg.feature_normalize_type in feature_normalize_types:
                    self.num_features += 1  
                    print("Using {} Flow Feature".format(self.model_cfg.feature_normalize_type))
        
        assert self.num_features in [1, 2, 4, 8], "Invalid total number of features!"
        per_hidden_size = int(hidden_size / self.num_features)
        print("Using total number of {} features in for DiT, with final encoder hidden size of {}".format(self.num_features, per_hidden_size))

        # Encoder for current timestep x features       
        if self.model_cfg.x_encoder == "mlp":
            # x_embedder is conv1d layer instead of 2d patch embedder
            self.x_embedder = nn.Conv1d(
                in_channels,
                per_hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")
        
        # Encoder for y features
        if self.model_cfg.y_encoder == "mlp":
            self.y_embedder = nn.Conv1d(
                in_channels,
                per_hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.y_encoder == "dgcnn":
            self.y_embedder = DGCNN(
                input_dims=in_channels, emb_dims=per_hidden_size
            )
        else:
            raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

        # Encoder for x0 features
        '''
        if self.model_cfg.x0_encoder == "mlp":
            self.x0_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.x0_encoder == "dgcnn":
            self.x0_embedder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif self.model_cfg.x0_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")
        '''

        # Encoder for additional features
        # encode one-hot
        if self.model_cfg.onehot_encoder == "mlp":
            self.onehot_encoder = nn.Conv1d(
                2,
                per_hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.onehot_encoder == "off":
            self.onehot_encoder = None
        else:
            raise ValueError(f"Invalid onehot_encoder: {self.model_cfg.onehot_encoder}")
        
        # encode context features
        if self.model_cfg.context_encoder == "mlp":
            assert self.model_cfg.feature_context_type in feature_context_types, "feature_context_type must be one of [\"anchor_mean\", \"action_mean\", \"all\"]"
            self.context_encoders = nn.ModuleDict()
            if self.model_cfg.feature_context_type == "all":
                for c_type in feature_context_types:
                    self.context_encoders[c_type] = nn.Conv1d(
                        in_channels,
                        per_hidden_size,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
            else:
                self.context_encoders[self.model_cfg.feature_context_type] = nn.Conv1d(
                    in_channels,
                    per_hidden_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
        elif self.model_cfg.context_encoder == "off":
            self.context_encoders = None
        else:
            raise ValueError(f"Invalid context_encoder: {self.model_cfg.context_encoder}")

        # encode context features
        if self.model_cfg.flow_encoder == "mlp":
            assert self.model_cfg.feature_normalize_type in feature_normalize_types, "feature_normalize_types must be one of [\"unnorm\", \"unit\", \"clip\", \"zeromean\", \"all\"]"
            self.flow_encoders = nn.ModuleDict()
            if self.model_cfg.feature_normalize_type == "all":
                for n_type in feature_normalize_types:
                    self.flow_encoders[n_type] = nn.Conv1d(
                        in_channels,
                        per_hidden_size,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
            else:
                self.flow_encoders[self.model_cfg.feature_normalize_type] = nn.Conv1d(
                    in_channels,
                    per_hidden_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
        elif self.model_cfg.flow_encoder == "off":
            self.flow_encoders = None
        else:
            raise ValueError(f"Invalid flow_encoder: {self.model_cfg.flow_encoder}")
        
        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTRelativeCrossBlock if self.model_cfg.rotary else DiTCrossBlock
        self.blocks = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
            P_A: Optional[torch.Tensor] = None,
            P_B: Optional[torch.Tensor] = None,
            y_ref: Optional[torch.Tensor] = None,
            y_action: Optional[torch.Tensor] = None,
            P_A_one_hot: Optional[torch.Tensor] = None,
            P_B_one_hot: Optional[torch.Tensor] = None,       
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
            P_A (Optional[torch.Tensor]): (B, D, N) tensor of un-noised un-centered x (e.g. action) features
            P_B (Optional[torch.Tensor]): (B, D, N) tensor of un-noised un-centered scene (e.g. anchor) features
            y_ref (Optional[torch.Tensor]): (B, D) center tensor of un-noised x (e.g. action) features
            y_action (Optional[torch.Tensor]): (B, D) center tensor of un-noised x (e.g. action) features
            P_A_one_hot (Optional[torch.Tensor]): (B, 2, N) one-hot encoder for action features
            P_B_one_hot (Optional[torch.Tensor]): (B, 2, N) one-hot encoder for anchor features
        """
        ################################
        # For now, do not use y and x0 #
        ################################
        
        # rotary position embedding, if enabled
        if self.model_cfg.rotary:
            x_pos = self.rotary_pos_enc(x.permute(0, 2, 1))
            y_pos = self.rotary_pos_enc(y.permute(0, 2, 1))

        # encode x features
        x_emb = self.x_embedder(x)
        '''
        if self.model_cfg.x0_encoder is not None:
            assert x0 is not None, "x0 features must be provided if x0_encoder is not None"
            x0_emb = self.x0_embedder(x0)
            x_emb = torch.cat([x_emb, x0_emb], dim=1)
        '''
        if self.onehot_encoder is not None:
            assert P_A_one_hot is not None, "P_A_one_hot features must be provided if onehot_encoder is not None"
            add_emb = self.onehot_encoder(P_A_one_hot)
            x_emb = torch.cat([x_emb, add_emb], dim=1)

        if self.context_encoders is not None:
            for c_type, context_encoder in self.context_encoders.items():
                if c_type == "anchor_mean":
                    assert y_ref is not None and P_A is not None, "y_ref and P_A features must be provided if context_encoder of type anchor_mean is not None"
                    add_emb = context_encoder(P_A - y_ref.unsqueeze(-1))
                    x_emb = torch.cat([x_emb, add_emb], dim=1)
                if c_type == "action_mean":
                    assert y_action is not None and P_A is not None, "y_action and P_A features must be provided if context_encoder of type action_mean is not None"
                    add_emb = context_encoder(P_A - y_action.unsqueeze(-1))
                    x_emb = torch.cat([x_emb, add_emb], dim=1)

        if self.flow_encoders is not None:
            assert y_ref is not None and P_A is not None, "y_ref and P_A features must be provided if using flow features"

            for n_type, flow_encoder in self.flow_encoders.items():
                P_A_hat = x + y_ref.unsqueeze(-1)   # broadcast y_ref from [B, C] to [B, C, 1]
                flow = P_A_hat - P_A

                if n_type == "unnorm":
                    add_emb = flow_encoder(flow)
                    x_emb = torch.cat([x_emb, add_emb], dim=1)
                if n_type == "unit":
                    # compute the L2 norm across the last dimension
                    norm = torch.norm(flow, dim=1, keepdim=True)  # Shape: [16, 1, 512]
                    # avoid division by zero
                    norm = torch.where(norm == 0, torch.tensor(1.0, device=flow.device), norm)
                    # Normalize flow to unit vectors
                    flow = flow / norm
                    add_emb = flow_encoder(flow)
                    x_emb = torch.cat([x_emb, add_emb], dim=1)
                if n_type == "clip":
                    flow = torch.clamp(flow, min=-1.0, max=1.0)
                    add_emb = flow_encoder(flow)
                    x_emb = torch.cat([x_emb, add_emb], dim=1)
                if n_type == "zeromean":
                    mean = flow.mean(dim=2, keepdim=True)
                    flow = flow - mean
                    add_emb = flow_encoder(flow)
                    x_emb = torch.cat([x_emb, add_emb], dim=1)

        x = x_emb.permute(0, 2, 1)

        # encode y features
        y_emb = self.y_embedder(y)
        
        if self.onehot_encoder is not None:
            assert P_B_one_hot is not None, "P_B_one_hot features must be provided if onehot_encoder is not None"
            add_emb = self.onehot_encoder(P_B_one_hot)
            y_emb = torch.cat([y_emb, add_emb], dim=1)

        zero_emb = torch.zeros_like(add_emb)
        for _ in range(self.num_features-2):
            y_emb = torch.cat([y_emb, zero_emb], dim=1)
        y_emb = y_emb.permute(0, 2, 1)

        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks:
            if self.model_cfg.rotary:
                x = block(x, y_emb, t_emb, x_pos, y_pos)
            else:
                x = block(x, y_emb, t_emb)

        # final layer
        x = self.final_layer(x, t_emb)
        x = x.permute(0, 2, 1)
        return x

class DiT_PointCloud_Cross_Flow_Feature(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg
        self.num_features = 0
        
        # define possible feature context types and normalization types
        feature_context_types = ["anchor_mean", "action_mean", "all"]
        feature_normalize_types = ["unnorm", "unit", "clip", "zeromean", "all"]

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        ################################
        # For now, do not use y and x0 #
        ################################
        print("Building Upgrade Architecture Option 2")

        if self.model_cfg.encoder_style == 'indiv':
            print("First individually encode features, then concatinate")
            if self.model_cfg.x_encoder not in [None, False]:
                self.num_features += 1
                print("Using Noisy-flow Encoding Feature with Encoder {}".format(self.model_cfg.x_encoder))

            if self.model_cfg.onehot_encoder not in [None, False]:
                self.num_features += 1
                print("Using One-Hot Encoding Feature with Encoder {}".format(self.model_cfg.onehot_encoder))

            if self.model_cfg.recon_encoder not in [None, False]:
                self.num_features += 1
                print("Using Reconstructed Goal PCD Feature with Encoder {}".format(self.model_cfg.recon_encoder))

            if self.model_cfg.context_encoder not in [None, False]:
                if self.model_cfg.feature_context_type == "all":
                    self.num_features += len(feature_context_types) -1
                    print("Using All Contextual Features with Encoder {}".format(self.model_cfg.context_encoder))
                elif self.model_cfg.feature_context_type in feature_context_types:
                    self.num_features += 1
                    print("Using {} Context Feature with Encoder {}".format(self.model_cfg.feature_context_type, self.model_cfg.context_encoder))

            if self.model_cfg.flow_encoder not in [None, False]:
                if self.model_cfg.feature_normalize_type == "all":
                    self.num_features += len(feature_normalize_types) -1
                    print("Using All Flow Features with Encoder {}".format(self.model_cfg.flow_encoder))
                elif self.model_cfg.feature_normalize_type in feature_normalize_types:
                    self.num_features += 1  
                    print("Using {} Flow Feature with Encoder {}".format(self.model_cfg.feature_normalize_type, self.model_cfg.flow_encoder))
            
            assert self.num_features in [1, 2, 3, 4, 8], "Invalid total number of {} features!".format(self.num_features)
            
            #TODO: do not hard code this
            if self.num_features != 3:
                per_hidden_size = int(hidden_size / self.num_features)
                main_hidden_size = per_hidden_size
                print("Using total number of {} features in for DiT, with final encoder hidden size of {}".format(self.num_features, per_hidden_size))
            else:
                # x/y encoder takes 50%, the other 2 features take 25%
                per_hidden_size = int(hidden_size / 4)
                main_hidden_size = int(hidden_size / 2)
                print("Using total number of {} features in for DiT, with x/y encoder of hidden size of {}, and feature encoder hidden size of {}".format(self.num_features, main_hidden_size, per_hidden_size))
            self.per_hidden_size = per_hidden_size
            self.main_hidden_size = main_hidden_size

            # Encoder for current timestep x features       
            if self.model_cfg.x_encoder == "mlp":
                # x_embedder is conv1d layer instead of 2d patch embedder
                self.x_embedder = nn.Conv1d(
                    in_channels,
                    main_hidden_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            elif self.model_cfg.x_encoder == "dgcnn":
                self.x_embedder = DGCNN(input_dims=in_channels, emb_dims=main_hidden_size)
            elif not self.model_cfg.x_encoder:
                self.x_embedder = None
            else:
                raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")
            
            # Encoder for y features
            if self.model_cfg.y_encoder == "mlp":
                self.y_embedder = nn.Conv1d(
                    in_channels,
                    main_hidden_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            elif self.model_cfg.y_encoder == "dgcnn":
                self.y_embedder = DGCNN(input_dims=in_channels, emb_dims=main_hidden_size)
            else:
                raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

            # Encoder for x0 features
            '''
            if self.model_cfg.x0_encoder == "mlp":
                self.x0_embedder = nn.Conv1d(
                    in_channels,
                    x_encoder_hidden_dims,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            elif self.model_cfg.x0_encoder == "dgcnn":
                self.x0_embedder = DGCNN(
                    input_dims=in_channels, emb_dims=x_encoder_hidden_dims
                )
            elif self.model_cfg.x0_encoder is None:
                pass
            else:
                raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")
            '''

            # Encoder for additional features
            # encode one-hot
            if self.model_cfg.onehot_encoder == "mlp":
                self.onehot_encoder = nn.Conv1d(
                    2,
                    per_hidden_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            elif self.model_cfg.onehot_encoder == "dgcnn":
                self.onehot_encoder = DGCNN(input_dims=2, emb_dims=per_hidden_size)
            elif not self.model_cfg.onehot_encoder:
                self.onehot_encoder = None
            else:
                raise ValueError(f"Invalid onehot_encoder: {self.model_cfg.onehot_encoder}")
            
            if self.model_cfg.recon_encoder == "mlp":
                self.recon_encoder = nn.Conv1d(
                    in_channels,
                    per_hidden_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            elif self.model_cfg.recon_encoder == "dgcnn":
                self.recon_encoder = DGCNN(input_dims=in_channels, emb_dims=per_hidden_size)
            elif not self.model_cfg.recon_encoder:
                self.recon_encoder = None
            else:
                raise ValueError(f"Invalid recon_encoder: {self.model_cfg.recon_encoder}")
            
            # encode context features
            if self.model_cfg.context_encoder == "mlp":
                assert self.model_cfg.feature_context_type in feature_context_types, "feature_context_type must be one of [\"anchor_mean\", \"action_mean\", \"all\"]"
                self.context_encoders = nn.ModuleDict()
                if self.model_cfg.feature_context_type == "all":
                    for c_type in feature_context_types:
                        self.context_encoders[c_type] = nn.Conv1d(
                            in_channels,
                            per_hidden_size,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        )
                else:
                    self.context_encoders[self.model_cfg.feature_context_type] = nn.Conv1d(
                        in_channels,
                        per_hidden_size,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
            elif self.model_cfg.context_encoder == "dgcnn":
                assert self.model_cfg.feature_context_type in feature_context_types, "feature_context_type must be one of [\"anchor_mean\", \"action_mean\", \"all\"]"
                self.context_encoders = nn.ModuleDict()
                if self.model_cfg.feature_context_type == "all":
                    for c_type in feature_context_types:
                        self.context_encoders[c_type] = DGCNN(input_dims=in_channels, emb_dims=per_hidden_size)
                else:
                    self.context_encoders[self.model_cfg.feature_context_type] = DGCNN(input_dims=in_channels, emb_dims=per_hidden_size)
            elif not self.model_cfg.context_encoder:
                self.context_encoders = None
            else:
                raise ValueError(f"Invalid context_encoder: {self.model_cfg.context_encoder}")

            # encode flow features
            if self.model_cfg.flow_encoder == "mlp":
                assert self.model_cfg.feature_normalize_type in feature_normalize_types, "feature_normalize_types must be one of [\"unnorm\", \"unit\", \"clip\", \"zeromean\", \"all\"]"
                self.flow_encoders = nn.ModuleDict()
                if self.model_cfg.feature_normalize_type == "all":
                    for n_type in feature_normalize_types:
                        self.flow_encoders[n_type] = nn.Conv1d(
                            in_channels,
                            per_hidden_size,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        )
                else:
                    self.flow_encoders[self.model_cfg.feature_normalize_type] = nn.Conv1d(
                        in_channels,
                        per_hidden_size,   # BUG: Need to set this to main_hidden_size when noise_flow is None and num_features=3
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
            elif self.model_cfg.flow_encoder == "dgcnn":
                assert self.model_cfg.feature_normalize_type in feature_normalize_types, "feature_normalize_types must be one of [\"unnorm\", \"unit\", \"clip\", \"zeromean\", \"all\"]"
                self.flow_encoders = nn.ModuleDict()
                if self.model_cfg.feature_normalize_type == "all":
                    for n_type in feature_normalize_types:
                        self.flow_encoders[n_type] = DGCNN(input_dims=in_channels, emb_dims=per_hidden_size)
                else:
                    self.flow_encoders[self.model_cfg.feature_normalize_type] = DGCNN(input_dims=in_channels, emb_dims=per_hidden_size) # BUG: Need to set this to main_hidden_size when noise_flow is None and num_features=3
            elif not self.model_cfg.flow_encoder:
                self.flow_encoders = None
            else:
                raise ValueError(f"Invalid flow_encoder: {self.model_cfg.flow_encoder}")
        
        elif self.model_cfg.encoder_style == 'concat':
            print("First concatinate features, then encode")
            if self.model_cfg.x_encoder not in [None, False]:
                self.num_features += 1
                print("Using Noisy-flow Encoding Feature with Encoder {}".format(self.model_cfg.x_encoder))

            if self.model_cfg.onehot_encoder not in [None, False]:
                self.num_features += 1
                print("Using One-Hot Encoding Feature with Encoder {}".format(self.model_cfg.onehot_encoder))

            if self.model_cfg.recon_encoder not in [None, False]:
                self.num_features += 1
                print("Using Reconstructed Goal PCD Feature with Encoder {}".format(self.model_cfg.recon_encoder))

            if self.model_cfg.context_encoder not in [None, False]:
                if self.model_cfg.feature_context_type == "all":
                    self.num_features += len(feature_context_types) -1
                    print("Using All Contextual Features with Encoder {}".format(self.model_cfg.context_encoder))
                elif self.model_cfg.feature_context_type in feature_context_types:
                    self.num_features += 1
                    print("Using {} Context Feature with Encoder {}".format(self.model_cfg.feature_context_type, self.model_cfg.context_encoder))

            if self.model_cfg.flow_encoder not in [None, False]:
                if self.model_cfg.feature_normalize_type == "all":
                    self.num_features += len(feature_normalize_types) -1
                    print("Using All Flow Features with Encoder {}".format(self.model_cfg.flow_encoder))
                elif self.model_cfg.feature_normalize_type in feature_normalize_types:
                    self.num_features += 1  
                    print("Using {} Flow Feature with Encoder {}".format(self.model_cfg.feature_normalize_type, self.model_cfg.flow_encoder))

            if self.model_cfg.onehot_encoder not in [None, False]:
                x_in_channels = 3 * (self.num_features - 1) + 2   # one-hot encoding only takes in 2 channels
            else:
                x_in_channels = 3 * self.num_features
        
            # Encoder for current timestep x features       
            if self.model_cfg.x_encoder == "mlp":
                # x_embedder is conv1d layer instead of 2d patch embedder
                self.x_embedder = nn.Conv1d(
                    x_in_channels,
                    hidden_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            elif self.model_cfg.x_encoder == "dgcnn":
                self.x_embedder = DGCNN(input_dims=in_channels, emb_dims=hidden_size)
            elif not self.model_cfg.x_encoder:
                self.x_embedder = None
            else:
                raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")
            
            # Encoder for y features
            if self.model_cfg.y_encoder == "mlp":
                self.y_embedder = nn.Conv1d(
                    in_channels,
                    hidden_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            elif self.model_cfg.y_encoder == "dgcnn":
                self.y_embedder = DGCNN(input_dims=in_channels, emb_dims=hidden_size)
            else:
                raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

            # Encoder for x0 features
            '''
            if self.model_cfg.x0_encoder == "mlp":
                self.x0_embedder = nn.Conv1d(
                    in_channels,
                    x_encoder_hidden_dims,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            elif self.model_cfg.x0_encoder == "dgcnn":
                self.x0_embedder = DGCNN(
                    input_dims=in_channels, emb_dims=x_encoder_hidden_dims
                )
            elif self.model_cfg.x0_encoder is None:
                pass
            else:
                raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")
            '''
            
        else:
            raise ValueError(f"Invalid encoder_style: {self.model_cfg.encoder_style}")

        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTRelativeCrossBlock if self.model_cfg.rotary else DiTCrossBlock
        self.blocks = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        if self.model_cfg.x_encoder == "mlp":
            w = self.x_embedder.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
            P_A: Optional[torch.Tensor] = None,
            P_B: Optional[torch.Tensor] = None,
            y_ref: Optional[torch.Tensor] = None,
            y_action: Optional[torch.Tensor] = None,
            P_A_one_hot: Optional[torch.Tensor] = None,
            P_B_one_hot: Optional[torch.Tensor] = None,       
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
            P_A (Optional[torch.Tensor]): (B, D, N) tensor of un-noised un-centered x (e.g. action) features
            P_B (Optional[torch.Tensor]): (B, D, N) tensor of un-noised un-centered scene (e.g. anchor) features
            y_ref (Optional[torch.Tensor]): (B, D) center tensor of un-noised x (e.g. action) features
            y_action (Optional[torch.Tensor]): (B, D) center tensor of un-noised x (e.g. action) features
            P_A_one_hot (Optional[torch.Tensor]): (B, 2, N) one-hot encoder for action features
            P_B_one_hot (Optional[torch.Tensor]): (B, 2, N) one-hot encoder for anchor features
        """
        ################################
        # For now, do not use y and x0 #
        ################################
        
        # rotary position embedding, if enabled
        if self.model_cfg.rotary:
            x_pos = self.rotary_pos_enc(x.permute(0, 2, 1))
            y_pos = self.rotary_pos_enc(y.permute(0, 2, 1))
        
        # initialize x_emb an an empty tensor
        B, _, N = x.shape
        x_emb = torch.empty((B, 0, N), device=x.device)


        if self.model_cfg.encoder_style == 'indiv':
            
            P_A_hat = x + P_A - y_action.unsqueeze(-1) + y_ref.unsqueeze(-1)

            # encode x features
            if self.x_embedder is not None:
                assert x is not None, "x features must be provided if onehot_encoder is not None"
                add_emb = self.x_embedder(x)
                x_emb = torch.cat([x_emb, add_emb], dim=1)

            #if self.model_cfg.x0_encoder is not None:
            #    assert x0 is not None, "x0 features must be provided if x0_encoder is not None"
            #    x0_emb = self.x0_embedder(x0)
            #    x_emb = torch.cat([x_emb, x0_emb], dim=1)
            
            if self.onehot_encoder is not None:
                assert P_A_one_hot is not None, "P_A_one_hot features must be provided if onehot_encoder is not None"
                add_emb = self.onehot_encoder(P_A_one_hot)
                x_emb = torch.cat([x_emb, add_emb], dim=1)

            if self.recon_encoder is not None:
                assert P_A is not None and y_action is not None, "y_action and P_A features must be provided if onehot_encoder is not None"
                
                # Center on Anchor
                P_A_hat_prime = x + P_A - y_action.unsqueeze(-1)
                add_emb = self.recon_encoder(P_A_hat_prime)
                x_emb = torch.cat([x_emb, add_emb], dim=1)
                
                '''
                # Center on Reconstruction
                P_A_hat_center_recon = P_A_hat - P_A_hat.mean(axis=0)
                add_emb = self.recon_encoder(P_A_hat_center_recon)
                x_emb = torch.cat([x_emb, add_emb], dim=1)
                '''
            if self.context_encoders is not None:
                for c_type, context_encoder in self.context_encoders.items():
                    if c_type == "anchor_mean":
                        assert y_ref is not None and P_A is not None, "y_ref and P_A features must be provided if context_encoder of type anchor_mean is not None"
                        add_emb = context_encoder(P_A - y_ref.unsqueeze(-1))
                        x_emb = torch.cat([x_emb, add_emb], dim=1)
                    if c_type == "action_mean":
                        assert y_action is not None and P_A is not None, "y_action and P_A features must be provided if context_encoder of type action_mean is not None"
                        add_emb = context_encoder(P_A - y_action.unsqueeze(-1))
                        x_emb = torch.cat([x_emb, add_emb], dim=1)

            if self.flow_encoders is not None:
                assert y_ref is not None and P_A is not None, "y_ref and P_A features must be provided if using flow features"

                for n_type, flow_encoder in self.flow_encoders.items():
                    flow = P_A_hat - P_A

                    if n_type == "unnorm":
                        add_emb = flow_encoder(flow)
                        x_emb = torch.cat([x_emb, add_emb], dim=1)
                    if n_type == "unit":
                        # compute the L2 norm across the last dimension
                        norm = torch.norm(flow, dim=1, keepdim=True)  # Shape: [16, 1, 512]
                        # avoid division by zero
                        norm = torch.where(norm == 0, torch.tensor(1.0, device=flow.device), norm)
                        # Normalize flow to unit vectors
                        flow = flow / norm
                        add_emb = flow_encoder(flow)
                        x_emb = torch.cat([x_emb, add_emb], dim=1)
                    if n_type == "clip":
                        flow = torch.clamp(flow, min=-1.0, max=1.0)
                        add_emb = flow_encoder(flow)
                        x_emb = torch.cat([x_emb, add_emb], dim=1)
                    if n_type == "zeromean":
                        mean = flow.mean(dim=2, keepdim=True)
                        flow = flow - mean
                        add_emb = flow_encoder(flow)
                        x_emb = torch.cat([x_emb, add_emb], dim=1)

            x = x_emb.permute(0, 2, 1)

            # encode y features
            
            # Center on Anchor
            y_emb = self.y_embedder(y)
            '''
            # Center on Reconstruction
            y_world = y + y_ref.unsqueeze(-1)
            y_center_recon = y_world - P_A_hat.mean(axis=0)
            y_emb = self.y_embedder(y_center_recon)
            '''
            if self.onehot_encoder is not None:
                assert P_B_one_hot is not None, "P_B_one_hot features must be provided if onehot_encoder is not None"
                add_emb = self.onehot_encoder(P_B_one_hot)
                y_emb = torch.cat([y_emb, add_emb], dim=1)
                
                # zero_emb = torch.zeros_like(add_emb)
                zero_emb = torch.zeros((B, self.per_hidden_size, N), device=add_emb.device, dtype=add_emb.dtype)  # BUG here. Also dont't forget to restore context_encoder hidden size
                for _ in range(self.num_features-2):
                    y_emb = torch.cat([y_emb, zero_emb], dim=1)
                y_emb = y_emb.permute(0, 2, 1)
            else:
                zero_emb = torch.zeros((B, self.per_hidden_size, N), device=add_emb.device, dtype=add_emb.dtype)  # BUG here. Also dont't forget to restore context_encoder hidden size
                for _ in range(self.num_features-1):
                    y_emb = torch.cat([y_emb, zero_emb], dim=1)
                y_emb = y_emb.permute(0, 2, 1)

        elif self.model_cfg.encoder_style == 'concat':
            B, _, N = x.shape
            x_feat = torch.empty((B, 0, N), device=x.device)

            if self.model_cfg.x_encoder:
                x_feat = torch.cat([x_feat, x], dim=1)

            if self.model_cfg.onehot_encoder:
                assert P_A_one_hot is not None, "P_A_one_hot features must be provided if onehot_encoder is not None"
                x_feat = torch.cat([x_feat, P_A_one_hot], dim=1)

            if self.model_cfg.recon_encoder:
                assert P_A is not None and y_action is not None, "y_action and P_A features must be provided if onehot_encoder is not None"
                P_A_hat_prime = x + P_A - y_action.unsqueeze(-1)
                x_feat = torch.cat([x_feat, P_A_hat_prime], dim=1)
            
            # encode context features
            if self.model_cfg.context_encoder:
                if self.model_cfg.feature_context_type in ["anchor_mean", "all"]:
                    assert y_ref is not None and P_A is not None, "y_ref and P_A features must be provided if context_encoder of type anchor_mean is not None"
                    anchor_mean = P_A - y_ref.unsqueeze(-1)
                    x_feat = torch.cat([x_feat, anchor_mean], dim=1)
                if self.model_cfg.feature_context_type in ["action_mean", "all"]:
                    assert y_action is not None and P_A is not None, "y_action and P_A features must be provided if context_encoder of type action_mean is not None"
                    action_mean = P_A - y_action.unsqueeze(-1)
                    x_feat = torch.cat([x_feat, action_mean], dim=1)

            # encode flow features
            if self.model_cfg.flow_encoder:
                assert y_ref is not None and P_A is not None, "y_ref and P_A features must be provided if using flow features"
                P_A_hat = x + P_A + (y_ref.unsqueeze(-1) - y_action.unsqueeze(-1))   # broadcast y_ref from [B, C] to [B, C, 1]
                flow = P_A_hat - P_A

                if self.model_cfg.feature_normalize_type in ["unnorm", "all"]:
                    x_feat = torch.cat([x_feat, flow], dim=1)
                if self.model_cfg.feature_normalize_type in ["unit", "all"]:
                    # compute the L2 norm across the last dimension
                    norm = torch.norm(flow, dim=1, keepdim=True)  # Shape: [16, 1, 512]
                    # avoid division by zero
                    norm = torch.where(norm == 0, torch.tensor(1.0, device=flow.device), norm)
                    # Normalize flow to unit vectors
                    flow_unitnorm = flow / norm
                    x_feat = torch.cat([x_feat, flow_unitnorm], dim=1)
                if self.model_cfg.feature_normalize_type in ["clip", "all"]:
                    flow_clip = torch.clamp(flow, min=-1.0, max=1.0)
                    x_feat = torch.cat([x_feat, flow_clip], dim=1)
                if self.model_cfg.feature_normalize_type in ["zeromean", "all"]:
                    mean = flow.mean(dim=2, keepdim=True)
                    flow_zeromean = flow - mean
                    x_feat = torch.cat([x_feat, flow_zeromean], dim=1)
            
            x_feat_l2 = x_feat / (torch.norm(x_feat, p=2, dim=1, keepdim=True) + 1e-8)  # Normalize each feature across C
            x_emb = self.x_embedder(x_feat_l2)
            x = x_emb.permute(0, 2, 1)
            
            y_emb = self.y_embedder(y)
            y_emb = y_emb.permute(0, 2, 1)

        else:
            raise ValueError(f"Invalid encoder_style: {self.model_cfg.encoder_style}")


        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks:
            if self.model_cfg.rotary:
                x = block(x, y_emb, t_emb, x_pos, y_pos)
            else:
                x = block(x, y_emb, t_emb)

        # final layer
        x = self.final_layer(x, t_emb)
        x = x.permute(0, 2, 1)
        return x


class PN2_DiT_PointCloud_Cross_Flow_Feature(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention.
    """
    def __init__(
            self,
            input_size=[30, 40],
            patch_size=1,
            n_points=1200,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg
        self.num_features = 1
        self.in_channels = 0

        # define possible feature context types and normalization types
        feature_context_types = ["anchor_mean", "action_mean"]
        feature_flow_types = ["unnorm", "unit", "clip", "zeromean"]
        
        assert self.model_cfg.feature_context_type in feature_context_types, "feature_context_type must be one of [\"anchor_mean\", \"action_mean\"]"
        assert self.model_cfg.feature_flow_type in feature_flow_types, "feature_flow_types must be one of [\"anchor_mean\", \"action_mean\"]"

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        ################################
        # For now, do not use y and x0 #
        ################################
        print("Building Feature-TAX3D with Pointnet++ backbone")

        if self.model_cfg.encoder_style == "indiv":
            print("Only encoding xyz positions with Pointnet++! Individually encode xyz and features and then concatinate...")

            if self.model_cfg.noise_encoder:
                self.num_features += 1
                print("Using Noise Encoding Feature with Encoder {}".format(self.model_cfg.noise_encoder))

            '''
            if self.model_cfg.onehot_encoder:
                self.num_features += 1
                print("Using One-Hot Encoding Feature with Encoder {}".format(self.model_cfg.onehot_encoder))
            '''

            if self.model_cfg.recon_encoder:
                self.num_features += 1
                print("Using Reconstructed Goal PCD Feature with Encoder {}".format(self.model_cfg.recon_encoder))

            if self.model_cfg.context_encoder:
                self.num_features += 1
                print("Using {} Context Feature with Encoder {}".format(self.model_cfg.feature_context_type, self.model_cfg.context_encoder))

            if self.model_cfg.flow_encoder:
                self.num_features += 1  
                print("Using {} Flow Feature with Encoder {}".format(self.model_cfg.feature_flow_type, self.model_cfg.flow_encoder))

            assert self.num_features in [4, 5], "Invalid total number of {} features!".format(self.num_features)
            
            # Update encoders hidden size and channels
            #TODO: do not hard code this
            if self.num_features != 5:
                per_hidden_size = int(hidden_size / self.num_features)
                main_hidden_size = per_hidden_size
                print("Using total number of {} features in for DiT, with final encoder hidden size of {}".format(self.num_features, per_hidden_size))
            else:
                per_hidden_size = 16
                main_hidden_size = 64
                print("Using total number of {} features in for DiT, with x/y encoder of hidden size of {}, and feature encoder hidden size of {}".format(self.num_features, main_hidden_size, per_hidden_size))
            self.per_hidden_size = per_hidden_size
            self.main_hidden_size = main_hidden_size

            self.in_channels = 3

            # x embedding (pcd features)
            self.x_embedder = pn2_pyg.PN2Dense(
                in_channels=0,
                out_channels=self.main_hidden_size,
                p=pn2_pyg.PN2DenseParams(),
            )

            # y embedding (pcd features)
            self.y_embedder = pn2_pyg.PN2Dense(
                in_channels=0,  # no additional feature
                out_channels=self.main_hidden_size,
                p=pn2_pyg.PN2DenseParams(),
            )
            
            
            # encode noise feature       
            if self.model_cfg.noise_encoder == "mlp":
                # x_embedder is conv1d layer instead of 2d patch embedder
                self.noise_encoder = nn.Conv1d(
                    self.in_channels,
                    self.per_hidden_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            elif self.model_cfg.noise_encoder == "dgcnn":
                self.noise_encoder = DGCNN(input_dims=self.in_channels, emb_dims=self.per_hidden_size)
            elif not self.model_cfg.noise_encoder:
                self.noise_encoder = None
            else:
                raise ValueError(f"Invalid noise_encoder: {self.model_cfg.noise_encoder}")
            '''
            # encode one-hot
            if self.model_cfg.onehot_encoder == "mlp":
                self.onehot_encoder = nn.Conv1d(
                    2,
                    self.per_hidden_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            elif self.model_cfg.onehot_encoder == "dgcnn":
                self.onehot_encoder = DGCNN(input_dims=2, emb_dims=self.per_hidden_size)
            elif not self.model_cfg.onehot_encoder:
                self.onehot_encoder = None
            else:
                raise ValueError(f"Invalid onehot_encoder: {self.model_cfg.onehot_encoder}")
            '''
            if self.model_cfg.recon_encoder == "mlp":
                self.recon_encoder = nn.Conv1d(
                    self.in_channels,
                    self.per_hidden_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            elif self.model_cfg.recon_encoder == "dgcnn":
                self.recon_encoder = DGCNN(input_dims=self.in_channels, emb_dims=self.per_hidden_size)
            elif not self.model_cfg.recon_encoder:
                self.recon_encoder = None
            else:
                raise ValueError(f"Invalid recon_encoder: {self.model_cfg.recon_encoder}")
            
            # encode context features
            if self.model_cfg.context_encoder == "mlp":
                self.context_encoder = nn.Conv1d(
                            self.in_channels,
                            self.per_hidden_size,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        )
            elif self.model_cfg.context_encoder == "dgcnn":
                self.context_encoder = DGCNN(input_dims=self.in_channels, emb_dims=self.per_hidden_size)
            elif not self.model_cfg.context_encoder:
                self.context_encoder = None
            else:
                raise ValueError(f"Invalid context_encoder: {self.model_cfg.context_encoder}")

            # encode flow features
            if self.model_cfg.flow_encoder == "mlp":
                self.flow_encoder = nn.Conv1d(
                    self.in_channels,
                    self.per_hidden_size,   # BUG: Need to set this to main_hidden_size when noise_flow is None and num_features=3
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            elif self.model_cfg.flow_encoder == "dgcnn":
                self.flow_encoder = DGCNN(input_dims=self.in_channels, emb_dims=self.per_hidden_size)
            elif not self.model_cfg.flow_encoder:
                self.flow_encoder = None
            else:
                raise ValueError(f"Invalid flow_encoder: {self.model_cfg.flow_encoder}")
        

        elif self.model_cfg.encoder_style == "concat":
            print("Encoding xyz positions and features with Pointnet++! Concatinating xyz and features and then encode...")

            if self.model_cfg.noise_encoder:
                self.in_channels += 3
                print("Using Noise Encoding Feature with Encoder {}".format(self.model_cfg.encoder_backbone))
            '''
            if self.model_cfg.onehot_encoder:
                self.in_channels += 2
                print("Using One-Hot Encoding Feature with Encoder {}".format(self.model_cfg.encoder_backbone))
            '''
            if self.model_cfg.recon_encoder:
                self.in_channels += 3
                print("Using Reconstructed Goal PCD Feature with Encoder {}".format(self.model_cfg.encoder_backbone))

            if self.model_cfg.context_encoder:
                self.in_channels += 3
                print("Using {} Context Feature with Encoder {}".format(self.model_cfg.feature_context_type, self.model_cfg.encoder_backbone))

            if self.model_cfg.flow_encoder:
                self.in_channels += 3
                print("Using {} Flow Feature with Encoder {}".format(self.model_cfg.feature_flow_type, self.model_cfg.encoder_backbone))
            
            print("Using total dimensions of {} features in for Pointnet++".format(self.in_channels))

            # Update encoders hidden size and channels
            self.num_features = 1

            # x embedding (pcd features)
            self.x_embedder = pn2_pyg.PN2Dense(
                in_channels=self.in_channels,  # additional feature channels for x0 feature
                out_channels=hidden_size,
                p=pn2_pyg.PN2DenseParams(),
            )

            # y embedding (pcd features)
            self.y_embedder = pn2_pyg.PN2Dense(
                in_channels=0,  # no additional feature
                out_channels=hidden_size,
                p=pn2_pyg.PN2DenseParams(),
            )


        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTRelativeCrossBlock if self.model_cfg.rotary else DiTCrossBlock
        self.blocks = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
            P_A: Optional[torch.Tensor] = None,
            P_B: Optional[torch.Tensor] = None,
            y_ref: Optional[torch.Tensor] = None,
            y_action: Optional[torch.Tensor] = None,
            P_A_one_hot: Optional[torch.Tensor] = None,
            P_B_one_hot: Optional[torch.Tensor] = None,       
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
            P_A (Optional[torch.Tensor]): (B, D, N) tensor of un-noised un-centered x (e.g. action) features
            P_B (Optional[torch.Tensor]): (B, D, N) tensor of un-noised un-centered scene (e.g. anchor) features
            y_ref (Optional[torch.Tensor]): (B, D) center tensor of un-noised x (e.g. action) features
            y_action (Optional[torch.Tensor]): (B, D) center tensor of un-noised x (e.g. action) features
            P_A_one_hot (Optional[torch.Tensor]): (B, 2, N) one-hot encoder for action features
            P_B_one_hot (Optional[torch.Tensor]): (B, 2, N) one-hot encoder for anchor features
        """
        ################################
        # For now, do not use y and x0 #
        ################################
        
        # rotary position embedding, if enabled
        if self.model_cfg.rotary:
            x_pos = self.rotary_pos_enc(x.permute(0, 2, 1))
            y_pos = self.rotary_pos_enc(y.permute(0, 2, 1))
        
        # initialize x_emb an an empty tensor
        B, D, N = y.shape

        if self.model_cfg.encoder_style == 'indiv':
            # construct features
            f_x = None
            f_y = None

            # construct pos
            pos_x = x0.permute(0, 2, 1).reshape(-1, D)  # using original action as positional grouping for PN2
            pos_y = y.permute(0, 2, 1).reshape(-1, D)   # using original anchor as positional grouping for PN2

            # construct batch
            batch_idx_x = torch.arange(B).repeat_interleave(N)  # Shape: (B * N,)
            batch_idx_y = torch.arange(B).repeat_interleave(N)  # Shape: (B * N,)

            data_x = Data(x=f_x, pos=pos_x, batch=batch_idx_x)
            data_y = Data(x=f_y, pos=pos_y, batch=batch_idx_y)

            # PN2 embedding
            x_emb = self.x_embedder(data_x.cuda())
            y_emb = self.y_embedder(data_y.cuda())

            x_emb = x_emb.reshape(B, -1, N)     # reshape back to (B, d_embed, N)
            y_emb = y_emb.reshape(B, -1, N)     # reshape back to (B, d_embed, N)

            if self.noise_encoder is not None:
                add_emb = self.noise_encoder(x)
                x_emb = torch.cat([x_emb, add_emb], dim=1)
            '''
            if self.onehot_encoder is not None:
                assert P_A_one_hot is not None, "P_A_one_hot features must be provided if onehot_encoder is not None"
                add_emb = self.onehot_encoder(P_A_one_hot)
                x_emb = torch.cat([x_emb, add_emb], dim=1)
            '''
            if self.recon_encoder is not None:
                assert P_A is not None and y_action is not None, "y_action and P_A features must be provided if onehot_encoder is not None"
                P_A_hat_prime = x + P_A - y_action.unsqueeze(-1)
                add_emb = self.recon_encoder(P_A_hat_prime)
                x_emb = torch.cat([x_emb, add_emb], dim=1)

            # encode context features
            if self.context_encoder is not None:
                if self.model_cfg.feature_context_type == "anchor_mean":
                    assert y_ref is not None and P_A is not None, "y_ref and P_A features must be provided if context_encoder of type anchor_mean is not None"
                    anchor_mean = P_A - y_ref.unsqueeze(-1)
                    add_emb = self.context_encoder(anchor_mean)
                    x_emb = torch.cat([x_emb, add_emb], dim=1)
                if self.model_cfg.feature_context_type == "action_mean":
                    assert y_action is not None and P_A is not None, "y_action and P_A features must be provided if context_encoder of type action_mean is not None"
                    action_mean = P_A - y_action.unsqueeze(-1)
                    add_emb = self.context_encoder(action_mean)
                    x_emb = torch.cat([x_emb, add_emb], dim=1)

            # encode flow features
            if self.flow_encoder is not None:
                assert y_ref is not None and P_A is not None, "y_ref and P_A features must be provided if using flow features"
                P_A_hat = x + P_A + (y_ref.unsqueeze(-1) - y_action.unsqueeze(-1))   # broadcast y_ref from [B, C] to [B, C, 1]
                flow = P_A_hat - P_A

                if self.model_cfg.feature_flow_type == "unnorm":
                    add_emb = self.flow_encoder(flow)
                    x_emb = torch.cat([x_emb, add_emb], dim=1)
                if self.model_cfg.feature_flow_type == "unit":
                    # compute the L2 norm across the last dimension
                    norm = torch.norm(flow, dim=1, keepdim=True)  # Shape: [16, 1, 512]
                    # avoid division by zero
                    norm = torch.where(norm == 0, torch.tensor(1.0, device=flow.device), norm)
                    # Normalize flow to unit vectors
                    flow_unitnorm = flow / norm
                    add_emb = self.flow_encoder(flow_unitnorm)
                    x_emb = torch.cat([x_emb, add_emb], dim=1)
                if self.model_cfg.feature_flow_type == "clip":
                    flow_clip = torch.clamp(flow, min=-1.0, max=1.0)
                    add_emb = self.flow_encoder(flow_clip)
                    x_emb = torch.cat([x_emb, add_emb], dim=1)
                if self.model_cfg.feature_flow_type == "zeromean":
                    mean = flow.mean(dim=2, keepdim=True)
                    flow_zeromean = flow - mean
                    add_emb = self.flow_encoder(flow_zeromean)
                    x_emb = torch.cat([x_emb, add_emb], dim=1)

            x = x_emb.permute(0, 2, 1)
 
            zero_emb = torch.zeros((B, self.per_hidden_size, N), device=add_emb.device, dtype=add_emb.dtype)  # BUG here. Also dont't forget to restore context_encoder hidden size
            for _ in range(self.num_features-1):
                y_emb = torch.cat([y_emb, zero_emb], dim=1)
            y_emb = y_emb.permute(0, 2, 1)

        elif self.model_cfg.encoder_style == 'concat':
            x_feat = torch.empty((B, 0, N), device=x.device)

            if self.model_cfg.noise_encoder:
                x_feat = torch.cat([x_feat, x], dim=1)

            '''
            if self.model_cfg.onehot_encoder:
                assert P_A_one_hot is not None, "P_A_one_hot features must be provided if onehot_encoder is not None"
                x_feat = torch.cat([x_feat, P_A_one_hot], dim=1)
            '''
            if self.model_cfg.recon_encoder:
                assert P_A is not None and y_action is not None, "y_action and P_A features must be provided if onehot_encoder is not None"
                P_A_hat_prime = x + P_A - y_action.unsqueeze(-1)
                x_feat = torch.cat([x_feat, P_A_hat_prime], dim=1)
            
            # encode context features
            if self.model_cfg.context_encoder:
                if self.model_cfg.feature_context_type == "anchor_mean":
                    assert y_ref is not None and P_A is not None, "y_ref and P_A features must be provided if context_encoder of type anchor_mean is not None"
                    anchor_mean = P_A - y_ref.unsqueeze(-1)
                    x_feat = torch.cat([x_feat, anchor_mean], dim=1)
                if self.model_cfg.feature_context_type == "action_mean":
                    assert y_action is not None and P_A is not None, "y_action and P_A features must be provided if context_encoder of type action_mean is not None"
                    action_mean = P_A - y_action.unsqueeze(-1)
                    x_feat = torch.cat([x_feat, action_mean], dim=1)

            # encode flow features
            if self.model_cfg.flow_encoder:
                assert y_ref is not None and P_A is not None, "y_ref and P_A features must be provided if using flow features"
                P_A_hat = x + P_A + (y_ref.unsqueeze(-1) - y_action.unsqueeze(-1))   # broadcast y_ref from [B, C] to [B, C, 1]
                flow = P_A_hat - P_A

                if self.model_cfg.feature_flow_type == "unnorm":
                    x_feat = torch.cat([x_feat, flow], dim=1)
                if self.model_cfg.feature_flow_type == "unit":
                    # compute the L2 norm across the last dimension
                    norm = torch.norm(flow, dim=1, keepdim=True)  # Shape: [16, 1, 512]
                    # avoid division by zero
                    norm = torch.where(norm == 0, torch.tensor(1.0, device=flow.device), norm)
                    # Normalize flow to unit vectors
                    flow_unitnorm = flow / norm
                    x_feat = torch.cat([x_feat, flow_unitnorm], dim=1)
                if self.model_cfg.feature_flow_type == "clip":
                    flow_clip = torch.clamp(flow, min=-1.0, max=1.0)
                    x_feat = torch.cat([x_feat, flow_clip], dim=1)
                if self.model_cfg.feature_flow_type == "zeromean":
                    mean = flow.mean(dim=2, keepdim=True)
                    flow_zeromean = flow - mean
                    x_feat = torch.cat([x_feat, flow_zeromean], dim=1)
            
            x_feat_l2 = x_feat / (torch.norm(x_feat, p=2, dim=1, keepdim=True) + 1e-8)  # Normalize each feature across C

            '''
            x_emb = self.x_embedder(x_feat_l2)
            x = x_emb.permute(0, 2, 1)
            
            y_emb = self.y_embedder(y)
            y_emb = y_emb.permute(0, 2, 1)
            '''
        
            # check x_fea_l2 dim here
            f_x = x_feat_l2.permute(0, 2, 1).reshape(-1, self.in_channels)  # Shape: (B * N, self.in_channels)
            f_y = None

            # construct pos
            pos_x = x0.permute(0, 2, 1).reshape(-1, D)  # using original action as positional grouping for PN2
            pos_y = y.permute(0, 2, 1).reshape(-1, D)   # using original anchor as positional grouping for PN2

            # construct batch
            batch_idx_x = torch.arange(B).repeat_interleave(N)  # Shape: (B * N,)
            batch_idx_y = torch.arange(B).repeat_interleave(N)  # Shape: (B * N,)

            data_x = Data(x=f_x, pos=pos_x, batch=batch_idx_x)
            data_y = Data(x=f_y, pos=pos_y, batch=batch_idx_y)

            # PN2 embedding
            #breakpoint()
            x = self.x_embedder(data_x.cuda())
            #breakpoint()
            y_emb = self.y_embedder(data_y.cuda())

            x = x.reshape(B, N, -1)             # reshape back to (B, N, d_embed)
            y_emb = y_emb.reshape(B, N, -1)     # reshape back to (B, N, d_embed)


        else:
            raise ValueError(f"Invalid encoder_style: {self.model_cfg.encoder_style}")


        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks:
            if self.model_cfg.rotary:
                x = block(x, y_emb, t_emb, x_pos, y_pos)
            else:
                x = block(x, y_emb, t_emb)

        # final layer
        x = self.final_layer(x, t_emb)
        x = x.permute(0, 2, 1)
        return x

class DiT_PointCloud_Unc_Cross(nn.Module):
    """
    Diffusion model with a Transformer backbone - point cloud, unconditional, with scene cross attention
    """

    def __init__(
        self,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        x_encoder_hidden_dims = hidden_size
        if self.model_cfg.x_encoder is not None and self.model_cfg.x0_encoder is not None:
            # We are concatenating x and x0 features so we halve the hidden size
            x_encoder_hidden_dims = hidden_size // 2

        # Encoder for current timestep x features       
        if self.model_cfg.x_encoder == "mlp":
            # x_embedder is conv1d layer instead of 2d patch embedder
            self.x_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")
        
        # Encoder for y features
        if self.model_cfg.y_encoder == "mlp":
            self.y_embedder = nn.Conv1d(
                in_channels,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.y_encoder == "dgcnn":
            self.y_embedder = DGCNN(
                input_dims=in_channels, emb_dims=hidden_size
            )
        else:
            raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

        # Encoder for x0 features
        if self.model_cfg.x0_encoder == "mlp":
            self.x0_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.x0_encoder == "dgcnn":
            self.x0_embedder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif self.model_cfg.x0_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")

        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTCrossBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        
        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        if self.model_cfg.center_noise:
            relative_center = torch.mean(x, dim=2, keepdim=True)
            x = x - relative_center
            y = y - relative_center
        
        x_emb = self.x_embedder(x)

        if self.model_cfg.x0_encoder is not None:
            assert x0 is not None, "x0 must be provided if x0_encoder is not None"
            x0_emb = self.x0_embedder(x0)
            x_emb = torch.cat((x_emb, x0_emb), dim=1)

        if self.model_cfg.y_encoder is not None:
            y_emb = self.y_embedder(y)
            y_emb = y_emb.permute(0, 2, 1)

        x = x_emb.permute(0, 2, 1)

        c = self.t_embedder(t)

        for block in self.blocks:
            x = block(x, y_emb, c)

        x = self.final_layer(x, c)

        x = x.permute(0, 2, 1)

        return x

class Rel3D_DiT_PointCloud_Unc_Cross(nn.Module):
    """
    Diffusion model with a Transformer backbone - point cloud, unconditional, with scene cross attention, and relative 3D positional encoding and attention.
    """

    def __init__(
        self,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        self.relative_3d_encoding = RotaryPositionEncoding3D(hidden_size)

        x_encoder_hidden_dims = hidden_size
        if self.model_cfg.x_encoder is not None and self.model_cfg.x0_encoder is not None:
            # We are concatenating x and x0 features so we halve the hidden size
            x_encoder_hidden_dims = hidden_size // 2

        # Encoder for current timestep x features       
        if self.model_cfg.x_encoder == "mlp":
            # x_embedder is conv1d layer instead of 2d patch embedder
            self.x_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")
        
        # Encoder for y features
        if self.model_cfg.y_encoder == "mlp":
            self.y_embedder = nn.Conv1d(
                in_channels,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.y_encoder == "dgcnn":
            self.y_embedder = DGCNN(
                input_dims=in_channels, emb_dims=hidden_size
            )
        else:
            raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

        # Encoder for x0 features
        if self.model_cfg.x0_encoder == "mlp":
            self.x0_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.x0_encoder == "dgcnn":
            self.x0_embedder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif self.model_cfg.x0_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")

        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTRelativeCrossBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        
        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
        """

        if self.model_cfg.center_noise:
            relative_center = torch.mean(x, dim=2, keepdim=True)
            x = x - relative_center
            y = y - relative_center

        # Get x and y relative 3D positional encoding
        x_pos = self.relative_3d_encoding(x.permute(0, 2, 1))
        y_pos = self.relative_3d_encoding(y.permute(0, 2, 1))
        
        # Get x and y features        
        x_emb = self.x_embedder(x)

        if self.model_cfg.x0_encoder is not None:
            assert x0 is not None, "x0 must be provided if x0_encoder is not None"
            x0_emb = self.x0_embedder(x0)
            x_emb = torch.cat((x_emb, x0_emb), dim=1)

        if self.model_cfg.y_encoder is not None:
            y_emb = self.y_embedder(y)
            y_emb = y_emb.permute(0, 2, 1)

        x = x_emb.permute(0, 2, 1)
        
        c = self.t_embedder(t)

        for i, block in enumerate(self.blocks):
            x = block(x, y_emb, c, x_pos, y_pos)

        x = self.final_layer(x, c)

        x = x.permute(0, 2, 1)
            
        return x