import numpy as np
import torch

from pytorch3d.loss import chamfer_distance

########## CORRESPONDENCE-BASED METRICS ##########
# TODO: implement these with reduction
def flow_cos_sim(pred_flow, gt_flow, mask, seg):
    """
        pred_flow: [n, sample_size, 3]
        gt_flow: [n, sample_size, 3]
    """
    cos_sims = torch.cosine_similarity(pred_flow, gt_flow, dim=-1)
    if mask:
        cos_sims = torch.mul(seg, cos_sims)
        n_points = seg.sum(dim=-1)
        cos_sims = cos_sims.sum(dim=-1) / n_points
    else:
        cos_sims = cos_sims.mean(dim=-1)
    return cos_sims


def flow_rmse(pred_flow, gt_flow, mask, seg):
    """
        pred_flow: [n, sample_size, 3]
        gt_flow: [n, sample_size, 3]
    """
    mse = (pred_flow - gt_flow).pow(2).sum(dim=-1)
    if mask:
        mse = torch.mul(seg, mse)
        n_points = seg.sum(dim=-1)
        mse = mse.sum(dim=-1) / n_points
    else:
        mse = mse.mean(dim=-1)
    return torch.sqrt(mse)


def pcd_rmse(pred_pc, gt_pc):
    """
        pred_pc: [n, sample_size, 3]
        gt_pc: [n, sample_size, 3]
    """
    mse = (pred_pc - gt_pc).pow(2).sum(dim=-1)
    mse = mse.mean(dim=-1)
    return torch.sqrt(mse)

########## CORRESPONDENCE-FREE METRICS ##########

def get_chamfer_pairs(inference_pcs, reference_pcs):
    """
    Helper function to compute pairwise Chamfer distances between two sets of point clouds.
    """
    # TODO: vectorize within memory constraints?
    n_inf = inference_pcs.shape[0]
    n_ref = reference_pcs.shape[0]
    pairwise_dists = torch.zeros(n_inf, n_ref)
    for i in range(n_inf):
        pred_pc_i = inference_pcs[[i]*n_ref]
        loss, _ = chamfer_distance(pred_pc_i, reference_pcs, batch_reduction=None, single_directional=False)
        pairwise_dists[i] = loss
    return pairwise_dists

def pc_nn(pairwise_dists):
    """
    Helper function to convert pairwise distances to nearest neighbors.
    """
    inference_nn = torch.argmin(pairwise_dists, dim=1)
    reference_nn = torch.argmin(pairwise_dists, dim=0)
    return inference_nn, reference_nn

def pc_coverage(inference_nn, reference_nn, bidirectional=False):
    inference_cov = torch.unique(inference_nn).shape[0] / reference_nn.shape[0]
    reference_cov = torch.unique(reference_nn).shape[0] / inference_nn.shape[0]
    return inference_cov, reference_cov

