import numpy as np
import torch
import omegaconf
import json
import os
import shutil
import hydra

from plotly import graph_objects as go

from tqdm import tqdm

from non_rigid.utils.script_utils import create_datamodule
from non_rigid.models.gmm_predictor import FrameGMMPredictor, GMMLoss, viz_gmm

# ignore TypedStorage warnings
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning)

@hydra.main(config_path="../configs", config_name="eval_gmm", version_base="1.3")
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
    torch.manual_seed(cfg.seed)

    # Setting device.
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

    ######################################################################
    # Create the datamodule.
    ######################################################################
    if cfg.dataset.name != "dedo":
        raise ValueError(f"This evaluation script only works with the DEDO dataset.")
    cfg, datamodule = create_datamodule(cfg)
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset

    ######################################################################
    # Create the network.
    ######################################################################
    model = FrameGMMPredictor(cfg.model, device)

    ######################################################################
    # Evaluation loop.
    ######################################################################
    # Creating logging directory.
    exp_name = os.path.join(os.path.expanduser(cfg.gmm_log_dir), f"{cfg.job_type}_{cfg.epochs}")
    if not os.path.exists(exp_name):
        raise ValueError(f"Experiment directory {exp_name} does not exist - train this model first.")
    
    viz_path = os.path.join(exp_name, "viz")
    ckpt_path = os.path.join(exp_name, "checkpoints", f"epoch_{cfg.epochs}.pt")

    os.makedirs(os.path.join(viz_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(viz_path, "val"), exist_ok=True)

    # Loading model checkpoint.
    model.load_state_dict(torch.load(ckpt_path))

    # Helper function to compute evaluation metrics.
    @torch.no_grad()
    def eval_dataset(dataset, model, split):
        rmses = []
        coverage_rmses = []
        precision_rmses = []
        num_batches = len(dataset) // bs
        data_keys = ["pc_action", "pc_anchor", "pc"]

        for i in tqdm(range(num_batches), desc=f"Evaluating {split} set", leave=False):
            gt_means = []
            sampled_means = []

            for j in range(bs):
                item_j = dataset[i * bs + j]
                item_j = {key: value.unsqueeze(0) for key, value in item_j.items() if key in data_keys}
                pred_j = model(item_j)

                # Computing ground truth action mean in anchor frame.
                anchor_frame = pred_j["anchor_frame"].cpu()
                targets = item_j["pc"] - anchor_frame
                targets = targets.mean(dim=1)[0].cpu()
                gt_means.append(targets.unsqueeze(0))

                # Sampling predictions.
                probs, means = pred_j["probs"], pred_j["means"]
                idxs = torch.multinomial(probs.squeeze(-1), cfg.num_samples // bs, replacement=True).squeeze()
                sampled_means.append(means[:, idxs].squeeze())

            # Concatenating all samples.
            gt_means = torch.cat(gt_means, dim=0).to(device)
            sampled_means = torch.cat(sampled_means, dim=0)

            # Compute pairwise RMSEs, and update statistics.
            dists = torch.cdist(gt_means, sampled_means, p=2)
            rmses.append(dists.flatten().cpu().numpy())
            coverage_rmses.append(dists.min(dim=1).values.cpu().numpy())
            precision_rmses.append(dists.min(dim=0).values.cpu().numpy())
            
            # Plot point clouds.
            fig = go.Figure()
            anchor_pc = (item_j["pc_anchor"] - anchor_frame)[0].cpu().numpy()
            gt_means_pc = gt_means.cpu().numpy()
            sampled_means_pc = sampled_means.cpu().numpy()
            fig.add_trace(
                go.Scatter3d(
                    mode="markers",
                    marker=dict(size=2, color="blue"),
                    x=anchor_pc[:, 0],
                    y=anchor_pc[:, 1],
                    z=anchor_pc[:, 2],
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    mode="markers",
                    marker=dict(size=4, color="red"),
                    x=gt_means_pc[:, 0],
                    y=gt_means_pc[:, 1],
                    z=gt_means_pc[:, 2],
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    mode="markers",
                    marker=dict(size=4, color="green"),
                    x=sampled_means_pc[:, 0],
                    y=sampled_means_pc[:, 1],
                    z=sampled_means_pc[:, 2],
                )
            )
            fig.write_html(os.path.join(viz_path, split, f"{i}.html"))

        # Plot statistics.
        rmses = np.concatenate(rmses)
        coverage_rmses = np.concatenate(coverage_rmses)
        precision_rmses = np.concatenate(precision_rmses)
        hist = go.Figure()
        hist.add_trace(
            go.Histogram(
                x=rmses,
                name="RMSE",
            )
        )
        hist.add_trace(
            go.Histogram(
                x=precision_rmses,
                name="Precision RMSE",
            )
        )
        hist.add_trace(
            go.Histogram(
                x=coverage_rmses,
                name="Coverage RMSE",
            )
        )
        hist.update_layout(barmode="overlay")
        hist.update_traces(opacity=0.75)
        hist.update_layout(title_text=f"{split} Histograms")
        hist.write_html(os.path.join(viz_path, split, "hist.html"))

        rmses = np.mean(rmses)
        coverage_rmses = np.mean(coverage_rmses)
        precision_rmses = np.mean(precision_rmses)
        print(f"RMSE: {rmses}, Coverage RMSE: {coverage_rmses}, Precision RMSE: {precision_rmses}")


    # Eavluating datasets.
    eval_dataset(train_dataset, model, "train")
    eval_dataset(val_dataset, model, "val")

if __name__ == "__main__":
    main()