import json
import os

import hydra
import lightning as L
import omegaconf
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from non_rigid.utils.script_utils import (
    PROJECT_ROOT,
    create_datamodule,
    create_model,
    match_fn,
)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    print(
        json.dumps(
            omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
            sort_keys=True,
            indent=4,
        )
    )

    TESTING = "PYTEST_CURRENT_TEST" in os.environ

    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("medium")

    # Global seed for reproducibility.
    L.seed_everything(cfg.seed)

    ######################################################################
    # Create the datamodule.
    # The datamodule is responsible for all the data loading, including
    # downloading the data, and splitting it into train/val/test.
    #
    # This could be swapped out for a different datamodule in-place,
    # or with an if statement, or by using hydra.instantiate.
    ######################################################################
    cfg, datamodule = create_datamodule(cfg)

    ######################################################################
    # Create the network(s) which will be trained by the Training Module.
    # The network should (ideally) be lightning-independent. This allows
    # us to use the network in other projects, or in other training
    # configurations. Also create the Training Module.
    #
    # This might get a bit more complicated if we have multiple networks,
    # but we can just customize the training module and the Hydra configs
    # to handle that case. No need to over-engineer it. You might
    # want to put this into a "create_network" function somewhere so train
    # and eval can be the same.
    #
    # If it's a custom network, a good idea is to put the custom network
    # in `python_ml_project_template.nets.my_net`.
    ######################################################################

    # Model architecture is dataset-dependent, so we have a helper
    # function to create the model (while separating out relevant vals).
    network, model = create_model(cfg)

    # datamodule.setup(stage="fit")
    # cfg.training.num_training_steps = (
    #     len(datamodule.train_dataloader()) * cfg.training.epochs
    # )
    # # updating the training sample size
    # # cfg.training.training_sample_size = cfg.dataset.sample_size

    # TODO: compiling model doesn't work with lightning out of the box?
    # model = torch.compile(model)

    ######################################################################
    # Set up logging in WandB.
    # This is a bit complicated, because we want to log the codebase,
    # the model, and the checkpoints.
    ######################################################################

    # If no group is provided, then we should create a new one (so we can allocate)
    # evaluations to this group later.
    if cfg.wandb.group is None:
        id = wandb.util.generate_id()
        group = "experiment-" + id
    else:
        group = cfg.wandb.group

    if cfg.checkpoint.run_id:
        resume = 'must'
        id = cfg.checkpoint.run_id
    else:
        resume = 'allow'
        id = None

    logger = WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        log_model=True,  # Only log the last checkpoint to wandb, and only the LAST model checkpoint.
        save_dir=cfg.wandb.save_dir,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=False
        ),
        job_type=cfg.job_type,
        save_code=True,  # This just has the main script.
        group=group,
        resume=resume,
        id = id,
    )

    ######################################################################
    # Create the trainer.
    # The trainer is responsible for running the training loop, and
    # logging the results.
    #
    # There are a few callbacks (which we could customize):
    # - LogPredictionSamplesCallback: Logs some examples from the dataset,
    #       and the model's predictions.
    # - ModelCheckpoint #1: Saves the latest model.
    # - ModelCheckpoint #2: Saves the best model (according to validation
    #       loss), and logs it to wandb.
    ######################################################################

    trainer = L.Trainer(
        accelerator="gpu",
        devices=cfg.resources.gpus,
        # precision="16-mixed",
        precision="32-true",
        max_epochs=cfg.training.epochs,
        logger=logger if not TESTING else False,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epochs,
        # log_every_n_steps=2, # TODO: MOVE THIS TO TRAINING CFG
        log_every_n_steps=len(datamodule.train_dataloader()),
        gradient_clip_val=cfg.training.grad_clip_norm,
        callbacks=(
            [
                # Callback which logs whatever visuals (i.e. dataset examples, preds, etc.) we want.
                # LogPredictionSamplesCallback(logger),
                # This checkpoint callback saves the latest model during training, i.e. so we can resume if it crashes.
                # It saves everything, and you can load by referencing last.ckpt.
                # CustomModelPlotsCallback(logger),
                ModelCheckpoint(
                    dirpath=cfg.lightning.checkpoint_dir,
                    filename="{epoch}-{step}",
                    monitor="step",
                    mode="max",
                    save_weights_only=False,
                    save_last=True,
                ),
                ModelCheckpoint(
                    dirpath=cfg.lightning.checkpoint_dir,
                    filename="{epoch}-{step}-{val_rmse_0:.3f}",
                    monitor="val_rmse_0",
                    mode="min",
                    save_weights_only=False,
                    save_last=False,
                    # auto_insert_metric_name=False,
                ),
                ModelCheckpoint(
                    dirpath=cfg.lightning.checkpoint_dir,
                    filename="{epoch}-{step}-{val_rmse_wta_0:.3f}",
                    monitor="val_rmse_wta_0",
                    mode="min",
                    save_weights_only=False,
                    save_last=False,
                    # auto_insert_metric_name=False,
                ),
                # ModelCheckpoint(
                #     dirpath=cfg.lightning.checkpoint_dir,
                #     filename="{epoch}-{step}-{val_rmse_0:.3f}",
                #     monitor="val_rmse_0",
                #     mode="min",
                #     save_weights_only=False,
                #     save_last=False,
                # )
                # This checkpoint will get saved to WandB. The Callback mechanism in lightning is poorly designed, so we have to put it last.
                # ModelCheckpoint(
                #     dirpath=cfg.lightning.checkpoint_dir,
                #     filename="{epoch}-{step}-{val_loss:.2f}-weights-only",
                #     monitor="val_loss",
                #     mode="min",
                #     save_weights_only=True,
                # ),
            ]
            if not TESTING
            else []
        ),
        fast_dev_run=5 if TESTING else False,
        num_sanity_val_steps=0,
    )

    ######################################################################
    # Log the code to wandb.
    # This is somewhat custom, you'll have to edit this to include whatever
    # additional files you want, but basically it just logs all the files
    # in the project root inside dirs, and with extensions.
    ######################################################################

    # Log the code used to train the model. Make sure not to log too much, because it will be too big.
    wandb.run.log_code(
        root=PROJECT_ROOT,
        include_fn=match_fn(
            dirs=["configs", "scripts", "src"],
            extensions=[".py", ".yaml"],
        ),
    )

    ######################################################################
    # Train the model.
    ######################################################################
    '''
    # this might be a little too "pythonic"
    if cfg.checkpoint.run_id:
        print(
            "Attempting to resume training from checkpoint: ", cfg.checkpoint.reference
        )

        api = wandb.Api()
        artifact_dir = cfg.wandb.artifact_dir
        artifact = api.artifact(cfg.checkpoint.reference, type="model")
        ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
        # ckpt = torch.load(ckpt_file)
        # # model.load_state_dict(
        # #     {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
        # # )
        # model.load_state_dict(ckpt["state_dict"])
    

    else:
        print("Starting training from scratch.")
        ckpt_file = None
    '''
    if cfg.checkpoint.run_id:
        print("Attempting to resume training from WandB checkpoint:", cfg.checkpoint.reference)

        api = wandb.Api()
        artifact_dir = cfg.wandb.artifact_dir

        try:
            artifact = api.artifact(cfg.checkpoint.reference, type="model")
            ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
            print(f"Successfully downloaded checkpoint from WandB: {ckpt_file}")

        except Exception as e:
            print(f"Failed to load checkpoint from WandB. Error: {e}")

            if cfg.checkpoint.local_ckpt:
                print(f"Attempting to load from local path: {cfg.checkpoint.local_ckpt}")

                try:
                    ckpt_file = cfg.checkpoint.local_ckpt
                    with open(ckpt_file, "rb") as f:
                        print(f"Successfully loaded local checkpoint: {ckpt_file}")
                except Exception as e:
                    print(f"Failed to load checkpoint from local path. Error: {e}")
                    ckpt_file = None  # Fallback to training from scratch

    else:
        print("Starting training from scratch.")
        ckpt_file = None


    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_file)

    ######################################################################
    # Log additional model checkpoints to wandb.
    ######################################################################
    monitors = ["val_rmse_wta_0", "val_rmse_0"]
    model_artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
    # iterate through each file in checkpoint dir
    for file in os.listdir(cfg.lightning.checkpoint_dir):
        if file.endswith(".ckpt"):
            # check if metric name is in monitors
            metric_name = file.split("-")[-1].split("=")[0]
            if metric_name in monitors:
                # add checkpoint to artifact
                model_artifact.add_file(os.path.join(cfg.lightning.checkpoint_dir, file))
    wandb.run.log_artifact(model_artifact, aliases=["monitor"])


if __name__ == "__main__":
    main()
