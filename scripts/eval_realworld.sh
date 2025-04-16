#!/bin/bash

# This should take in 3 arguments:
# 1. the index of which GPU to use
# 2. model checkpoint
# 3. the rest of the arguments for the eval script

# Example usage:
# Load from Wandb: ./eval_realworld.sh 0 `CHECKPOINT`
# Load from local: ./eval_realworld.sh 0 `CHECKPOINT` checkpoint.reference="/home/lyuxing/Desktop/tax3d_upgrade/scripts/logs/train_rpdiff_df_cross/2025-04-11/12-50-17/checkpoints/last.ckpt"
# Evaluate with GMM: ./eval_realworld.sh 0 `CHECKPOINT` gmm=1000 exp_name=rpdiff_mug_single_easy_rack

########## Insertion ##########
### 12-15-ssd ### *PCD_Scale=50*
## Anchor Centroid ##
# ./eval_realworld.sh 0 er7k5kam

GPU_INDEX=$1
CHECKPOINT=$2
shift
shift
COMMAND=$@

echo "Evaluating model at checkpoint $CHECKPOINT with command: $COMMAND."
python eval.py \
    resources.gpus=[${GPU_INDEX}] \
    checkpoint.run_id=${CHECKPOINT} \
    wandb.group=rigid \
    wandb.project=corl2025_tax3dv2 \
    $COMMAND