#!/bin/bash

# This should take in 3 arguments:
# 1. the index of which GPU to use
# 2. model checkpoint
# 3. the rest of the arguments for the eval script

# Example usage:
# ./tax3dv2_cli.sh 0 kwcsrp5k gmm_log_dir=/home/lyuxing/Desktop/tax3d_upgrade/scripts/logs/gmm_outputs/ gmm_pcd_scale=finetune exp_name=book_bookshelf gmm=1000 data_in_path=/home/lyuxing/Desktop/third_party/rpdiff/src/rpdiff/eval_data/eval_data/book_bookshelf/tax3dv2_fixed_kwcsrp5k_gmm1000_pfinetune_psf15.0/seed_0/data/demo_aug_0.npz data_out_path=/home/lyuxing/Desktop/tax3d_upgrade/scripts/tmp/demo_aug_0_pred.npz vis_path=/home/lyuxing/Desktop/tax3d_upgrade/scripts/tmp/vis/

# Get absolute path to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_INDEX=$1
CHECKPOINT=$2
shift
shift
COMMAND=$@

echo "Evaluating model at checkpoint $CHECKPOINT with command: $COMMAND."
python "$SCRIPT_DIR/tax3dv2_cli.py" \
    resources.gpus=[${GPU_INDEX}] \
    checkpoint.run_id=${CHECKPOINT} \
    wandb.group=rigid \
    wandb.project=corl2025_tax3dv2 \
    $COMMAND