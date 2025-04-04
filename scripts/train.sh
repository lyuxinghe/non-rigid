#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. the rest of the arguments for the train.py script

# Anchor Centroid
# ./train.sh 0 cross_point_relative rpdiff online model.pred_ref_frame=anchor model.ref_error_scale=0.0
# ./train.sh 0 cross_flow_relative rpdiff online model.pred_ref_frame=anchor model.ref_error_scale=0.0

# Noisy Goal Centroid
# ./train_upgrade.sh 0 cross_point_relative rpdiff_fit online model.pred_ref_frame=noisy_goal model.ref_error_scale=1.0
# ./train_upgrade.sh 0 cross_flow_relative rpdiff_fit online model.pred_ref_frame=noisy_goal model.ref_error_scale=1.0

# Resuming from a crashed run:
#./train.sh 0 cross_point_relative rpdiff_fit False online checkpoint.run_id=k8iy8vfo checkpoint.local_ckpt='/home/lyuxing/Desktop/tax3d_upgrade/scripts/logs/train_rpdiff_feature_df_cross/2025-03-02/17-41-55/checkpoints/last.ckpt'

GPU_INDEX=$1
MODEL_TYPE=$2
DATASET_NAME=$3
WANDB_MODE=$4
shift
shift
shift
shift
COMMAND=$@


if [ $MODEL_TYPE == "cross_flow_relative" ]; then
  echo "Training cross relative flow model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=flow"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=flow"

elif [ $MODEL_TYPE == "cross_point_relative" ]; then
  echo "Training cross relative point model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=point"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=point"

elif [ $MODEL_TYPE == "tax3dv2_point" ]; then
  echo "Training TAX3Dv2 point model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=tax3dv2 model.type=point"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=point dataset.scene=False dataset.world_frame=False"

elif [ $MODEL_TYPE == "tax3dv2_flow" ]; then
  echo "Training TAX3Dv2 flow model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=tax3dv2 model.type=flow"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=flow dataset.scene=False dataset.world_frame=False"
fi



WANDB_MODE=$WANDB_MODE python train.py \
  $MODEL_PARAMS \
  $DATASET_PARAMS \
  wandb.group=rigid \
  wandb.project=corl2025_tax3dv2 \
  resources.gpus=[${GPU_INDEX}] \
  $COMMAND
