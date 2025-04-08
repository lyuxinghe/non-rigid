#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. dataset name
# 5. the rest of the arguments for the train.py script

# Example usage:
# ./train_deform.sh 0 cross_flow offline dedo

# Resuming from a crashed run:
#./train_rigid.sh 0 ddrd_flow_separate rpdiff_fit online checkpoint.run_id=k8iy8vfo checkpoint.local_ckpt='/home/lyuxing/Desktop/tax3d_upgrade/scripts/logs/train_rpdiff_feature_df_cross/2025-03-02/17-41-55/checkpoints/last.ckpt'

GPU_INDEX=$1
MODEL_TYPE=$2
WANDB_MODE=$3
DATASET_NAME=$4
shift
shift
shift
shift
COMMAND=$@
DATASET_PARAMS="dataset=$DATASET_NAME"

if [ $MODEL_TYPE == "cross_flow" ]; then
  echo "Training cross flow model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=flow"

elif [ $MODEL_TYPE == "cross_point" ]; then
  echo "Training cross point model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=point"

elif [ $MODEL_TYPE == "ddrd" ]; then
  echo "Training shape-frame model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=ddrd model.type=point model.model_take=separate"

elif [ $MODEL_TYPE == "ddrd_test" ]; then
  echo "Training shape-frame model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=ddrd model.type=point model.model_take=joint"

elif [ $MODEL_TYPE == "tax3dv2" ]; then
  echo "Training tax3dv2 model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=tax3dv2 model.type=point"

fi

WANDB_MODE=$WANDB_MODE python train.py \
  $MODEL_PARAMS \
  $DATASET_PARAMS \
  wandb.group=tax3d_upgrade_rigid \
  resources.gpus=[${GPU_INDEX}] \
  $COMMAND
