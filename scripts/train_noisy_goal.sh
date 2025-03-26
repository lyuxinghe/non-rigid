#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. the rest of the arguments for the train.py script

# Example usage:
#./train_noisy_goal.sh 0 ddrd_point rpdiff_fit offline dataset.anchor_translation_variance=0.0

# Resuming from a crashed run:
#./train_noisy_goal.sh 0 ddrd_point rpdiff_fit online checkpoint.run_id=9ozopbwx checkpoint.local_ckpt='/home/lyuxing/Desktop/tax3d_upgrade/scripts/logs/train_rpdiff_fit_ddrd/2025-03-16/01-26-44/checkpoints/last.ckpt'


GPU_INDEX=$1
MODEL_TYPE=$2
DATASET_NAME=$3
WANDB_MODE=$4
shift
shift
shift
shift
COMMAND=$@

if [ $MODEL_TYPE == "cross_point_relative" ]; then
  echo "Training Cross Relative Point model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=point"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=point dataset.scene=False dataset.world_frame=False dataset.noisy_goal=True"
elif [ $MODEL_TYPE == "cross_flow_relative" ]; then
  echo "Training Cross Relative Flow model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=flow"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=flow dataset.scene=False dataset.world_frame=False dataset.noisy_goal=True"
elif [ $MODEL_TYPE == "ddrd_point_joint" ]; then
  echo "Training DDRD point joint model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=ddrd model.type=point model.model_take=joint"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=point dataset.scene=False dataset.world_frame=False dataset.noisy_goal=True"
elif [ $MODEL_TYPE == "ddrd_flow_joint" ]; then
  echo "Training DDRD flow joint model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=ddrd model.type=flow model.model_take=joint"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=flow dataset.scene=False dataset.world_frame=False dataset.noisy_goal=True"
elif [ $MODEL_TYPE == "ddrd_point_separate" ]; then
  echo "Training DDRD point separate model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=ddrd model.type=point model.model_take=separate"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=point dataset.scene=False dataset.world_frame=False dataset.noisy_goal=True"
elif [ $MODEL_TYPE == "ddrd_flow_separate" ]; then
  echo "Training DDRD flow separate model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=ddrd model.type=flow model.model_take=separate"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=flow dataset.scene=False dataset.world_frame=False dataset.noisy_goal=True"
fi

WANDB_MODE=$WANDB_MODE python train.py \
  $MODEL_PARAMS \
  $DATASET_PARAMS \
  wandb.group=tax3d_upgrade_rigid \
  resources.gpus=[${GPU_INDEX}] \
  $COMMAND


