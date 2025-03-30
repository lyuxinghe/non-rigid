#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. dataset name
# 5. the rest of the arguments for the train.py script

# Example usage:
# ./train_deform.sh 0 cross_flow_relative offline dedo

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

if [ $MODEL_TYPE == "cross_flow_relative" ]; then
  echo "Training cross relative flow model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=flow"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.noisy_goal=False"

elif [ $MODEL_TYPE == "cross_point_relative" ]; then
  echo "Training cross relative point model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=point"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.noisy_goal=False"

elif [ $MODEL_TYPE == "feature_df_cross" ]; then
  echo "Training feature cross relative flow model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=feature_df_cross model.type=flow"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.noisy_goal=False"
elif [ $MODEL_TYPE == "pn2_df_cross" ]; then
  echo "Training pointnet++ w/ cross relative flow model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=pn2_df_cross model.type=flow"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.noisy_goal=False"
elif [ $MODEL_TYPE == "pn2_feature_df_cross" ]; then
  echo "Training pointnet++ w/ feature cross relative flow model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=pn2_feature_df_cross model.type=flow"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.noisy_goal=False"
elif [ $MODEL_TYPE == "ddrd_point_joint" ]; then
  echo "Training DDRD point joint model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=ddrd model.type=point model.model_take=joint"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.noisy_goal=False"
elif [ $MODEL_TYPE == "ddrd_flow_joint" ]; then
  echo "Training DDRD flow joint model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=ddrd model.type=flow model.model_take=joint"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.noisy_goal=False"
elif [ $MODEL_TYPE == "ddrd_point_separate" ]; then
  echo "Training DDRD point separate model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=ddrd model.type=point model.model_take=separate"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.noisy_goal=False"
elif [ $MODEL_TYPE == "ddrd_flow_separate" ]; then
  echo "Training DDRD flow separate model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=ddrd model.type=flow model.model_take=separate"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.noisy_goal=False"
fi

WANDB_MODE=$WANDB_MODE python train.py \
  $MODEL_PARAMS \
  $DATASET_PARAMS \
  wandb.group=tax3d_upgrade_rigid \
  resources.gpus=[${GPU_INDEX}] \
  $COMMAND
