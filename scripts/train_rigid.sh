#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. the rest of the arguments for the train.py script

# Example usage:
# ./train_rigid.sh 0 cross_flow_relative ndf offline
# ./train_rigid.sh 0 cross_flow_relative rpdiff offline

# Resuming from a crashed run:
#./train_rigid.sh 0 feature_df_cross rpdiff online checkpoint.run_id=2nekrf5u checkpoint.local_ckpt='/home/lyuxing/Desktop/tax3d_upgrade/scripts/logs/train_rpdiff_feature_df_cross/2025-02-02/12-14-50/checkpoints/last.ckpt'


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
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=flow dataset.scene=False dataset.world_frame=False"

elif [ $MODEL_TYPE == "feature_df_cross" ]; then
  echo "Training feature cross relative flow model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=feature_df_cross model.type=flow"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=flow dataset.scene=False dataset.world_frame=False"
elif [ $MODEL_TYPE == "pn2_df_cross" ]; then
  echo "Training pointnet++ w/ cross relative flow model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=pn2_df_cross model.type=flow"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=flow dataset.scene=False dataset.world_frame=False"
elif [ $MODEL_TYPE == "pn2_feature_df_cross" ]; then
  echo "Training pointnet++ w/ feature cross relative flow model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=pn2_feature_df_cross model.type=flow"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=flow dataset.scene=False dataset.world_frame=False"
fi

WANDB_MODE=$WANDB_MODE python train.py \
  $MODEL_PARAMS \
  $DATASET_PARAMS \
  wandb.group=tax3d_upgrade_rigid \
  resources.gpus=[${GPU_INDEX}] \
  $COMMAND




<<COMMENT
# scene flow model - no object centric processing
if [ $MODEL_TYPE == "scene_flow" ]; then
  echo "Training scene flow model with command: $COMMAND."

  MODEL_PARAMS="model=df_base model.type=flow"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=flow dataset.scene=True dataset.world_frame=True"
# scene point model - no object centric processing
elif [ $MODEL_TYPE == "scene_point" ]; then
  echo "Training scene point model with command: $COMMAND."

  MODEL_PARAMS="model=df_base model.type=point"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=point dataset.scene=True dataset.world_frame=True"
# world frame cross flow
elif [ $MODEL_TYPE == "cross_flow_absolute" ]; then
  echo "Training absolute flow model with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=flow"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=flow dataset.scene=False dataset.world_frame=True"
# relative frame cross flow
elif [ $MODEL_TYPE == "cross_flow_relative" ]; then
  echo "Training relative flow model with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=flow"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=flow dataset.scene=False dataset.world_frame=False"
# world frame cross point
elif [ $MODEL_TYPE == "cross_point_absolute" ]; then
  echo "Training absolute point model with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=point"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=point dataset.scene=False dataset.world_frame=True"
# relative frame cross point
elif [ $MODEL_TYPE == "cross_point_relative" ]; then
  echo "Training relative point model with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=point"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=point dataset.scene=False dataset.world_frame=False"
# flow regression baseline
elif [ $MODEL_TYPE == "regression_flow" ]; then
  echo "Training flow regression model with command: $COMMAND."

  MODEL_PARAMS="model=regression model.type=flow"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=flow dataset.scene=False dataset.world_frame=False"
# point regression baseline
elif [ $MODEL_TYPE == "regression_point" ]; then
  echo "Training point regression model with command: $COMMAND."

  MODEL_PARAMS="model=regression model.type=point"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=point dataset.scene=False dataset.world_frame=False"
else
  echo "Invalid model type."
fi

WANDB_MODE=$WANDB_MODE python train.py \
  $MODEL_PARAMS \
  $DATASET_PARAMS \
  wandb.group=proc_cloth \
  resources.gpus=[${GPU_INDEX}] \
  $COMMAND