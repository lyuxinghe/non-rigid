#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. the rest of the arguments for the train.py script

# Anchor Centroid
# ./train_upgrade.sh 0 cross_point_relative rpdiff_fit online model.pred_ref_frame=anchor model.ref_error_scale=0.0 resources.num_workers=16
# ./train_upgrade.sh 0 cross_flow_relative rpdiff_fit online model.pred_ref_frame=anchor model.ref_error_scale=0.0 resources.num_workers=16

# Noisy Goal Centroid
# ./train_upgrade.sh 0 cross_point_relative rpdiff_fit online model.pred_ref_frame=noisy_goal model.ref_error_scale=1.0 resources.num_workers=16
# ./train_upgrade.sh 0 cross_flow_relative rpdiff_fit online model.pred_ref_frame=noisy_goal model.ref_error_scale=1.0 resources.num_workers=16

# Resuming from a crashed run:
#./train_rigid.sh 0 ddrd_flow_separate rpdiff_fit online checkpoint.run_id=k8iy8vfo checkpoint.local_ckpt='/home/lyuxing/Desktop/tax3d_upgrade/scripts/logs/train_rpdiff_feature_df_cross/2025-03-02/17-41-55/checkpoints/last.ckpt'


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
elif [ $MODEL_TYPE == "mu_point_take1" ]; then
  echo "Training Mu-Frame point model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=mu model.type=point model.model_take=1"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=point"
elif [ $MODEL_TYPE == "mu_point_take2" ]; then
  echo "Training Mu-Frame point model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=mu model.type=point model.model_take=2"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=point"
elif [ $MODEL_TYPE == "mu_point_take3" ]; then
  echo "Training Mu-Frame point model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=mu model.type=point model.model_take=3"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=point"
elif [ $MODEL_TYPE == "mu_point_take4" ]; then
  echo "Training Mu-Frame point model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=mu model.type=point model.model_take=4"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=point"
elif [ $MODEL_TYPE == "ddrd_point_separate" ]; then
  echo "Training DDRD point separate model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=ddrd model.type=point model.model_take=separate"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=point"
fi

WANDB_MODE=$WANDB_MODE python train.py \
  $MODEL_PARAMS \
  $DATASET_PARAMS \
  wandb.group=tax3d_upgrade_rigid \
  resources.gpus=[${GPU_INDEX}] \
  $COMMAND
