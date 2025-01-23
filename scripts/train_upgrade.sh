#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. the rest of the arguments for the train.py script

# Example usage:
# ./train_upgrade.sh 0 feature_df_cross online dataset.train_size=400 dataset.data_dir=/data/lyuxing/tax3d/proccloth/ dataset.cloth_geometry=multi dataset.hole=single wandb.group=tax3d_upgrade

GPU_INDEX=$1
MODEL_TYPE=$2
WANDB_MODE=$3
shift
shift
shift
COMMAND=$@


if [ $MODEL_TYPE == "feature_df_cross" ]; then
  echo "Training feature_df_cross model with command: $COMMAND."

  MODEL_PARAMS="model=feature_df_cross model.type=flow"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=flow dataset.scene=False dataset.world_frame=False"

elif [ $MODEL_TYPE == "pn2_df_cross" ]; then
  echo "Training pn2_df_cross model with command: $COMMAND."

  MODEL_PARAMS="model=pn2_df_cross model.type=flow"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=flow dataset.scene=False dataset.world_frame=False"

elif [ $MODEL_TYPE == "pn2_df" ]; then
  echo "Training pn2_df model with command: $COMMAND."

  MODEL_PARAMS="model=pn2_df model.type=flow"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=flow dataset.scene=False dataset.world_frame=False"
else
  echo "Invalid model type."
fi

WANDB_MODE=$WANDB_MODE python train.py \
  $MODEL_PARAMS \
  $DATASET_PARAMS \
  wandb.group=proc_cloth \
  resources.gpus=[${GPU_INDEX}] \
  $COMMAND