#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. the rest of the arguments for the train.py script

# Example usage:
# ./multi_cloth_train.sh 0 cross_point_relative online
# ./multi_cloth_train.sh 1 scene_flow disabled dataset.multi_cloth.hole=single dataset.multi_cloth.size=100

GPU_INDEX=$1
MODEL_TYPE=$2
WANDB_MODE=$3
shift
shift
shift
COMMAND=$@


if [ $MODEL_TYPE == "upgrade" ]; then
  echo "Training upgrade model with command: $COMMAND."

  MODEL_PARAMS="model=upgrade model.type=point"
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