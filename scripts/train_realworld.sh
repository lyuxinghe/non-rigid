#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. the rest of the arguments for the train.py script

########## Insertion ##########
### 12-15-ssd ### *PCD_Scale=50*
## Anchor Centroid ##
# ./train_realworld.sh 0 tax3dv2 online insertion model.frame_type=fixed model.pred_frame=anchor_center model.point_encoder=pn2 model.diff_rotation_noise_scale=45 resources.num_workers=32


#------ Run On AutoBot with Singularity ------#
# singularity exec --nv -B /home/lyuxingh/code/tax3d_upgrade:/opt/lyuxingh/code/tax3d_upgrade -B /scratch/lyuxingh/data:/opt/lyuxingh/data -B /scratch/lyuxingh/logs:/opt/lyuxingh/logs /scratch/lyuxingh/singularity/tax3d_lyuxing.sif bash -c "cd /opt/lyuxingh/code/tax3d_upgrade/scripts && CUDA_VISIBLE_DEVICES=？ ./train.sh <YOUR TRAIN COMMAND HERE> dataset.data_dir=/opt/lyuxingh/data/rpdiff/data/task_demos/ log_dir=/opt/lyuxingh/logs"

#------ Resuming from a crashed run ------#
#./train.sh 0 cross_point_relative rpdiff False online checkpoint.run_id=k8iy8vfo checkpoint.local_ckpt='/home/lyuxing/Desktop/tax3d_upgrade/scripts/logs/train_rpdiff_feature_df_cross/2025-03-02/17-41-55/checkpoints/last.ckpt'

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

elif [ $MODEL_TYPE == "tax3dv2" ]; then
  echo "Training tax3dv2 model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=tax3dv2 model.type=point"

fi



WANDB_MODE=$WANDB_MODE python train.py \
  $MODEL_PARAMS \
  $DATASET_PARAMS \
  wandb.group=rigid \
  wandb.project=corl2025_tax3dv2 \
  resources.gpus=[${GPU_INDEX}] \
  $COMMAND


