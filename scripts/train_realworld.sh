#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. the rest of the arguments for the train.py script

########## Insertion ##########
### 04-21-dsub-1 ### 
## Anchor Centroid ##
# ./train_realworld.sh 0 tax3dv2 online insertion model.frame_type=fixed model.pred_frame=anchor_center model.point_encoder=pn2 model.diff_rotation_noise_scale=45 dataset.connector_type=04-21-dsub-1 resources.num_workers=16


#------ Run On AutoBot with Singularity ------#
#singularity exec --nv -B /home/lyuxingh/code/tax3d_realworld:/opt/lyuxingh/code/tax3d_realworld -B /scratch/lyuxingh/data:/opt/lyuxingh/data -B /scratch/lyuxingh/logs:/opt/lyuxingh/logs /scratch/lyuxingh/singularity/tax3d_realworld.sif bash -c "cd /opt/lyuxingh/code/tax3d_realworld/scripts && CUDA_VISIBLE_DEVICES=0 ./train_realworld.sh 0 tax3dv2 online insertion model.point_encoder=pn2 model.one_hot_recon=True model.diff_rotation_noise_scale=45 dataset.connector_type=0503-wp-1 model.pcd_scale=1.0 dataset.pcd_scale=1.0 dataset.preprocess=True dataset.sample_size_action=2048 dataset.sample_size_anchor=2048 resources.num_workers=16 dataset.data_dir=/opt/lyuxingh/data/insertion/demonstrations/ log_dir=/opt/lyuxingh/logs"
#singularity exec --nv -B /home/lyuxingh/code/tax3d_realworld:/opt/lyuxingh/code/tax3d_realworld -B /scratch/lyuxingh/data:/opt/lyuxingh/data -B /scratch/lyuxingh/logs:/opt/lyuxingh/logs /scratch/lyuxingh/singularity/tax3d_realworld.sif bash -c "cd /opt/lyuxingh/code/tax3d_realworld/scripts && CUDA_VISIBLE_DEVICES=0 ./train_realworld.sh 0 cross_flow online insertion dataset.connector_type=0428-wp-3 dataset.preprocess=True dataset.sample_size_action=2048 dataset.sample_size_anchor=2048 resources.num_workers=16 dataset.data_dir=/opt/lyuxingh/data/insertion/demonstrations/ log_dir=/opt/lyuxingh/logs"

#------ Resuming from a crashed run (u want to keep the run id, and log to the original entry)------#
# ./train.sh 0 tax3dv2 online insertion model.frame_type=fixed model.pred_frame=anchor_center model.point_encoder=pn2 model.diff_rotation_noise_scale=45 dataset.connector_type=0428-wp-2 resources.num_workers=16 checkpoint.run_id=k8iy8vfo checkpoint.local_ckpt=/home/lyuxing/Desktop/tax3d_realworld/checkpoints/annia0md.ckpt

#------ Finetune (we dont keep the run id, and log to a new entry)------#
#./train.sh 0 tax3dv2 online insertion model.frame_type=fixed model.pred_frame=anchor_center model.point_encoder=pn2 model.diff_rotation_noise_scale=45 dataset.connector_type=0428-wp-2 resources.num_workers=16 checkpoint.local_ckpt=/home/lyuxing/Desktop/tax3d_realworld/checkpoints/annia0md.ckpt dataset.action_rotation_variance=1.04 dataset.anchor_rotation_variance=0.262 training.epochs=22000

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
  wandb.group=insertion \
  wandb.project=corl2025_tax3dv2 \
  resources.gpus=[${GPU_INDEX}] \
  $COMMAND


