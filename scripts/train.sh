#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. the rest of the arguments for the train.py script

########## RPDiff ##########
### Mug_Rack_Easy_Single ###
# Anchor Centroid
# ./train.sh 0 cross_point_relative rpdiff online model.pred_ref_frame=anchor model.ref_error_scale=0.0 dataset.rpdiff_task_name=mug_rack_easy_single dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
# ./train.sh 0 cross_flow_relative rpdiff online model.pred_ref_frame=anchor model.ref_error_scale=0.0 dataset.rpdiff_task_name=mug_rack_easy_single dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32

# Noisy Goal Centroid
# TAX3D Models
# ./train.sh 0 cross_point_relative rpdiff online model.pred_ref_frame=noisy_goal model.ref_error_scale=1.0 resources.num_workers=32
# ./train.sh 0 cross_flow_relative rpdiff online model.pred_ref_frame=noisy_goal model.ref_error_scale=1.0 resources.num_workers=32
# TAX3Dv2 Models
# ./train.sh 0 tax3dv2_muframe rpdiff online model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 dataset.rpdiff_task_name=mug_rack_easy_single dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32

### Mug_Rack_Med_Single ###
# Anchor Centroid
# ./train.sh 0 cross_point_relative rpdiff online model.pred_ref_frame=anchor model.ref_error_scale=0.0 dataset.rpdiff_task_name=mug_rack_med_single dataset.train_dataset_size=1600 dataset.val_dataset_size=200 dataset.test_dataset_size=200 training.batch_size=16 training.val_batch_size=16 resources.num_workers=32
# ./train.sh 0 cross_flow_relative rpdiff online model.pred_ref_frame=anchor model.ref_error_scale=0.0 dataset.rpdiff_task_name=mug_rack_med_single dataset.train_dataset_size=1600 dataset.val_dataset_size=200 dataset.test_dataset_size=200 training.batch_size=16 training.val_batch_size=16 resources.num_workers=32

### Mug_Rack_Med_Multi ###
# Anchor Centroid
# ./train.sh 0 cross_flow_relative rpdiff online model.pred_ref_frame=anchor model.ref_error_scale=0.0 dataset.rpdiff_task_name=mug_on_rack_multi_large_proc_gen_demos dataset.rpdiff_task_type=task_name_mug_on_rack_multi resources.num_workers=32 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32

# Noisy Goal Centroid
# TAX3Dv2 Models
# ./train.sh 0 tax3dv2_muframe rpdiff online model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 dataset.rpdiff_task_name=mug_on_rack_multi_large_proc_gen_demos dataset.rpdiff_task_type=task_name_mug_on_rack_multi resources.num_workers=32 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32


#------ Run On AutoBot with Singularity ------#
# singularity exec --nv -B /home/lyuxingh/code/tax3d_upgrade:/opt/lyuxingh/code/tax3d_upgrade -B /scratch/lyuxingh/data:/opt/lyuxingh/data -B /scratch/lyuxingh/logs:/opt/lyuxingh/logs /scratch/lyuxingh/singularity/tax3d_lyuxing.sif bash -c "cd /opt/lyuxingh/code/tax3d_upgrade/scripts && CUDA_VISIBLE_DEVICES=3 ./train.sh 0 cross_flow_relative rpdiff online model.pred_ref_frame=anchor model.ref_error_scale=0.0 dataset.rpdiff_task_name=mug_rack_med_single dataset.train_dataset_size=1600 dataset.val_dataset_size=200 dataset.test_dataset_size=200 training.batch_size=16 resources.num_workers=32 dataset.data_dir=/opt/lyuxingh/data/rpdiff/data/task_demos/"

#------ Resuming from a crashed run ------#
#./train.sh 0 cross_point_relative rpdiff False online checkpoint.run_id=k8iy8vfo checkpoint.local_ckpt='/home/lyuxing/Desktop/tax3d_upgrade/scripts/logs/train_rpdiff_feature_df_cross/2025-03-02/17-41-55/checkpoints/last.ckpt'

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

elif [ $MODEL_TYPE == "tax3dv2_muframe" ]; then
  echo "Training TAX3Dv2 Mu-Frame point model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=tax3dv2_muframe model.type=point"
  DATASET_PARAMS="dataset=$DATASET_NAME dataset.type=point"

fi



WANDB_MODE=$WANDB_MODE python train.py \
  $MODEL_PARAMS \
  $DATASET_PARAMS \
  wandb.group=rigid \
  wandb.project=corl2025_tax3dv2 \
  resources.gpus=[${GPU_INDEX}] \
  $COMMAND


