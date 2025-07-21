#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. the rest of the arguments for the train.py script

########## RPDiff ##########
### Mug_Rack_Easy_Single ###
# ./train.sh 0 tax3dv2 online rpdiff model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.object_scale=3.0 dataset.rpdiff_task_name=mug_rack_easy_single dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
### Mug_Rack_Med_Single ###
# ./train.sh 0 tax3dv2 online rpdiff model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.object_scale=3.0 dataset.rpdiff_task_name=mug_rack_med_single dataset.train_dataset_size=1600 dataset.val_dataset_size=200 dataset.test_dataset_size=200 training.batch_size=16 training.val_batch_size=16 resources.num_workers=32
### Mug_Rack_Med_Multi ###
# ./train.sh 0 tax3dv2 online rpdiff model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.object_scale=3.0 dataset.rpdiff_task_name=mug_rack_med_multi dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.sample_size_anchor=1024 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
### Mug_Rack_Hard_Multi ###
# ./train.sh 0 tax3dv2 online rpdiff model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.object_scale=3.0 dataset.rpdiff_task_name=mug_on_rack_multi_large_proc_gen_demos dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.sample_size_anchor=1024 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
### Can_Cabinet ###
# ./train.sh 0 tax3dv2 online rpdiff model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.object_scale=3.0 dataset.rpdiff_task_name=can_in_cabinet_stack dataset.rpdiff_task_type=task_name_stack_can_in_cabinet dataset.sample_size_anchor=1024 dataset.sample_size_action=256 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
### Book_Bookshelf ###
# ./train.sh 0 tax3dv2 online rpdiff model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.object_scale=3.0 dataset.rpdiff_task_name=book_on_bookshelf_double_view_rnd_ori dataset.rpdiff_task_type=task_name_book_in_bookshelf dataset.sample_size_anchor=1024 dataset.train_dataset_size=1600 dataset.val_dataset_size=200 dataset.test_dataset_size=200 training.batch_size=16 training.val_batch_size=16 resources.num_workers=32

########## Real World ##########
# ./train.sh 0 tax3dv2 online insertion model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.object_scale=3.0 model.point_encoder=pn2 model.diff_rotation_noise_scale=45 model.one_hot_recon=True dataset.connector_type=0629-wp-mixed resources.num_workers=32

#------ Run On AutoBot with Singularity ------#
# singularity exec --nv -B /home/lyuxingh/code/tax3dv2:/opt/lyuxingh/code/tax3dv2 -B /scratch/lyuxingh/data:/opt/lyuxingh/data -B /scratch/lyuxingh/logs:/opt/lyuxingh/logs /scratch/lyuxingh/singularity/tax3dv2_lyuxing.sif bash -c "cd /opt/lyuxingh/code/tax3dv2/scripts && CUDA_VISIBLE_DEVICES=？ ./train.sh <YOUR TRAIN COMMAND HERE> dataset.data_dir=/opt/lyuxingh/data/rpdiff/data/task_demos/ log_dir=/opt/lyuxingh/logs"

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


