#!/bin/bash

# Really simple helper script to elevate policy evaluation code to primary TAX3D repo.
# This should take in 5 arguments:
# 1. the algorithm name
# 2. the task name
# 3. the additional information (e.g. checkpoint name)
# 4. the seed
# 5. the index of which GPU to use
# 6. DEDO task type tag

# Example usage:
# ./eval_policy.sh dp3 adroit_hammer 0322 0 0
# ./eval_dedo_policy.sh dp3 dedo_proccloth none 1 [GPU_ID] easy

alg_name=${1}
task_name=${2}
addition_info=${3}
seed=${4}
gpu_id=${5}
dedo_type=${6}
data_size=${7}
goal_model=${8}

# Setting up goal-conditioning.
if [[ $addition_info == "none" ]]; then
    overrides="policy.goal_conditioning=none "
elif [[ $addition_info == "gt" ]]; then
    overrides="policy.goal_conditioning=gt_pcd "
elif [[ $addition_info == "none_seg" ]]; then
    overrides="policy.goal_conditioning=none "
    overrides+="policy.pointcloud_encoder_cfg.one_hot=True policy.pointcloud_encoder_cfg.in_channels=5 "
elif [[ $addition_info == "gt_seg" ]]; then
    overrides="policy.goal_conditioning=gt_pcd "
    overrides+="policy.pointcloud_encoder_cfg.one_hot=True policy.pointcloud_encoder_cfg.in_channels=6 "
else
    echo "Invalid goal conditioning type. Exiting."
    exit 1
fi

# Setting task-specific dataset overrides.
if [[ $dedo_type == "easy" ]]; then
    hole="single"
elif [[ $dedo_type == "hard" ]]; then
    hole="double"
fi

if [[ $data_size == "small" ]] || [[ $data_size == "medium" ]] || [[ $data_size == "big" ]]; then
    overrides+="task.dataset.size=${data_size} "
else
    echo "Invalid data size. Exiting."
    exit 1
fi

overrides+="task.dataset.cloth_geometry=multi task.dataset.cloth_pose=random "
overrides+="task.dataset.anchor_geometry=single task.dataset.anchor_pose=random "
overrides+="task.dataset.num_anchors=2 task.dataset.hole=${hole} "
overrides+="task.dataset.robot=True "

# Concatenate addition_info and dedo_type for more informative run name.
addition_info="${addition_info}_${dedo_type}_${data_size}"

# If goal model is not an empty string, add model id to overrides.
if [[ -n $goal_model ]]; then
    overrides+="task.env_runner.goal_conditioning=tax3d_pcd "
    overrides+="task.env_runner.goal_model=${goal_model} "
fi

# Kind of hacky for now - if evaluating TAX3D, set addition_info to checkpoint name.
if [[ $alg_name == "tax3d" ]]; then
    addition_info=$goal_model
fi

cd ../third_party/3D-Diffusion-Policy
bash scripts/eval_policy.sh $alg_name $task_name $addition_info $seed $gpu_id $overrides