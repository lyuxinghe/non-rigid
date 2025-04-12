#!/bin/bash

# Really simple helper script to elevate policy training code to primary TAX3D repo.
# This should take in 5 arguments:
# 1. the algorithm name
# 2. the task name
# 3. the additional information (e.g. checkpoint name)
# 4. the seed
# 5. the index of which GPU to use
# 6. DEDO task type tag

# Example usage:
# ./train_policy.sh dp3 adroit_hammer 0322 0 0
# ./train_policy.sh dp3 dedo_proccloth [GOAL_TYPE] 1 [GPU_ID]

alg_name=${1}
task_name=${2}
addition_info=${3}
seed=${4}
gpu_id=${5}
dedo_type=${6}

# Setting up goal-conditioning.
if [[ $addition_info == "none" ]]; then
    overrides="policy.goal_conditioning=none "
elif [[ $addition_info == "gt" ]]; then
    overrides="policy.goal_conditioning=gt_pcd "
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

overrides+="task.dataset.cloth_geometry=multi task.dataset.cloth_pose=random "
overrides+="task.dataset.anchor_geometry=single task.dataset.anchor_pose=random "
overrides+="task.dataset.num_anchors=2 task.dataset.hole=${hole} "
overrides+="task.dataset.robot=True "

# Setting up logging.
overrides+="logging.entity=r-pad logging.project=corl2025_tax3dv2 logging.group=dp3_dedo "

# Concatenate addition_info and dedo_type for more informative run name.
addition_info="${addition_info}_${dedo_type}"

cd ../third_party/3D-Diffusion-Policy
bash scripts/train_policy.sh $alg_name $task_name $addition_info $seed $gpu_id $overrides