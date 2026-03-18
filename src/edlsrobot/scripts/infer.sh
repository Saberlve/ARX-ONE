#!/bin/bash

# export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH

export HF_ENDPOINT=https://hf-mirror.com

## pi05
# python inference.py \
#     --robot.type=ACone \
#     --robot.id=my_ACone_arm \
#     --robot.frame_rate=20 \
#     --robot.cameras='{"head":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":480,"fps":30},"left_wrist":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":480,"fps":30},"right_wrist":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":480,"fps":30}}' \
#     --display_data=false \
#     --dataset.repo_id=eval/eval_ACone_pi05 \
#     --dataset.single_task="Pick up the teapot and pour one-third of the tea into the glass." \
#     --policy.path=/home/ubuntu/edlsrobot/repos/ROS2_AC-one_Play/models/train/pour_tea/checkpoints/100000/pretrained_model

##smolvla
# python inference.py \
#     --robot.type=ACone \
#     --robot.id=my_ACone_arm \
#     --robot.frame_rate=25 \
#     --robot.cameras='{"camera1":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":480,"fps":30},"camera2":{"type":"opencv","index_or_path":"/dev/video11","width":640,"height":480,"fps":30},"camera3":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":480,"fps":30}}' \
#     --display_data=false \
#     --dataset.repo_id=eval/eval_ACone_smolvla \
#     --dataset.single_task="Pick up the teapot and pour one-third of the tea into the glass." \
#     --policy.path=/home/ubuntu/edlsrobot/repos/ROS2_AC-one_Play/models/train/pour_tea_smol/checkpoints/100000/pretrained_model


## act
python inference.py \
    --robot.type=ACone \
    --robot.id=my_ACone_arm \
    --robot.frame_rate=25 \
    --robot.cameras='{"head":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":480,"fps":30},"left_wrist":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":480,"fps":30},"right_wrist":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":480,"fps":30}}' \
    --display_data=false \
    --dataset.repo_id=eval/eval_ACoone_act \
    --dataset.single_task="Pick up the teapot and pour one-third of the tea into the glass." \
    --policy.path=/home/ubuntu/edlsrobot/repos/ROS2_AC-one_Play/models/train/pour_tea_act/checkpoints/100000/pretrained_model