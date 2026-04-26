#!/bin/bash
set -euo pipefail

export LD_LIBRARY_PATH=/opt/ros/humble/lib:${LD_LIBRARY_PATH:-}

OPENPI_HOST=${OPENPI_HOST:-localhost}
OPENPI_PORT=${OPENPI_PORT:-8000}
ROBOT_TYPE=${ROBOT_TYPE:-ACone}
ROBOT_ID=${ROBOT_ID:-my_ACone_arm}
FRAME_RATE=${FRAME_RATE:-20}
DATASET_REPO_ID=${DATASET_REPO_ID:-eval/eval_ACone_openpi_pi05}
TASK_PROMPT=${TASK_PROMPT:-"Pick up the teapot and pour one-third of the tea into the glass."}

HEAD_CAMERA=${HEAD_CAMERA:-/dev/video10}
LEFT_WRIST_CAMERA=${LEFT_WRIST_CAMERA:-/dev/video11}
RIGHT_WRIST_CAMERA=${RIGHT_WRIST_CAMERA:-/dev/video12}

python inference.py \
    --robot.type="${ROBOT_TYPE}" \
    --robot.id="${ROBOT_ID}" \
    --robot.frame_rate="${FRAME_RATE}" \
    --robot.cameras="{\"head\":{\"type\":\"opencv\",\"index_or_path\":\"${HEAD_CAMERA}\",\"width\":640,\"height\":480,\"fps\":30},\"left_wrist\":{\"type\":\"opencv\",\"index_or_path\":\"${LEFT_WRIST_CAMERA}\",\"width\":640,\"height\":480,\"fps\":30},\"right_wrist\":{\"type\":\"opencv\",\"index_or_path\":\"${RIGHT_WRIST_CAMERA}\",\"width\":640,\"height\":480,\"fps\":30}}" \
    --display_data=false \
    --use_openpi_server=true \
    --openpi_server_host="${OPENPI_HOST}" \
    --openpi_server_port="${OPENPI_PORT}" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.single_task="${TASK_PROMPT}"
