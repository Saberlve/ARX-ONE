#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
EXPORT_PYTHON="${REPO_ROOT}/.venv_lerobot/bin/python"
EXPORT_SCRIPT="${REPO_ROOT}/src/edlsrobot/datasets/export_lerobot_episode_to_excel.py"

LEROBOT_EPISODE_PATH="${LEROBOT_EPISODE_PATH:-${REPO_ROOT}/All_datas/pour_tea100/data/chunk-000/episode_000009.parquet}"
LEROBOT_EXPORT_OUTPUT="${LEROBOT_EXPORT_OUTPUT:-${REPO_ROOT}/outputs/episode_000009.xlsx}"
LEROBOT_IMAGE_WIDTH="${LEROBOT_IMAGE_WIDTH:-160}"

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        exit 1
    fi
}

require_file "${EXPORT_PYTHON}"
require_file "${EXPORT_SCRIPT}"
require_file "${LEROBOT_EPISODE_PATH}"

COMMAND=(
    "${EXPORT_PYTHON}"
    "${EXPORT_SCRIPT}"
    "${LEROBOT_EPISODE_PATH}"
    -o "${LEROBOT_EXPORT_OUTPUT}"
    --image-width "${LEROBOT_IMAGE_WIDTH}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf "[DRY_RUN]"
    printf " %q" "${COMMAND[@]}"
    printf "\n"
    exit 0
fi

cd "${REPO_ROOT}"
PYTHONPATH=src "${COMMAND[@]}"
