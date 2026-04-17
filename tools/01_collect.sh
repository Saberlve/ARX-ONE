#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
ARX_ROOT=$(cd "${REPO_ROOT}/../ARX_X5" && pwd)
CAN_ROOT="${ARX_ROOT}/ARX_CAN/arx_can"
ROS2_ROOT="${ARX_ROOT}/ROS2/X5_ws"
JOY_ROOT="${ARX_ROOT}/arx_joy"
REALSENSE_ROOT="${REPO_ROOT}/realsense"
COLLECTOR_ROOT="${REPO_ROOT}/src/edlsrobot/datasets"
COLLECT_PYTHON="${REPO_ROOT}/.venv_lerobot/bin/python"

# LeRobot v2.1 collect settings. Edit these values directly when needed.
LEROBOT_ROOT="${REPO_ROOT}/All_datas"
LEROBOT_REPO_ID="pickXtimes_v21"
LEROBOT_EPISODE_NUMS="2"
LEROBOT_MAX_TIMESTEPS="1000"
LEROBOT_FRAME_RATE="30"
LEROBOT_TASK="Pick up the black pouch three times, then touch the green grommet"

shell_type=${SHELL##*/}
if [[ -z "${shell_type}" ]]; then
    shell_type="bash"
fi

require_dir() {
    local path="$1"
    if [[ ! -d "${path}" ]]; then
        echo "Missing required directory: ${path}" >&2
        exit 1
    fi
}

launch_terminal() {
    local title="$1"
    local command="$2"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[DRY_RUN] ${title}: ${command}"
        return 0
    fi

    if [[ -z "${DISPLAY:-}" ]]; then
        echo "DISPLAY is not set. Run this script inside a graphical desktop terminal, or use DRY_RUN=1 to verify commands." >&2
        exit 1
    fi

    if ! xdpyinfo >/dev/null 2>&1; then
        echo "DISPLAY=${DISPLAY} is not reachable. Log in to the desktop locally/remotely and run this script there, or use DRY_RUN=1 to verify commands." >&2
        exit 1
    fi

    gnome-terminal --title="${title}" -- "${shell_type}" -lc "${command}; exec ${shell_type} -i"
}

require_dir "${CAN_ROOT}"
require_dir "${ROS2_ROOT}"
require_dir "${JOY_ROOT}"
require_dir "${REALSENSE_ROOT}"
require_dir "${COLLECTOR_ROOT}"
if [[ ! -x "${COLLECT_PYTHON}" ]]; then
    echo "Missing collector python: ${COLLECT_PYTHON}" >&2
    exit 1
fi

# CAN
launch_terminal "can1" "cd '${CAN_ROOT}' && ./arx_can1.sh"
sleep 0.3
launch_terminal "can3" "cd '${CAN_ROOT}' && ./arx_can3.sh"
sleep 0.3
launch_terminal "can6" "cd '${CAN_ROOT}' && ./arx_can6.sh"
sleep 0.3

# Ac_one
launch_terminal "lift" "cd '${ROS2_ROOT}' && source install/setup.bash && ros2 launch arx_x5_controller v2_collect.launch.py"
sleep 0.3
launch_terminal "joy" "cd '${JOY_ROOT}' && source install/setup.bash && ros2 run arx_joy arx_joy"
sleep 1

# Realsense
launch_terminal "realsense" "cd '${REALSENSE_ROOT}' && ./realsense.sh"
sleep 3

# Collect directly to LeRobot v2.1 format.
launch_terminal "collect_lerobot_v21" "cd '${REPO_ROOT}' && source /opt/ros/humble/setup.bash && source '${ROS2_ROOT}/install/setup.bash' && '${COLLECT_PYTHON}' src/edlsrobot/datasets/collect_ledatav21.py --root_path '${LEROBOT_ROOT}' --repo_id '${LEROBOT_REPO_ID}' --episode_nums '${LEROBOT_EPISODE_NUMS}' --max_timesteps '${LEROBOT_MAX_TIMESTEPS}' --frame_rate '${LEROBOT_FRAME_RATE}' --task '${LEROBOT_TASK}'"

 
