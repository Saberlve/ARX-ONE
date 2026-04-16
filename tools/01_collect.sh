#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
ARX_ROOT=$(cd "${REPO_ROOT}/../ARX_X5" && pwd)
CAN_ROOT="${ARX_ROOT}/ARX_CAN/arx_can"
ROS2_ROOT="${ARX_ROOT}/ROS2/X5_ws"
JOY_ROOT="${ARX_ROOT}/arx_joy"
REALSENSE_ROOT="${REPO_ROOT}/realsense"

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


# Collect
# gnome-terminal --title="collect" -x $shell_type -i -c "cd ${workspace}; cd ../act; source /opt/ros/humble/setup.bash; source /home/ubuntu/edlsrobot/repos/ARX_X5/ROS2/X5_ws/install/setup.bash; source ./.venv_ros/bin/activate; python collect.py --episode_idx -1; $shell_exec"

 
