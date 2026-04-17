# -- coding: UTF-8
import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
ROOT1 = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    sys.path.append(str(ROOT1))
    os.chdir(str(ROOT))

import yaml
import h5py
import argparse
import signal

import rclpy

import threading
import time
import numpy as np

np.set_printoptions(linewidth=200)

from functools import partial

from act.utils.ros_operator import Rate, RosOperator
from act.utils.setup_loader import setup_loader
from src.edlsrobot.datasets.lerobot_v21.lerobot_dataset import LeRobotDataset


def load_yaml(yaml_file):
    try:
        with open(yaml_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File not found - {yaml_file}")

        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file - {e}")

        return None

def load_ledata(args):
    try:
        print(f"Loading data from {args.root_path}")
        dataset = LeRobotDataset(args.repo_id, root=args.root_path, episodes=[args.episode])
        qpos = dataset.hf_dataset.select_columns("observation.state")
        qvel = dataset.hf_dataset.select_columns("observation.velocity")
        actions = dataset.hf_dataset.select_columns("action")
    except Exception as e:
        raise RuntimeError(f"Error occured while loading the LerobotDataset file: {e}")
    return qpos, qvel, actions


def robot_action(ros_operator, args, action, action_base, actions_velocity):
    gripper_idx = [6, 13]

    left_action = action[:gripper_idx[0] + 1]  # 取8维度
    right_action = action[gripper_idx[0] + 1 : gripper_idx[1] + 1]  # action[7:14]

    print(f'{left_action=}')

    ros_operator.follow_arm_publish(left_action, right_action)  # follow_arm_publish_continuous_thread

    if args.use_base:
        ros_operator.set_robot_base_target(np.concatenate([action_base, actions_velocity]))


def init_robot(ros_operator, use_base, rate):
    # init0 = [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, 0.0]
    # init1 = [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, 0.0]
    ## second pos
    init0 = [0.0, 0.650, 0.700, -0.900, 0.0, 0.0, 0.0]    #go_home_position
    init1 = [0.0, 0.650, 0.700, -0.900, 0.0, 0.0, 0.0]

    ## 修改了里面的 follow_arm_publish_continuous 和 _update_arm_position， 缓慢移动到目标
    ros_operator.follow_arm_publish_continuous(init0, init1)
    ros_operator.robot_base_shutdown()

    if use_base:
        input("Enter any key to continue :")

        ros_operator.start_base_control_thread()
        ros_operator.follow_arm_publish_continuous(init1, init1)


def signal_handler(signal, frame, ros_operator):
    print('Caught Ctrl+C / SIGINT signal')

    # 底盘给零
    ros_operator.robot_base_shutdown()
    ros_operator.base_control_thread.join()

    sys.exit(0)


def main(args):
    setup_loader(ROOT)

    rclpy.init()

    config = load_yaml(args.data)
    ros_operator = RosOperator(args, config, in_collect=False)

    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_operator,), daemon=True)
    spin_thread.start()

    signal.signal(signal.SIGINT, partial(signal_handler, ros_operator=ros_operator))

    # qpoes, eefs, actions, actions_eefs, action_base, actions_velocity = load_hdf5(args.episode_path)
    qpos, qvel, actions = load_ledata(args)
    action_base = []

    rate = Rate(args.frame_rate)
    # rate = Rate(20)
    init_robot(ros_operator, args.use_base, rate)

    if args.states_replay:
        replay_actions = actions
    else:
        replay_actions = qpos

    # rate = Rate(25)
    for idx in range(len(replay_actions)):
        replay_action = replay_actions[idx]["observation.state"].numpy()
        print(f'{replay_action=}')
        robot_action(ros_operator, args, replay_action, action_base, idx)
        rate.sleep()

    ros_operator.base_enable = False

    ros_operator.destroy_node()
    rclpy.shutdown()
    spin_thread.join()


def parse_args(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_path', type=str, default=Path.joinpath(ROOT, './All_datas/test'), help='dataset dir')
    parser.add_argument('--repo_id', type=str, default='', help='dataset dir name')
    parser.add_argument('--episode', type=int, default=0, help='episode in datasets')

    parser.add_argument('--frame_rate', type=int, default=25,help='frame rate')
    parser.add_argument('--data', type=str, default=Path.joinpath(ROOT, 'act/data/config.yaml'), help='config file')

    parser.add_argument('--use_base', action='store_true', help='use base')
    parser.add_argument('--record', choices=['Distance', 'Speed'], default='Distance',
                        help='record data')

    parser.add_argument('--states_replay', action='store_true', help='use qpos replay')

    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')
    parser.add_argument('--is_compress', action='store_true', help='compress image')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)


""" ziqi 3.18 add:
python ./src/edlsrobot/datasets/replay.py --root_path './All_datas/test_old'
"""