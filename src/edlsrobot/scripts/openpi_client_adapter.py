import numpy as np


class ActionSafetyError(RuntimeError):
    pass


def build_openpi_arx_observation(obs_dict: dict, prompt: str) -> dict:
    images = obs_dict["images"]
    return {
        "observation/images/head": images["head"],
        "observation/images/left_wrist": images["left_wrist"],
        "observation/images/right_wrist": images["right_wrist"],
        "observation/state": obs_dict["qpos"],
        "prompt": prompt,
    }


def select_first_openpi_action(result: dict) -> np.ndarray:
    actions = np.asarray(result["actions"])
    if len(actions) == 0:
        raise ValueError("OpenPI returned an empty action chunk")
    return actions[0]


def check_arx_action_safety(
    action: np.ndarray,
    current_qpos: np.ndarray,
    *,
    max_joint_step: float,
    max_gripper_step: float,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    current_qpos = np.asarray(current_qpos, dtype=np.float32)
    if action.ndim != 1 or action.shape[0] < 14:
        raise ActionSafetyError(f"ARX action must be a 1D vector with at least 14 values, got {action.shape}")
    if current_qpos.shape != (14,):
        raise ActionSafetyError(f"ARX qpos must have shape (14,), got {current_qpos.shape}")
    if not np.all(np.isfinite(action[:14])):
        raise ActionSafetyError("ARX action must contain only finite values")
    if not np.all(np.isfinite(current_qpos)):
        raise ActionSafetyError("ARX qpos must contain only finite values")

    arm_action = action[:14]
    delta = np.abs(arm_action - current_qpos)
    joint_mask = np.ones(14, dtype=bool)
    joint_mask[[6, 13]] = False
    max_observed_joint_step = float(np.max(delta[joint_mask]))
    if max_observed_joint_step > max_joint_step:
        raise ActionSafetyError(
            f"ARX action joint step {max_observed_joint_step:.4f} exceeds limit {max_joint_step:.4f}"
        )

    max_observed_gripper_step = float(np.max(delta[[6, 13]]))
    if max_observed_gripper_step > max_gripper_step:
        raise ActionSafetyError(
            f"ARX action gripper step {max_observed_gripper_step:.4f} exceeds limit {max_gripper_step:.4f}"
        )

    return action
