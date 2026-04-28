import pathlib

import numpy as np


class ActionSafetyError(RuntimeError):
    pass


def describe_debug_payload(payload, *, max_vector_values=14):
    if isinstance(payload, dict):
        return {key: describe_debug_payload(value, max_vector_values=max_vector_values) for key, value in payload.items()}
    if isinstance(payload, bytes):
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            return {"type": "bytes", "length": len(payload)}
    if isinstance(payload, str):
        return payload

    try:
        array = np.asarray(payload)
    except Exception:
        return repr(payload)

    if array.dtype == object:
        return repr(payload)

    summary = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }
    if array.size == 0:
        return summary
    if np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        if array.ndim <= 1 and array.size <= max_vector_values:
            summary["values"] = array.tolist()
        elif array.ndim <= 1:
            head_count = max_vector_values // 2
            tail_count = max_vector_values - head_count
            summary["values"] = array[:head_count].tolist() + ["..."] + array[-tail_count:].tolist()
        else:
            summary["min"] = float(np.min(array))
            summary["max"] = float(np.max(array))
    return summary


def save_debug_observation_images(payload: dict, debug_dir: str, *, prefix: str, step: int) -> list[pathlib.Path]:
    import cv2

    output_dir = pathlib.Path(debug_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    image_keys = [
        "observation/images/head",
        "observation/images/left_wrist",
        "observation/images/right_wrist",
    ]
    for key in image_keys:
        if key not in payload:
            continue
        image = np.asarray(payload[key])
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.moveaxis(image, 0, -1)
        if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
            continue
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[-1] == 3:
            image_to_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.shape[-1] == 4:
            image_to_write = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        else:
            image_to_write = image.squeeze(-1)

        camera = key.rsplit("/", 1)[-1]
        path = output_dir / f"{prefix}_step_{step:06d}_{camera}.jpg"
        cv2.imwrite(str(path), image_to_write)
        saved_paths.append(path)
    return saved_paths


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
        max_joint_index = int(np.arange(14)[joint_mask][np.argmax(delta[joint_mask])])
        raise ActionSafetyError(
            "ARX action joint step "
            f"{max_observed_joint_step:.4f} exceeds limit {max_joint_step:.4f} "
            f"at dim {max_joint_index}: current={current_qpos[max_joint_index]:.4f}, "
            f"target={arm_action[max_joint_index]:.4f}"
        )

    max_observed_gripper_step = float(np.max(delta[[6, 13]]))
    if max_observed_gripper_step > max_gripper_step:
        max_gripper_index = int(np.asarray([6, 13])[np.argmax(delta[[6, 13]])])
        raise ActionSafetyError(
            "ARX action gripper step "
            f"{max_observed_gripper_step:.4f} exceeds limit {max_gripper_step:.4f} "
            f"at dim {max_gripper_index}: current={current_qpos[max_gripper_index]:.4f}, "
            f"target={arm_action[max_gripper_index]:.4f}"
        )

    return action
