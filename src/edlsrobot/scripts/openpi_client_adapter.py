import numpy as np


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
