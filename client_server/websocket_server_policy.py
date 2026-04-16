import asyncio
import http
import logging
import time
import traceback
from typing import Union
import torch
import numpy

import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import ACTION, OBS_STR

from lerobot.processor.factory import make_default_robot_observation_processor, make_default_robot_action_processor
from lerobot.datasets.utils import build_dataset_frame

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        config,
        dataset,
        robot,
        policy,
        preprocessor,
        postprocessor,
        host: str = "0.0.0.0",
        port: Union[int, None] = None,
        metadata: Union[dict, None] = None,
    ) -> None:
        self._config = config
        self._dataset = dataset
        self._robot = robot
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        robot_observation_processor = make_default_robot_observation_processor()
        robot_action_processor = make_default_robot_action_processor()
        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                obs_processed = robot_observation_processor(obs)
                infer_time = time.monotonic()
                action_values = predict_action(
                    observation=obs_processed,
                    policy=self._policy,
                    device=get_safe_torch_device(self._policy.config.device),
                    preprocessor=self._preprocessor,
                    postprocessor=self._postprocessor,
                    use_amp=self._policy.config.use_amp,
                    task=self._config.dataset.single_task,
                    robot_type=self._robot.robot_type,
                )
                infer_time = time.monotonic() - infer_time
                action_values = action_values.squeeze(0).to("cpu")
                action_values: RobotAction = {"action": action_values.numpy()}
                robot_action_to_send = robot_action_processor((action_values, obs))

                robot_action_to_send["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    robot_action_to_send["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(robot_action_to_send))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> Union[_server.Response, None]:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
