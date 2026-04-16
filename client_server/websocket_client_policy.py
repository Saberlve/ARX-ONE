import json
import logging
import time
from typing import Dict, Optional, Tuple
import pathlib
import polars as pl

from typing_extensions import override
import websockets.sync.client

# from openpi_client import base_policy as _base_policy
from src.policy.DP.client_server import msgpack_numpy

logger = logging.getLogger(__name__)

class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement for the given key."""
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)


    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)

class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data ={
            "data": obs,
            "type": "inference"
        }
        data = self._packer.pack(data)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset_model(self) -> None:
        data = {
            "type": "reset_model"
        }
        data = self._packer.pack(data)
        self._ws.send(data)
        # response = self._ws.recv()


    def update_server_obs(self, obs):
        data ={
            "data": obs,
            "type": "update_obs"
        }
        data = self._packer.pack(data)
        self._ws.send(data)
        response = self._ws.recv()
        # if isinstance(response, str):
        #     # we're expecting bytes; if the server sends a string, it's an error.
        #     raise RuntimeError(f"Error in inference server:\n{response}")
        # return msgpack_numpy.unpackb(response)
