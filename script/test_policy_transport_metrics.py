import json
import socket
import sys
import threading
import types
import unittest

import numpy as np

envs = types.ModuleType("envs")
envs.CONFIGS_PATH = ""
sys.modules.setdefault("envs", envs)
create_actor = types.ModuleType("envs.utils.create_actor")
create_actor.UnStableError = RuntimeError
sys.modules.setdefault("envs.utils", types.ModuleType("envs.utils"))
sys.modules.setdefault("envs.utils.create_actor", create_actor)
sys.modules.setdefault(
    "generate_episode_instructions", types.ModuleType("generate_episode_instructions")
)
global_configs = types.ModuleType("envs._GLOBAL_CONFIGS")
global_configs.CONFIGS_PATH = ""
sys.modules.setdefault("envs._GLOBAL_CONFIGS", global_configs)

from script.eval_policy_client import ModelClient, numpy_to_json
from script.policy_model_server import ModelServer, numpy_schema


class _Socket:
    def __init__(self, response):
        self.sent = bytearray()
        self.response = bytearray(response)

    def sendall(self, payload):
        self.sent.extend(payload)

    def recv(self, size):
        payload = self.response[:size]
        del self.response[:size]
        return bytes(payload)

    def close(self):
        pass


class PolicyTransportMetricsTest(unittest.TestCase):
    def test_client_reports_actual_wire_bytes_and_round_trip(self):
        response = numpy_to_json({"res": {"ok": True}}).encode("utf-8")
        client = object.__new__(ModelClient)
        client.sock = _Socket(len(response).to_bytes(4, "big") + response)
        client.last_rpc_metrics = {}
        client.decode_errors = 0
        client.connection_resets = 0

        result = client._send_recv(
            {"cmd": "step", "obs": np.zeros((2, 3), dtype=np.float32)}
        )

        self.assertEqual(result, {"res": {"ok": True}})
        self.assertEqual(client.last_rpc_metrics["response_wire_bytes"], len(response))
        self.assertEqual(
            client.last_rpc_metrics["request_wire_bytes"],
            int.from_bytes(client.sock.sent[:4], "big"),
        )
        self.assertGreaterEqual(client.last_rpc_metrics["socket_round_trip_s"], 0.0)

    def test_server_schema_reports_paths_shapes_and_dtypes(self):
        schema = numpy_schema(
            {"obs": {"depth": np.zeros((240, 320), dtype=np.float32)}}
        )
        self.assertEqual(
            schema,
            [{"path": "obs.depth", "dtype": "float32", "shape": [240, 320]}],
        )
        json.dumps(schema)

    def test_client_server_round_trip_preserves_numpy_payload(self):
        class EchoModel:
            @staticmethod
            def echo(payload):
                return payload

        server_socket, client_socket = socket.socketpair()
        server = ModelServer(EchoModel())
        server.running = True
        thread = threading.Thread(target=server._handle_client, args=(server_socket,))
        thread.start()
        client = object.__new__(ModelClient)
        client.host = "local"
        client.port = 0
        client.sock = client_socket
        client.last_rpc_metrics = {}
        client.decode_errors = 0
        client.connection_resets = 0
        depth = np.arange(12, dtype=np.float32).reshape(3, 4)

        response = client._send_recv({"cmd": "echo", "obs": {"depth": depth}})
        client.close()
        thread.join(2)

        np.testing.assert_array_equal(response["res"]["depth"], depth)
        self.assertFalse(thread.is_alive())
        self.assertEqual(client.last_rpc_metrics["decode_errors"], 0)
        self.assertEqual(client.last_rpc_metrics["connection_resets"], 0)


if __name__ == "__main__":
    unittest.main()
