from __future__ import annotations

import os
import socket
import time
import uuid

from Protocol.constants import (
    DEFAULT_AGGREGATOR_HOST,
    DEFAULT_AGGREGATOR_PORT,
    DEFAULT_INGRESS_HOST,
    DEFAULT_INGRESS_PORT,
    MSG_ERROR,
    MSG_NO_TASK,
    MSG_RESULT_ACK,
    MSG_RESULT_RETURN,
    MSG_TASK_DENSE,
    MSG_TASK_REQUEST,
)
from Protocol.framing import recv_message, send_message
from Protocol.messages import matrix_to_bytes, unpack_two_matrices
from Worker.kernels import dense_multiply


def _connect_forever(host: str, port: int, label: str) -> socket.socket:
    delay = 0.25
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=10.0)
            sock.settimeout(None)
            print(f"Connected to {label} at {host}:{port}", flush=True)
            return sock
        except OSError as exc:
            print(f"Waiting for {label} at {host}:{port}: {exc}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 5.0)


def run_worker() -> None:
    worker_id = os.getenv("WORKER_ID", f"worker-{uuid.uuid4().hex[:8]}")
    ingress_host = os.getenv("INGRESS_HOST", DEFAULT_INGRESS_HOST)
    ingress_port = int(os.getenv("INGRESS_PORT", str(DEFAULT_INGRESS_PORT)))
    aggregator_host = os.getenv("AGGREGATOR_HOST", DEFAULT_AGGREGATOR_HOST)
    aggregator_port = int(
        os.getenv("AGGREGATOR_PORT", str(DEFAULT_AGGREGATOR_PORT))
    )

    ingress = _connect_forever(ingress_host, ingress_port, "ingress")
    aggregator = _connect_forever(aggregator_host, aggregator_port, "aggregator")

    print(f"{worker_id} ready", flush=True)

    while True:
        try:
            send_message(ingress, MSG_TASK_REQUEST, {"worker_id": worker_id})
            header, payload = recv_message(ingress)

            if header["type"] == MSG_NO_TASK:
                time.sleep(float(header.get("retry_after_ms", 100)) / 1000.0)
                continue

            if header["type"] == MSG_ERROR:
                raise RuntimeError(header.get("message", "Ingress returned an error."))

            if header["type"] != MSG_TASK_DENSE:
                raise RuntimeError(f"Unexpected ingress message: {header['type']}")

            a_block, b_block = unpack_two_matrices(payload, header)
            started = time.perf_counter()
            result = dense_multiply(a_block, b_block)
            compute_seconds = time.perf_counter() - started

            send_message(
                aggregator,
                MSG_RESULT_RETURN,
                {
                    "worker_id": worker_id,
                    "job_id": header["job_id"],
                    "task_id": header["task_id"],
                    "i": header["i"],
                    "j": header["j"],
                    "k": header["k"],
                    "row_start": header["row_start"],
                    "col_start": header["col_start"],
                    "result_shape": list(result.shape),
                    "compute_seconds": compute_seconds,
                },
                matrix_to_bytes(result),
            )

            ack, _ = recv_message(aggregator)
            if ack["type"] == MSG_ERROR:
                raise RuntimeError(ack.get("message", "Aggregator returned an error."))
            if ack["type"] != MSG_RESULT_ACK:
                raise RuntimeError(f"Unexpected aggregator response: {ack['type']}")

        except (ConnectionError, OSError) as exc:
            print(f"{worker_id} reconnecting after connection failure: {exc}", flush=True)
            try:
                ingress.close()
            except OSError:
                pass
            try:
                aggregator.close()
            except OSError:
                pass
            ingress = _connect_forever(ingress_host, ingress_port, "ingress")
            aggregator = _connect_forever(
                aggregator_host, aggregator_port, "aggregator"
            )


def main() -> None:
    run_worker()


if __name__ == "__main__":
    main()
