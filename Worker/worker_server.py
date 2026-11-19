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
    MSG_RESULT_BATCH,
    MSG_RESULT_BATCH_ACK,
    MSG_TASK_BATCH,
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
    completed_task_ids: list[str] = []

    print(f"{worker_id} ready", flush=True)

    while True:
        try:
            send_message(
                ingress,
                MSG_TASK_REQUEST,
                {
                    "worker_id": worker_id,
                    "completed_task_ids": completed_task_ids,
                },
            )
            completed_task_ids = []
            header, payload = recv_message(ingress)

            if header["type"] == MSG_NO_TASK:
                time.sleep(float(header.get("retry_after_ms", 25)) / 1000.0)
                continue

            if header["type"] == MSG_ERROR:
                raise RuntimeError(header.get("message", "Ingress returned an error."))

            if header["type"] != MSG_TASK_BATCH:
                raise RuntimeError(f"Unexpected ingress message: {header['type']}")

            result_payload_parts: list[bytes] = []
            result_items: list[dict] = []
            result_offset = 0

            for item in header.get("items", []):
                offset = int(item["payload_offset"])
                length = int(item["payload_length"])
                task_payload = payload[offset : offset + length]
                a_block, b_block = unpack_two_matrices(task_payload, item)

                started = time.perf_counter()
                result = dense_multiply(a_block, b_block)
                compute_seconds = time.perf_counter() - started
                result_bytes = matrix_to_bytes(result)

                result_items.append(
                    {
                        "worker_id": worker_id,
                        "job_id": item["job_id"],
                        "task_id": item["task_id"],
                        "i": item["i"],
                        "j": item["j"],
                        "k": item["k"],
                        "row_start": item["row_start"],
                        "col_start": item["col_start"],
                        "result_shape": list(result.shape),
                        "compute_seconds": compute_seconds,
                        "payload_offset": result_offset,
                        "payload_length": len(result_bytes),
                    }
                )
                result_payload_parts.append(result_bytes)
                result_offset += len(result_bytes)

            send_message(
                aggregator,
                MSG_RESULT_BATCH,
                {
                    "worker_id": worker_id,
                    "task_count": len(result_items),
                    "items": result_items,
                },
                b"".join(result_payload_parts),
            )

            ack, _ = recv_message(aggregator)
            if ack["type"] == MSG_ERROR:
                raise RuntimeError(ack.get("message", "Aggregator returned an error."))
            if ack["type"] != MSG_RESULT_BATCH_ACK:
                raise RuntimeError(f"Unexpected aggregator response: {ack['type']}")

            completed_task_ids = [
                str(task_id) for task_id in ack.get("accepted_task_ids", [])
            ] + [
                str(task_id) for task_id in ack.get("duplicate_task_ids", [])
            ]

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
