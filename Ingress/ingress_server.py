from __future__ import annotations

import os
import queue
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict

import numpy as np

from Ingress.adaptive import should_prune
from Ingress.decomposition import BlockTask, iter_block_tasks, pad_for_blocking
from Protocol.constants import (
    DEFAULT_AGGREGATOR_HOST,
    DEFAULT_AGGREGATOR_PORT,
    DEFAULT_INGRESS_PORT,
    MSG_ERROR,
    MSG_JOB_INIT,
    MSG_JOB_INIT_ACK,
    MSG_NO_TASK,
    MSG_SUBMIT_ACK,
    MSG_SUBMIT_JOB,
    MSG_TASK_DENSE,
    MSG_TASK_REQUEST,
)
from Protocol.framing import ProtocolError, recv_message, send_message
from Protocol.messages import pack_two_matrices, unpack_two_matrices


@dataclass
class JobStats:
    job_id: str
    total_candidate_tasks: int
    dispatched_tasks: int
    pruned_tasks: int
    block_size: int
    adaptive: bool
    epsilon: float


class SchedulerState:
    def __init__(self) -> None:
        self.tasks: queue.Queue[BlockTask] = queue.Queue()
        self.jobs: Dict[str, JobStats] = {}
        self.lock = threading.RLock()

    def add_job(
        self,
        job_id: str,
        tasks: list[BlockTask],
        total_candidate_tasks: int,
        pruned_tasks: int,
        block_size: int,
        adaptive: bool,
        epsilon: float,
    ) -> JobStats:
        for task in tasks:
            self.tasks.put(task)

        stats = JobStats(
            job_id=job_id,
            total_candidate_tasks=total_candidate_tasks,
            dispatched_tasks=len(tasks),
            pruned_tasks=pruned_tasks,
            block_size=block_size,
            adaptive=adaptive,
            epsilon=epsilon,
        )
        with self.lock:
            self.jobs[job_id] = stats
        return stats


class IngressServer:
    def __init__(
        self,
        host: str,
        port: int,
        aggregator_host: str,
        aggregator_port: int,
    ) -> None:
        self.host = host
        self.port = port
        self.aggregator_host = aggregator_host
        self.aggregator_port = aggregator_port
        self.state = SchedulerState()

    def _send_error(self, conn: socket.socket, message: str) -> None:
        send_message(conn, MSG_ERROR, {"message": message})

    def _register_job_with_aggregator(
        self,
        job_id: str,
        output_shape: tuple[int, int],
        original_shape: tuple[int, int],
        block_size: int,
        expected_tasks: int,
        total_candidate_tasks: int,
        pruned_tasks: int,
    ) -> None:
        with socket.create_connection(
            (self.aggregator_host, self.aggregator_port),
            timeout=10.0,
        ) as agg:
            send_message(
                agg,
                MSG_JOB_INIT,
                {
                    "job_id": job_id,
                    "output_shape": list(output_shape),
                    "original_shape": list(original_shape),
                    "block_size": block_size,
                    "expected_tasks": expected_tasks,
                    "total_candidate_tasks": total_candidate_tasks,
                    "pruned_tasks": pruned_tasks,
                },
            )
            header, _ = recv_message(agg)
            if header["type"] != MSG_JOB_INIT_ACK:
                raise RuntimeError(
                    f"Aggregator rejected job initialization: {header.get('message', header)}"
                )

    def _handle_submit(self, conn: socket.socket, header: dict, payload: bytes) -> None:
        a, b = unpack_two_matrices(payload, header)
        block_size = int(header.get("block_size", 128))
        adaptive = bool(header.get("adaptive", False))
        epsilon = float(header.get("epsilon", 0.0))
        job_id = str(header.get("job_id") or uuid.uuid4())

        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")

        a_pad, b_pad, original_shape = pad_for_blocking(a, b, block_size)
        output_shape = (a_pad.shape[0], b_pad.shape[1])

        tasks: list[BlockTask] = []
        total_candidate_tasks = 0
        pruned_tasks = 0

        for task in iter_block_tasks(a, b, block_size, job_id):
            total_candidate_tasks += 1
            if adaptive:
                decision = should_prune(task.a_block, task.b_block, epsilon)
                if decision.prune:
                    pruned_tasks += 1
                    continue
            tasks.append(task)

        self._register_job_with_aggregator(
            job_id=job_id,
            output_shape=output_shape,
            original_shape=original_shape,
            block_size=block_size,
            expected_tasks=len(tasks),
            total_candidate_tasks=total_candidate_tasks,
            pruned_tasks=pruned_tasks,
        )

        stats = self.state.add_job(
            job_id=job_id,
            tasks=tasks,
            total_candidate_tasks=total_candidate_tasks,
            pruned_tasks=pruned_tasks,
            block_size=block_size,
            adaptive=adaptive,
            epsilon=epsilon,
        )

        send_message(
            conn,
            MSG_SUBMIT_ACK,
            {
                "job_id": job_id,
                "total_candidate_tasks": stats.total_candidate_tasks,
                "queued_tasks": stats.dispatched_tasks,
                "pruned_tasks": stats.pruned_tasks,
                "adaptive": stats.adaptive,
                "epsilon": stats.epsilon,
            },
        )

    def _handle_task_request(self, conn: socket.socket, header: dict) -> None:
        worker_id = str(header.get("worker_id", "unknown-worker"))

        try:
            task = self.state.tasks.get_nowait()
        except queue.Empty:
            send_message(conn, MSG_NO_TASK, {"worker_id": worker_id, "retry_after_ms": 100})
            return

        payload, matrix_meta = pack_two_matrices(task.a_block, task.b_block)
        metadata = {
            "worker_id": worker_id,
            "job_id": task.job_id,
            "task_id": task.task_id,
            "i": task.i,
            "j": task.j,
            "k": task.k,
            "row_start": task.row_start,
            "col_start": task.col_start,
            "result_shape": [task.a_block.shape[0], task.b_block.shape[1]],
            **matrix_meta,
        }
        send_message(conn, MSG_TASK_DENSE, metadata, payload)

    def handle_connection(self, conn: socket.socket, address: tuple[str, int]) -> None:
        with conn:
            while True:
                try:
                    header, payload = recv_message(conn)
                except ConnectionError:
                    return
                except (ProtocolError, ValueError, KeyError) as exc:
                    try:
                        self._send_error(conn, str(exc))
                    finally:
                        return

                try:
                    if header["type"] == MSG_SUBMIT_JOB:
                        self._handle_submit(conn, header, payload)
                    elif header["type"] == MSG_TASK_REQUEST:
                        self._handle_task_request(conn, header)
                    else:
                        self._send_error(conn, f"Unsupported message type: {header['type']}")
                except Exception as exc:
                    self._send_error(conn, f"{type(exc).__name__}: {exc}")

    def serve_forever(self) -> None:
        print(
            f"Ingress listening on {self.host}:{self.port}; "
            f"aggregator={self.aggregator_host}:{self.aggregator_port}",
            flush=True,
        )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen()

            while True:
                conn, address = server.accept()
                thread = threading.Thread(
                    target=self.handle_connection,
                    args=(conn, address),
                    daemon=True,
                )
                thread.start()


def main() -> None:
    host = os.getenv("INGRESS_BIND_HOST", "0.0.0.0")
    port = int(os.getenv("INGRESS_PORT", str(DEFAULT_INGRESS_PORT)))
    aggregator_host = os.getenv("AGGREGATOR_HOST", DEFAULT_AGGREGATOR_HOST)
    aggregator_port = int(
        os.getenv("AGGREGATOR_PORT", str(DEFAULT_AGGREGATOR_PORT))
    )
    IngressServer(host, port, aggregator_host, aggregator_port).serve_forever()


if __name__ == "__main__":
    main()
