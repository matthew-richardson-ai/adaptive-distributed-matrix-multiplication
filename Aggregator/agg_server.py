from __future__ import annotations

import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Set

import numpy as np

from Protocol.constants import (
    DEFAULT_AGGREGATOR_HOST,
    DEFAULT_AGGREGATOR_PORT,
    MSG_ERROR,
    MSG_JOB_INIT,
    MSG_JOB_INIT_ACK,
    MSG_JOB_RESULT,
    MSG_JOB_STATUS,
    MSG_RESULT_ACK,
    MSG_RESULT_BATCH,
    MSG_RESULT_BATCH_ACK,
    MSG_RESULT_RETURN,
)
from Protocol.framing import ProtocolError, recv_message, send_message
from Protocol.messages import matrix_from_bytes, matrix_to_bytes


@dataclass
class JobState:
    job_id: str
    output_shape: tuple[int, int]
    original_shape: tuple[int, int]
    expected_tasks: int
    block_size: int
    total_candidate_tasks: int = 0
    pruned_tasks: int = 0
    created_at: float = field(default_factory=time.time)
    completed_tasks: Set[str] = field(default_factory=set)
    result: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.result = np.zeros(self.output_shape, dtype=np.float64)

    @property
    def complete(self) -> bool:
        return len(self.completed_tasks) >= self.expected_tasks

    def add_result(
        self,
        task_id: str,
        row_start: int,
        col_start: int,
        partial: np.ndarray,
    ) -> bool:
        if task_id in self.completed_tasks:
            return False

        row_end = row_start + partial.shape[0]
        col_end = col_start + partial.shape[1]
        if row_start < 0 or col_start < 0:
            raise ValueError("Negative result offset.")
        if row_end > self.result.shape[0] or col_end > self.result.shape[1]:
            raise ValueError("Result block exceeds output matrix bounds.")

        self.result[row_start:row_end, col_start:col_end] += partial
        self.completed_tasks.add(task_id)
        return True

    def trimmed_result(self) -> np.ndarray:
        rows, cols = self.original_shape
        return np.ascontiguousarray(self.result[:rows, :cols])


class AggregatorState:
    def __init__(self) -> None:
        self.jobs: Dict[str, JobState] = {}
        self.lock = threading.RLock()

    def create_job(self, metadata: dict) -> JobState:
        job_id = str(metadata["job_id"])
        output_shape = tuple(int(x) for x in metadata["output_shape"])
        original_shape = tuple(int(x) for x in metadata["original_shape"])
        expected_tasks = int(metadata["expected_tasks"])
        block_size = int(metadata["block_size"])
        total_candidate_tasks = int(metadata.get("total_candidate_tasks", expected_tasks))
        pruned_tasks = int(metadata.get("pruned_tasks", 0))

        if len(output_shape) != 2 or len(original_shape) != 2:
            raise ValueError("Output shapes must be two-dimensional.")
        if expected_tasks < 0:
            raise ValueError("expected_tasks cannot be negative.")
        if block_size <= 0:
            raise ValueError("block_size must be positive.")

        with self.lock:
            if job_id in self.jobs:
                raise ValueError(f"Job {job_id} already exists.")
            job = JobState(
                job_id=job_id,
                output_shape=output_shape,
                original_shape=original_shape,
                expected_tasks=expected_tasks,
                block_size=block_size,
                total_candidate_tasks=total_candidate_tasks,
                pruned_tasks=pruned_tasks,
            )
            self.jobs[job_id] = job
            return job

    def get_job(self, job_id: str) -> JobState | None:
        with self.lock:
            return self.jobs.get(job_id)


class AggregatorServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.state = AggregatorState()

    def _send_error(self, conn: socket.socket, message: str, task_id: str | None = None) -> None:
        metadata = {"message": message}
        if task_id:
            metadata["task_id"] = task_id
        send_message(conn, MSG_ERROR, metadata)

    def _handle_job_init(self, conn: socket.socket, header: dict) -> None:
        job = self.state.create_job(header)
        send_message(
            conn,
            MSG_JOB_INIT_ACK,
            {
                "job_id": job.job_id,
                "expected_tasks": job.expected_tasks,
                "complete": job.complete,
            },
        )

    def _handle_result(self, conn: socket.socket, header: dict, payload: bytes) -> None:
        job_id = str(header["job_id"])
        task_id = str(header["task_id"])
        job = self.state.get_job(job_id)
        if job is None:
            self._send_error(conn, f"Unknown job_id: {job_id}", task_id)
            return

        partial = matrix_from_bytes(payload, header["result_shape"])
        with self.state.lock:
            accepted = job.add_result(
                task_id=task_id,
                row_start=int(header["row_start"]),
                col_start=int(header["col_start"]),
                partial=partial,
            )
            completed = len(job.completed_tasks)
            complete = job.complete

        send_message(
            conn,
            MSG_RESULT_ACK,
            {
                "job_id": job_id,
                "task_id": task_id,
                "accepted": accepted,
                "completed_tasks": completed,
                "expected_tasks": job.expected_tasks,
                "complete": complete,
            },
        )


    def _handle_result_batch(self, conn: socket.socket, header: dict, payload: bytes) -> None:
        items = header.get("items", [])
        accepted_task_ids: list[str] = []
        duplicate_task_ids: list[str] = []
        job_summaries: dict[str, dict] = {}

        for item in items:
            job_id = str(item["job_id"])
            task_id = str(item["task_id"])
            job = self.state.get_job(job_id)
            if job is None:
                raise ValueError(f"Unknown job_id: {job_id}")

            offset = int(item["payload_offset"])
            length = int(item["payload_length"])
            if offset < 0 or length < 0 or offset + length > len(payload):
                raise ValueError(f"Invalid result payload slice for task {task_id}.")

            partial = matrix_from_bytes(
                payload[offset : offset + length],
                item["result_shape"],
            )

            with self.state.lock:
                accepted = job.add_result(
                    task_id=task_id,
                    row_start=int(item["row_start"]),
                    col_start=int(item["col_start"]),
                    partial=partial,
                )
                job_summaries[job_id] = {
                    "completed_tasks": len(job.completed_tasks),
                    "expected_tasks": job.expected_tasks,
                    "complete": job.complete,
                }

            if accepted:
                accepted_task_ids.append(task_id)
            else:
                duplicate_task_ids.append(task_id)

        send_message(
            conn,
            MSG_RESULT_BATCH_ACK,
            {
                "accepted_task_ids": accepted_task_ids,
                "duplicate_task_ids": duplicate_task_ids,
                "jobs": job_summaries,
            },
        )

    def _handle_status(self, conn: socket.socket, header: dict) -> None:
        job_id = str(header["job_id"])
        job = self.state.get_job(job_id)
        if job is None:
            self._send_error(conn, f"Unknown job_id: {job_id}")
            return

        with self.state.lock:
            metadata = {
                "job_id": job.job_id,
                "complete": job.complete,
                "completed_tasks": len(job.completed_tasks),
                "expected_tasks": job.expected_tasks,
                "total_candidate_tasks": job.total_candidate_tasks,
                "pruned_tasks": job.pruned_tasks,
                "elapsed_seconds": max(0.0, time.time() - job.created_at),
            }
            if job.complete:
                matrix = job.trimmed_result()
                metadata["result_shape"] = list(matrix.shape)
                payload = matrix_to_bytes(matrix)
            else:
                payload = b""

        send_message(conn, MSG_JOB_RESULT, metadata, payload)

    def handle_connection(self, conn: socket.socket, address: tuple[str, int]) -> None:
        with conn:
            while True:
                try:
                    header, payload = recv_message(conn)
                except ConnectionError:
                    return
                except (ProtocolError, ValueError, KeyError) as exc:
                    self._send_error(conn, str(exc))
                    return

                message_type = header["type"]
                try:
                    if message_type == MSG_JOB_INIT:
                        self._handle_job_init(conn, header)
                    elif message_type == MSG_RESULT_RETURN:
                        self._handle_result(conn, header, payload)
                    elif message_type == MSG_RESULT_BATCH:
                        self._handle_result_batch(conn, header, payload)
                    elif message_type == MSG_JOB_STATUS:
                        self._handle_status(conn, header)
                    else:
                        self._send_error(conn, f"Unsupported message type: {message_type}")
                except Exception as exc:
                    self._send_error(conn, f"{type(exc).__name__}: {exc}")

    def serve_forever(self) -> None:
        print(f"Aggregator listening on {self.host}:{self.port}", flush=True)
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
    host = os.getenv("AGGREGATOR_BIND_HOST", "0.0.0.0")
    port = int(os.getenv("AGGREGATOR_PORT", str(DEFAULT_AGGREGATOR_PORT)))
    AggregatorServer(host, port).serve_forever()


if __name__ == "__main__":
    main()
