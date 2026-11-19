from __future__ import annotations

import argparse
import socket
import time
from dataclasses import dataclass

import numpy as np

from Benchmarks.generate_data import generate_pair
from Benchmarks.metrics import BenchmarkRecord, relative_frobenius_error, save_record
from Benchmarks.serial_reference import multiply_blocked_serial, multiply_serial, validate_result
from Protocol.constants import (
    MSG_ERROR,
    MSG_JOB_RESULT,
    MSG_JOB_STATUS,
    MSG_SUBMIT_ACK,
    MSG_SUBMIT_JOB,
)
from Protocol.framing import recv_message, send_message
from Protocol.messages import matrix_from_bytes, pack_two_matrices


@dataclass(frozen=True)
class DistributedRun:
    matrix: np.ndarray
    elapsed_seconds: float
    total_candidate_tasks: int
    queued_tasks: int
    pruned_tasks: int


def submit_job(
    a: np.ndarray,
    b: np.ndarray,
    ingress_host: str,
    ingress_port: int,
    block_size: int,
    adaptive: bool,
    epsilon: float,
) -> dict:
    payload, matrix_meta = pack_two_matrices(a, b)

    with socket.create_connection((ingress_host, ingress_port), timeout=30.0) as sock:
        send_message(
            sock,
            MSG_SUBMIT_JOB,
            {
                "block_size": block_size,
                "adaptive": adaptive,
                "epsilon": epsilon,
                **matrix_meta,
            },
            payload,
        )
        header, _ = recv_message(sock)

    if header["type"] == MSG_ERROR:
        raise RuntimeError(header.get("message", "Ingress rejected the job."))
    if header["type"] != MSG_SUBMIT_ACK:
        raise RuntimeError(f"Unexpected ingress response: {header['type']}")
    return header


def wait_for_result(
    job_id: str,
    aggregator_host: str,
    aggregator_port: int,
    timeout_seconds: float,
    poll_interval: float = 0.05,
) -> tuple[np.ndarray, dict]:
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        with socket.create_connection(
            (aggregator_host, aggregator_port),
            timeout=10.0,
        ) as sock:
            send_message(sock, MSG_JOB_STATUS, {"job_id": job_id})
            header, payload = recv_message(sock)

        if header["type"] == MSG_ERROR:
            raise RuntimeError(header.get("message", "Aggregator returned an error."))
        if header["type"] != MSG_JOB_RESULT:
            raise RuntimeError(f"Unexpected aggregator response: {header['type']}")

        if header.get("complete", False):
            return matrix_from_bytes(payload, header["result_shape"]), header

        time.sleep(poll_interval)

    raise TimeoutError(f"Timed out waiting for distributed job {job_id}.")


def run_distributed(
    a: np.ndarray,
    b: np.ndarray,
    ingress_host: str,
    ingress_port: int,
    aggregator_host: str,
    aggregator_port: int,
    block_size: int,
    adaptive: bool,
    epsilon: float,
    timeout_seconds: float,
) -> DistributedRun:
    started = time.perf_counter()
    ack = submit_job(
        a=a,
        b=b,
        ingress_host=ingress_host,
        ingress_port=ingress_port,
        block_size=block_size,
        adaptive=adaptive,
        epsilon=epsilon,
    )
    matrix, status = wait_for_result(
        job_id=ack["job_id"],
        aggregator_host=aggregator_host,
        aggregator_port=aggregator_port,
        timeout_seconds=timeout_seconds,
    )
    elapsed = time.perf_counter() - started

    return DistributedRun(
        matrix=matrix,
        elapsed_seconds=elapsed,
        total_candidate_tasks=int(ack["total_candidate_tasks"]),
        queued_tasks=int(ack["queued_tasks"]),
        pruned_tasks=int(ack["pruned_tasks"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one distributed matrix benchmark.")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1, help="Label for the active Docker worker count.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--matrix-type",
        choices=["dense", "block_sparse", "low_rank"],
        default="dense",
    )
    parser.add_argument("--zero-block-fraction", type=float, default=0.5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--ingress-host", default="127.0.0.1")
    parser.add_argument("--ingress-port", type=int, default=9001)
    parser.add_argument("--aggregator-host", default="127.0.0.1")
    parser.add_argument("--aggregator-port", type=int, default=9002)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", default="Results/benchmark_results.json")
    args = parser.parse_args()

    generated = generate_pair(
        matrix_type=args.matrix_type,
        size=args.size,
        seed=args.seed,
        block_size=args.block_size,
        zero_block_fraction=args.zero_block_fraction,
        rank=args.rank,
    )

    serial = multiply_serial(generated.a, generated.b)
    blocked_serial = multiply_blocked_serial(
        generated.a,
        generated.b,
        block_size=args.block_size,
    )
    if not validate_result(blocked_serial.matrix, serial.matrix):
        raise AssertionError("Blocked serial result does not match the NumPy reference.")

    distributed = run_distributed(
        a=generated.a,
        b=generated.b,
        ingress_host=args.ingress_host,
        ingress_port=args.ingress_port,
        aggregator_host=args.aggregator_host,
        aggregator_port=args.aggregator_port,
        block_size=args.block_size,
        adaptive=args.adaptive,
        epsilon=args.epsilon,
        timeout_seconds=args.timeout,
    )

    exact_mode = (not args.adaptive) or args.epsilon == 0.0
    if exact_mode and not validate_result(distributed.matrix, serial.matrix):
        raise AssertionError("Distributed result does not match the serial reference.")

    rel_error = relative_frobenius_error(distributed.matrix, serial.matrix)
    speedup_vs_numpy = (
        serial.elapsed_seconds / distributed.elapsed_seconds
        if distributed.elapsed_seconds > 0
        else float("inf")
    )
    speedup_vs_blocked = (
        blocked_serial.elapsed_seconds / distributed.elapsed_seconds
        if distributed.elapsed_seconds > 0
        else float("inf")
    )
    efficiency_vs_blocked = speedup_vs_blocked / max(args.workers, 1)

    record = BenchmarkRecord(
        matrix_type=args.matrix_type,
        matrix_size=args.size,
        block_size=args.block_size,
        worker_count=args.workers,
        adaptive=args.adaptive,
        epsilon=args.epsilon,
        seed=args.seed,
        serial_seconds=serial.elapsed_seconds,
        blocked_serial_seconds=blocked_serial.elapsed_seconds,
        distributed_seconds=distributed.elapsed_seconds,
        speedup_vs_numpy=speedup_vs_numpy,
        speedup_vs_blocked=speedup_vs_blocked,
        efficiency_vs_blocked=efficiency_vs_blocked,
        total_candidate_tasks=distributed.total_candidate_tasks,
        queued_tasks=distributed.queued_tasks,
        pruned_tasks=distributed.pruned_tasks,
        relative_error=rel_error,
    )
    save_record(record, args.output)

    print(f"NumPy serial time:       {serial.elapsed_seconds:.6f} s")
    print(f"Blocked serial time:     {blocked_serial.elapsed_seconds:.6f} s")
    print(f"Distributed time:        {distributed.elapsed_seconds:.6f} s")
    print(f"Speedup vs NumPy:        {speedup_vs_numpy:.4f}x")
    print(f"Speedup vs blocked:      {speedup_vs_blocked:.4f}x")
    print(f"Efficiency vs blocked:   {efficiency_vs_blocked:.4f}")
    print(f"Tasks candidate:   {distributed.total_candidate_tasks}")
    print(f"Tasks queued:      {distributed.queued_tasks}")
    print(f"Tasks pruned:      {distributed.pruned_tasks}")
    print(f"Relative error:    {rel_error:.6e}")
    print(f"Saved record to:   {args.output}")


if __name__ == "__main__":
    main()
