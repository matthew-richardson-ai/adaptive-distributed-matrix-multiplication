from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_command(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a benchmark matrix from a JSON config.")
    parser.add_argument("--config", default="Benchmarks/Configs/dev.json")
    parser.add_argument("--manage-docker", action="store_true")
    parser.add_argument("--output", default="Results/benchmark_results.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    sizes = config["sizes"]
    block_sizes = config["block_sizes"]
    worker_counts = config["worker_counts"]
    matrix_types = config["matrix_types"]
    adaptive_modes = config.get(
        "adaptive_modes",
        [{"adaptive": False, "epsilon": 0.0}, {"adaptive": True, "epsilon": 0.0}],
    )
    seed = int(config.get("seed", 1234))
    repeats = int(config.get("repeats", 1))
    sparse_fraction = float(config.get("zero_block_fraction", 0.5))
    timeout = float(config.get("timeout_seconds", 300))

    for workers in worker_counts:
        if args.manage_docker:
            run_command(
                [
                    "docker",
                    "compose",
                    "up",
                    "-d",
                    "--build",
                    "--scale",
                    f"worker={workers}",
                ]
            )
            time.sleep(3)

        for size in sizes:
            for block_size in block_sizes:
                for matrix_type in matrix_types:
                    for mode in adaptive_modes:
                        for repeat in range(repeats):
                            command = [
                                sys.executable,
                                "-m",
                                "Benchmarks.run_benchmarks",
                                "--size",
                                str(size),
                                "--block-size",
                                str(block_size),
                                "--workers",
                                str(workers),
                                "--seed",
                                str(seed + repeat),
                                "--matrix-type",
                                matrix_type,
                                "--zero-block-fraction",
                                str(sparse_fraction),
                                "--timeout",
                                str(timeout),
                                "--output",
                                args.output,
                            ]
                            if bool(mode.get("adaptive", False)):
                                command.append("--adaptive")
                                command.extend(["--epsilon", str(mode.get("epsilon", 0.0))])
                            run_command(command)

    print("Benchmark suite complete.", flush=True)


if __name__ == "__main__":
    main()
