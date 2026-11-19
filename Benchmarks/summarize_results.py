from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


GROUP_KEYS = (
    "matrix_type",
    "matrix_size",
    "block_size",
    "worker_count",
    "adaptive",
    "epsilon",
)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(record.get(k) for k in GROUP_KEYS)
        groups[key].append(record)

    output: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        distributed = [float(r["distributed_seconds"]) for r in rows]
        numpy_serial = [float(r["serial_seconds"]) for r in rows]
        blocked_serial = [
            float(r["blocked_serial_seconds"])
            for r in rows
            if "blocked_serial_seconds" in r
        ]
        speedup_blocked = [
            float(r["speedup_vs_blocked"])
            for r in rows
            if "speedup_vs_blocked" in r
        ]

        summary = dict(zip(GROUP_KEYS, key))
        summary.update(
            {
                "trials": len(rows),
                "distributed_mean_seconds": _mean(distributed),
                "distributed_median_seconds": _median(distributed),
                "distributed_stdev_seconds": _stdev(distributed),
                "numpy_serial_mean_seconds": _mean(numpy_serial),
                "blocked_serial_mean_seconds": _mean(blocked_serial),
                "speedup_vs_blocked_mean": _mean(speedup_blocked),
                "relative_error_max": max(float(r["relative_error"]) for r in rows),
                "pruned_tasks_mean": _mean(
                    [float(r["pruned_tasks"]) for r in rows]
                ),
            }
        )
        output.append(summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize repeated benchmark records.")
    parser.add_argument("--input", default="Results/benchmark_results.json")
    parser.add_argument("--output", default="Results/benchmark_summary.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    records = json.loads(input_path.read_text(encoding="utf-8"))
    summaries = summarize(records)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    for row in summaries:
        print(
            f"{row['matrix_type']} N={row['matrix_size']} "
            f"b={row['block_size']} workers={row['worker_count']} "
            f"adaptive={row['adaptive']} trials={row['trials']} | "
            f"dist mean={row['distributed_mean_seconds']:.6f}s "
            f"median={row['distributed_median_seconds']:.6f}s "
            f"stdev={row['distributed_stdev_seconds']:.6f}s "
            f"speedup(blocked)={row['speedup_vs_blocked_mean']:.4f}x"
        )

    print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
