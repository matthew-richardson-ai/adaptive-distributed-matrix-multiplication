from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BenchmarkRecord:
    matrix_type: str
    matrix_size: int
    block_size: int
    worker_count: int
    adaptive: bool
    epsilon: float
    seed: int
    serial_seconds: float
    blocked_serial_seconds: float
    distributed_seconds: float
    speedup_vs_numpy: float
    speedup_vs_blocked: float
    efficiency_vs_blocked: float
    total_candidate_tasks: int
    queued_tasks: int
    pruned_tasks: int
    relative_error: float
    numpy_version: str = np.__version__
    python_version: str = platform.python_version()
    platform: str = platform.platform()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def relative_frobenius_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denom = float(np.linalg.norm(expected, ord="fro"))
    diff = float(np.linalg.norm(actual - expected, ord="fro"))
    if denom == 0.0:
        return 0.0 if diff == 0.0 else float("inf")
    return diff / denom


def save_record(record: BenchmarkRecord, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []

    existing.append(record.to_dict())
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
