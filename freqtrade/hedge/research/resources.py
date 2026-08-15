"""Resource telemetry for the local research executor."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import psutil


@dataclass(frozen=True, slots=True)
class ProcessResourceSnapshot:
    pid: int
    cpu_percent: float
    rss_bytes: int
    threads: int
    children: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "pid": self.pid,
            "cpu_percent": self.cpu_percent,
            "rss_bytes": self.rss_bytes,
            "threads": self.threads,
            "children": self.children,
        }


def process_snapshot(pid: int) -> ProcessResourceSnapshot | None:
    try:
        process = psutil.Process(pid)
        children = process.children(recursive=True)
        rss = process.memory_info().rss
        cpu = process.cpu_percent(interval=None)
        threads = process.num_threads()
        for child in children:
            try:
                rss += child.memory_info().rss
                cpu += child.cpu_percent(interval=None)
                threads += child.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return ProcessResourceSnapshot(
            pid=pid,
            cpu_percent=round(float(cpu), 2),
            rss_bytes=int(rss),
            threads=int(threads),
            children=len(children),
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _gpu_rows() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    command = [
        executable,
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []

    rows: list[dict[str, Any]] = []
    for raw in completed.stdout.splitlines():
        parts = [item.strip() for item in raw.split(",")]
        if len(parts) != 6:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_percent": float(parts[2]),
                    "memory_used_mb": float(parts[3]),
                    "memory_total_mb": float(parts[4]),
                    "memory_free_mb": max(0.0, float(parts[4]) - float(parts[3])),
                    "temperature_c": float(parts[5]),
                }
            )
        except ValueError:
            continue
    return rows


def system_snapshot(*, include_gpu: bool = True) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.abspath(os.sep))
    return {
        "cpu_percent": round(float(psutil.cpu_percent(interval=None)), 2),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_used_bytes": int(memory.used),
        "memory_total_bytes": int(memory.total),
        "memory_percent": round(float(memory.percent), 2),
        "disk_used_bytes": int(disk.used),
        "disk_total_bytes": int(disk.total),
        "disk_percent": round(float(disk.percent), 2),
        "gpus": _gpu_rows() if include_gpu else [],
    }
