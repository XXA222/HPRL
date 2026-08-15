"""Bounded asynchronous logging/checkpoint persistence for HPRL training benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
import time
from typing import Mapping

from .checkpoint import capture_checkpoint_payload, write_checkpoint_payload


class AsyncArtifactError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AsyncArtifactStats:
    submitted: int
    completed: int
    metrics_events: int
    checkpoints: int
    log_bytes: int
    checkpoint_bytes: int
    deletions: int
    queue_high_water: int
    producer_block_seconds: float
    worker_seconds: float
    pending: int
    failed: bool


class AsyncArtifactWriter:
    """One bounded FIFO worker with lossless checkpoint semantics and flush-on-close.

    Metrics and checkpoints share ordering.  Checkpoint snapshots are captured before enqueueing,
    so the worker never observes parameters or optimizer state while they are being mutated by a
    later training update.  Queue saturation applies explicit producer backpressure instead of
    silently dropping checkpoints or metrics.
    """

    _STOP = object()

    def __init__(self, *, queue_size: int = 8) -> None:
        if not isinstance(queue_size, int) or isinstance(queue_size, bool) or queue_size < 1:
            raise ValueError("async artifact queue_size must be a positive integer")
        self._queue: Queue[object] = Queue(maxsize=queue_size)
        self._lock = Lock()
        self._handles: dict[Path, object] = {}
        self._submitted = 0
        self._completed = 0
        self._metrics_events = 0
        self._checkpoints = 0
        self._log_bytes = 0
        self._checkpoint_bytes = 0
        self._deletions = 0
        self._queue_high_water = 0
        self._producer_block_seconds = 0.0
        self._worker_seconds = 0.0
        self._error: BaseException | None = None
        self._closed = False
        self._thread = Thread(target=self._run, name="hprl-artifact-writer", daemon=False)
        self._thread.start()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise AsyncArtifactError("asynchronous artifact writer failed") from self._error

    def _put(self, task: object) -> None:
        if self._closed:
            raise RuntimeError("asynchronous artifact writer is closed")
        self._raise_if_failed()
        started = time.perf_counter()
        self._queue.put(task, block=True)
        blocked = time.perf_counter() - started
        with self._lock:
            self._producer_block_seconds += blocked
            self._submitted += 1
            self._queue_high_water = max(self._queue_high_water, self._queue.qsize())

    def submit_metrics(self, path: str | Path, values: Mapping[str, object]) -> None:
        # JSON conversion intentionally stays in the worker.  Copy the mapping itself so callers
        # can safely reuse their metrics container after submission.
        self._put(("metrics", Path(path), dict(values)))

    def submit_checkpoint(
        self,
        path: str | Path,
        agent,
        metadata: Mapping[str, object],
        *,
        cpu_snapshot: bool = True,
    ) -> None:
        state, metadata_text = capture_checkpoint_payload(
            agent, metadata, cpu_snapshot=cpu_snapshot
        )
        self._put(("checkpoint", Path(path), state, metadata_text))


    def submit_delete(self, path: str | Path) -> None:
        """Delete an older checkpoint after all previously queued writes complete."""
        self._put(("delete", Path(path)))

    def _handle_metrics(self, path: Path, values: Mapping[str, object]) -> None:
        handle = self._handles.get(path)
        if handle is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", encoding="utf-8")
            self._handles[path] = handle
        text = json.dumps(dict(values), sort_keys=True) + "\n"
        handle.write(text)
        self._metrics_events += 1
        self._log_bytes += len(text.encode("utf-8"))

    def _handle_checkpoint(
        self, path: Path, state: Mapping[str, object], metadata_text: str
    ) -> None:
        write_checkpoint_payload(path, state, metadata_text)
        self._checkpoints += 1
        if path.exists():
            self._checkpoint_bytes += path.stat().st_size

    def _run(self) -> None:
        try:
            while True:
                task = self._queue.get()
                try:
                    if task is self._STOP:
                        return
                    started = time.perf_counter()
                    kind = task[0]
                    if kind == "metrics":
                        self._handle_metrics(task[1], task[2])
                    elif kind == "checkpoint":
                        self._handle_checkpoint(task[1], task[2], task[3])
                    elif kind == "delete":
                        path = task[1]
                        path.unlink(missing_ok=True)
                        path.with_suffix(path.suffix + ".json").unlink(missing_ok=True)
                        self._deletions += 1
                    else:
                        raise RuntimeError(f"unknown async artifact task: {kind!r}")
                    with self._lock:
                        self._worker_seconds += time.perf_counter() - started
                        self._completed += 1
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._error = exc
            # Drain any already queued tasks so queue.join() cannot deadlock after worker failure.
            while True:
                try:
                    self._queue.get_nowait()
                except Exception:
                    break
                else:
                    self._queue.task_done()
        finally:
            for handle in self._handles.values():
                try:
                    handle.flush()
                    handle.close()
                except Exception:
                    pass
            self._handles.clear()

    def flush(self) -> None:
        self._queue.join()
        self._raise_if_failed()
        for handle in tuple(self._handles.values()):
            handle.flush()

    def close(self) -> None:
        if self._closed:
            self._raise_if_failed()
            return
        self.flush()
        self._queue.put(self._STOP)
        self._thread.join()
        self._closed = True
        self._raise_if_failed()

    def stats(self) -> AsyncArtifactStats:
        with self._lock:
            return AsyncArtifactStats(
                submitted=self._submitted,
                completed=self._completed,
                metrics_events=self._metrics_events,
                checkpoints=self._checkpoints,
                log_bytes=self._log_bytes,
                checkpoint_bytes=self._checkpoint_bytes,
                deletions=self._deletions,
                queue_high_water=self._queue_high_water,
                producer_block_seconds=self._producer_block_seconds,
                worker_seconds=self._worker_seconds,
                pending=self._queue.unfinished_tasks,
                failed=self._error is not None,
            )

    def __enter__(self) -> "AsyncArtifactWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

class SynchronousArtifactWriter:
    """Interface-compatible synchronous writer used by workload-aware auto policy."""
    def __init__(self) -> None:
        self._handles: dict[Path, object] = {}
        self._submitted = self._completed = 0
        self._metrics_events = self._checkpoints = 0
        self._log_bytes = self._checkpoint_bytes = self._deletions = 0
        self._worker_seconds = 0.0
        self._closed = False

    def _timed(self, fn) -> None:
        started = time.perf_counter(); fn(); self._worker_seconds += time.perf_counter() - started
        self._submitted += 1; self._completed += 1

    def submit_metrics(self, path: str | Path, values: Mapping[str, object]) -> None:
        path = Path(path)
        def work():
            handle = self._handles.get(path)
            if handle is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("a", encoding="utf-8"); self._handles[path] = handle
            text = json.dumps(dict(values), sort_keys=True) + "\n"
            handle.write(text); self._metrics_events += 1; self._log_bytes += len(text.encode("utf-8"))
        self._timed(work)

    def submit_checkpoint(self, path: str | Path, agent, metadata: Mapping[str, object], *, cpu_snapshot: bool = True) -> None:
        state, metadata_text = capture_checkpoint_payload(agent, metadata, cpu_snapshot=cpu_snapshot)
        path = Path(path)
        def work():
            write_checkpoint_payload(path, state, metadata_text); self._checkpoints += 1
            if path.exists(): self._checkpoint_bytes += path.stat().st_size
        self._timed(work)

    def submit_delete(self, path: str | Path) -> None:
        path = Path(path)
        def work():
            path.unlink(missing_ok=True); path.with_suffix(path.suffix + ".json").unlink(missing_ok=True); self._deletions += 1
        self._timed(work)

    def flush(self) -> None:
        for handle in self._handles.values(): handle.flush()

    def close(self) -> None:
        if self._closed: return
        self.flush()
        for handle in self._handles.values(): handle.close()
        self._handles.clear(); self._closed = True

    def stats(self) -> AsyncArtifactStats:
        return AsyncArtifactStats(
            submitted=self._submitted, completed=self._completed,
            metrics_events=self._metrics_events, checkpoints=self._checkpoints,
            log_bytes=self._log_bytes, checkpoint_bytes=self._checkpoint_bytes,
            deletions=self._deletions, queue_high_water=0, producer_block_seconds=0.0,
            worker_seconds=self._worker_seconds, pending=0, failed=False,
        )

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.close()
