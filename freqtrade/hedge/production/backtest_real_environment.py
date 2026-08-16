"""Measured two-year backtest execution and durable evidence journal for R3.

The runner measures a real child process tree with psutil, hashes source/result artifacts,
and appends immutable chunk evidence.  The existing ``backtest_stability`` evaluator remains
the policy authority; this module provides the missing measured execution layer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Sequence

import psutil

from .backtest_stability import (
    BacktestChunkEvidence,
    TwoYearBacktestPolicy,
    TwoYearBacktestStabilityReport,
    evaluate_two_year_backtest_stability,
)

ZERO_HASH = "0" * 64


def _chunk_payload(chunk: BacktestChunkEvidence) -> dict[str, object]:
    return {
        "started_at": chunk.started_at.isoformat(),
        "ended_at": chunk.ended_at.isoformat(),
        "bars": int(chunk.bars),
        "events": int(chunk.events),
        "elapsed_seconds": float(chunk.elapsed_seconds),
        "peak_rss_bytes": int(chunk.peak_rss_bytes),
        "exit_code": int(chunk.exit_code),
        "result_sha256": chunk.result_sha256,
        "source_data_sha256": chunk.source_data_sha256,
    }


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_rss_bytes(process: psutil.Process) -> int:
    total = 0
    rows = [process]
    try:
        rows.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    for item in rows:
        try:
            total += int(item.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


@dataclass(frozen=True, slots=True)
class MeasuredBacktestCommand:
    argv: tuple[str, ...]
    cwd: str
    started_at: datetime
    ended_at: datetime
    source_data_path: str
    result_path: str
    metrics_path: str
    timeout_seconds: int
    poll_interval_seconds: float = 0.2

    def __post_init__(self) -> None:
        if not self.argv or not self.argv[0]:
            raise ValueError("backtest argv is required")
        object.__setattr__(self, "started_at", _aware(self.started_at))
        object.__setattr__(self, "ended_at", _aware(self.ended_at))
        if self.ended_at <= self.started_at:
            raise ValueError("backtest market coverage must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("backtest timeout must be positive")
        if self.poll_interval_seconds <= 0 or self.poll_interval_seconds > 5:
            raise ValueError("backtest poll interval must be within (0,5]")


@dataclass(frozen=True, slots=True)
class MeasuredBacktestCommandReport:
    chunk: BacktestChunkEvidence | None
    command_sha256: str
    stdout_path: str
    stderr_path: str
    timeout_killed: bool
    metrics_loaded: bool
    reasons: tuple[str, ...]
    source_unchanged: bool = False
    rss_samples: int = 0
    result_fresh: bool = False
    metrics_fresh: bool = False

    @property
    def passed(self) -> bool:
        return self.chunk is not None and self.chunk.exit_code == 0 and not self.reasons


def run_measured_backtest_command(
    command: MeasuredBacktestCommand,
    *,
    output_dir: str | Path,
) -> MeasuredBacktestCommandReport:
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stdout_path = out / "backtest.stdout.log"
    stderr_path = out / "backtest.stderr.log"
    source = Path(command.source_data_path).resolve()
    result = Path(command.result_path).resolve()
    metrics = Path(command.metrics_path).resolve()
    reasons: list[str] = []
    if not source.is_file():
        reasons.append("BACKTEST_SOURCE_DATA_MISSING")
        return MeasuredBacktestCommandReport(None, "", str(stdout_path), str(stderr_path), False, False, tuple(reasons))

    # R3 evidence must be produced by this invocation, never inherited from a stale
    # previous run.  Operators should direct each measured chunk to a fresh evidence
    # directory or explicitly remove old result/metrics artifacts before execution.
    if result.exists():
        reasons.append("BACKTEST_RESULT_PREEXISTS")
    if metrics.exists():
        reasons.append("BACKTEST_METRICS_PREEXISTS")
    if reasons:
        return MeasuredBacktestCommandReport(
            None, "", str(stdout_path), str(stderr_path), False, False, tuple(reasons),
            False, 0, False, False,
        )

    source_sha_before = _sha256_path(source)
    command_digest = sha256(json.dumps({
        "argv": command.argv,
        "cwd": str(Path(command.cwd).resolve()),
        "start": command.started_at.isoformat(),
        "end": command.ended_at.isoformat(),
        "source": source_sha_before,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    timeout_killed = False
    started_monotonic = time.monotonic()
    peak_rss = 0
    rss_samples = 0
    rc = -1

    def sample(root: psutil.Process) -> None:
        nonlocal peak_rss, rss_samples
        try:
            value = _tree_rss_bytes(root)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return
        if value > 0:
            peak_rss = max(peak_rss, value)
            rss_samples += 1

    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            list(command.argv),
            cwd=str(Path(command.cwd).resolve()),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        root = psutil.Process(process.pid)
        # Sample immediately so very short commands still have a real process RSS
        # observation rather than a fabricated non-zero placeholder.
        sample(root)
        deadline = started_monotonic + command.timeout_seconds
        while process.poll() is None:
            sample(root)
            if time.monotonic() >= deadline:
                timeout_killed = True
                try:
                    for child in root.children(recursive=True):
                        child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        for child in root.children(recursive=True):
                            child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    process.kill()
                break
            time.sleep(command.poll_interval_seconds)
        rc = int(process.wait())
        sample(root)

    elapsed = max(time.monotonic() - started_monotonic, 1e-9)
    source_sha_after = _sha256_path(source) if source.is_file() else ""
    source_unchanged = bool(source_sha_after and source_sha_before == source_sha_after)
    if not source_unchanged:
        reasons.append("BACKTEST_SOURCE_DATA_CHANGED_DURING_RUN")
    if rss_samples <= 0 or peak_rss <= 0:
        reasons.append("BACKTEST_RSS_NOT_OBSERVED")
    if timeout_killed:
        reasons.append("BACKTEST_PROCESS_TIMEOUT")

    metrics_fresh = metrics.is_file() and metrics.stat().st_size > 0
    result_fresh = result.is_file() and result.stat().st_size > 0
    if not metrics_fresh:
        reasons.append("BACKTEST_METRICS_FILE_MISSING")
        return MeasuredBacktestCommandReport(
            None, command_digest, str(stdout_path), str(stderr_path), timeout_killed, False,
            tuple(dict.fromkeys(reasons)), source_unchanged, rss_samples, result_fresh, False,
        )
    try:
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        bars = int(payload["bars"])
        events = int(payload.get("events", bars))
    except Exception as exc:
        reasons.append(f"BACKTEST_METRICS_INVALID:{type(exc).__name__}")
        return MeasuredBacktestCommandReport(
            None, command_digest, str(stdout_path), str(stderr_path), timeout_killed, False,
            tuple(dict.fromkeys(reasons)), source_unchanged, rss_samples, result_fresh, True,
        )
    if not result_fresh:
        reasons.append("BACKTEST_RESULT_MISSING")
        return MeasuredBacktestCommandReport(
            None, command_digest, str(stdout_path), str(stderr_path), timeout_killed, True,
            tuple(dict.fromkeys(reasons)), source_unchanged, rss_samples, False, True,
        )
    chunk = BacktestChunkEvidence(
        started_at=command.started_at,
        ended_at=command.ended_at,
        bars=bars,
        events=events,
        elapsed_seconds=elapsed,
        peak_rss_bytes=peak_rss,
        exit_code=rc,
        result_sha256=_sha256_path(result),
        source_data_sha256=source_sha_after,
    )
    if rc != 0:
        reasons.append(f"BACKTEST_PROCESS_EXIT_NONZERO:{rc}")
    return MeasuredBacktestCommandReport(
        chunk=chunk,
        command_sha256=command_digest,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        timeout_killed=timeout_killed,
        metrics_loaded=True,
        reasons=tuple(dict.fromkeys(reasons)),
        source_unchanged=source_unchanged,
        rss_samples=rss_samples,
        result_fresh=result_fresh,
        metrics_fresh=metrics_fresh,
    )


@dataclass(frozen=True, slots=True)
class BacktestJournalRecord:
    sequence: int
    chunk: BacktestChunkEvidence
    command_sha256: str
    previous_sha256: str

    @property
    def record_sha256(self) -> str:
        return sha256(json.dumps({
            "sequence": self.sequence,
            "chunk": _chunk_payload(self.chunk),
            "command_sha256": self.command_sha256,
            "previous_sha256": self.previous_sha256,
        }, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class BacktestJournalState:
    valid: bool
    records: tuple[BacktestJournalRecord, ...]
    tip_sha256: str
    reasons: tuple[str, ...]


class JsonlBacktestEvidenceJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> BacktestJournalState:
        if not self.path.exists():
            return BacktestJournalState(True, (), ZERO_HASH, ())
        records: list[BacktestJournalRecord] = []
        reasons: list[str] = []
        previous = ZERO_HASH
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                wrapper = json.loads(line)
                raw = wrapper["record"]
                chunk_raw = raw["chunk"]
                chunk = BacktestChunkEvidence(
                    started_at=datetime.fromisoformat(str(chunk_raw["started_at"])),
                    ended_at=datetime.fromisoformat(str(chunk_raw["ended_at"])),
                    bars=int(chunk_raw["bars"]),
                    events=int(chunk_raw["events"]),
                    elapsed_seconds=float(chunk_raw["elapsed_seconds"]),
                    peak_rss_bytes=int(chunk_raw["peak_rss_bytes"]),
                    exit_code=int(chunk_raw["exit_code"]),
                    result_sha256=str(chunk_raw["result_sha256"]),
                    source_data_sha256=str(chunk_raw["source_data_sha256"]),
                )
                record = BacktestJournalRecord(
                    sequence=int(raw["sequence"]),
                    chunk=chunk,
                    command_sha256=str(raw["command_sha256"]),
                    previous_sha256=str(raw["previous_sha256"]),
                )
                claimed = str(wrapper["record_sha256"])
            except Exception as exc:
                reasons.append(f"BACKTEST_JOURNAL_PARSE:{line_number}:{type(exc).__name__}")
                continue
            if record.sequence != len(records) + 1:
                reasons.append(f"BACKTEST_JOURNAL_SEQUENCE:{line_number}")
            if record.previous_sha256 != previous:
                reasons.append(f"BACKTEST_JOURNAL_PREVIOUS_HASH:{line_number}")
            if record.record_sha256 != claimed:
                reasons.append(f"BACKTEST_JOURNAL_RECORD_HASH:{line_number}")
            previous = record.record_sha256
            records.append(record)
        return BacktestJournalState(not reasons, tuple(records), previous, tuple(dict.fromkeys(reasons)))

    def append(self, report: MeasuredBacktestCommandReport) -> BacktestJournalRecord:
        if report.chunk is None:
            raise ValueError("cannot journal a missing measured chunk")
        state = self.load()
        if not state.valid:
            raise RuntimeError("backtest evidence journal is corrupt: " + ",".join(state.reasons))
        if state.records and report.chunk.started_at < state.records[-1].chunk.ended_at:
            raise ValueError("backtest evidence chunks cannot overlap/regress")
        record = BacktestJournalRecord(len(state.records) + 1, report.chunk, report.command_sha256, state.tip_sha256)
        wrapper = {
            "record": {
                "sequence": record.sequence,
                "chunk": _chunk_payload(record.chunk),
                "command_sha256": record.command_sha256,
                "previous_sha256": record.previous_sha256,
            },
            "record_sha256": record.record_sha256,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(wrapper, sort_keys=True, default=str, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def qualify(
        self,
        *,
        repeat_aggregate_sha256: str | None,
        policy: TwoYearBacktestPolicy | None = None,
    ) -> TwoYearBacktestStabilityReport:
        state = self.load()
        if not state.valid:
            return TwoYearBacktestStabilityReport(
                False, len(state.records), timedelta(0), 0, 0, 0.0, 0, 0, False,
                state.tip_sha256, state.reasons,
            )
        return evaluate_two_year_backtest_stability(
            (record.chunk for record in state.records),
            repeat_result_sha256=repeat_aggregate_sha256,
            policy=policy,
        )

@dataclass(frozen=True, slots=True)
class R3TwoYearBacktestQualification:
    passed: bool
    primary_report: TwoYearBacktestStabilityReport
    repeat_report: TwoYearBacktestStabilityReport
    independent_journals: bool
    same_chunk_count: bool
    semantic_repeat_match: bool
    primary_semantic_sha256: str
    repeat_semantic_sha256: str
    reasons: tuple[str, ...]


def _semantic_repeat_payload(state: BacktestJournalState) -> list[dict[str, object]]:
    rows = sorted((record.chunk for record in state.records), key=lambda item: item.started_at)
    return [
        {
            "start": row.started_at.isoformat(),
            "end": row.ended_at.isoformat(),
            "bars": int(row.bars),
            "events": int(row.events),
            "source": row.source_data_sha256,
            "result": row.result_sha256,
        }
        for row in rows
    ]


def _semantic_repeat_sha256(state: BacktestJournalState) -> str:
    payload = _semantic_repeat_payload(state)
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def qualify_r3_two_year_backtest(
    primary: JsonlBacktestEvidenceJournal,
    repeat: JsonlBacktestEvidenceJournal,
    *,
    policy: TwoYearBacktestPolicy | None = None,
) -> R3TwoYearBacktestQualification:
    """Qualify two independently measured backtest evidence journals.

    Resource measurements are evaluated independently for both runs.  Deterministic
    repeat proof compares only semantic outputs and immutable market coverage facts;
    elapsed time and RSS are intentionally excluded because they may vary across runs.
    A journal cannot be compared with itself.
    """
    effective = policy or TwoYearBacktestPolicy()
    no_repeat_policy = replace(effective, require_repeat_digest=False)
    primary_state = primary.load()
    repeat_state = repeat.load()
    reasons: list[str] = []

    try:
        independent = primary.path.resolve() != repeat.path.resolve()
    except OSError:
        independent = str(primary.path.absolute()) != str(repeat.path.absolute())
    if not independent:
        reasons.append("TWO_YEAR_REPEAT_JOURNAL_MUST_BE_INDEPENDENT")

    if not primary_state.valid:
        reasons.extend(f"PRIMARY_{reason}" for reason in primary_state.reasons)
    if not repeat_state.valid:
        reasons.extend(f"REPEAT_{reason}" for reason in repeat_state.reasons)

    primary_report = evaluate_two_year_backtest_stability(
        (record.chunk for record in primary_state.records),
        repeat_result_sha256=None,
        policy=no_repeat_policy,
    )
    repeat_report = evaluate_two_year_backtest_stability(
        (record.chunk for record in repeat_state.records),
        repeat_result_sha256=None,
        policy=no_repeat_policy,
    )
    if not primary_report.passed:
        reasons.extend(f"PRIMARY_{reason}" for reason in primary_report.reasons)
    if not repeat_report.passed:
        reasons.extend(f"REPEAT_{reason}" for reason in repeat_report.reasons)

    same_chunk_count = bool(primary_state.records and repeat_state.records) and (
        len(primary_state.records) == len(repeat_state.records)
    )
    if not same_chunk_count:
        reasons.append("TWO_YEAR_REPEAT_CHUNK_COUNT_MISMATCH")

    primary_semantic = _semantic_repeat_sha256(primary_state)
    repeat_semantic = _semantic_repeat_sha256(repeat_state)
    semantic_match = bool(
        same_chunk_count
        and primary_semantic == repeat_semantic
        and primary_state.valid
        and repeat_state.valid
    )
    if not semantic_match:
        reasons.append("TWO_YEAR_DETERMINISTIC_REPEAT_SEMANTIC_MISMATCH")

    return R3TwoYearBacktestQualification(
        passed=not reasons,
        primary_report=primary_report,
        repeat_report=repeat_report,
        independent_journals=independent,
        same_chunk_count=same_chunk_count,
        semantic_repeat_match=semantic_match,
        primary_semantic_sha256=primary_semantic,
        repeat_semantic_sha256=repeat_semantic,
        reasons=tuple(dict.fromkeys(reasons)),
    )

