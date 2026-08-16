"""Durable 24h/72h shadow evidence binding for Runtime Closure R3.

R2 proved that elapsed time cannot be synthesized from one short window.  R3 additionally
binds every durable shadow window to real-market/model-target evidence.  Acceptance-probe
or manually unlabeled windows may be useful diagnostics, but they cannot satisfy the final
24h/72h model shadow gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Iterable

import psutil

from .risk_behavior_real_environment import JsonlR3BehaviorJournal

from .shadow import ShadowMetrics, ShadowPolicy
from .shadow_runtime import ShadowRunPolicy, ShadowWindow, qualify_shadow_run

ZERO_HASH = "0" * 64


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _sha(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _sha_hex(value: str, *, field: str) -> str:
    text = value.lower().strip()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field} must be SHA-256 hex")
    return text


@dataclass(frozen=True, slots=True)
class R3ShadowWindowEvidence:
    window: ShadowWindow
    source_release: str
    target_source: str
    model_id: str
    model_observations: int
    real_market_evidence_sha256: str
    behavior_chain_sha256: str
    process_rss_start_bytes: int
    process_rss_end_bytes: int
    recorded_at: datetime

    def __post_init__(self) -> None:
        if not self.source_release.strip() or not self.model_id.strip():
            raise ValueError("shadow source_release/model_id are required")
        if self.model_observations <= 0:
            raise ValueError("model_observations must be positive")
        if self.process_rss_start_bytes <= 0 or self.process_rss_end_bytes <= 0:
            raise ValueError("shadow RSS observations must be positive")
        actual_duration = self.window.ended_at - self.window.started_at
        if abs((self.window.metrics.duration - actual_duration).total_seconds()) > 1.0:
            raise ValueError("shadow metrics duration must match real window duration")
        observed_growth = max(0.0, (self.process_rss_end_bytes - self.process_rss_start_bytes) / self.process_rss_start_bytes)
        if abs(self.window.metrics.memory_growth_ratio - observed_growth) > 0.01:
            raise ValueError("shadow memory_growth_ratio must match observed RSS growth")
        object.__setattr__(self, "recorded_at", _aware(self.recorded_at))
        object.__setattr__(self, "real_market_evidence_sha256", _sha_hex(self.real_market_evidence_sha256, field="real_market_evidence_sha256"))
        object.__setattr__(self, "behavior_chain_sha256", _sha_hex(self.behavior_chain_sha256, field="behavior_chain_sha256"))

    @property
    def model_target_feed(self) -> bool:
        return self.target_source == "model-target-feed"

    def payload(self) -> dict[str, object]:
        return {
            "window": {
                "started_at": self.window.started_at.isoformat(),
                "ended_at": self.window.ended_at.isoformat(),
                "metrics": {
                    key: (value.total_seconds() if isinstance(value, timedelta) else value)
                    for key, value in asdict(self.window.metrics).items()
                },
                "restart_boundary": self.window.restart_boundary,
                "source_cursor_start": self.window.source_cursor_start,
                "source_cursor_end": self.window.source_cursor_end,
            },
            "source_release": self.source_release,
            "target_source": self.target_source,
            "model_id": self.model_id,
            "model_observations": self.model_observations,
            "real_market_evidence_sha256": self.real_market_evidence_sha256,
            "behavior_chain_sha256": self.behavior_chain_sha256,
            "process_rss_start_bytes": self.process_rss_start_bytes,
            "process_rss_end_bytes": self.process_rss_end_bytes,
            "recorded_at": self.recorded_at.isoformat(),
        }

    @classmethod
    def from_payload(cls, raw: dict[str, object]) -> "R3ShadowWindowEvidence":
        window_raw = raw["window"]
        if not isinstance(window_raw, dict):
            raise ValueError("shadow window payload must be an object")
        metrics_raw = window_raw["metrics"]
        if not isinstance(metrics_raw, dict):
            raise ValueError("shadow metrics payload must be an object")
        metrics = ShadowMetrics(
            duration=timedelta(seconds=float(metrics_raw["duration"])),
            rest_ws_position_divergences=int(metrics_raw.get("rest_ws_position_divergences", 0)),
            unknown_orders_peak=int(metrics_raw.get("unknown_orders_peak", 0)),
            unresolved_unknown_orders=int(metrics_raw.get("unresolved_unknown_orders", 0)),
            sequence_gaps_unrecovered=int(metrics_raw.get("sequence_gaps_unrecovered", 0)),
            candle_gaps_unrecovered=int(metrics_raw.get("candle_gaps_unrecovered", 0)),
            duplicate_effects=int(metrics_raw.get("duplicate_effects", 0)),
            reconciliation_p99_seconds=float(metrics_raw.get("reconciliation_p99_seconds", 0.0)),
            loop_p99_ms=float(metrics_raw.get("loop_p99_ms", 0.0)),
            db_p99_ms=float(metrics_raw.get("db_p99_ms", 0.0)),
            model_p99_ms=float(metrics_raw.get("model_p99_ms", 0.0)),
            model_fallbacks=int(metrics_raw.get("model_fallbacks", 0)),
            memory_growth_ratio=float(metrics_raw.get("memory_growth_ratio", 0.0)),
            restart_recoveries=int(metrics_raw.get("restart_recoveries", 0)),
            restart_recovery_failures=int(metrics_raw.get("restart_recovery_failures", 0)),
            funding_cycles_observed=int(metrics_raw.get("funding_cycles_observed", 0)),
            planner_churn_ratio=float(metrics_raw.get("planner_churn_ratio", 0.0)),
            risk_reject_ratio=float(metrics_raw.get("risk_reject_ratio", 0.0)),
        )
        window = ShadowWindow(
            started_at=datetime.fromisoformat(str(window_raw["started_at"])),
            ended_at=datetime.fromisoformat(str(window_raw["ended_at"])),
            metrics=metrics,
            restart_boundary=bool(window_raw.get("restart_boundary", False)),
            source_cursor_start=int(window_raw.get("source_cursor_start", 0)),
            source_cursor_end=int(window_raw.get("source_cursor_end", 0)),
        )
        return cls(
            window=window,
            source_release=str(raw["source_release"]),
            target_source=str(raw["target_source"]),
            model_id=str(raw["model_id"]),
            model_observations=int(raw["model_observations"]),
            real_market_evidence_sha256=str(raw["real_market_evidence_sha256"]),
            behavior_chain_sha256=str(raw["behavior_chain_sha256"]),
            process_rss_start_bytes=int(raw["process_rss_start_bytes"]),
            process_rss_end_bytes=int(raw["process_rss_end_bytes"]),
            recorded_at=datetime.fromisoformat(str(raw["recorded_at"])),
        )


@dataclass(frozen=True, slots=True)
class R3ShadowJournalRecord:
    sequence: int
    evidence: R3ShadowWindowEvidence
    previous_sha256: str

    @property
    def record_sha256(self) -> str:
        return _sha({
            "sequence": self.sequence,
            "evidence": self.evidence.payload(),
            "previous_sha256": self.previous_sha256,
        })


@dataclass(frozen=True, slots=True)
class R3ShadowJournalState:
    valid: bool
    records: tuple[R3ShadowJournalRecord, ...]
    tip_sha256: str
    reasons: tuple[str, ...]


class JsonlR3ShadowJournal:
    def __init__(self, path: str | Path, *, source_release: str) -> None:
        self.path = Path(path)
        self.source_release = source_release

    def load(self) -> R3ShadowJournalState:
        if not self.path.exists():
            return R3ShadowJournalState(True, (), ZERO_HASH, ())
        records: list[R3ShadowJournalRecord] = []
        reasons: list[str] = []
        previous = ZERO_HASH
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                wrapper = json.loads(line)
                record_raw = wrapper["record"]
                evidence = R3ShadowWindowEvidence.from_payload(record_raw["evidence"])
                record = R3ShadowJournalRecord(
                    sequence=int(record_raw["sequence"]),
                    evidence=evidence,
                    previous_sha256=str(record_raw["previous_sha256"]),
                )
                claimed = str(wrapper["record_sha256"])
            except Exception as exc:
                reasons.append(f"SHADOW_R3_PARSE:{line_number}:{type(exc).__name__}")
                continue
            if record.sequence != len(records) + 1:
                reasons.append(f"SHADOW_R3_SEQUENCE:{line_number}")
            if record.previous_sha256 != previous:
                reasons.append(f"SHADOW_R3_PREVIOUS_HASH:{line_number}")
            if record.record_sha256 != claimed:
                reasons.append(f"SHADOW_R3_RECORD_HASH:{line_number}")
            if evidence.source_release != self.source_release:
                reasons.append(f"SHADOW_R3_SOURCE_RELEASE:{line_number}")
            previous = record.record_sha256
            records.append(record)
        return R3ShadowJournalState(not reasons, tuple(records), previous, tuple(dict.fromkeys(reasons)))

    def append(self, evidence: R3ShadowWindowEvidence) -> R3ShadowJournalRecord:
        state = self.load()
        if not state.valid:
            raise RuntimeError("R3 shadow journal is corrupt: " + ",".join(state.reasons))
        if evidence.source_release != self.source_release:
            raise ValueError("R3 shadow source release mismatch")
        if state.records:
            previous = state.records[-1].evidence.window
            if evidence.window.started_at < previous.started_at:
                raise ValueError("R3 shadow windows cannot regress")
            if evidence.window.source_cursor_end <= previous.source_cursor_end:
                raise ValueError("R3 shadow source cursor must advance")
        record = R3ShadowJournalRecord(len(state.records) + 1, evidence, state.tip_sha256)
        wrapper = {
            "record": {
                "sequence": record.sequence,
                "evidence": evidence.payload(),
                "previous_sha256": record.previous_sha256,
            },
            "record_sha256": record.record_sha256,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(wrapper, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record


@dataclass(frozen=True, slots=True)
class MeasuredR3ShadowCommand:
    argv: tuple[str, ...]
    cwd: str
    metrics_path: str
    real_market_evidence_path: str
    behavior_journal_path: str
    source_release: str
    model_id: str
    timeout_seconds: int
    poll_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.argv or not self.argv[0]:
            raise ValueError("shadow argv is required")
        if not self.source_release.strip() or not self.model_id.strip():
            raise ValueError("shadow source release/model id are required")
        if self.timeout_seconds <= 0:
            raise ValueError("shadow timeout must be positive")
        if self.poll_interval_seconds <= 0 or self.poll_interval_seconds > 30:
            raise ValueError("shadow poll interval must be within (0,30]")


@dataclass(frozen=True, slots=True)
class MeasuredR3ShadowReport:
    passed: bool
    evidence: R3ShadowWindowEvidence | None
    journal_record_sha256: str
    process_exit_code: int
    peak_rss_bytes: int
    rss_samples: int
    stdout_path: str
    stderr_path: str
    reasons: tuple[str, ...]


def _process_tree_rss(root: psutil.Process) -> int:
    rows = [root]
    try:
        rows.extend(root.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    total = 0
    for item in rows:
        try:
            total += int(item.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _metrics_from_payload(raw: dict[str, object], *, duration: timedelta, memory_growth_ratio: float) -> ShadowMetrics:
    return ShadowMetrics(
        duration=duration,
        rest_ws_position_divergences=int(raw.get("rest_ws_position_divergences", 0)),
        unknown_orders_peak=int(raw.get("unknown_orders_peak", 0)),
        unresolved_unknown_orders=int(raw.get("unresolved_unknown_orders", 0)),
        sequence_gaps_unrecovered=int(raw.get("sequence_gaps_unrecovered", 0)),
        candle_gaps_unrecovered=int(raw.get("candle_gaps_unrecovered", 0)),
        duplicate_effects=int(raw.get("duplicate_effects", 0)),
        reconciliation_p99_seconds=float(raw.get("reconciliation_p99_seconds", 0.0)),
        loop_p99_ms=float(raw.get("loop_p99_ms", 0.0)),
        db_p99_ms=float(raw.get("db_p99_ms", 0.0)),
        model_p99_ms=float(raw.get("model_p99_ms", 0.0)),
        model_fallbacks=int(raw.get("model_fallbacks", 0)),
        memory_growth_ratio=memory_growth_ratio,
        restart_recoveries=int(raw.get("restart_recoveries", 0)),
        restart_recovery_failures=int(raw.get("restart_recovery_failures", 0)),
        funding_cycles_observed=int(raw.get("funding_cycles_observed", 0)),
        planner_churn_ratio=float(raw.get("planner_churn_ratio", 0.0)),
        risk_reject_ratio=float(raw.get("risk_reject_ratio", 0.0)),
    )


def run_measured_r3_shadow_command(
    command: MeasuredR3ShadowCommand,
    *,
    shadow_journal: JsonlR3ShadowJournal,
    output_dir: str | Path,
) -> MeasuredR3ShadowReport:
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stdout_path = out / "shadow.stdout.log"
    stderr_path = out / "shadow.stderr.log"
    metrics_path = Path(command.metrics_path).resolve()
    market_path = Path(command.real_market_evidence_path).resolve()
    reasons: list[str] = []
    if metrics_path.exists(): reasons.append("SHADOW_METRICS_PREEXISTS")
    if market_path.exists(): reasons.append("SHADOW_REAL_MARKET_EVIDENCE_PREEXISTS")
    behavior_journal = JsonlR3BehaviorJournal(command.behavior_journal_path)
    before = behavior_journal.load()
    if not before.valid:
        reasons.extend(before.reasons)
    if reasons:
        return MeasuredR3ShadowReport(False, None, "", -1, 0, 0, str(stdout_path), str(stderr_path), tuple(dict.fromkeys(reasons)))

    started = datetime.now(UTC)
    started_mono = time.monotonic()
    rss_start = peak_rss = rss_end = rss_samples = 0
    timeout = False
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            list(command.argv), cwd=str(Path(command.cwd).resolve()),
            stdout=stdout_handle, stderr=stderr_handle,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        root = psutil.Process(process.pid)
        deadline = started_mono + command.timeout_seconds
        while True:
            value = _process_tree_rss(root)
            if value > 0:
                if rss_samples == 0: rss_start = value
                rss_end = value; peak_rss = max(peak_rss, value); rss_samples += 1
            if process.poll() is not None:
                break
            if time.monotonic() >= deadline:
                timeout = True
                try:
                    for child in root.children(recursive=True): child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill()
                break
            time.sleep(command.poll_interval_seconds)
        rc = int(process.wait())
    ended = datetime.now(UTC)
    if timeout: reasons.append("SHADOW_PROCESS_TIMEOUT")
    if rc != 0: reasons.append(f"SHADOW_PROCESS_EXIT_NONZERO:{rc}")
    if rss_samples <= 0 or rss_start <= 0 or rss_end <= 0: reasons.append("SHADOW_RSS_NOT_OBSERVED")
    if not metrics_path.is_file(): reasons.append("SHADOW_METRICS_MISSING")
    if not market_path.is_file() or market_path.stat().st_size <= 0: reasons.append("SHADOW_REAL_MARKET_EVIDENCE_MISSING")

    after = behavior_journal.load()
    if not after.valid: reasons.extend(after.reasons)
    if len(after.records) <= len(before.records): reasons.append("SHADOW_MODEL_OBSERVATIONS_MISSING")
    new_rows = after.records[len(before.records):] if len(after.records) >= len(before.records) else ()
    if new_rows and any(row.row.model_id != command.model_id for row in new_rows): reasons.append("SHADOW_MODEL_ID_MISMATCH")
    if new_rows and any(not row.row.model_target_feed for row in new_rows): reasons.append("SHADOW_MODEL_TARGET_FEED_REQUIRED")

    metrics_raw: dict[str, object] = {}
    market_raw: dict[str, object] = {}
    if metrics_path.is_file():
        try:
            loaded = json.loads(metrics_path.read_text(encoding="utf-8")); metrics_raw = loaded if isinstance(loaded, dict) else {}
        except Exception as exc: reasons.append(f"SHADOW_METRICS_INVALID:{type(exc).__name__}")
    if market_path.is_file():
        try:
            loaded = json.loads(market_path.read_text(encoding="utf-8")); market_raw = loaded if isinstance(loaded, dict) else {}
        except Exception as exc: reasons.append(f"SHADOW_MARKET_EVIDENCE_INVALID:{type(exc).__name__}")
    if market_raw:
        if not bool(market_raw.get("passed")): reasons.append("SHADOW_REAL_MARKET_RUN_NOT_PASSED")
        if not bool(market_raw.get("production_evidence_eligible")): reasons.append("SHADOW_REAL_MARKET_NOT_PRODUCTION_ELIGIBLE")
        if not bool(market_raw.get("model_target_feed")): reasons.append("SHADOW_MARKET_MODEL_FEED_REQUIRED")
        if int(market_raw.get("real_trade_write_count", -1)) != 0: reasons.append("SHADOW_REAL_TRADE_WRITE_NONZERO")

    evidence: R3ShadowWindowEvidence | None = None
    record_sha = ""
    if not reasons:
        duration = ended - started
        growth = max(0.0, (rss_end - rss_start) / rss_start)
        metrics = _metrics_from_payload(metrics_raw, duration=duration, memory_growth_ratio=growth)
        cursor_start = int(metrics_raw.get("source_cursor_start", len(before.records)))
        cursor_end = int(metrics_raw.get("source_cursor_end", len(after.records) - 1))
        evidence = R3ShadowWindowEvidence(
            window=ShadowWindow(
                started_at=started, ended_at=ended, metrics=metrics,
                restart_boundary=bool(metrics_raw.get("restart_boundary", False)),
                source_cursor_start=cursor_start, source_cursor_end=cursor_end,
            ),
            source_release=command.source_release, target_source="model-target-feed",
            model_id=command.model_id, model_observations=len(new_rows),
            real_market_evidence_sha256=sha256(market_path.read_bytes()).hexdigest(),
            behavior_chain_sha256=after.tip_sha256,
            process_rss_start_bytes=rss_start, process_rss_end_bytes=rss_end,
            recorded_at=ended,
        )
        record_sha = shadow_journal.append(evidence).record_sha256
    return MeasuredR3ShadowReport(
        passed=not reasons, evidence=evidence, journal_record_sha256=record_sha,
        process_exit_code=rc, peak_rss_bytes=peak_rss, rss_samples=rss_samples,
        stdout_path=str(stdout_path), stderr_path=str(stderr_path),
        reasons=tuple(dict.fromkeys(reasons)),
    )


@dataclass(frozen=True, slots=True)
class R3ShadowQualification:
    passed: bool
    target: str
    journal_valid: bool
    model_target_only: bool
    real_market_bound: bool
    model_observations: int
    covered_duration: timedelta
    windows: int
    journal_tip_sha256: str
    semantic_sha256: str
    reasons: tuple[str, ...]


def qualify_r3_shadow(
    journal: JsonlR3ShadowJournal,
    *,
    target: str,
    run_policy: ShadowRunPolicy | None = None,
    shadow_policy: ShadowPolicy | None = None,
) -> R3ShadowQualification:
    state = journal.load()
    reasons = list(state.reasons)
    windows = tuple(record.evidence.window for record in state.records)
    base = qualify_shadow_run(windows, target=target, run_policy=run_policy, shadow_policy=shadow_policy)
    reasons.extend(base.reasons)
    model_only = bool(state.records) and all(record.evidence.model_target_feed for record in state.records)
    if not model_only:
        reasons.append("SHADOW_R3_MODEL_TARGET_FEED_REQUIRED")
    market_bound = bool(state.records) and all(bool(record.evidence.real_market_evidence_sha256) for record in state.records)
    if not market_bound:
        reasons.append("SHADOW_R3_REAL_MARKET_EVIDENCE_REQUIRED")
    observations = sum(record.evidence.model_observations for record in state.records)
    if observations <= 0:
        reasons.append("SHADOW_R3_MODEL_OBSERVATIONS_MISSING")
    semantic = _sha({
        "target": target,
        "base": base.semantic_hash,
        "tip": state.tip_sha256,
        "market": [record.evidence.real_market_evidence_sha256 for record in state.records],
        "behavior": [record.evidence.behavior_chain_sha256 for record in state.records],
        "observations": observations,
    })
    return R3ShadowQualification(
        passed=not reasons,
        target=target,
        journal_valid=state.valid,
        model_target_only=model_only,
        real_market_bound=market_bound,
        model_observations=observations,
        covered_duration=base.covered_duration,
        windows=base.windows,
        journal_tip_sha256=state.tip_sha256,
        semantic_sha256=semantic,
        reasons=tuple(dict.fromkeys(reasons)),
    )
