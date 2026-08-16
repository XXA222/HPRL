"""Crash-safe append-only evidence journal for 24h/72h shadow qualification."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Iterable

from .shadow import ShadowMetrics, ShadowPolicy
from .shadow_runtime import ShadowRunPolicy, ShadowRunQualification, ShadowWindow, qualify_shadow_run

ZERO_HASH = "0" * 64


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(raw).hexdigest()


def _metrics_payload(metrics: ShadowMetrics) -> dict[str, object]:
    payload = asdict(metrics)
    payload["duration_seconds"] = metrics.duration.total_seconds()
    payload.pop("duration", None)
    return payload


def _metrics_from_payload(payload: dict[str, object]) -> ShadowMetrics:
    values = dict(payload)
    duration_seconds = float(values.pop("duration_seconds"))
    return ShadowMetrics(duration=timedelta(seconds=duration_seconds), **values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ShadowJournalRecord:
    sequence: int
    window: ShadowWindow
    previous_sha256: str
    source_release: str
    observed_at: datetime

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("shadow journal sequence must start at 1")
        if len(self.previous_sha256) != 64:
            raise ValueError("previous_sha256 must be SHA-256")
        if not self.source_release.strip():
            raise ValueError("source_release is required")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))

    def payload(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "window": {
                "started_at": self.window.started_at.isoformat(),
                "ended_at": self.window.ended_at.isoformat(),
                "metrics": _metrics_payload(self.window.metrics),
                "restart_boundary": self.window.restart_boundary,
                "source_cursor_start": self.window.source_cursor_start,
                "source_cursor_end": self.window.source_cursor_end,
            },
            "previous_sha256": self.previous_sha256,
            "source_release": self.source_release,
            "observed_at": self.observed_at.isoformat(),
        }

    @property
    def record_sha256(self) -> str:
        return _sha(self.payload())

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "ShadowJournalRecord":
        raw_window = payload["window"]
        if not isinstance(raw_window, dict):
            raise ValueError("shadow journal window must be an object")
        metrics_raw = raw_window["metrics"]
        if not isinstance(metrics_raw, dict):
            raise ValueError("shadow journal metrics must be an object")
        window = ShadowWindow(
            started_at=datetime.fromisoformat(str(raw_window["started_at"])),
            ended_at=datetime.fromisoformat(str(raw_window["ended_at"])),
            metrics=_metrics_from_payload(metrics_raw),
            restart_boundary=bool(raw_window.get("restart_boundary", False)),
            source_cursor_start=int(raw_window.get("source_cursor_start", 0)),
            source_cursor_end=int(raw_window.get("source_cursor_end", 0)),
        )
        return cls(
            sequence=int(payload["sequence"]),
            window=window,
            previous_sha256=str(payload["previous_sha256"]),
            source_release=str(payload["source_release"]),
            observed_at=datetime.fromisoformat(str(payload["observed_at"])),
        )


@dataclass(frozen=True, slots=True)
class ShadowJournalState:
    records: tuple[ShadowJournalRecord, ...]
    valid: bool
    reasons: tuple[str, ...]
    tip_sha256: str

    @property
    def windows(self) -> tuple[ShadowWindow, ...]:
        return tuple(record.window for record in self.records)


class JsonlShadowWindowJournal:
    """Append-only fsync journal; one durable record must represent one real window."""

    def __init__(self, path: str | Path, *, source_release: str) -> None:
        self.path = Path(path)
        self.source_release = source_release.strip()
        if not self.source_release:
            raise ValueError("source_release is required")

    def load(self) -> ShadowJournalState:
        if not self.path.exists():
            return ShadowJournalState((), True, (), ZERO_HASH)
        records: list[ShadowJournalRecord] = []
        reasons: list[str] = []
        previous = ZERO_HASH
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                wrapper = json.loads(line)
                payload = wrapper["record"]
                claimed = str(wrapper["record_sha256"])
                if not isinstance(payload, dict):
                    raise ValueError("record payload must be object")
                record = ShadowJournalRecord.from_payload(payload)
            except Exception as exc:
                reasons.append(f"INVALID_LINE:{line_number}:{type(exc).__name__}")
                break
            if record.sequence != len(records) + 1:
                reasons.append(f"SEQUENCE_GAP:{line_number}")
            if record.previous_sha256 != previous:
                reasons.append(f"PREVIOUS_HASH_MISMATCH:{line_number}")
            if record.record_sha256 != claimed:
                reasons.append(f"RECORD_HASH_MISMATCH:{line_number}")
            if record.source_release != self.source_release:
                reasons.append(f"SOURCE_RELEASE_MISMATCH:{line_number}")
            records.append(record)
            previous = record.record_sha256
        return ShadowJournalState(tuple(records), not reasons, tuple(reasons), previous)

    def append(self, window: ShadowWindow, *, observed_at: datetime | None = None) -> ShadowJournalRecord:
        state = self.load()
        if not state.valid:
            raise RuntimeError("shadow journal is corrupt: " + ",".join(state.reasons))
        if state.records:
            previous = state.records[-1].window
            if window.started_at < previous.started_at:
                raise ValueError("shadow windows cannot regress in time")
            if window.source_cursor_end <= previous.source_cursor_end:
                raise ValueError("shadow journal requires forward source cursor progress")
        record = ShadowJournalRecord(
            sequence=len(state.records) + 1,
            window=window,
            previous_sha256=state.tip_sha256,
            source_release=self.source_release,
            observed_at=(observed_at or datetime.now(UTC)),
        )
        wrapper = {"record": record.payload(), "record_sha256": record.record_sha256}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(wrapper, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def qualify(
        self,
        *,
        target: str,
        run_policy: ShadowRunPolicy | None = None,
        shadow_policy: ShadowPolicy | None = None,
    ) -> ShadowRunQualification:
        state = self.load()
        if not state.valid:
            return ShadowRunQualification(False, state.reasons, timedelta(0), len(state.records), state.tip_sha256)
        qualification = qualify_shadow_run(
            state.windows,
            target=target,
            run_policy=run_policy,
            shadow_policy=shadow_policy,
        )
        # Bind the acceptance semantic hash to the durable chain tip as well as windows.
        digest = _sha({"qualification": qualification.semantic_hash, "journal_tip": state.tip_sha256})
        return ShadowRunQualification(
            qualification.passed,
            qualification.reasons,
            qualification.covered_duration,
            qualification.windows,
            digest,
        )
