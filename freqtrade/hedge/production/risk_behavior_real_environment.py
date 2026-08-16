"""Durable real-observation position-behavior evidence for Runtime Closure R3."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path

from .risk_behavior import HprlBehaviorObservation, HprlBehaviorPolicy, HprlBehaviorReport, analyze_hprl_position_behavior

ZERO_HASH = "0" * 64


def _sha(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _sha_hex(value: str, *, field: str) -> str:
    text = value.lower().strip()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field} must be SHA-256 hex")
    return text


@dataclass(frozen=True, slots=True)
class R3BehaviorObservation:
    cycle_id: str
    model_id: str
    target_source: str
    target_sha256: str
    market_evidence_sha256: str
    observation: HprlBehaviorObservation

    def __post_init__(self) -> None:
        if not self.cycle_id.strip() or not self.model_id.strip():
            raise ValueError("behavior cycle_id/model_id are required")
        object.__setattr__(self, "target_sha256", _sha_hex(self.target_sha256, field="target_sha256"))
        object.__setattr__(self, "market_evidence_sha256", _sha_hex(self.market_evidence_sha256, field="market_evidence_sha256"))

    @property
    def model_target_feed(self) -> bool:
        return self.target_source == "model-target-feed"

    def payload(self) -> dict[str, object]:
        return {
            "cycle_id": self.cycle_id,
            "model_id": self.model_id,
            "target_source": self.target_source,
            "target_sha256": self.target_sha256,
            "market_evidence_sha256": self.market_evidence_sha256,
            "observation": asdict(self.observation),
        }

    @classmethod
    def from_payload(cls, raw: dict[str, object]) -> "R3BehaviorObservation":
        obs = raw["observation"]
        if not isinstance(obs, dict):
            raise ValueError("behavior observation payload must be an object")
        from decimal import Decimal
        return cls(
            cycle_id=str(raw["cycle_id"]),
            model_id=str(raw["model_id"]),
            target_source=str(raw["target_source"]),
            target_sha256=str(raw["target_sha256"]),
            market_evidence_sha256=str(raw["market_evidence_sha256"]),
            observation=HprlBehaviorObservation(
                timestamp=datetime.fromisoformat(str(obs["timestamp"])),
                long_margin_ratio=Decimal(str(obs["long_margin_ratio"])),
                short_margin_ratio=Decimal(str(obs["short_margin_ratio"])),
                equity_return=float(obs["equity_return"]),
                drawdown=float(obs["drawdown"]),
                uncertainty=float(obs.get("uncertainty", 0.0)),
            ),
        )


@dataclass(frozen=True, slots=True)
class R3BehaviorRecord:
    sequence: int
    row: R3BehaviorObservation
    previous_sha256: str

    @property
    def record_sha256(self) -> str:
        return _sha({"sequence": self.sequence, "row": self.row.payload(), "previous_sha256": self.previous_sha256})


@dataclass(frozen=True, slots=True)
class R3BehaviorJournalState:
    valid: bool
    records: tuple[R3BehaviorRecord, ...]
    tip_sha256: str
    reasons: tuple[str, ...]


class JsonlR3BehaviorJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> R3BehaviorJournalState:
        if not self.path.exists():
            return R3BehaviorJournalState(True, (), ZERO_HASH, ())
        records: list[R3BehaviorRecord] = []
        reasons: list[str] = []
        previous = ZERO_HASH
        seen_cycles: set[str] = set()
        last_time = None
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                wrapper = json.loads(line)
                raw = wrapper["record"]
                row = R3BehaviorObservation.from_payload(raw["row"])
                record = R3BehaviorRecord(int(raw["sequence"]), row, str(raw["previous_sha256"]))
                claimed = str(wrapper["record_sha256"])
            except Exception as exc:
                reasons.append(f"BEHAVIOR_JOURNAL_PARSE:{line_number}:{type(exc).__name__}")
                continue
            if record.sequence != len(records) + 1:
                reasons.append(f"BEHAVIOR_JOURNAL_SEQUENCE:{line_number}")
            if record.previous_sha256 != previous:
                reasons.append(f"BEHAVIOR_JOURNAL_PREVIOUS_HASH:{line_number}")
            if record.record_sha256 != claimed:
                reasons.append(f"BEHAVIOR_JOURNAL_RECORD_HASH:{line_number}")
            if row.cycle_id in seen_cycles:
                reasons.append(f"BEHAVIOR_JOURNAL_DUPLICATE_CYCLE:{line_number}")
            if last_time is not None and row.observation.timestamp < last_time:
                reasons.append(f"BEHAVIOR_JOURNAL_TIME_REGRESSION:{line_number}")
            previous = record.record_sha256
            seen_cycles.add(row.cycle_id)
            last_time = row.observation.timestamp
            records.append(record)
        return R3BehaviorJournalState(not reasons, tuple(records), previous, tuple(dict.fromkeys(reasons)))

    def append(self, rows: tuple[R3BehaviorObservation, ...]) -> tuple[R3BehaviorRecord, ...]:
        if not rows:
            raise ValueError("behavior append requires observations")
        state = self.load()
        if not state.valid:
            raise RuntimeError("behavior journal is corrupt: " + ",".join(state.reasons))
        existing_cycles = {record.row.cycle_id for record in state.records}
        result: list[R3BehaviorRecord] = []
        previous = state.tip_sha256
        sequence = len(state.records)
        last_time = state.records[-1].row.observation.timestamp if state.records else None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                if not row.model_target_feed:
                    raise ValueError("final behavior journal accepts model-target-feed observations only")
                if row.cycle_id in existing_cycles:
                    raise ValueError(f"duplicate behavior cycle: {row.cycle_id}")
                if last_time is not None and row.observation.timestamp < last_time:
                    raise ValueError("behavior observation time regression")
                sequence += 1
                record = R3BehaviorRecord(sequence, row, previous)
                wrapper = {
                    "record": {"sequence": sequence, "row": row.payload(), "previous_sha256": previous},
                    "record_sha256": record.record_sha256,
                }
                handle.write(json.dumps(wrapper, sort_keys=True, default=str, separators=(",", ":")) + "\n")
                result.append(record)
                existing_cycles.add(row.cycle_id)
                last_time = row.observation.timestamp
                previous = record.record_sha256
            handle.flush()
            os.fsync(handle.fileno())
        return tuple(result)


@dataclass(frozen=True, slots=True)
class R3BehaviorQualification:
    passed: bool
    journal_valid: bool
    model_target_only: bool
    market_bound: bool
    observations: int
    journal_tip_sha256: str
    behavior: HprlBehaviorReport
    reasons: tuple[str, ...]


def qualify_r3_behavior(
    journal: JsonlR3BehaviorJournal,
    *,
    policy: HprlBehaviorPolicy | None = None,
) -> R3BehaviorQualification:
    state = journal.load()
    rows = tuple(record.row for record in state.records)
    model_only = bool(rows) and all(row.model_target_feed for row in rows)
    market_bound = bool(rows) and all(bool(row.market_evidence_sha256) for row in rows)
    behavior = analyze_hprl_position_behavior((row.observation for row in rows), policy=policy)
    reasons = list(state.reasons)
    if not model_only:
        reasons.append("BEHAVIOR_R3_MODEL_TARGET_FEED_REQUIRED")
    if not market_bound:
        reasons.append("BEHAVIOR_R3_REAL_MARKET_BINDING_REQUIRED")
    model_ids = {row.model_id for row in rows}
    if len(model_ids) != 1:
        reasons.append("BEHAVIOR_R3_SINGLE_MODEL_REQUIRED")
    reasons.extend(behavior.reasons)
    return R3BehaviorQualification(
        passed=not reasons,
        journal_valid=state.valid,
        model_target_only=model_only,
        market_bound=market_bound,
        observations=len(rows),
        journal_tip_sha256=state.tip_sha256,
        behavior=behavior,
        reasons=tuple(dict.fromkeys(reasons)),
    )
