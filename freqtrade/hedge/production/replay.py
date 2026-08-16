"""Recorded-fact replay manifest and deterministic semantic hashing."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class RecordedFact:
    sequence: int
    observed_at: datetime
    stream: str
    event_type: str
    identity: str
    payload_sha256: str

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be nonnegative")
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        if not self.stream.strip() or not self.event_type.strip() or not self.identity.strip():
            raise ValueError("stream, event_type and identity are required")
        digest = self.payload_sha256.lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("payload_sha256 must be sha256")
        object.__setattr__(self, "payload_sha256", digest)
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))


@dataclass(frozen=True, slots=True)
class ReplayManifest:
    exchange: str
    account_fingerprint: str
    symbols: tuple[str, ...]
    started_at: datetime
    ended_at: datetime
    facts: tuple[RecordedFact, ...]
    feature_schema_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.started_at.tzinfo is None or self.ended_at.tzinfo is None:
            raise ValueError("replay timestamps must be timezone-aware")
        if self.ended_at < self.started_at:
            raise ValueError("ended_at cannot precede started_at")
        if not self.exchange.strip() or not self.account_fingerprint.strip():
            raise ValueError("exchange/account_fingerprint required")
        if not self.symbols:
            raise ValueError("at least one symbol is required")
        if tuple(sorted(set(self.symbols))) != tuple(sorted(self.symbols)):
            raise ValueError("symbols must be unique")
        sequences = [item.sequence for item in self.facts]
        if sequences != sorted(sequences):
            raise ValueError("facts must be sequence ordered")
        if len(sequences) != len(set(sequences)):
            raise ValueError("duplicate recorded fact sequence")
        if self.feature_schema_sha256 is not None:
            digest = self.feature_schema_sha256.lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("feature_schema_sha256 must be sha256")
            object.__setattr__(self, "feature_schema_sha256", digest)

    @property
    def sequence_gaps(self) -> tuple[tuple[int, int], ...]:
        gaps: list[tuple[int, int]] = []
        for left, right in zip(self.facts, self.facts[1:], strict=False):
            if right.sequence != left.sequence + 1:
                gaps.append((left.sequence, right.sequence))
        return tuple(gaps)

    @property
    def timestamp_regressions(self) -> tuple[tuple[int, int], ...]:
        regressions: list[tuple[int, int]] = []
        for left, right in zip(self.facts, self.facts[1:], strict=False):
            if right.observed_at < left.observed_at:
                regressions.append((left.sequence, right.sequence))
        return tuple(regressions)

    @property
    def duplicate_identities(self) -> tuple[tuple[str, str, str], ...]:
        seen: set[tuple[str, str, str]] = set()
        duplicate: set[tuple[str, str, str]] = set()
        for fact in self.facts:
            key = (fact.stream, fact.event_type, fact.identity)
            if key in seen:
                duplicate.add(key)
            seen.add(key)
        return tuple(sorted(duplicate))

    @property
    def semantic_hash(self) -> str:
        payload = {
            "exchange": self.exchange,
            "account_fingerprint": self.account_fingerprint,
            "symbols": sorted(self.symbols),
            "started_at": self.started_at.astimezone(UTC).isoformat(),
            "ended_at": self.ended_at.astimezone(UTC).isoformat(),
            "feature_schema_sha256": self.feature_schema_sha256,
            "facts": [
                {
                    "sequence": f.sequence,
                    "observed_at": f.observed_at.isoformat(),
                    "stream": f.stream,
                    "event_type": f.event_type,
                    "identity": f.identity,
                    "payload_sha256": f.payload_sha256,
                }
                for f in self.facts
            ],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        return sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class ReplayComparison:
    passed: bool
    expected_semantic_hash: str
    actual_semantic_hash: str
    gaps: tuple[tuple[int, int], ...]
    state_hash_match: bool
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReplayIntegrityPolicy:
    required_streams: frozenset[str] = frozenset({"market", "user", "account"})
    required_event_types: frozenset[str] = frozenset({"ORDER", "POSITION", "BALANCE"})
    allow_duplicate_identities: bool = False
    require_monotonic_timestamps: bool = True
    require_feature_schema: bool = True


@dataclass(frozen=True, slots=True)
class ReplayIntegrityResult:
    passed: bool
    reasons: tuple[str, ...]
    streams: tuple[str, ...]
    event_types: tuple[str, ...]
    facts: int


def evaluate_replay_integrity(
    manifest: ReplayManifest,
    policy: ReplayIntegrityPolicy | None = None,
) -> ReplayIntegrityResult:
    policy = policy or ReplayIntegrityPolicy()
    reasons: list[str] = []
    streams = frozenset(x.stream for x in manifest.facts)
    event_types = frozenset(x.event_type for x in manifest.facts)
    if not manifest.facts:
        reasons.append("REPLAY_HAS_NO_FACTS")
    missing_streams = sorted(policy.required_streams - streams)
    if missing_streams:
        reasons.extend(f"MISSING_STREAM:{x}" for x in missing_streams)
    missing_types = sorted(policy.required_event_types - event_types)
    if missing_types:
        reasons.extend(f"MISSING_EVENT_TYPE:{x}" for x in missing_types)
    if manifest.sequence_gaps:
        reasons.append("SEQUENCE_GAPS")
    if policy.require_monotonic_timestamps and manifest.timestamp_regressions:
        reasons.append("TIMESTAMP_REGRESSION")
    if not policy.allow_duplicate_identities and manifest.duplicate_identities:
        reasons.append("DUPLICATE_FACT_IDENTITY")
    if policy.require_feature_schema and manifest.feature_schema_sha256 is None:
        reasons.append("FEATURE_SCHEMA_PROVENANCE_MISSING")
    return ReplayIntegrityResult(
        not reasons,
        tuple(reasons),
        tuple(sorted(streams)),
        tuple(sorted(event_types)),
        len(manifest.facts),
    )


def compare_replay(
    manifest: ReplayManifest,
    *,
    expected_semantic_hash: str,
    actual_state_hash: str,
    expected_state_hash: str,
    assumptions: Iterable[str] = (),
) -> ReplayComparison:
    gaps = manifest.sequence_gaps
    state_match = actual_state_hash == expected_state_hash
    passed = (
        manifest.semantic_hash == expected_semantic_hash
        and state_match
        and not gaps
        and not manifest.timestamp_regressions
    )
    return ReplayComparison(
        passed,
        expected_semantic_hash,
        manifest.semantic_hash,
        gaps,
        state_match,
        tuple(str(x) for x in assumptions),
    )


def payload_hash(payload: Mapping[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return sha256(raw).hexdigest()
