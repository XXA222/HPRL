"""FreqAI-to-Hedge target, signal and model-readiness contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from json import dumps
from typing import Any, Mapping, Sequence

from .models import (
    AdmissionCode,
    AdmissionDecision,
    ModelReadinessSnapshot,
    NativeOrderIntent,
    ONE,
    ZERO,
    finite_decimal,
    utc_datetime,
)


@dataclass(frozen=True, slots=True)
class HedgeFreqAITarget:
    """One timestamp-aligned supervised target for dual-leg planning."""

    long_score: Decimal
    short_score: Decimal
    target_net_ratio: Decimal
    target_gross_ratio: Decimal
    funding_edge: Decimal = ZERO
    risk_confidence: Decimal = ONE

    def __post_init__(self) -> None:
        for name in (
            "long_score", "short_score", "target_net_ratio", "target_gross_ratio",
            "funding_edge", "risk_confidence",
        ):
            value = finite_decimal(getattr(self, name), field_name=name)
            object.__setattr__(self, name, value)
        if not ZERO <= self.long_score <= ONE or not ZERO <= self.short_score <= ONE:
            raise ValueError("long_score and short_score must be in [0, 1]")
        if not -ONE <= self.target_net_ratio <= ONE:
            raise ValueError("target_net_ratio must be in [-1, 1]")
        if self.target_gross_ratio < ZERO:
            raise ValueError("target_gross_ratio cannot be negative")
        if not ZERO <= self.risk_confidence <= ONE:
            raise ValueError("risk_confidence must be in [0, 1]")

    def to_columns(self) -> dict[str, float]:
        return {
            "&-hedge_long_score": float(self.long_score),
            "&-hedge_short_score": float(self.short_score),
            "&-hedge_target_net_ratio": float(self.target_net_ratio),
            "&-hedge_target_gross_ratio": float(self.target_gross_ratio),
            "&-hedge_funding_edge": float(self.funding_edge),
            "&-hedge_risk_confidence": float(self.risk_confidence),
        }


@dataclass(frozen=True, slots=True)
class HedgeSignalEnvelope:
    pair: str
    timestamp: datetime
    long_score: Decimal
    short_score: Decimal
    target_net_ratio: Decimal | None
    target_gross_ratio: Decimal | None
    confidence: Decimal
    risk_scale: Decimal
    model_version: str
    feature_schema: str
    candle_fingerprint: str
    producer_id: str = "local-freqai"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair", str(self.pair).strip().upper())
        object.__setattr__(self, "timestamp", utc_datetime(self.timestamp))
        for name in ("long_score", "short_score", "confidence", "risk_scale"):
            value = finite_decimal(getattr(self, name), field_name=name)
            object.__setattr__(self, name, value)
        for name in ("target_net_ratio", "target_gross_ratio"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, finite_decimal(value, field_name=name))
        if not ZERO <= self.long_score <= ONE or not ZERO <= self.short_score <= ONE:
            raise ValueError("signal scores must be in [0, 1]")
        if not ZERO <= self.confidence <= ONE or self.risk_scale < ZERO:
            raise ValueError("confidence/risk_scale out of range")
        if self.target_net_ratio is not None and not -ONE <= self.target_net_ratio <= ONE:
            raise ValueError("target_net_ratio must be in [-1, 1]")
        if self.target_gross_ratio is not None and self.target_gross_ratio < ZERO:
            raise ValueError("target_gross_ratio cannot be negative")
        for name in ("model_version", "feature_schema", "candle_fingerprint", "producer_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def evidence_hash(self) -> str:
        payload = {
            "pair": self.pair,
            "timestamp": self.timestamp.isoformat(),
            "long_score": str(self.long_score),
            "short_score": str(self.short_score),
            "target_net_ratio": None if self.target_net_ratio is None else str(self.target_net_ratio),
            "target_gross_ratio": None if self.target_gross_ratio is None else str(self.target_gross_ratio),
            "confidence": str(self.confidence),
            "risk_scale": str(self.risk_scale),
            "model_version": self.model_version,
            "feature_schema": self.feature_schema,
            "candle_fingerprint": self.candle_fingerprint,
            "producer_id": self.producer_id,
            "metadata": dict(self.metadata),
        }
        return sha256(dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class HedgeModelManifest:
    model_version: str
    trained_at: datetime
    expires_at: datetime
    feature_schema: str
    target_schema: str = "hedge-freqai-target-v1"
    training_data_hash: str = ""
    model_file_hash: str = ""
    compatible_pairs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.model_version).strip() or not str(self.feature_schema).strip():
            raise ValueError("model_version and feature_schema are required")
        trained = utc_datetime(self.trained_at)
        expires = utc_datetime(self.expires_at)
        if expires <= trained:
            raise ValueError("model expiry must be after training")
        object.__setattr__(self, "trained_at", trained)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "compatible_pairs", tuple(str(item).upper() for item in self.compatible_pairs))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def with_ttl(
        cls,
        *,
        model_version: str,
        trained_at: datetime,
        ttl: timedelta,
        feature_schema: str,
        **kwargs: Any,
    ) -> "HedgeModelManifest":
        if ttl <= timedelta(0):
            raise ValueError("model TTL must be positive")
        trained = utc_datetime(trained_at)
        return cls(model_version, trained, trained + ttl, feature_schema, **kwargs)


class HedgeModelReadinessGate:
    """Fail closed when FreqAI evidence is stale, incompatible or absent."""

    def __init__(
        self,
        manifest: HedgeModelManifest | None = None,
        *,
        required: bool = False,
        expected_feature_schema: str | None = None,
        maximum_signal_age: timedelta = timedelta(minutes=2),
    ) -> None:
        if maximum_signal_age <= timedelta(0):
            raise ValueError("maximum_signal_age must be positive")
        self.manifest = manifest
        self.required = bool(required)
        self.expected_feature_schema = expected_feature_schema
        self.maximum_signal_age = maximum_signal_age
        self.latest_signal: HedgeSignalEnvelope | None = None

    def update_manifest(self, manifest: HedgeModelManifest | None) -> None:
        self.manifest = manifest

    def observe(self, signal: HedgeSignalEnvelope) -> None:
        manifest = self.manifest
        if manifest is not None:
            if signal.model_version != manifest.model_version:
                raise ValueError("signal model version does not match manifest")
            if signal.feature_schema != manifest.feature_schema:
                raise ValueError("signal feature schema does not match manifest")
            if manifest.compatible_pairs and signal.pair not in manifest.compatible_pairs:
                raise ValueError("signal pair is not declared compatible")
        self.latest_signal = signal

    def snapshot(
        self,
        *,
        pair: str | None = None,
        at: datetime | None = None,
        candle_fingerprint: str | None = None,
    ) -> ModelReadinessSnapshot:
        now = utc_datetime(at)
        reasons: list[str] = []
        manifest = self.manifest
        signal = self.latest_signal
        if manifest is None:
            if self.required:
                reasons.append("MODEL_MANIFEST_MISSING")
        else:
            if now >= manifest.expires_at:
                reasons.append("MODEL_EXPIRED")
            if self.expected_feature_schema and manifest.feature_schema != self.expected_feature_schema:
                reasons.append("FEATURE_SCHEMA_MISMATCH")
            if pair and manifest.compatible_pairs and pair.upper() not in manifest.compatible_pairs:
                reasons.append("PAIR_NOT_COMPATIBLE")
        if self.required and signal is None:
            reasons.append("SIGNAL_MISSING")
        if signal is not None:
            if now - signal.timestamp > self.maximum_signal_age:
                reasons.append("SIGNAL_STALE")
            if pair and signal.pair != pair.upper():
                reasons.append("SIGNAL_PAIR_MISMATCH")
            if candle_fingerprint and signal.candle_fingerprint != candle_fingerprint:
                reasons.append("CANDLE_FINGERPRINT_MISMATCH")
        return ModelReadinessSnapshot(
            ready=not reasons,
            model_version="" if manifest is None else manifest.model_version,
            trained_at=None if manifest is None else manifest.trained_at,
            expires_at=None if manifest is None else manifest.expires_at,
            feature_schema="" if manifest is None else manifest.feature_schema,
            reasons=tuple(reasons),
        )

    def admit(self, intent: NativeOrderIntent) -> AdmissionDecision:
        if intent.reduce_only:
            return AdmissionDecision.allow(reason="MODEL_GATE_REDUCE_ONLY", reduce_only_exempt=True)
        snapshot = self.snapshot(pair=intent.pair)
        if snapshot.ready:
            return AdmissionDecision.allow(reason="MODEL_READY")
        return AdmissionDecision.block(
            AdmissionCode.MODEL_NOT_READY,
            ";".join(snapshot.reasons) or "model is not ready",
        )


class HedgeFreqAISignalAdapter:
    """Extract a strict signal envelope from the final analyzed dataframe row."""

    COLUMN_ALIASES: Mapping[str, Sequence[str]] = {
        "long_score": ("hedge_long_score", "&-hedge_long_score", "enter_long"),
        "short_score": ("hedge_short_score", "&-hedge_short_score", "enter_short"),
        "target_net_ratio": ("hedge_target_net_ratio", "&-hedge_target_net_ratio"),
        "target_gross_ratio": ("hedge_target_gross_ratio", "&-hedge_target_gross_ratio"),
        "confidence": ("hedge_confidence", "&-hedge_risk_confidence", "do_predict"),
        "risk_scale": ("hedge_risk_scale",),
        "model_version": ("hedge_model_version", "model_version"),
        "feature_schema": ("hedge_feature_schema", "feature_schema"),
    }

    @staticmethod
    def _lookup(row: Mapping[str, Any], aliases: Sequence[str], default: Any = None) -> Any:
        for name in aliases:
            value = row.get(name)
            if value is not None:
                return value
        return default

    def from_row(
        self,
        row: Mapping[str, Any],
        *,
        pair: str,
        timestamp: datetime,
        candle_fingerprint: str,
        producer_id: str = "local-freqai",
        default_model_version: str = "strategy-signal",
        default_feature_schema: str = "strategy-columns-v1",
    ) -> HedgeSignalEnvelope:
        long_score = self._lookup(row, self.COLUMN_ALIASES["long_score"], ZERO)
        short_score = self._lookup(row, self.COLUMN_ALIASES["short_score"], ZERO)
        target_net = self._lookup(row, self.COLUMN_ALIASES["target_net_ratio"])
        target_gross = self._lookup(row, self.COLUMN_ALIASES["target_gross_ratio"])
        confidence = self._lookup(row, self.COLUMN_ALIASES["confidence"], ONE)
        # FreqAI do_predict uses 1 for valid, 0/-1 for invalid. Clamp to [0,1].
        confidence = min(max(Decimal(str(confidence)), ZERO), ONE)
        return HedgeSignalEnvelope(
            pair=pair,
            timestamp=timestamp,
            long_score=min(max(Decimal(str(long_score)), ZERO), ONE),
            short_score=min(max(Decimal(str(short_score)), ZERO), ONE),
            target_net_ratio=None if target_net is None else Decimal(str(target_net)),
            target_gross_ratio=None if target_gross is None else Decimal(str(target_gross)),
            confidence=confidence,
            risk_scale=Decimal(str(self._lookup(row, self.COLUMN_ALIASES["risk_scale"], ONE))),
            model_version=str(self._lookup(row, self.COLUMN_ALIASES["model_version"], default_model_version)),
            feature_schema=str(self._lookup(row, self.COLUMN_ALIASES["feature_schema"], default_feature_schema)),
            candle_fingerprint=candle_fingerprint,
            producer_id=producer_id,
        )


def manifest_from_mapping(values: Mapping[str, Any]) -> HedgeModelManifest:
    """Load a strict manifest from JSON-compatible values."""
    return HedgeModelManifest(
        model_version=str(values["model_version"]),
        trained_at=datetime.fromisoformat(str(values["trained_at"])),
        expires_at=datetime.fromisoformat(str(values["expires_at"])),
        feature_schema=str(values["feature_schema"]),
        target_schema=str(values.get("target_schema", "hedge-freqai-target-v1")),
        training_data_hash=str(values.get("training_data_hash", "")),
        model_file_hash=str(values.get("model_file_hash", "")),
        compatible_pairs=tuple(str(item) for item in values.get("compatible_pairs", ())),
        metadata=dict(values.get("metadata", {})),
    )
