"""HPRL dual-leg replay/backtest integration over the canonical Hedge simulator.

No second trading simulator is introduced.  HPRL targets are converted to ordinary
SignalEvent values, then replayed by EventReplayEngine and HedgeBacktestRunner.  This is
important for semantic parity: fees, funding, cross-wallet margin, partial fills,
liquidation, target planning and next-bar activation remain identical to the non-HPRL
Hedge path.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Iterable, Sequence

from freqtrade.hedge.backtesting.contracts import (
    BacktestDataset,
    BacktestEvaluation,
    Candidate,
    EngineConfig,
    ObjectiveConfig,
)
from freqtrade.hedge.backtesting.dataset import build_dataset
from freqtrade.hedge.backtesting.decimal_utils import canonical_json
from freqtrade.hedge.backtesting.runner import HedgeBacktestRunner
from freqtrade.hedge.hprl.contracts import PlannedExecutionIntent
from freqtrade.hedge.simulation.exchange import BarEvent, FundingEvent, SignalEvent, SimulationInputEvent

from .hprl_hedge_adapter import HprlHedgeAdapter, HprlTargetProjection


@dataclass(frozen=True, slots=True)
class TimedHprlTarget:
    timestamp: datetime
    sequence: int
    intent: PlannedExecutionIntent
    allow_new_risk: bool = True

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("target timestamp must be timezone-aware")
        if self.sequence < 0:
            raise ValueError("target sequence must be nonnegative")
        if not isinstance(self.intent, PlannedExecutionIntent):
            raise TypeError("intent must be PlannedExecutionIntent")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(UTC))


@dataclass(frozen=True, slots=True)
class HprlReplayBuildReport:
    target_count: int
    accepted_targets: int
    rejected_targets: int
    first_sequence: int | None
    last_sequence: int | None
    planner_profile_sha256: str
    dataset_fingerprint: str
    projection_chain_sha256: str
    rejection_reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.target_count > 0 and self.rejected_targets == 0


@dataclass(frozen=True, slots=True)
class HprlReplayBundle:
    dataset: BacktestDataset
    projections: tuple[HprlTargetProjection, ...]
    report: HprlReplayBuildReport


@dataclass(frozen=True, slots=True)
class HprlReplayParityReport:
    passed: bool
    first_state_sha256: str
    second_state_sha256: str
    first_event_sha256: str
    second_event_sha256: str
    final_long_quantity: str
    final_short_quantity: str
    final_equity: str


def _projection_chain_hash(projections: Sequence[HprlTargetProjection]) -> str:
    payload = [
        {
            "sequence": item.sequence,
            "observed_at": item.observed_at.isoformat(),
            "symbol": item.symbol,
            "model_id": item.model_id,
            "long_margin_ratio": str(item.long_margin_ratio),
            "short_margin_ratio": str(item.short_margin_ratio),
            "long_notional_ratio": str(item.long_notional_ratio),
            "short_notional_ratio": str(item.short_notional_ratio),
            "accepted": item.accepted,
            "reasons": list(item.reasons),
            "source_sha256": item.source_sha256,
        }
        for item in projections
    ]
    return sha256(canonical_json(payload)).hexdigest()


def _result_state_hash(result) -> str:
    payload = {
        "report": result.report,
        "snapshots": result.snapshots,
    }
    return sha256(canonical_json(payload)).hexdigest()


def _result_event_hash(result) -> str:
    return sha256(canonical_json(result.events)).hexdigest()


class HprlReplayDatasetBuilder:
    def __init__(self, adapter: HprlHedgeAdapter) -> None:
        if not isinstance(adapter, HprlHedgeAdapter):
            raise TypeError("adapter must be HprlHedgeAdapter")
        self.adapter = adapter

    def build(
        self,
        *,
        base_dataset: BacktestDataset,
        targets: Iterable[TimedHprlTarget],
        dataset_id: str | None = None,
        reject_invalid_targets: bool = True,
    ) -> HprlReplayBundle:
        materialized = tuple(targets)
        if not materialized:
            raise ValueError("at least one HPRL target is required")
        if any(item.intent.symbol != base_dataset.symbol for item in materialized):
            raise ValueError("HPRL target symbol must match replay dataset symbol")
        ordered = tuple(sorted(materialized, key=lambda item: (item.timestamp, item.sequence)))
        sequences = [item.sequence for item in ordered]
        if len(sequences) != len(set(sequences)) or sequences != sorted(sequences):
            raise ValueError("HPRL target sequences must be unique and monotonic")
        bar_times = {
            item.timestamp
            for item in base_dataset.events
            if isinstance(item, BarEvent)
        }
        missing = [item.timestamp for item in ordered if item.timestamp not in bar_times]
        if missing:
            raise ValueError(
                "every HPRL target must align to a replay bar timestamp: "
                + ",".join(value.isoformat() for value in missing[:5])
            )

        projections: list[HprlTargetProjection] = []
        signals: list[SignalEvent] = []
        previous: HprlTargetProjection | None = None
        rejection_reasons: list[str] = []
        for item in ordered:
            projection = self.adapter.adapt(
                item.intent,
                sequence=item.sequence,
                observed_at=item.timestamp,
                now=item.timestamp,
                previous=previous,
            )
            projections.append(projection)
            if projection.accepted:
                previous = projection
            else:
                rejection_reasons.extend(projection.reasons)
                if reject_invalid_targets:
                    raise ValueError(
                        f"HPRL target sequence {item.sequence} rejected: "
                        + ",".join(projection.reasons)
                    )
            signals.append(
                self.adapter.to_signal_event(
                    projection,
                    allow_new_risk=item.allow_new_risk,
                )
            )

        # Replace any pre-existing strategy signals; bars and funding facts remain unchanged.
        facts: list[SimulationInputEvent] = [
            item for item in base_dataset.events if not isinstance(item, SignalEvent)
        ]
        facts.extend(signals)
        facts.sort(
            key=lambda item: (
                item.timestamp,
                0 if isinstance(item, SignalEvent) else 1 if isinstance(item, FundingEvent) else 2,
            )
        )
        rebuilt = build_dataset(
            events=facts,
            dataset_id=dataset_id or f"{base_dataset.dataset_id}:hprl-v3",
            timeframe=base_dataset.timeframe,
            metadata={
                **dict(base_dataset.metadata),
                "hprl_v3": "true",
                "source_dataset_fingerprint": base_dataset.fingerprint,
                "hprl_target_unit": self.adapter.policy.target_unit.value,
            },
        )
        profile = self.adapter.planner_profile()
        rejected = sum(not item.accepted for item in projections)
        report = HprlReplayBuildReport(
            target_count=len(projections),
            accepted_targets=len(projections) - rejected,
            rejected_targets=rejected,
            first_sequence=projections[0].sequence if projections else None,
            last_sequence=projections[-1].sequence if projections else None,
            planner_profile_sha256=profile.semantic_sha256,
            dataset_fingerprint=rebuilt.fingerprint,
            projection_chain_sha256=_projection_chain_hash(projections),
            rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
        )
        return HprlReplayBundle(rebuilt, tuple(projections), report)


class HprlReplayBacktestRunner:
    """Thin wrapper around the existing deterministic HedgeBacktestRunner."""

    def __init__(
        self,
        *,
        bundle: HprlReplayBundle,
        adapter: HprlHedgeAdapter,
        engine_config: EngineConfig | None = None,
        objective_config: ObjectiveConfig | None = None,
        periods_per_year: int = 365 * 24 * 60,
    ) -> None:
        self.bundle = bundle
        self.adapter = adapter
        self.engine_config = engine_config or EngineConfig(leverage=adapter.policy.leverage)
        self.objective_config = objective_config or ObjectiveConfig()
        self.periods_per_year = periods_per_year

    def _runner(self) -> HedgeBacktestRunner:
        return HedgeBacktestRunner(
            dataset=self.bundle.dataset,
            engine_config=self.engine_config,
            planner_config=self.adapter.planner_profile().planner_config,
            objective_config=self.objective_config,
            periods_per_year=self.periods_per_year,
        )

    def evaluate(self, candidate: Candidate | None = None) -> BacktestEvaluation:
        selected = candidate or Candidate("hprl-v3-exact-targets", {})
        return self._runner().evaluate(selected)

    def parity(self, candidate: Candidate | None = None) -> HprlReplayParityReport:
        selected = candidate or Candidate("hprl-v3-parity", {})
        first = self._runner().evaluate(selected)
        second = self._runner().evaluate(selected)
        if first.result is None or second.result is None:  # pragma: no cover
            raise RuntimeError("HPRL replay returned no simulation result")
        first_state = _result_state_hash(first.result)
        second_state = _result_state_hash(second.result)
        first_events = _result_event_hash(first.result)
        second_events = _result_event_hash(second.result)
        report = first.result.report
        return HprlReplayParityReport(
            passed=first_state == second_state and first_events == second_events,
            first_state_sha256=first_state,
            second_state_sha256=second_state,
            first_event_sha256=first_events,
            second_event_sha256=second_events,
            final_long_quantity=str(report.get("final_long_quantity", "0")),
            final_short_quantity=str(report.get("final_short_quantity", "0")),
            final_equity=str(report.get("final_equity", "0")),
        )
