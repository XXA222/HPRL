from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from freqtrade.hedge.exchange.base import CalibrationKind, Clock, SystemClock


DEFAULT_DRIFT_TOLERANCE = Decimal("0.00000001")


@dataclass(frozen=True, slots=True)
class SoakObservation:
    observed_at: datetime
    service_state: str
    stream_age_seconds: float | None
    fast_diff_count: int
    full_diff_count: int
    wallet_delta: Decimal
    funding_income: Decimal
    realized_pnl: Decimal
    commissions: Decimal
    transfers: Decimal
    funding_period_observed: bool = False
    funding_event_count_delta: int = 0
    run_id: str | None = None
    account_id: str | None = None
    baseline_id: str | None = None
    contracts_version: str = "2.0"

    def __post_init__(self) -> None:
        normalized_state = str(self.service_state).strip().upper()
        if not normalized_state:
            raise ValueError("service_state is required")
        object.__setattr__(self, "service_state", normalized_state)
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        if self.stream_age_seconds is not None and (
            not math.isfinite(self.stream_age_seconds)
            or self.stream_age_seconds < 0
        ):
            raise ValueError("stream_age_seconds must be finite, nonnegative or None")
        if self.fast_diff_count < 0 or self.full_diff_count < 0:
            raise ValueError("diff counts must be nonnegative")
        if not isinstance(self.funding_period_observed, bool):
            raise ValueError("funding_period_observed must be a boolean")
        if self.funding_event_count_delta < 0:
            raise ValueError("funding_event_count_delta must be nonnegative")
        for name in (
            "wallet_delta",
            "funding_income",
            "realized_pnl",
            "commissions",
            "transfers",
        ):
            if not getattr(self, name).is_finite():
                raise ValueError(f"{name} must be finite")

    @property
    def expected_wallet_delta(self) -> Decimal:
        return (
            self.funding_income
            + self.realized_pnl
            - self.commissions
            + self.transfers
        )

    @property
    def unexplained_drift(self) -> Decimal:
        return self.wallet_delta - self.expected_wallet_delta


@dataclass(frozen=True, slots=True)
class SoakSummary:
    duration_hours: float
    observation_count: int
    expected_observation_count: int
    coverage_ratio: float
    max_gap_seconds: float
    continuous: bool
    non_ready_observation_count: int
    reconciliation_diff_observation_count: int
    unexplained_drift_count: int
    max_abs_unexplained_drift: Decimal
    funding_observation_count: int
    meets_24h: bool
    meets_72h: bool
    passed_24h: bool
    passed_72h: bool


class SoakMonitor:
    def __init__(
        self,
        path: str | Path,
        *,
        drift_tolerance: Decimal = DEFAULT_DRIFT_TOLERANCE,
        expected_interval_seconds: float = 60.0,
        max_gap_multiplier: float = 2.5,
        minimum_coverage_ratio: float = 0.95,
    ) -> None:
        if not drift_tolerance.is_finite() or drift_tolerance < 0:
            raise ValueError("drift_tolerance must be finite and nonnegative")
        if not math.isfinite(expected_interval_seconds) or expected_interval_seconds <= 0:
            raise ValueError("expected_interval_seconds must be finite and positive")
        if not math.isfinite(max_gap_multiplier) or max_gap_multiplier < 1:
            raise ValueError("max_gap_multiplier must be finite and at least 1")
        if not math.isfinite(minimum_coverage_ratio) or not 0 < minimum_coverage_ratio <= 1:
            raise ValueError("minimum_coverage_ratio must be in (0, 1]")
        self.path = Path(path)
        self.drift_tolerance = drift_tolerance
        self.expected_interval_seconds = expected_interval_seconds
        self.max_gap_multiplier = max_gap_multiplier
        self.minimum_coverage_ratio = minimum_coverage_ratio

    def append(self, observation: SoakObservation) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(observation)
        payload["observed_at"] = observation.observed_at.astimezone(UTC).isoformat()
        for key in (
            "wallet_delta",
            "funding_income",
            "realized_pnl",
            "commissions",
            "transfers",
        ):
            payload[key] = str(payload[key])
        payload["expected_wallet_delta"] = str(observation.expected_wallet_delta)
        payload["unexplained_drift"] = str(observation.unexplained_drift)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def load(self) -> tuple[SoakObservation, ...]:
        if not self.path.exists():
            return ()
        observations: list[SoakObservation] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                observed_at = datetime.fromisoformat(str(raw["observed_at"]))
                observation = SoakObservation(
                    observed_at=observed_at,
                    service_state=str(raw["service_state"]).upper(),
                    stream_age_seconds=(
                        None
                        if raw.get("stream_age_seconds") is None
                        else float(raw["stream_age_seconds"])
                    ),
                    fast_diff_count=int(raw.get("fast_diff_count", 0)),
                    full_diff_count=int(raw.get("full_diff_count", 0)),
                    wallet_delta=Decimal(str(raw.get("wallet_delta", "0"))),
                    funding_income=Decimal(str(raw.get("funding_income", "0"))),
                    realized_pnl=Decimal(str(raw.get("realized_pnl", "0"))),
                    commissions=Decimal(str(raw.get("commissions", "0"))),
                    transfers=Decimal(str(raw.get("transfers", "0"))),
                    funding_period_observed=raw.get(
                        "funding_period_observed", False
                    ),
                    funding_event_count_delta=int(
                        raw.get("funding_event_count_delta", 0)
                    ),
                    run_id=raw.get("run_id"),
                    account_id=raw.get("account_id"),
                    baseline_id=raw.get("baseline_id"),
                    contracts_version=str(raw.get("contracts_version", "2.0")),
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                InvalidOperation,
            ) as exc:
                raise ValueError(
                    f"Invalid soak observation at line {line_number}: {exc}"
                ) from exc
            observations.append(observation)
        return tuple(observations)

    def _timeline_metrics(
        self,
        ordered: tuple[SoakObservation, ...],
    ) -> tuple[float, int, float, float, bool]:
        if len(ordered) < 2:
            duration_seconds = 0.0
            gaps: tuple[float, ...] = ()
        else:
            duration_seconds = max(
                0.0,
                (ordered[-1].observed_at - ordered[0].observed_at).total_seconds(),
            )
            gaps = tuple(
                (right.observed_at - left.observed_at).total_seconds()
                for left, right in zip(ordered, ordered[1:], strict=False)
            )
        expected_count = (
            0
            if not ordered
            else max(
                1,
                math.floor(duration_seconds / self.expected_interval_seconds) + 1,
            )
        )
        coverage_ratio = (
            0.0
            if expected_count == 0
            else min(1.0, len(ordered) / expected_count)
        )
        max_gap = max(gaps, default=0.0)
        gap_limit = self.expected_interval_seconds * self.max_gap_multiplier
        continuous = (
            bool(ordered)
            and all(gap > 0 for gap in gaps)
            and (not gaps or max_gap <= gap_limit)
            and coverage_ratio >= self.minimum_coverage_ratio
        )
        return duration_seconds, expected_count, coverage_ratio, max_gap, continuous

    def _quality_metrics(
        self,
        ordered: tuple[SoakObservation, ...],
    ) -> tuple[int, Decimal, int, int, int]:
        drifts = tuple(abs(item.unexplained_drift) for item in ordered)
        drift_count = sum(1 for value in drifts if value > self.drift_tolerance)
        max_drift = max(drifts, default=Decimal("0"))
        funding_count = sum(
            item.funding_event_count_delta
            if item.funding_event_count_delta > 0
            else int(item.funding_period_observed or item.funding_income != 0)
            for item in ordered
        )
        non_ready_count = sum(
            1 for item in ordered if item.service_state != "READY"
        )
        diff_count = sum(
            1
            for item in ordered
            if item.fast_diff_count != 0 or item.full_diff_count != 0
        )
        return drift_count, max_drift, funding_count, non_ready_count, diff_count

    def summarize(
        self,
        observations: Iterable[SoakObservation] | None = None,
    ) -> SoakSummary:
        items = tuple(observations if observations is not None else self.load())
        ordered = tuple(sorted(items, key=lambda item: item.observed_at))
        (
            duration_seconds,
            expected_count,
            coverage_ratio,
            max_gap,
            continuous,
        ) = self._timeline_metrics(ordered)
        (
            drift_count,
            max_drift,
            funding_count,
            non_ready_count,
            diff_count,
        ) = self._quality_metrics(ordered)
        duration_hours = duration_seconds / 3600.0
        clean = (
            continuous
            and drift_count == 0
            and non_ready_count == 0
            and diff_count == 0
        )
        meets_24h = duration_hours >= 24 and continuous
        meets_72h = duration_hours >= 72 and continuous
        return SoakSummary(
            duration_hours=duration_hours,
            observation_count=len(ordered),
            expected_observation_count=expected_count,
            coverage_ratio=coverage_ratio,
            max_gap_seconds=max_gap,
            continuous=continuous,
            non_ready_observation_count=non_ready_count,
            reconciliation_diff_observation_count=diff_count,
            unexplained_drift_count=drift_count,
            max_abs_unexplained_drift=max_drift,
            funding_observation_count=funding_count,
            meets_24h=meets_24h,
            meets_72h=meets_72h,
            passed_24h=meets_24h and clean,
            passed_72h=meets_72h and funding_count >= 1 and clean,
        )


@dataclass(frozen=True, slots=True)
class SoakAccountingTotals:
    wallet_balance: Decimal
    funding_income: Decimal
    realized_pnl: Decimal
    commissions: Decimal
    transfers: Decimal
    funding_event_count: int = 0

    def __post_init__(self) -> None:
        for name in (
            "wallet_balance",
            "funding_income",
            "realized_pnl",
            "commissions",
            "transfers",
        ):
            if not getattr(self, name).is_finite():
                raise ValueError(f"{name} must be finite")
        if self.funding_event_count < 0:
            raise ValueError("funding_event_count must be nonnegative")


class SoakAccountingSource(Protocol):
    async def read_soak_accounting_totals(
        self, account_id: str
    ) -> SoakAccountingTotals: ...


class ReadonlyServiceObservationSource(Protocol):
    @property
    def status(self) -> Any: ...

    def runtime_snapshot(self) -> Any: ...

    def latest_calibration(self, kind: CalibrationKind) -> Any | None: ...


class ReadonlyServiceSoakProvider:
    """Build soak observations from the running service and repository totals."""

    def __init__(
        self,
        *,
        account_id: str,
        service: ReadonlyServiceObservationSource,
        accounting_source: SoakAccountingSource,
        clock: Clock | None = None,
        baseline_path: str | Path | None = None,
        run_id: str | None = None,
    ) -> None:
        normalized_account_id = account_id.strip()
        if not normalized_account_id:
            raise ValueError("account_id is required")
        self.account_id = normalized_account_id
        self.service = service
        self.accounting_source = accounting_source
        self.clock = clock or SystemClock()
        self.baseline_path = None if baseline_path is None else Path(baseline_path)
        self.run_id = run_id
        self._baseline: SoakAccountingTotals | None = self._load_baseline()
        self._last_funding_event_count = (
            None if self._baseline is None else self._baseline.funding_event_count
        )
        self._baseline_id: str | None = None

    def _load_baseline(self) -> SoakAccountingTotals | None:
        if self.baseline_path is None or not self.baseline_path.exists():
            return None
        raw = json.loads(self.baseline_path.read_text(encoding="utf-8"))
        return SoakAccountingTotals(
            wallet_balance=Decimal(str(raw["wallet_balance"])),
            funding_income=Decimal(str(raw["funding_income"])),
            realized_pnl=Decimal(str(raw["realized_pnl"])),
            commissions=Decimal(str(raw["commissions"])),
            transfers=Decimal(str(raw["transfers"])),
            funding_event_count=int(raw.get("funding_event_count", 0)),
        )

    def _save_baseline(self, baseline: SoakAccountingTotals) -> None:
        if self.baseline_path is None:
            return
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "account_id": self.account_id,
            "created_at": self.clock.now().astimezone(UTC).isoformat(),
            "wallet_balance": str(baseline.wallet_balance),
            "funding_income": str(baseline.funding_income),
            "realized_pnl": str(baseline.realized_pnl),
            "commissions": str(baseline.commissions),
            "transfers": str(baseline.transfers),
            "funding_event_count": baseline.funding_event_count,
        }
        self.baseline_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._baseline_id = str(payload["created_at"])

    @staticmethod
    def _diff_count(result: Any | None) -> int:
        return 0 if result is None else int(result.diff_count)

    async def read_observation(self) -> SoakObservation:
        totals = await self.accounting_source.read_soak_accounting_totals(
            self.account_id
        )
        if self._baseline is None:
            self._baseline = totals
            self._last_funding_event_count = totals.funding_event_count
            self._save_baseline(totals)
        baseline = self._baseline
        previous_funding_count = self._last_funding_event_count
        funding_delta = max(
            0,
            totals.funding_event_count
            - (previous_funding_count if previous_funding_count is not None else 0),
        )
        self._last_funding_event_count = totals.funding_event_count
        runtime = self.service.runtime_snapshot()
        fast = self.service.latest_calibration(CalibrationKind.FAST)
        full = self.service.latest_calibration(CalibrationKind.FULL)
        return SoakObservation(
            observed_at=self.clock.now(),
            service_state=runtime.status.state.value,
            stream_age_seconds=runtime.freshness.event_age_seconds,
            fast_diff_count=self._diff_count(fast),
            full_diff_count=self._diff_count(full),
            wallet_delta=totals.wallet_balance - baseline.wallet_balance,
            funding_income=totals.funding_income - baseline.funding_income,
            realized_pnl=totals.realized_pnl - baseline.realized_pnl,
            commissions=totals.commissions - baseline.commissions,
            transfers=totals.transfers - baseline.transfers,
            funding_period_observed=funding_delta > 0,
            funding_event_count_delta=funding_delta,
            run_id=self.run_id,
            account_id=self.account_id,
            baseline_id=self._baseline_id,
        )


class SoakObservationProvider(Protocol):
    async def read_observation(self) -> SoakObservation: ...


class SoakRunner:
    def __init__(
        self,
        *,
        provider: SoakObservationProvider,
        monitor: SoakMonitor,
        interval_seconds: float = 60.0,
        clock: Clock | None = None,
    ) -> None:
        if interval_seconds < 1:
            raise ValueError("interval_seconds must be at least 1")
        self.provider = provider
        self.monitor = monitor
        self.interval_seconds = interval_seconds
        self.clock = clock or SystemClock()

    async def run(self, *, duration_hours: float, stop_event=None) -> SoakSummary:
        if not math.isfinite(duration_hours) or duration_hours <= 0:
            raise ValueError("duration_hours must be finite and positive")
        deadline = self.clock.monotonic() + duration_hours * 3600.0
        current_run: list[SoakObservation] = []
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            observation = await self.provider.read_observation()
            self.monitor.append(observation)
            current_run.append(observation)
            remaining = deadline - self.clock.monotonic()
            if remaining <= 0:
                break
            await self.clock.sleep(min(self.interval_seconds, remaining))
        return self.monitor.summarize(current_run)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Binance hedge read-only soak observations"
    )
    parser.add_argument("path")
    parser.add_argument("--require-hours", type=float, default=24.0)
    parser.add_argument("--require-funding", action="store_true")
    parser.add_argument("--expected-interval-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.require_hours <= 0:
        parser.error("--require-hours must be positive")
    monitor = SoakMonitor(
        args.path, expected_interval_seconds=args.expected_interval_seconds
    )
    summary = monitor.summarize()
    print(
        json.dumps(
            {
                **asdict(summary),
                "max_abs_unexplained_drift": str(
                    summary.max_abs_unexplained_drift
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_hours >= 72:
        passed = summary.passed_72h
    elif args.require_hours >= 24:
        passed = summary.passed_24h
    else:
        passed = (
            summary.duration_hours >= args.require_hours
            and summary.continuous
            and summary.non_ready_observation_count == 0
            and summary.reconciliation_diff_observation_count == 0
            and summary.unexplained_drift_count == 0
        )
    if args.require_funding:
        passed = passed and summary.funding_observation_count >= 1
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
