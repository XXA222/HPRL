from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from freqtrade.hedge.exchange.base import (
    CalibrationKind,
    CalibrationResult,
    Clock,
    Direction2HealthFact,
    ReadonlyAccountView,
    ReadonlyReasonCode,
    ReadonlyState,
    StreamHealth,
    SystemClock,
    maybe_await,
)
from freqtrade.hedge.exchange.binance_readonly import BinanceReadonlyClient, PermissionPolicy
from freqtrade.hedge.exchange.binance_user_stream import BinanceUserStream
from freqtrade.hedge.readonly.calibration import ReadonlyCalibration, ReadonlySafetyHalt
from freqtrade.hedge.readonly.freshness import FreshnessAssessment, UserStreamFreshness
from freqtrade.hedge.readonly.scheduler import ReconciliationScheduler


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    state: ReadonlyState
    reason: str
    changed_at: datetime
    last_calibration_run_id: str | None


@dataclass(frozen=True, slots=True)
class ReadonlyRuntimeSnapshot:
    status: ServiceStatus
    stream_health: StreamHealth
    freshness: FreshnessAssessment
    last_fast_calibration: CalibrationResult | None
    last_full_calibration: CalibrationResult | None
    last_reconnect_calibration: CalibrationResult | None
    direction2_health: Direction2HealthFact
    transport_telemetry: Any | None = None
    clock_sync_status: Any | None = None
    listen_key_generation: int | None = None
    listen_key_created_at: datetime | None = None
    listen_key_renewed_at: datetime | None = None
    listen_key_expires_at: datetime | None = None
    account_view: ReadonlyAccountView | None = None



class BinanceReadonlyService:
    """Coordinate preflight, REST facts, stream and reconciliation without writes."""

    def __init__(
        self,
        *,
        client: BinanceReadonlyClient,
        calibration: ReadonlyCalibration,
        stream: BinanceUserStream,
        scheduler: ReconciliationScheduler | None = None,
        freshness: UserStreamFreshness | None = None,
        permission_policy: PermissionPolicy | None = None,
        clock: Clock | None = None,
        drift_verification_attempts: int = 1,
        target_leverage: int | None = None,
    ) -> None:
        self.client = client
        self.calibration = calibration
        self.stream = stream
        self.scheduler = scheduler
        self.freshness = freshness or UserStreamFreshness()
        self.permission_policy = permission_policy or PermissionPolicy()
        if drift_verification_attempts < 0:
            raise ValueError("drift_verification_attempts must be nonnegative")
        self.clock = clock or SystemClock()
        self.drift_verification_attempts = drift_verification_attempts
        if target_leverage is not None and int(target_leverage) <= 0:
            raise ValueError("target_leverage must be positive when configured")
        self.target_leverage = None if target_leverage is None else int(target_leverage)
        self._status = ServiceStatus(
            ReadonlyState.STARTING, "NOT_STARTED", self.clock.now(), None
        )
        self._recovery_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []
        self._supervisor_tasks: set[asyncio.Task[Any]] = set()
        self._started = False
        self._stopping = False
        self._freshness_action_pending = False
        self._last_calibration_by_kind: dict[
            CalibrationKind, CalibrationResult
        ] = {}
        self.stream.set_callbacks(
            on_connected=self.on_stream_connected,
            on_disconnected=self.on_stream_disconnected,
            on_integrity_fault=self.on_integrity_fault,
            on_recalibration_required=self.on_recalibration_required,
            on_event=self.on_stream_event,
        )
        if self.scheduler is not None:
            self.scheduler.set_callbacks(
                on_result=self.on_scheduled_calibration,
                on_error=self.on_scheduler_error,
            )

    @property
    def status(self) -> ServiceStatus:
        return self._status

    def _set_state(
        self, state: ReadonlyState, reason: str, run_id: str | None = None
    ) -> None:
        self._status = ServiceStatus(
            state,
            reason,
            self.clock.now(),
            run_id or self._status.last_calibration_run_id,
        )

    @staticmethod
    def _seed_argument_count(method: Any) -> int | None:
        try:
            parameters = inspect.signature(method).parameters.values()
        except (TypeError, ValueError):
            return None
        if any(item.kind is inspect.Parameter.VAR_POSITIONAL for item in parameters):
            return 3
        return sum(
            1
            for item in parameters
            if item.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        )

    async def _call_stream_seed(self, method: Any, bundle: Any) -> None:
        positions = bundle.positions
        orders = bundle.open_orders
        balances = getattr(bundle, "balances", ())
        argument_count = self._seed_argument_count(method)
        if argument_count is not None and argument_count < 3:
            await maybe_await(method(positions, orders))
            return
        await maybe_await(method(positions, orders, balances))

    def _record_calibration(
        self, result: CalibrationResult
    ) -> CalibrationResult:
        self._last_calibration_by_kind[result.kind] = result
        return result

    async def _run_calibration(
        self, kind: CalibrationKind
    ) -> CalibrationResult:
        result = await self.calibration.run(kind)
        return self._record_calibration(result)

    def latest_calibration(
        self, kind: CalibrationKind
    ) -> CalibrationResult | None:
        return self._last_calibration_by_kind.get(kind)

    def _latest_health_calibration(self) -> CalibrationResult | None:
        if not self._last_calibration_by_kind:
            return None
        return max(
            self._last_calibration_by_kind.values(),
            key=lambda item: (item.completed_at, item.started_at, item.run_id),
        )

    def _leverage_configuration_valid(self, leverage: Any) -> bool:
        if not leverage:
            return False
        try:
            values = tuple(int(value) for value in leverage.values())
        except (AttributeError, TypeError, ValueError):
            return False
        if any(value <= 0 for value in values):
            return False
        if self.target_leverage is None:
            return True
        return all(value == self.target_leverage for value in values)

    def _bundle_health_values(self) -> tuple[int, bool]:
        bundle = self.calibration.last_bundle
        if bundle is None:
            return 0, False
        external_orders = sum(
            1
            for item in bundle.open_orders
            if item.active and item.quarantined
        )
        configuration = getattr(bundle, "configuration", None)
        leverage = (
            {}
            if configuration is None
            else getattr(configuration, "leverage_by_symbol_side", {})
        )
        configuration_valid = bool(
            configuration is not None
            and getattr(configuration, "hedge_mode", False)
            and getattr(configuration, "active_margin_modes", ()) == ("cross",)
            and self._leverage_configuration_valid(leverage)
        )
        return external_orders, configuration_valid

    @staticmethod
    def _freshness_reason_codes(
        freshness: FreshnessAssessment,
    ) -> list[str]:
        if freshness.fresh:
            return []
        reasons: list[str] = []
        if "USER_STREAM" in freshness.reason:
            reasons.append(ReadonlyReasonCode.STALE_USER_STREAM.value)
        if "CALIBRATION" in freshness.reason:
            reasons.append(ReadonlyReasonCode.STALE_REST_SNAPSHOT.value)
        return reasons

    @staticmethod
    def _calibration_reason_codes(
        latest: CalibrationResult | None,
    ) -> list[str]:
        if latest is None:
            return []
        reasons: list[str] = []
        if latest.unmanaged_positions:
            reasons.append(ReadonlyReasonCode.UNMANAGED_POSITION.value)
        if latest.unmanaged_orders:
            reasons.append(ReadonlyReasonCode.UNMANAGED_ORDER.value)
        if not latest.consistent:
            reasons.append(ReadonlyReasonCode.RECONCILIATION_DRIFT.value)
        return reasons

    def _configuration_reason_codes(self) -> list[str]:
        bundle = self.calibration.last_bundle
        configuration = None if bundle is None else getattr(bundle, "configuration", None)
        if configuration is None:
            return []
        reasons: list[str] = []
        if not getattr(configuration, "hedge_mode", False):
            reasons.append(ReadonlyReasonCode.POSITION_MODE_MISMATCH.value)
        if getattr(configuration, "active_margin_modes", ()) != ("cross",):
            reasons.append(ReadonlyReasonCode.MARGIN_MODE_MISMATCH.value)
        leverage = getattr(configuration, "leverage_by_symbol_side", {})
        if not self._leverage_configuration_valid(leverage):
            reasons.append(ReadonlyReasonCode.LEVERAGE_MISMATCH.value)
        return reasons

    def _stream_is_fresh(self, freshness: FreshnessAssessment) -> bool:
        health = self.stream.health
        if not health.connected or health.last_connected_at is None:
            return False
        threshold = self.freshness.policy.event_stale_after
        if threshold is None:
            return True
        age = freshness.event_age_seconds
        return age is not None and age <= threshold.total_seconds()

    def _account_id(self) -> str:
        return str(
            getattr(self.client, "account_id", None)
            or getattr(self.stream, "_account_id", "unknown")
        )

    def _clock_sync_health(self) -> tuple[bool, list[str]]:
        clock_sync = getattr(self.client, "clock_sync", None)
        clock_status = getattr(clock_sync, "status", None)
        synchronized = bool(
            clock_status is not None
            and getattr(clock_status, "synchronized", False)
        )
        reasons: list[str] = []
        if clock_status is not None and not synchronized:
            reasons.append(ReadonlyReasonCode.CLOCK_SKEW_EXCEEDED.value)
        return synchronized, reasons

    def _freshness_health(
        self, freshness: FreshnessAssessment
    ) -> tuple[bool, bool, list[str]]:
        calibration_age = freshness.calibration_age_seconds
        max_age = self.freshness.policy.calibration_stale_after.total_seconds()
        rest_fresh = calibration_age is not None and calibration_age <= max_age
        stream_fresh = self._stream_is_fresh(freshness)
        reasons: list[str] = []
        if not rest_fresh:
            reasons.append(ReadonlyReasonCode.STALE_REST_SNAPSHOT.value)
        if not stream_fresh:
            reasons.append(ReadonlyReasonCode.STALE_USER_STREAM.value)
        return rest_fresh, stream_fresh, reasons

    def _direction2_health(
        self, freshness: FreshnessAssessment
    ) -> Direction2HealthFact:
        latest = self._latest_health_calibration()
        external_orders, configuration_valid = self._bundle_health_values()
        rest_fresh, stream_fresh, freshness_reasons = self._freshness_health(freshness)
        clock_synchronized, clock_reasons = self._clock_sync_health()
        reasons = self._freshness_reason_codes(freshness)
        reasons.extend(freshness_reasons)
        reasons.extend(self._calibration_reason_codes(latest))
        reasons.extend(self._configuration_reason_codes())
        reasons.extend(clock_reasons)
        if external_orders:
            reasons.append(ReadonlyReasonCode.EXTERNAL_ORDER.value)
        stream_health = self.stream.health
        return Direction2HealthFact(
            account_id=self._account_id(),
            rest_fresh=rest_fresh,
            stream_connected=stream_health.connected,
            stream_fresh=stream_fresh,
            clock_synchronized=clock_synchronized,
            configuration_valid=configuration_valid,
            reconciliation_consistent=bool(latest and latest.consistent),
            unmanaged_position_count=(
                0 if latest is None else len(latest.unmanaged_positions)
            ),
            unmanaged_order_count=(
                0 if latest is None else len(latest.unmanaged_orders)
            ),
            external_order_count=external_orders,
            reason_codes=tuple(dict.fromkeys(reasons)),
            observed_at=self.clock.now(),
            last_rest_at=None if latest is None else latest.completed_at,
            last_stream_event_at=stream_health.last_event_at,
            last_stream_connected_at=stream_health.last_connected_at,
            latest_reconciliation_run_id=None if latest is None else latest.run_id,
        )

    def _account_view_sources(
        self,
    ) -> tuple[Any, tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
        bundle = self.calibration.last_bundle
        positions = () if bundle is None else tuple(getattr(bundle, "positions", ()))
        balances = () if bundle is None else tuple(getattr(bundle, "balances", ()))
        orders = () if bundle is None else tuple(getattr(bundle, "open_orders", ()))
        positions = tuple(getattr(self.stream, "current_positions", positions))
        balances = tuple(getattr(self.stream, "current_balances", balances))
        orders = tuple(getattr(self.stream, "current_active_orders", orders))
        return bundle, positions, balances, orders

    def _account_view_observed_at(self, bundle: Any) -> datetime:
        observed_at = getattr(self.stream, "state_observed_at", None)
        if observed_at is None and bundle is not None:
            observed_at = getattr(bundle, "collection_completed_at", None)
        return observed_at or self.clock.now()

    def account_view(self) -> ReadonlyAccountView:
        bundle, positions, balances, active_orders = self._account_view_sources()
        latest = self._latest_health_calibration()
        return ReadonlyAccountView(
            account_id=self._account_id(),
            observed_at=self._account_view_observed_at(bundle),
            account_snapshot=(
                None if bundle is None else getattr(bundle, "account_snapshot", None)
            ),
            balances=tuple(sorted(balances, key=lambda item: item.asset)),
            positions=tuple(
                sorted(positions, key=lambda item: (item.symbol, item.position_side))
            ),
            active_orders=tuple(
                sorted(
                    (item for item in active_orders if item.active),
                    key=lambda item: (item.symbol, item.exchange_order_id),
                )
            ),
            configuration=(
                None if bundle is None else getattr(bundle, "configuration", None)
            ),
            revision=int(getattr(self.stream, "state_revision", 0)),
            last_calibration_run_id=None if latest is None else latest.run_id,
        )

    def runtime_snapshot(self) -> ReadonlyRuntimeSnapshot:
        freshness = self.freshness.assess(
            self.stream.health, now=self.clock.now()
        )
        transport = getattr(self.client, "transport", None)
        transport_telemetry = getattr(transport, "telemetry", None)
        clock_sync = getattr(self.client, "clock_sync", None)
        clock_sync_status = getattr(clock_sync, "status", None)
        lease = getattr(self.stream, "listen_key_lease", None)
        return ReadonlyRuntimeSnapshot(
            status=self._status,
            stream_health=self.stream.health,
            freshness=freshness,
            last_fast_calibration=self.latest_calibration(
                CalibrationKind.FAST
            ),
            last_full_calibration=self.latest_calibration(
                CalibrationKind.FULL
            ),
            last_reconnect_calibration=self.latest_calibration(
                CalibrationKind.RECONNECT
            ),
            direction2_health=self._direction2_health(freshness),
            transport_telemetry=transport_telemetry,
            clock_sync_status=clock_sync_status,
            listen_key_generation=(None if lease is None else lease.generation),
            listen_key_created_at=(None if lease is None else lease.created_at),
            listen_key_renewed_at=(None if lease is None else lease.renewed_at),
            listen_key_expires_at=(None if lease is None else lease.expires_at),
            account_view=self.account_view(),
        )

    async def preflight_and_bootstrap(self) -> None:
        self._set_state(ReadonlyState.PREFLIGHT, "CLOCK_AND_PERMISSION_PREFLIGHT")
        try:
            await self.client.synchronize_clock()
            await self.client.preflight_permissions(self.permission_policy)
            self._set_state(ReadonlyState.CALIBRATING, "STARTUP_REST_CALIBRATION")
            result = await self._run_calibration(CalibrationKind.STARTUP)
            if self.calibration.last_bundle is not None:
                await self._call_stream_seed(
                    self.stream.seed_from_rest,
                    self.calibration.last_bundle,
                )
            self.stream.mark_calibrated(result.completed_at)
            if self.scheduler is not None:
                self.scheduler.reset_after_bootstrap()
            self._set_state(
                ReadonlyState.RECOVERING,
                "WAITING_FOR_USER_STREAM_CONNECTION",
                result.run_id,
            )
        except ReadonlySafetyHalt as exc:
            self._set_state(
                ReadonlyState.HALT,
                exc.reason,
                exc.result.run_id if exc.result else None,
            )
            raise
        except Exception as exc:
            self._set_state(
                ReadonlyState.HALT, f"PREFLIGHT_FAILED:{type(exc).__name__}"
            )
            raise

    async def on_stream_connected(self, generation: int) -> None:
        # Every physical connection, including a listenKey rebuild, needs a new
        # REST calibration before READY.
        try:
            await self._recover(f"STREAM_CONNECTED_GENERATION_{generation}")
        finally:
            self._freshness_action_pending = False


    async def on_stream_event(self, _observed_at: datetime) -> None:
        self._freshness_action_pending = False
        if self._status.state in {ReadonlyState.DEGRADED, ReadonlyState.READY}:
            assessment = self.freshness.assess(
                self.stream.health, now=self.clock.now()
            )
            latest = self._latest_health_calibration()
            if assessment.fresh and latest is not None and latest.consistent:
                self._set_state(ReadonlyState.READY, "USER_STREAM_EVENT_FRESH")

    async def on_stream_disconnected(self) -> None:
        if self._status.state is ReadonlyState.HALT:
            return
        self._set_state(
            ReadonlyState.RECOVERING,
            "STREAM_DISCONNECTED_REST_CALIBRATION_REQUIRED",
        )

    async def on_recalibration_required(
        self, reason: str, _payload: Any
    ) -> None:
        if self._status.state is ReadonlyState.HALT:
            return
        self._set_state(
            ReadonlyState.RECOVERING,
            f"STREAM_RECALIBRATION_REQUIRED:{reason}",
        )

    async def on_integrity_fault(self, reason: str, _payload: Any) -> None:
        if self._status.state is ReadonlyState.HALT:
            return
        # BinanceUserStream closes the current socket after a proven integrity
        # fault. Recovery is intentionally deferred to the next physical
        # connection, avoiding a calibration while the bad stream is still live.
        self._set_state(ReadonlyState.RECOVERING, f"STREAM_INTEGRITY_FAULT:{reason}")

    async def _reseed_stream_from_last_bundle(self) -> None:
        bundle = self.calibration.last_bundle
        if bundle is None:
            return
        runtime_reseed = getattr(self.stream, "reseed_from_rest", None)
        if callable(runtime_reseed):
            await self._call_stream_seed(runtime_reseed, bundle)
            return
        # Compatibility for simple fake adapters used by downstream tests.
        # Production BinanceUserStream provides the locked async method.
        await self._call_stream_seed(self.stream.seed_from_rest, bundle)

    async def _verify_repaired_drift(
        self, result: CalibrationResult
    ) -> CalibrationResult:
        current = self._record_calibration(result)
        for _attempt in range(self.drift_verification_attempts):
            if current.consistent:
                break
            if current.unmanaged_positions or current.unmanaged_orders:
                break
            current = await self._run_calibration(CalibrationKind.FAST)
        return current

    async def on_scheduled_calibration(
        self, result: CalibrationResult
    ) -> None:
        if self._status.state is ReadonlyState.HALT:
            return
        result = await self._verify_repaired_drift(result)
        await self._reseed_stream_from_last_bundle()
        self.stream.mark_calibrated(result.completed_at)
        if result.consistent:
            # A periodic run must not satisfy the explicit reconnect barrier.
            # Only _recover(), which is tied to a physical connection callback,
            # may move RECOVERING/CALIBRATING back to READY.
            if (
                self.stream.health.connected
                and self._status.state not in {
                    ReadonlyState.RECOVERING, ReadonlyState.CALIBRATING
                }
            ):
                self._set_state(
                    ReadonlyState.READY,
                    "SCHEDULED_CALIBRATION_COMPLETE",
                    result.run_id,
                )
        else:
            self._set_state(
                ReadonlyState.DEGRADED,
                "SCHEDULED_RECONCILIATION_DRIFT",
                result.run_id,
            )

    async def calibrate_now(
        self, kind: CalibrationKind = CalibrationKind.FULL
    ) -> CalibrationResult:
        if kind not in {CalibrationKind.FAST, CalibrationKind.FULL}:
            raise ValueError("manual calibration kind must be FAST or FULL")
        async with self._recovery_lock:
            if self._status.state in {ReadonlyState.HALT, ReadonlyState.STOPPED}:
                raise RuntimeError(
                    "Cannot calibrate a HALT or STOPPED read-only service"
                )
            self._set_state(
                ReadonlyState.CALIBRATING,
                f"MANUAL_{kind.value}_CALIBRATION",
            )
            try:
                result = await self._run_calibration(kind)
                result = await self._verify_repaired_drift(result)
                await self._reseed_stream_from_last_bundle()
                self.stream.mark_calibrated(result.completed_at)
                if not result.consistent:
                    state = ReadonlyState.DEGRADED
                    reason = "MANUAL_RECONCILIATION_DRIFT"
                elif self.stream.health.connected:
                    state = ReadonlyState.READY
                    reason = "MANUAL_CALIBRATION_COMPLETE"
                else:
                    state = ReadonlyState.RECOVERING
                    reason = "MANUAL_CALIBRATION_WAITING_FOR_STREAM"
                self._set_state(state, reason, result.run_id)
                return result
            except ReadonlySafetyHalt as exc:
                self._set_state(
                    ReadonlyState.HALT,
                    exc.reason,
                    exc.result.run_id if exc.result else None,
                )
                raise
            except Exception as exc:
                self._set_state(
                    ReadonlyState.DEGRADED,
                    f"MANUAL_CALIBRATION_FAILED:{type(exc).__name__}",
                )
                raise

    async def on_scheduler_error(self, exc: Exception) -> None:
        if isinstance(exc, ReadonlySafetyHalt):
            self._set_state(
                ReadonlyState.HALT,
                exc.reason,
                exc.result.run_id if exc.result else None,
            )
        elif self._status.state is not ReadonlyState.HALT:
            self._set_state(
                ReadonlyState.DEGRADED,
                f"SCHEDULED_CALIBRATION_FAILED:{type(exc).__name__}",
            )

    async def _recover(self, reason: str) -> None:
        async with self._recovery_lock:
            if self._status.state is ReadonlyState.HALT:
                return
            self._set_state(ReadonlyState.CALIBRATING, reason)
            try:
                result = await self._run_calibration(CalibrationKind.RECONNECT)
                result = await self._verify_repaired_drift(result)
                await self._reseed_stream_from_last_bundle()
                self.stream.mark_calibrated(result.completed_at)
                if result.consistent:
                    self._set_state(
                        ReadonlyState.READY,
                        "RECONNECT_REST_CALIBRATION_COMPLETE",
                        result.run_id,
                    )
                else:
                    self._set_state(
                        ReadonlyState.DEGRADED,
                        "RECONNECT_RECONCILIATION_DRIFT",
                        result.run_id,
                    )
            except ReadonlySafetyHalt as exc:
                self._set_state(
                    ReadonlyState.HALT,
                    exc.reason,
                    exc.result.run_id if exc.result else None,
                )
                raise
            except Exception as exc:
                self._set_state(
                    ReadonlyState.RECOVERING,
                    f"RECONNECT_CALIBRATION_FAILED:{type(exc).__name__}",
                )
                raise

    def assess_freshness(self) -> FreshnessAssessment:
        assessment = self.freshness.assess(
            self.stream.health, now=self.clock.now()
        )
        if (
            self._status.state not in {ReadonlyState.HALT, ReadonlyState.STOPPED}
            and not assessment.fresh
        ):
            self._set_state(assessment.state, assessment.reason)
        return assessment

    def _spawn(self, coroutine: Any, *, name: str) -> None:
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.append(task)
        task.add_done_callback(self._background_task_done)

    def _background_task_done(self, task: asyncio.Task[Any]) -> None:
        if self._stopping or self._stop_event.is_set() or task.cancelled():
            return
        supervisor = asyncio.create_task(
            self._handle_background_task_done(task),
            name=f"supervise-{task.get_name()}",
        )
        self._supervisor_tasks.add(supervisor)
        supervisor.add_done_callback(self._supervisor_tasks.discard)

    async def _handle_background_task_done(self, task: asyncio.Task[Any]) -> None:
        try:
            exception = task.exception()
        except asyncio.CancelledError:
            return
        reason = (
            f"BACKGROUND_TASK_FAILED:{task.get_name()}:{type(exception).__name__}"
            if exception is not None
            else f"BACKGROUND_TASK_EXITED:{task.get_name()}"
        )
        self._set_state(ReadonlyState.HALT, reason)
        self._stop_event.set()
        try:
            await self.stream.stop()
        except Exception:
            logger.exception("Failed to stop Binance user stream after task failure")
        current = asyncio.current_task()
        for other in self._tasks:
            if other is not task and other is not current and not other.done():
                other.cancel()
        try:
            await self._close_transport()
        except Exception:
            logger.exception("Failed to close Binance transport after task failure")

    async def start(self) -> None:
        if self._status.state is ReadonlyState.STOPPED:
            raise RuntimeError(
                "A stopped BinanceReadonlyService cannot be restarted; create a new instance"
            )
        if self._started:
            raise RuntimeError("BinanceReadonlyService is already started")
        self._started = True
        self._stopping = False
        try:
            await self.preflight_and_bootstrap()
        except Exception:
            self._started = False
            try:
                await self._close_transport()
            except Exception:
                logger.exception("Failed to close Binance transport after startup failure")
            raise
        self._stop_event.clear()
        self._spawn(self.stream.run(), name="binance-user-stream")
        self._spawn(
            self.stream.run_listen_key_renewal(self._stop_event),
            name="binance-listen-key-renewal",
        )
        if self.scheduler is not None:
            self._spawn(
                self.scheduler.run(self._stop_event),
                name="hedge-reconciliation",
            )
        self._spawn(self._freshness_loop(), name="hedge-stream-freshness")

    async def _handle_freshness_failure(
        self, assessment: FreshnessAssessment
    ) -> None:
        if self._freshness_action_pending:
            return
        if assessment.reason == "USER_STREAM_STALE":
            reconnect = getattr(self.stream, "request_reconnect", None)
            if callable(reconnect) and self.stream.health.connected:
                self._freshness_action_pending = True
                self._set_state(
                    ReadonlyState.RECOVERING,
                    "USER_STREAM_STALE_RECONNECT_REQUESTED",
                )
                await maybe_await(reconnect())
        elif (
            assessment.reason == "REST_CALIBRATION_STALE"
            and self.stream.health.connected
            and self._status.state not in {
                ReadonlyState.CALIBRATING,
                ReadonlyState.RECOVERING,
            }
        ):
            self._freshness_action_pending = True
            try:
                async with self._recovery_lock:
                    self._set_state(
                        ReadonlyState.CALIBRATING,
                        "FRESHNESS_FULL_CALIBRATION",
                    )
                    result = await self._run_calibration(CalibrationKind.FULL)
                    result = await self._verify_repaired_drift(result)
                    await self._reseed_stream_from_last_bundle()
                    self.stream.mark_calibrated(result.completed_at)
                    state = (
                        ReadonlyState.READY
                        if result.consistent
                        else ReadonlyState.DEGRADED
                    )
                    self._set_state(
                        state,
                        "FRESHNESS_FULL_CALIBRATION_COMPLETE",
                        result.run_id,
                    )
            finally:
                self._freshness_action_pending = False

    async def _freshness_iteration(self) -> None:
        try:
            assessment = self.assess_freshness()
            if not assessment.fresh:
                await self._handle_freshness_failure(assessment)
        except asyncio.CancelledError:
            raise
        except ReadonlySafetyHalt as exc:
            self._set_state(
                ReadonlyState.HALT,
                exc.reason,
                exc.result.run_id if exc.result else None,
            )
        except Exception as exc:
            self._freshness_action_pending = False
            self._set_state(
                ReadonlyState.DEGRADED,
                f"FRESHNESS_RECOVERY_FAILED:{type(exc).__name__}",
            )
            logger.exception("Binance readonly freshness recovery failed")

    async def _freshness_loop(self) -> None:
        while not self._stop_event.is_set():
            await self._freshness_iteration()
            if self._status.state is ReadonlyState.HALT:
                return
            await self.clock.sleep(10.0)

    async def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self._stop_event.set()
        errors: list[Exception] = []
        try:
            await self.stream.stop()
        except Exception as exc:
            errors.append(exc)
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        for task in tuple(self._supervisor_tasks):
            task.cancel()
        if self._supervisor_tasks:
            await asyncio.gather(*self._supervisor_tasks, return_exceptions=True)
        self._supervisor_tasks.clear()
        try:
            await self._close_transport()
        except Exception as exc:
            errors.append(exc)
        self._started = False
        self._set_state(ReadonlyState.STOPPED, "STOPPED")
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("Binance readonly service cleanup failed", errors)

    async def _close_transport(self) -> None:
        transport = getattr(self.client, "transport", None)
        closer = getattr(transport, "close", None)
        if callable(closer):
            await maybe_await(closer())
