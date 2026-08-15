"""Adapters that expose the central fail-closed HedgeRuntime through RPC."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
import json
import logging
from uuid import UUID
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from freqtrade.hedge.runtime import HedgeRuntime
from freqtrade.hedge.symbols import canonicalize_symbol, raw_symbol
from freqtrade.rpc.api_server.hedge_schemas import (
    DualLegPositionsResponse,
    LegPositionSchema,
    ActionGroupStatusResponse,
    ExecutionOrderListResponse,
    ExecutionOrderSchema,
    HedgeEventListResponse,
    OperationAuditSchema,
    PairSummaryResponse,
    ReadinessStatusSchema,
    ReconciliationStatusSchema,
    RiskStatusSchema,
    UserStreamStatusSchema,
)

logger = logging.getLogger(__name__)


class HedgeRuntimeQuery:
    def __init__(
        self,
        runtime_provider: Callable[[], HedgeRuntime] | None = None,
    ) -> None:
        self._runtime_provider = runtime_provider or self._runtime_from_rpc

    @staticmethod
    def _runtime_from_rpc() -> HedgeRuntime:
        from freqtrade.rpc.api_server.deps import get_rpc_optional

        rpc = get_rpc_optional()
        runtime = getattr(getattr(rpc, "_freqtrade", None), "hedge_runtime", None)
        if not isinstance(runtime, HedgeRuntime):
            raise RuntimeError("Hedge runtime is not attached")
        return runtime

    @property
    def runtime(self) -> HedgeRuntime:
        return self._runtime_provider()

    def _validate_account(self, account_id: str) -> None:
        if account_id != self.runtime.config.account_id:
            raise ValueError("Unknown hedge account_id")

    def _view(self, source: str | None = None):
        return self.runtime.view(source)

    def _metadata(self, view) -> dict[str, object]:
        return {
            "source": view.source.value,
            "source_version": view.source_version,
            "sequence": view.sequence,
            "event_time": view.source_event_time,
            "observed_at": view.observed_at,
            "stale": view.stale,
            "operation_mode": self.runtime.config.operation_mode,
        }

    def positions(
        self,
        *,
        account_id: str,
        symbol: str,
        source: str | None = None,
    ) -> DualLegPositionsResponse:
        self._validate_account(account_id)
        view = self._view(source)
        requested_symbol = canonicalize_symbol(symbol)
        legs = tuple(
            LegPositionSchema(
                account_id=account_id,
                symbol=position.symbol,
                position_side=position.position_side.value,
                quantity=position.amount,
                entry_price=position.entry_price,
                mark_price=position.mark_price,
                unrealized_pnl=position.unrealized_pnl,
                leverage=position.leverage,
            )
            for position in view.positions
            if canonicalize_symbol(position.symbol) == requested_symbol
        )
        return DualLegPositionsResponse(
            account_id=account_id,
            symbol=symbol,
            legs=legs,
            as_of=view.observed_at,
            **self._metadata(view),
        )

    def risk(self, *, account_id: str, source: str | None = None) -> RiskStatusSchema:
        self._validate_account(account_id)
        view = self._view(source)
        risk = view.risk or self.runtime.empty_risk(account_id)
        reasons = view.reasons
        if not risk.risk_data_valid and "RISK_DATA_INVALID" not in reasons:
            reasons = (*reasons, "RISK_DATA_INVALID")
        return RiskStatusSchema(
            account_id=account_id,
            equity=risk.equity if view.risk is not None else Decimal("0"),
            gross_notional=risk.gross_notional,
            gross_exposure_ratio=(
                risk.gross_exposure_ratio if view.risk is not None else Decimal("0")
            ),
            margin_utilization=(risk.margin_utilization if view.risk is not None else Decimal("0")),
            liquidation_buffer_ratio=risk.liquidation_buffer_ratio,
            halted=view.halted or not risk.risk_data_valid,
            reasons=reasons,
            **self._metadata(view),
        )

    def reconciliation(
        self,
        *,
        account_id: str,
        source: str | None = None,
    ) -> ReconciliationStatusSchema:
        self._validate_account(account_id)
        view = self._view(source)
        drift = int(view.reconciliation_status == "DRIFT")
        return ReconciliationStatusSchema(
            status=view.reconciliation_status,
            last_run_at=view.reconciliation_at,
            drift_count=drift,
            unresolved_count=drift,
            details=view.reconciliation_details,
            **self._metadata(view),
        )

    def readiness(
        self,
        *,
        account_id: str,
        source: str | None = None,
    ) -> ReadinessStatusSchema:
        self._validate_account(account_id)
        view = self._view(source)
        return ReadinessStatusSchema(
            ready=view.ready,
            read_only=self.runtime.config.read_only,
            live_trading_enabled=self.runtime.config.live_trading_enabled,
            kill_switch="HALTED" if view.halted else "RUNNING",
            checks=dict(view.checks),
            reasons=view.reasons,
            **self._metadata(view),
        )

    def user_stream(
        self,
        *,
        account_id: str,
        source: str | None = None,
    ) -> UserStreamStatusSchema:
        self._validate_account(account_id)
        view = self._view(source)
        age_ms = None
        if view.stream_last_event_at is not None:
            age_ms = max(
                0,
                int((datetime.now(UTC) - view.stream_last_event_at).total_seconds() * 1000),
            )
        return UserStreamStatusSchema(
            state=view.stream_state,
            last_event_at=view.stream_last_event_at,
            age_ms=age_ms,
            reconnect_count=view.stream_reconnect_count,
            **self._metadata(view),
        )

    def audit(self, *, account_id: str, limit: int) -> tuple[OperationAuditSchema, ...]:
        """Return durable audit facts first, with an in-memory fallback for tests.

        Read APIs must not silently depend on process-local execution objects.  The
        mode-specific composition owns the authoritative persistence service, so its
        ``hedge_audit_events`` rows are queried whenever the composition is attached.
        """

        self._validate_account(account_id)
        from freqtrade.rpc.api_server.deps import get_rpc_optional

        rpc = get_rpc_optional()
        bot = getattr(rpc, "_freqtrade", None)
        composition = getattr(bot, "hedge_composition", None)
        persistence = getattr(composition, "persistence_service", None)
        session_factory = getattr(persistence, "session_factory", None)
        if callable(session_factory):
            try:
                from freqtrade.persistence.hedge_models import AuditEvent

                with session_factory() as session:
                    records = tuple(
                        session.scalars(
                            select(AuditEvent)
                            .where(AuditEvent.account_id == account_id)
                            .order_by(AuditEvent.occurred_at.desc(), AuditEvent.id.desc())
                            .limit(limit)
                        )
                    )
                return tuple(
                    OperationAuditSchema(
                        audit_id=row.audit_id,
                        actor=row.actor or "hedge-runtime",
                        action=row.event_type,
                        outcome=row.reason_code or row.severity,
                        occurred_at=row.occurred_at,
                        correlation_id=row.correlation_id,
                        details=json.loads(row.payload_json or "{}"),
                    )
                    for row in records
                )
            except (SQLAlchemyError, ValueError, TypeError, json.JSONDecodeError):
                logger.exception("Unable to query durable Hedge audit events")
                # Fail closed for a production composition.  Returning a partial
                # process-local audit history would misrepresent the durable ledger.
                raise RuntimeError("HEDGE_AUDIT_STORE_UNAVAILABLE")

        application = getattr(bot, "hedge_application", None)
        execution = getattr(application, "execution", None)
        ledger = getattr(execution, "ledger", None)
        if ledger is None:
            return ()
        rows = []
        for item in ledger.audit(limit=limit):
            payload = dict(item.payload)
            if str(payload.get("account_id", account_id)) != account_id:
                continue
            digest = sha256(
                f"{item.event_type}|{item.occurred_at.isoformat()}|{payload}".encode()
            ).hexdigest()[:24]
            rows.append(
                OperationAuditSchema(
                    audit_id=f"paper-{digest}",
                    actor="hedge-runtime",
                    action=item.event_type,
                    outcome=str(payload.get("status", "RECORDED")),
                    occurred_at=item.occurred_at,
                    correlation_id=(
                        str(payload.get("correlation_id"))
                        if payload.get("correlation_id") is not None
                        else None
                    ),
                    details=payload,
                )
            )
        return tuple(rows)



class HedgeExecutionRuntimeQuery:
    """Expose paper execution data and useful readonly-account fallbacks.

    In paper/combined mode the direction-five execution ledger is authoritative.
    In readonly mode there is intentionally no execution ledger, so active
    Binance orders and the central position projection are mapped into the same
    control-plane schemas instead of making these endpoints fail.
    """

    @staticmethod
    def _bot():
        from freqtrade.rpc.api_server.deps import get_rpc_optional

        rpc = get_rpc_optional()
        return getattr(rpc, "_freqtrade", None)

    def _paper_adapter(self):
        from freqtrade.rpc.api_server.hedge_readonly import ExecutionLedgerReadonlyAdapter

        application = getattr(self._bot(), "hedge_application", None)
        execution = getattr(application, "execution", None)
        if execution is None:
            return None
        return ExecutionLedgerReadonlyAdapter(execution.core, execution.ledger)

    def _readonly_account_view(self):
        coordinator = getattr(self._bot(), "hedge_coordinator", None)
        runtime = getattr(coordinator, "readonly_runtime", None)
        account_view = getattr(runtime, "account_view", None)
        if not callable(account_view):
            return None
        return account_view()

    def _central_runtime(self) -> HedgeRuntime | None:
        runtime = getattr(self._bot(), "hedge_runtime", None)
        return runtime if isinstance(runtime, HedgeRuntime) else None

    @staticmethod
    def _readonly_status(value: str) -> str:
        status = str(value).strip().upper()
        return {
            "NEW": "ACKNOWLEDGED",
            "PENDING_NEW": "SUBMITTING",
            "PARTIALLY_FILLED": "PARTIAL",
            "CANCELLED": "CANCELED",
            "EXPIRED": "CANCELED",
            "EXPIRED_IN_MATCH": "CANCELED",
        }.get(status, status if status in {
            "PREPARED", "SUBMITTING", "ACKNOWLEDGED", "PARTIAL",
            "FILLED", "CANCELED", "REJECTED", "UNKNOWN",
        } else "UNKNOWN")

    @staticmethod
    def _readonly_action(order: object) -> str:
        side = str(getattr(order, "side", "")).upper()
        position_side = str(getattr(order, "position_side", "")).upper()
        increases = (position_side == "LONG" and side == "BUY") or (
            position_side == "SHORT" and side == "SELL"
        )
        return "INCREASE" if increases else "REDUCE"

    @staticmethod
    def _positive_decimal(value: object) -> Decimal | None:
        if value in (None, ""):
            return None
        try:
            result = Decimal(str(value))
        except Exception:
            return None
        return result if result > 0 else None

    def _readonly_order_schema(self, order: object) -> ExecutionOrderSchema:
        requested = Decimal(str(getattr(order, "original_quantity")))
        filled = Decimal(str(getattr(order, "cumulative_filled_quantity")))
        raw = getattr(order, "raw", {})
        limit_price = None
        if isinstance(raw, dict) or hasattr(raw, "get"):
            limit_price = self._positive_decimal(raw.get("price"))
        average = self._positive_decimal(getattr(order, "average_price", None))
        observed = getattr(order, "observed_at")
        status = self._readonly_status(str(getattr(order, "status")))
        order_type = str(getattr(order, "order_type", "LIMIT")).upper()
        if order_type not in {"MARKET", "LIMIT"}:
            order_type = "LIMIT"
        exchange_order_id = str(getattr(order, "exchange_order_id"))
        return ExecutionOrderSchema(
            client_order_id=str(getattr(order, "client_order_id")),
            intent_id=f"readonly-{exchange_order_id}",
            action_group_id=None,
            account_id=str(getattr(order, "account_id")),
            symbol=canonicalize_symbol(str(getattr(order, "symbol"))),
            position_side=str(getattr(order, "position_side")).upper(),
            action=self._readonly_action(order),
            order_type=order_type,
            status=status,
            requested_quantity=requested,
            approved_quantity=requested,
            filled_quantity=filled,
            remaining_quantity=max(requested - filled, Decimal("0")),
            limit_price=limit_price,
            average_price=average,
            reduce_only=bool(getattr(order, "reduce_only", False)),
            exchange_order_id=exchange_order_id,
            reason="READONLY_EXCHANGE_FACT",
            created_at=observed,
            updated_at=observed,
        )

    def orders(
        self, *, account_id: str, symbol: str | None, status: str | None, limit: int
    ) -> ExecutionOrderListResponse:
        adapter = self._paper_adapter()
        if adapter is not None:
            return adapter.orders(
                account_id=account_id, symbol=symbol, status=status, limit=limit
            )
        view = self._readonly_account_view()
        as_of = datetime.now(UTC) if view is None else view.observed_at
        rows = []
        requested_symbol = None if symbol is None else canonicalize_symbol(symbol)
        requested_status = None if status is None else self._readonly_status(status)
        if view is not None and view.account_id == account_id:
            for item in view.active_orders:
                model = self._readonly_order_schema(item)
                if requested_symbol is not None and canonicalize_symbol(model.symbol) != requested_symbol:
                    continue
                if requested_status is not None and model.status != requested_status:
                    continue
                rows.append(model)
        rows = rows[-limit:]
        return ExecutionOrderListResponse(orders=tuple(rows), count=len(rows), as_of=as_of)

    def order(self, *, client_order_id: str) -> ExecutionOrderSchema:
        adapter = self._paper_adapter()
        if adapter is not None:
            return adapter.order(client_order_id=client_order_id)
        view = self._readonly_account_view()
        if view is not None:
            for item in view.active_orders:
                if str(item.client_order_id) == client_order_id:
                    return self._readonly_order_schema(item)
        raise KeyError(client_order_id)

    def action_group(self, *, action_group_id: UUID) -> ActionGroupStatusResponse:
        adapter = self._paper_adapter()
        if adapter is not None:
            return adapter.action_group(action_group_id=action_group_id)
        return ActionGroupStatusResponse(
            action_group_id=str(action_group_id),
            status="EMPTY",
            orders=(),
            filled_quantity=Decimal("0"),
            as_of=datetime.now(UTC),
        )

    def pair_summary(self, *, account_id: str, symbol: str) -> PairSummaryResponse:
        adapter = self._paper_adapter()
        if adapter is not None:
            return adapter.pair_summary(account_id=account_id, symbol=symbol)
        runtime = self._central_runtime()
        now = datetime.now(UTC)
        long_quantity = Decimal("0")
        short_quantity = Decimal("0")
        long_average = Decimal("0")
        short_average = Decimal("0")
        realized = Decimal("0")
        requested_symbol = canonicalize_symbol(symbol)
        if runtime is not None:
            view = runtime.view()
            now = view.observed_at
            if runtime.config.account_id != account_id:
                raise ValueError("Unknown hedge account_id")
            for position in view.positions:
                if canonicalize_symbol(position.symbol) != requested_symbol:
                    continue
                if position.position_side.value == "LONG":
                    long_quantity = position.amount
                    long_average = position.entry_price
                else:
                    short_quantity = position.amount
                    short_average = position.entry_price
        orders = self.orders(account_id=account_id, symbol=symbol, status=None, limit=1000)
        pending_entry = sum(
            (item.remaining_quantity for item in orders.orders if item.action in {"OPEN", "INCREASE"}),
            Decimal("0"),
        )
        pending_reduce = sum(
            (item.remaining_quantity for item in orders.orders if item.action in {"REDUCE", "CLOSE"}),
            Decimal("0"),
        )
        return PairSummaryResponse(
            account_id=account_id,
            symbol=symbol,
            long_quantity=long_quantity,
            short_quantity=short_quantity,
            net_quantity=long_quantity - short_quantity,
            gross_quantity=long_quantity + short_quantity,
            long_average_price=long_average,
            short_average_price=short_average,
            realized_pnl=realized,
            fees=Decimal("0"),
            funding=Decimal("0"),
            pending_entry_quantity=pending_entry,
            pending_reduce_quantity=pending_reduce,
            as_of=now,
        )

    def events(self, *, account_id: str, limit: int) -> HedgeEventListResponse:
        adapter = self._paper_adapter()
        if adapter is not None:
            return adapter.events(account_id=account_id, limit=limit)
        return HedgeEventListResponse(events=(), count=0, as_of=datetime.now(UTC))
