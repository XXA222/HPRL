"""Deterministic reconciliation between exchange facts and ledger projections."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Iterable

from freqtrade.persistence.hedge_contracts import canonical_symbol, normalize_order_status
from freqtrade.persistence.hedge_models import canonical_decimal, decimal_value
from freqtrade.persistence.hedge_repositories import HedgeLedgerRepository


@dataclass(frozen=True)
class PositionFact:
    symbol: str
    position_side: str
    quantity: str
    entry_price: str = "0"
    realized_pnl: str = "0"
    event_time: datetime | None = None
    sequence_number: int | None = None


@dataclass(frozen=True)
class OrderFact:
    symbol: str
    position_side: str
    exchange_order_id: str
    side: str
    status: str
    original_quantity: str
    executed_quantity: str
    cumulative_quote: str = "0"
    average_price: str | None = None
    client_order_id: str | None = None
    action: str | None = None
    event_time: datetime | None = None
    sequence_number: int | None = None


@dataclass(frozen=True)
class AccountModeFact:
    position_mode: str
    margin_mode: str
    leverage: str | None = None
    event_time: datetime | None = None


@dataclass(frozen=True)
class AccountReconciliationSummary:
    run_id: str
    status: str
    compared_positions: int
    compared_orders: int
    diff_count: int
    severe_diff_count: int
    repaired_count: int
    unmanaged_positions: tuple[str, ...] = field(default_factory=tuple)
    unmanaged_orders: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ReconciliationPolicy:
    quantity_tolerance: Decimal = Decimal("0")
    price_tolerance: Decimal = Decimal("0")
    auto_repair_positions: bool = False
    auto_repair_orders: bool = False

    def __post_init__(self) -> None:
        if self.quantity_tolerance < 0 or self.price_tolerance < 0:
            raise ValueError("Reconciliation tolerances must not be negative")


@dataclass(frozen=True)
class ReconciliationSummary:
    run_id: str
    status: str
    compared_positions: int
    diff_count: int
    severe_diff_count: int
    repaired_count: int
    missing_local: tuple[str, ...] = field(default_factory=tuple)
    missing_exchange: tuple[str, ...] = field(default_factory=tuple)


def _key(symbol: str, position_side: str) -> tuple[str, str]:
    normalized_side = position_side.upper()
    if normalized_side not in {"LONG", "SHORT"}:
        raise ValueError(f"Unsupported position side: {position_side!r}")
    return canonical_symbol(symbol), normalized_side


class HedgeReconciler:
    """Compare exchange position facts with replayed local ledger state."""

    def __init__(self, repository: HedgeLedgerRepository):
        self.repository = repository

    def reconcile_positions(
        self,
        *,
        account_id: str,
        exchange: str,
        facts: Iterable[PositionFact],
        trigger: str = "SCHEDULED",
        policy: ReconciliationPolicy | None = None,
        observed_at: datetime | None = None,
    ) -> ReconciliationSummary:
        policy = policy or ReconciliationPolicy()
        normalized_facts: dict[tuple[str, str], PositionFact] = {}
        for fact in facts:
            key = _key(fact.symbol, fact.position_side)
            if key in normalized_facts:
                raise ValueError(f"Duplicate exchange position fact: {key}")
            if decimal_value(fact.quantity) < 0:
                raise ValueError("Exchange position quantity must not be negative")
            normalized_facts[key] = fact

        run = self.repository.start_reconciliation(
            account_id=account_id,
            exchange=exchange,
            trigger=trigger,
            scope="POSITIONS",
        )
        recovery = self.repository.recover_projection(account_id=account_id)
        local = {
            _key(row.symbol, row.position_side): row
            for row in recovery.positions
            if row.exchange == exchange
        }
        all_keys = sorted(set(local) | set(normalized_facts))
        missing_local: list[str] = []
        missing_exchange: list[str] = []
        repaired = 0
        severe = 0

        for symbol, side in all_keys:
            entity_key = f"{exchange}|{account_id}|{symbol}|{side}"
            local_row = local.get((symbol, side))
            exchange_row = normalized_facts.get((symbol, side))
            local_quantity = decimal_value(local_row.quantity) if local_row else Decimal("0")
            exchange_quantity = (
                decimal_value(exchange_row.quantity) if exchange_row else Decimal("0")
            )
            local_price = decimal_value(local_row.entry_price) if local_row else Decimal("0")
            exchange_price = (
                decimal_value(exchange_row.entry_price) if exchange_row else Decimal("0")
            )

            if local_row is None and exchange_quantity != 0:
                missing_local.append(entity_key)
            if exchange_row is None and local_quantity != 0:
                missing_exchange.append(entity_key)

            quantity_drift = abs(local_quantity - exchange_quantity)
            price_drift = abs(local_price - exchange_price)
            fields: list[tuple[str, Decimal, Decimal, Decimal]] = []
            if quantity_drift > policy.quantity_tolerance:
                fields.append(("quantity", local_quantity, exchange_quantity, quantity_drift))
            if (
                max(local_quantity, exchange_quantity) > 0
                and price_drift > policy.price_tolerance
            ):
                fields.append(("entry_price", local_price, exchange_price, price_drift))

            for field_name, local_value, exchange_value, drift in fields:
                severity = "CRITICAL" if field_name == "quantity" else "ERROR"
                severe += 1
                resolution = "UNRESOLVED"
                repair_action = None
                if policy.auto_repair_positions:
                    resolution = "AUTO_REPAIRED"
                    repair_action = "APPEND_EXCHANGE_POSITION_FACT"
                self.repository.add_reconciliation_diff(
                    run_id=run.run_id,
                    diff_key=f"{entity_key}|{field_name}",
                    entity_type="POSITION",
                    entity_key=entity_key,
                    field_name=field_name,
                    local_value=canonical_decimal(local_value),
                    exchange_value=canonical_decimal(exchange_value),
                    severity=severity,
                    resolution=resolution,
                    repair_action=repair_action,
                )

            if policy.auto_repair_positions and fields:
                event_time = exchange_row.event_time if exchange_row is not None else None
                event_time = event_time or observed_at
                if event_time is None:
                    raise ValueError("Auto repair requires an exchange fact event time")
                repair_quantity = exchange_row.quantity if exchange_row is not None else "0"
                repair_entry_price = (
                    exchange_row.entry_price if exchange_row is not None else "0"
                )
                repair_realized_pnl = (
                    exchange_row.realized_pnl
                    if exchange_row is not None
                    else local_row.realized_pnl if local_row is not None else "0"
                )
                repair_sequence = (
                    exchange_row.sequence_number if exchange_row is not None else None
                )
                self.repository.append_position_snapshot(
                    snapshot_key=(
                        f"reconciliation:{run.run_id}:{symbol}:{side}:"
                        f"{repair_sequence or 0}"
                    ),
                    account_id=account_id,
                    exchange=exchange,
                    symbol=symbol,
                    position_side=side,
                    quantity=repair_quantity,
                    entry_price=repair_entry_price,
                    realized_pnl=repair_realized_pnl,
                    source="REST",
                    source_event_time=event_time,
                    sequence_number=repair_sequence,
                    raw_payload={
                        "reconciliation_run_id": run.run_id,
                        "exchange_position_missing": exchange_row is None,
                    },
                )
                repaired += 1

        status = "HEALTHY"
        if severe:
            status = "REPAIRED" if repaired else "DRIFT"
        run.automatic_repairs = repaired
        self.repository.complete_reconciliation(
            run_id=run.run_id,
            status=status,
            summary={
                "compared_positions": len(all_keys),
                "missing_local": missing_local,
                "missing_exchange": missing_exchange,
                "repaired_count": repaired,
            },
        )
        return ReconciliationSummary(
            run_id=run.run_id,
            status=status,
            compared_positions=len(all_keys),
            diff_count=run.diff_count,
            severe_diff_count=run.severe_diff_count,
            repaired_count=repaired,
            missing_local=tuple(missing_local),
            missing_exchange=tuple(missing_exchange),
        )

    def reconcile_account_facts(
        self,
        *,
        account_id: str,
        exchange: str,
        positions: Iterable[PositionFact],
        orders: Iterable[OrderFact],
        mode: AccountModeFact | None = None,
        expected_position_mode: str = "HEDGE",
        expected_margin_mode: str = "CROSS",
        expected_leverage: str | None = None,
        unmanaged_positions: Iterable[str] = (),
        unmanaged_orders: Iterable[str] = (),
        trigger: str = "SCHEDULED",
        policy: ReconciliationPolicy | None = None,
        observed_at: datetime | None = None,
    ) -> AccountReconciliationSummary:
        """Reconcile full account position, order, and mode facts in one run."""

        policy = policy or ReconciliationPolicy()
        position_facts: dict[tuple[str, str], PositionFact] = {}
        for fact in positions:
            key = _key(fact.symbol, fact.position_side)
            if key in position_facts:
                raise ValueError(f"Duplicate exchange position fact: {key}")
            position_facts[key] = PositionFact(
                symbol=key[0],
                position_side=key[1],
                quantity=fact.quantity,
                entry_price=fact.entry_price,
                realized_pnl=fact.realized_pnl,
                event_time=fact.event_time,
                sequence_number=fact.sequence_number,
            )

        order_facts: dict[tuple[str, str, str], OrderFact] = {}
        for fact in orders:
            symbol, side = _key(fact.symbol, fact.position_side)
            key = (symbol, side, fact.exchange_order_id)
            if key in order_facts:
                raise ValueError(f"Duplicate exchange order fact: {key}")
            order_facts[key] = OrderFact(
                symbol=symbol,
                position_side=side,
                exchange_order_id=fact.exchange_order_id,
                side=fact.side.upper(),
                status=normalize_order_status(fact.status),
                original_quantity=canonical_decimal(fact.original_quantity),
                executed_quantity=canonical_decimal(fact.executed_quantity),
                cumulative_quote=canonical_decimal(fact.cumulative_quote),
                average_price=(
                    canonical_decimal(fact.average_price)
                    if fact.average_price is not None
                    else None
                ),
                client_order_id=fact.client_order_id,
                action=fact.action,
                event_time=fact.event_time,
                sequence_number=fact.sequence_number,
            )

        run = self.repository.start_reconciliation(
            account_id=account_id,
            exchange=exchange,
            trigger=trigger,
            scope="FULL_ACCOUNT",
        )
        recovery = self.repository.recover_projection(account_id=account_id)
        local_positions = {
            _key(row.symbol, row.position_side): row
            for row in recovery.positions
            if row.exchange == exchange
        }
        local_orders = {
            (row.symbol, row.position_side, row.exchange_order_id): row
            for row in recovery.orders
            if row.exchange == exchange
        }
        repaired = 0
        severe = 0

        def add_diff(
            *,
            entity_type: str,
            entity_key: str,
            field_name: str,
            local_value: Any,
            exchange_value: Any,
            severity: str,
            repair_action: str | None = None,
        ) -> None:
            nonlocal severe
            if severity in {"ERROR", "CRITICAL"}:
                severe += 1
            self.repository.add_reconciliation_diff(
                run_id=run.run_id,
                diff_key=f"{entity_key}|{field_name}",
                entity_type=entity_type,
                entity_key=entity_key,
                field_name=field_name,
                local_value=local_value,
                exchange_value=exchange_value,
                severity=severity,
                resolution="UNRESOLVED" if repair_action is None else "AUTO_REPAIRED",
                repair_action=repair_action,
            )

        for symbol, side in sorted(set(local_positions) | set(position_facts)):
            local = local_positions.get((symbol, side))
            remote = position_facts.get((symbol, side))
            local_qty = decimal_value(local.quantity) if local else Decimal("0")
            remote_qty = decimal_value(remote.quantity) if remote else Decimal("0")
            local_price = decimal_value(local.entry_price) if local else Decimal("0")
            remote_price = decimal_value(remote.entry_price) if remote else Decimal("0")
            entity_key = f"{exchange}|{account_id}|{symbol}|{side}"
            changed = False
            if abs(local_qty - remote_qty) > policy.quantity_tolerance:
                changed = True
                add_diff(
                    entity_type="POSITION",
                    entity_key=entity_key,
                    field_name="quantity",
                    local_value=canonical_decimal(local_qty),
                    exchange_value=canonical_decimal(remote_qty),
                    severity="CRITICAL",
                    repair_action=(
                        "APPEND_EXCHANGE_POSITION_FACT"
                        if policy.auto_repair_positions
                        else None
                    ),
                )
            if (
                max(local_qty, remote_qty) > 0
                and abs(local_price - remote_price) > policy.price_tolerance
            ):
                changed = True
                add_diff(
                    entity_type="POSITION",
                    entity_key=entity_key,
                    field_name="entry_price",
                    local_value=canonical_decimal(local_price),
                    exchange_value=canonical_decimal(remote_price),
                    severity="ERROR",
                    repair_action=(
                        "APPEND_EXCHANGE_POSITION_FACT"
                        if policy.auto_repair_positions
                        else None
                    ),
                )
            if changed and policy.auto_repair_positions:
                event_time = (remote.event_time if remote else None) or observed_at
                if event_time is None:
                    raise ValueError("Position auto repair requires event time")
                self.repository.append_position_snapshot(
                    snapshot_key=f"reconciliation:{run.run_id}:{symbol}:{side}",
                    account_id=account_id,
                    exchange=exchange,
                    symbol=symbol,
                    position_side=side,
                    quantity=remote.quantity if remote else "0",
                    entry_price=remote.entry_price if remote else "0",
                    realized_pnl=(
                        remote.realized_pnl
                        if remote
                        else local.realized_pnl if local else "0"
                    ),
                    source="REST",
                    source_event_time=event_time,
                    sequence_number=remote.sequence_number if remote else None,
                    raw_payload={"reconciliation_run_id": run.run_id},
                )
                repaired += 1

        for key in sorted(set(local_orders) | set(order_facts)):
            local = local_orders.get(key)
            remote = order_facts.get(key)
            symbol, side, order_id = key
            entity_key = f"{exchange}|{account_id}|{symbol}|{side}|{order_id}"
            comparisons = {
                "status": (
                    normalize_order_status(local.status) if local else None,
                    remote.status if remote else None,
                ),
                "executed_quantity": (
                    local.executed_quantity if local else None,
                    remote.executed_quantity if remote else None,
                ),
                "remaining_quantity": (
                    local.remaining_quantity if local else None,
                    canonical_decimal(
                        max(
                            decimal_value(remote.original_quantity)
                            - decimal_value(remote.executed_quantity),
                            Decimal("0"),
                        )
                    )
                    if remote
                    else None,
                ),
            }
            order_changed = False
            for field_name, (local_value, remote_value) in comparisons.items():
                if local_value == remote_value:
                    continue
                order_changed = True
                add_diff(
                    entity_type="ORDER",
                    entity_key=entity_key,
                    field_name=field_name,
                    local_value=local_value,
                    exchange_value=remote_value,
                    severity="CRITICAL" if field_name == "status" else "ERROR",
                    repair_action=(
                        "APPEND_EXCHANGE_ORDER_FACT"
                        if policy.auto_repair_orders and remote is not None
                        else None
                    ),
                )
            if order_changed and policy.auto_repair_orders and remote is not None:
                event_time = remote.event_time or observed_at
                if event_time is None:
                    raise ValueError("Order auto repair requires event time")
                self.repository.append_order_snapshot(
                    snapshot_key=f"reconciliation:{run.run_id}:{order_id}",
                    account_id=account_id,
                    exchange=exchange,
                    symbol=symbol,
                    position_side=side,
                    exchange_order_id=order_id,
                    client_order_id=remote.client_order_id,
                    side=remote.side,
                    action=remote.action,
                    status=remote.status,
                    original_quantity=remote.original_quantity,
                    executed_quantity=remote.executed_quantity,
                    cumulative_quote=remote.cumulative_quote,
                    average_price=remote.average_price,
                    source="REST",
                    source_event_time=event_time,
                    sequence_number=remote.sequence_number,
                    raw_payload={"reconciliation_run_id": run.run_id},
                )
                repaired += 1

        if mode is not None:
            mode_values = {
                "position_mode": (
                    expected_position_mode.upper(),
                    mode.position_mode.upper(),
                ),
                "margin_mode": (
                    expected_margin_mode.upper(),
                    mode.margin_mode.upper(),
                ),
            }
            if expected_leverage is not None:
                mode_values["leverage"] = (str(expected_leverage), str(mode.leverage))
            for field_name, (expected, actual) in mode_values.items():
                if expected != actual:
                    add_diff(
                        entity_type="ACCOUNT_MODE",
                        entity_key=f"{exchange}|{account_id}|mode",
                        field_name=field_name,
                        local_value=expected,
                        exchange_value=actual,
                        severity="CRITICAL",
                    )

        unmanaged_position_keys = tuple(sorted(set(unmanaged_positions)))
        unmanaged_order_keys = tuple(sorted(set(unmanaged_orders)))
        for entity_type, values in (
            ("UNMANAGED_POSITION", unmanaged_position_keys),
            ("UNMANAGED_ORDER", unmanaged_order_keys),
        ):
            for value in values:
                add_diff(
                    entity_type=entity_type,
                    entity_key=value,
                    field_name="managed",
                    local_value=False,
                    exchange_value=True,
                    severity="CRITICAL",
                )

        status = "HEALTHY"
        if severe:
            status = "REPAIRED" if repaired and repaired >= severe else "DRIFT"
        run.automatic_repairs = repaired
        self.repository.complete_reconciliation(
            run_id=run.run_id,
            status=status,
            summary={
                "compared_positions": len(set(local_positions) | set(position_facts)),
                "compared_orders": len(set(local_orders) | set(order_facts)),
                "unmanaged_positions": unmanaged_position_keys,
                "unmanaged_orders": unmanaged_order_keys,
                "repaired_count": repaired,
            },
        )
        return AccountReconciliationSummary(
            run_id=run.run_id,
            status=status,
            compared_positions=len(set(local_positions) | set(position_facts)),
            compared_orders=len(set(local_orders) | set(order_facts)),
            diff_count=run.diff_count,
            severe_diff_count=run.severe_diff_count,
            repaired_count=repaired,
            unmanaged_positions=unmanaged_position_keys,
            unmanaged_orders=unmanaged_order_keys,
        )

