from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from freqtrade.hedge.acceptance.models import (
    FactPlane,
    ReconciliationDepth,
    ReconciliationIssue,
    ReconciliationOutcome,
    RuntimeSnapshotSet,
)


_CANONICAL_ORDER_STATUSES = frozenset(
    {
        "PLANNED",
        "APPROVED",
        "PREPARED",
        "SUBMITTING",
        "ACKNOWLEDGED",
        "PARTIAL",
        "FILLED",
        "CANCELED",
        "REJECTED",
        "UNKNOWN",
        "EXPIRED",
    }
)
_ORDER_STATUS_ALIASES = {
    "NEW": "ACKNOWLEDGED",
    "SUBMITTED": "ACKNOWLEDGED",
    "PARTIALLY_FILLED": "PARTIAL",
    "CANCELLED": "CANCELED",
    "FAILED": "UNKNOWN",
}


def _canonical_order_status(value: str) -> str | None:
    normalized = str(value).strip().upper()
    canonical = _ORDER_STATUS_ALIASES.get(normalized, normalized)
    return canonical if canonical in _CANONICAL_ORDER_STATUSES else None


def _different(left: Decimal, right: Decimal, tolerance: Decimal) -> bool:
    return abs(left - right) > tolerance


def _issue(
    depth: ReconciliationDepth,
    entity_type: str,
    key: str,
    reason: str,
    expected: Any,
    observed: Any,
) -> ReconciliationIssue:
    return ReconciliationIssue(depth, entity_type, key, reason, expected, observed)


def _position_issues(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    depth: ReconciliationDepth,
    quantity_tolerance: Decimal,
    financial_tolerance: Decimal,
    plane_name: str,
) -> list[ReconciliationIssue]:
    issues: list[ReconciliationIssue] = []
    for key in sorted(set(expected) | set(observed)):
        left = expected.get(key)
        right = observed.get(key)
        if left is None or right is None:
            issues.append(
                _issue(depth, "POSITION", key, f"MISSING_{plane_name}", left, right)
            )
            continue
        comparisons = (
            ("QUANTITY", left.quantity, right.quantity, quantity_tolerance),
            ("ENTRY_PRICE", left.entry_price, right.entry_price, financial_tolerance),
            (
                "UNREALIZED_PNL",
                left.unrealized_pnl,
                right.unrealized_pnl,
                financial_tolerance,
            ),
        )
        for field, left_value, right_value, tolerance in comparisons:
            if _different(left_value, right_value, tolerance):
                issues.append(
                    _issue(
                        depth,
                        "POSITION",
                        key,
                        f"{field}_{plane_name}",
                        str(left_value),
                        str(right_value),
                    )
                )
    return issues


def _balance_issues(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    depth: ReconciliationDepth,
    tolerance: Decimal,
    plane_name: str,
) -> list[ReconciliationIssue]:
    issues: list[ReconciliationIssue] = []
    for key in sorted(set(expected) | set(observed)):
        left = expected.get(key)
        right = observed.get(key)
        if left is None or right is None:
            issues.append(_issue(depth, "BALANCE", key, f"MISSING_{plane_name}", left, right))
            continue
        for field in ("wallet_balance", "available_balance", "unrealized_pnl"):
            left_value = getattr(left, field)
            right_value = getattr(right, field)
            if _different(left_value, right_value, tolerance):
                issues.append(
                    _issue(
                        depth,
                        "BALANCE",
                        key,
                        f"{field.upper()}_{plane_name}",
                        str(left_value),
                        str(right_value),
                    )
                )
    return issues


def _order_issues(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    depth: ReconciliationDepth,
    quantity_tolerance: Decimal,
    plane_name: str,
) -> list[ReconciliationIssue]:
    issues: list[ReconciliationIssue] = []
    for key in sorted(set(expected) | set(observed)):
        left = expected.get(key)
        right = observed.get(key)
        if left is None or right is None:
            issues.append(
                _issue(depth, "ACTIVE_ORDER", key, f"MISSING_{plane_name}", left, right)
            )
            continue
        for field in ("position_side", "client_order_id", "active"):
            left_value = getattr(left, field)
            right_value = getattr(right, field)
            if left_value != right_value:
                issues.append(
                    _issue(
                        depth,
                        "ACTIVE_ORDER",
                        key,
                        f"{field.upper()}_{plane_name}",
                        left_value,
                        right_value,
                    )
                )
        left_status = _canonical_order_status(left.status)
        right_status = _canonical_order_status(right.status)
        if left_status is None or right_status is None or left_status != right_status:
            issues.append(
                _issue(
                    depth,
                    "ACTIVE_ORDER",
                    key,
                    f"STATUS_{plane_name}",
                    {"raw": left.status, "canonical": left_status},
                    {"raw": right.status, "canonical": right_status},
                )
            )
        if _different(
            left.cumulative_filled_quantity,
            right.cumulative_filled_quantity,
            quantity_tolerance,
        ):
            issues.append(
                _issue(
                    depth,
                    "ACTIVE_ORDER",
                    key,
                    f"CUMULATIVE_FILLED_QUANTITY_{plane_name}",
                    str(left.cumulative_filled_quantity),
                    str(right.cumulative_filled_quantity),
                )
            )
    return issues


def _fill_issues(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    depth: ReconciliationDepth,
    quantity_tolerance: Decimal,
    financial_tolerance: Decimal,
    plane_name: str,
) -> list[ReconciliationIssue]:
    issues: list[ReconciliationIssue] = []
    # Fill history is append-only locally, while Binance REST history is collected
    # through a bounded/incremental window.  Durable Memory/DB planes may therefore
    # legitimately contain older fills that are outside the current REST window.
    # Reconcile directionally: every REST-window fill must exist locally, and
    # overlapping values must agree.  Local historical supersets are not drift.
    for key in sorted(expected):
        left = expected[key]
        right = observed.get(key)
        if right is None:
            issues.append(_issue(depth, "FILL", key, f"MISSING_{plane_name}", left, right))
            continue
        for field in ("position_side", "exchange_order_id"):
            left_value = getattr(left, field)
            right_value = getattr(right, field)
            if left_value != right_value:
                issues.append(
                    _issue(
                        depth,
                        "FILL",
                        key,
                        f"{field.upper()}_{plane_name}",
                        left_value,
                        right_value,
                    )
                )
        comparisons = (
            ("QUANTITY", left.quantity, right.quantity, quantity_tolerance),
            ("PRICE", left.price, right.price, financial_tolerance),
            ("COMMISSION", left.commission, right.commission, financial_tolerance),
            ("REALIZED_PNL", left.realized_pnl, right.realized_pnl, financial_tolerance),
        )
        for field, left_value, right_value, tolerance in comparisons:
            if _different(left_value, right_value, tolerance):
                issues.append(
                    _issue(
                        depth,
                        "FILL",
                        key,
                        f"{field}_{plane_name}",
                        str(left_value),
                        str(right_value),
                    )
                )
    return issues


def _income_issues(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    depth: ReconciliationDepth,
    financial_tolerance: Decimal,
    plane_name: str,
) -> list[ReconciliationIssue]:
    issues: list[ReconciliationIssue] = []
    # Income history has the same bounded/incremental REST-window semantics as
    # fills.  Funding/commission events already persisted in Memory/DB can outlive
    # the latest REST window and must not be reported as reconciliation drift.
    # Missing local copies of current REST-window events remain fail-closed.
    for key in sorted(expected):
        left = expected[key]
        right = observed.get(key)
        if right is None:
            issues.append(_issue(depth, "INCOME", key, f"MISSING_{plane_name}", left, right))
            continue
        for field in ("income_type", "asset", "symbol", "event_time_ms"):
            left_value = getattr(left, field)
            right_value = getattr(right, field)
            if left_value != right_value:
                issues.append(
                    _issue(
                        depth,
                        "INCOME",
                        key,
                        f"{field.upper()}_{plane_name}",
                        left_value,
                        right_value,
                    )
                )
        if _different(left.amount, right.amount, financial_tolerance):
            issues.append(
                _issue(
                    depth,
                    "INCOME",
                    key,
                    f"AMOUNT_{plane_name}",
                    str(left.amount),
                    str(right.amount),
                )
            )
    return issues


def reconcile_planes(
    snapshots: RuntimeSnapshotSet,
    *,
    depth: ReconciliationDepth,
    quantity_tolerance: Decimal,
    financial_tolerance: Decimal,
    wallet_drift_tolerance: Decimal | None = None,
) -> ReconciliationOutcome:
    wallet_tolerance = (
        financial_tolerance if wallet_drift_tolerance is None else wallet_drift_tolerance
    )
    issues: list[ReconciliationIssue] = []
    for plane_name, plane in (("MEMORY", snapshots.memory), ("DB", snapshots.database)):
        issues.extend(
            _position_issues(
                snapshots.rest.positions,
                plane.positions,
                depth=depth,
                quantity_tolerance=quantity_tolerance,
                financial_tolerance=financial_tolerance,
                plane_name=plane_name,
            )
        )
        issues.extend(
            _balance_issues(
                snapshots.rest.balances,
                plane.balances,
                depth=depth,
                tolerance=wallet_tolerance,
                plane_name=plane_name,
            )
        )
        issues.extend(
            _order_issues(
                snapshots.rest.active_orders,
                plane.active_orders,
                depth=depth,
                quantity_tolerance=quantity_tolerance,
                plane_name=plane_name,
            )
        )
        if depth in {ReconciliationDepth.DEEP, ReconciliationDepth.RECOVERY}:
            issues.extend(
                _fill_issues(
                    snapshots.rest.fills,
                    plane.fills,
                    depth=depth,
                    quantity_tolerance=quantity_tolerance,
                    financial_tolerance=financial_tolerance,
                    plane_name=plane_name,
                )
            )
            issues.extend(
                _income_issues(
                    snapshots.rest.income,
                    plane.income,
                    depth=depth,
                    financial_tolerance=financial_tolerance,
                    plane_name=plane_name,
                )
            )
    return ReconciliationOutcome(depth=depth, checked_at=datetime.now(UTC), issues=tuple(issues))


def count_position_diffs(outcome: ReconciliationOutcome, *, plane: str) -> int:
    suffix = f"_{plane.upper()}"
    return sum(
        1
        for issue in outcome.unexplained
        if issue.entity_type == "POSITION" and issue.reason.endswith(suffix)
    )


def count_wallet_drift(outcome: ReconciliationOutcome) -> int:
    return sum(1 for issue in outcome.unexplained if issue.entity_type == "BALANCE")


def _diagnostic_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _diagnostic_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_diagnostic_value(item) for item in value]
    return str(value)


def reconciliation_issue_metrics(
    outcome: ReconciliationOutcome, *, sample_limit: int = 50
) -> dict[str, Any]:
    if sample_limit < 1:
        raise ValueError("sample_limit must be positive")
    unexplained = outcome.unexplained
    reason_counts = Counter(issue.reason for issue in unexplained)
    samples = [
        {
            "entity_type": issue.entity_type,
            "entity_key": issue.entity_key,
            "reason": issue.reason,
            "expected": _diagnostic_value(issue.expected),
            "observed": _diagnostic_value(issue.observed),
        }
        for issue in unexplained[:sample_limit]
    ]
    return {
        "issues": len(unexplained),
        "reason_counts": dict(sorted(reason_counts.items())),
        "issue_samples": samples,
        "issue_samples_truncated": len(unexplained) > sample_limit,
    }


def clone_plane(plane: FactPlane, *, observed_at: datetime | None = None) -> FactPlane:
    return FactPlane(
        account_id=plane.account_id,
        observed_at=observed_at or plane.observed_at,
        positions=dict(plane.positions),
        balances=dict(plane.balances),
        active_orders=dict(plane.active_orders),
        fills=dict(plane.fills),
        income=dict(plane.income),
    )
