"""Finite Decimal conversion and unit-aware validation helpers."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from freqtrade.hedge.errors import HedgeDataError


def to_decimal(value: Any, *, field: str, allow_none: bool = False) -> Decimal | None:
    if value is None:
        if allow_none:
            return None
        raise HedgeDataError(f"{field} is required.")
    if isinstance(value, bool):
        raise HedgeDataError(f"{field} must not be a boolean.")
    if isinstance(value, str) and not value.strip():
        raise HedgeDataError(f"{field} must not be empty.")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError, AttributeError) as exc:
        raise HedgeDataError(f"{field} is not a valid decimal: {value!r}.") from exc
    if not result.is_finite():
        raise HedgeDataError(f"{field} must be finite.")
    return result


def require_nonnegative(value: Any, *, field: str) -> Decimal:
    result = to_decimal(value, field=field)
    assert result is not None
    if result < 0:
        raise HedgeDataError(f"{field} must be nonnegative.")
    return result


def require_positive(value: Any, *, field: str) -> Decimal:
    result = to_decimal(value, field=field)
    assert result is not None
    if result <= 0:
        raise HedgeDataError(f"{field} must be greater than zero.")
    return result


def require_unit_interval(value: Any, *, field: str) -> Decimal:
    result = require_nonnegative(value, field=field)
    if result > 1:
        raise HedgeDataError(f"{field} must be within [0, 1].")
    return result


def require_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise HedgeDataError(f"{field} must not be a boolean.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise HedgeDataError(f"{field} must be an integer.") from exc
    if result < 0:
        raise HedgeDataError(f"{field} must be nonnegative.")
    return result
