"""Non-authoritative Dry-run dashboard telemetry."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
import json
import os
from pathlib import Path
from threading import RLock

ZERO = Decimal(0)
ONE = Decimal(1)


@dataclass(frozen=True, slots=True)
class StrategyTelemetry:
    long_score: Decimal = ZERO
    short_score: Decimal = ZERO
    target_net_quantity: Decimal | None = None
    target_net_ratio: Decimal | None = None
    confidence: Decimal = ONE
    risk_scale: Decimal = ONE
    long_exposure_scale: Decimal = ONE
    short_exposure_scale: Decimal = ONE
    allow_new_risk: bool = True
    regime: str = "UNSPECIFIED"
    reason: str = ""
    model_version: str = "strategy"


@dataclass(frozen=True, slots=True)
class DryRunCycleTelemetry:
    cycle_id: str
    account_id: str
    symbol: str
    timestamp: datetime
    mark_price: Decimal
    equity: Decimal
    available_balance: Decimal
    gross_notional: Decimal
    net_quantity: Decimal
    target_net_quantity: Decimal = ZERO
    net_gap_quantity: Decimal = ZERO
    long_quantity: Decimal = ZERO
    short_quantity: Decimal = ZERO
    long_target_quantity: Decimal = ZERO
    short_target_quantity: Decimal = ZERO
    long_average_price: Decimal = ZERO
    short_average_price: Decimal = ZERO
    unrealized_pnl: Decimal = ZERO
    realized_pnl: Decimal = ZERO
    funding_pnl: Decimal = ZERO
    fees: Decimal = ZERO
    ideal_order_count: int = 0
    submit_order_count: int = 0
    cancel_order_count: int = 0
    fill_count: int = 0
    active_order_count: int = 0
    risk_blocked: bool = False
    diagnostics: tuple[str, ...] = ()
    strategy: StrategyTelemetry = field(default_factory=StrategyTelemetry)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("telemetry timestamp must be timezone-aware")


class DryRunTelemetryStore:
    def __init__(self, capacity: int = 2000) -> None:
        if capacity < 10 or capacity > 100000:
            raise ValueError("telemetry capacity must be within [10,100000]")
        self.capacity = capacity
        self._items: deque[DryRunCycleTelemetry] = deque(maxlen=capacity)
        self._lock = RLock()
        self.last_error: str | None = None

    def _record_error(self, exc: Exception) -> None:
        self.last_error = f"{type(exc).__name__}: {exc}"[:512]

    def append(self, item: DryRunCycleTelemetry) -> None:
        with self._lock:
            self._items.append(item)

    def latest(self) -> DryRunCycleTelemetry | None:
        with self._lock:
            return self._items[-1] if self._items else None

    def list(self, limit: int = 200) -> tuple[DryRunCycleTelemetry, ...]:
        with self._lock:
            bounded = max(1, min(limit, self.capacity))
            return tuple(list(self._items)[-bounded:])

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.last_error = None


def _encode(value: object) -> str | list[object]:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value).__name__)


def _strategy_from_raw(raw: dict[str, object]) -> StrategyTelemetry:
    decimal_fields = {
        "long_score",
        "short_score",
        "target_net_quantity",
        "target_net_ratio",
        "confidence",
        "risk_scale",
        "long_exposure_scale",
        "short_exposure_scale",
    }
    values = {
        key: Decimal(str(value)) if key in decimal_fields and value is not None else value
        for key, value in raw.items()
    }
    return StrategyTelemetry(**values)  # type: ignore[arg-type]


def _from(raw: dict[str, object]) -> DryRunCycleTelemetry:
    strategy_raw = raw.get("strategy", {})
    strategy = _strategy_from_raw(strategy_raw if isinstance(strategy_raw, dict) else {})
    decimal_fields = {
        "mark_price", "equity", "available_balance", "gross_notional", "net_quantity",
        "target_net_quantity", "net_gap_quantity", "long_quantity", "short_quantity",
        "long_target_quantity", "short_target_quantity", "long_average_price",
        "short_average_price", "unrealized_pnl", "realized_pnl", "funding_pnl", "fees",
    }
    defaults = {
        "target_net_quantity": ZERO,
        "net_gap_quantity": ZERO,
        "long_target_quantity": ZERO,
        "short_target_quantity": ZERO,
    }
    values: dict[str, object] = {}
    for key, value in raw.items():
        if key == "strategy":
            continue
        if key == "timestamp":
            values[key] = datetime.fromisoformat(str(value))
        elif key in decimal_fields:
            values[key] = Decimal(str(value))
        elif key == "diagnostics":
            values[key] = tuple(value) if isinstance(value, Iterable) and not isinstance(value, str) else ()
        else:
            values[key] = value
    for key, value in defaults.items():
        values.setdefault(key, value)
    values["strategy"] = strategy
    return DryRunCycleTelemetry(**values)  # type: ignore[arg-type]


class JsonlDryRunTelemetryStore(DryRunTelemetryStore):
    def __init__(
        self,
        path: str | Path,
        capacity: int = 2000,
        max_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        self.path = Path(path)
        self.max_bytes = max_bytes
        super().__init__(capacity)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines:
                try:
                    super().append(_from(json.loads(line)))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
        except OSError as exc:
            self._record_error(exc)

    def append(self, item: DryRunCycleTelemetry) -> None:
        super().append(item)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                payload = json.dumps(
                    asdict(item),
                    default=_encode,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if self.path.stat().st_size > self.max_bytes:
                self._compact()
        except OSError as exc:
            self._record_error(exc)

    def _compact(self) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for item in self.list(self.capacity):
                payload = json.dumps(
                    asdict(item),
                    default=_encode,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)

    def clear(self) -> None:
        super().clear()
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError as exc:
            self._record_error(exc)
