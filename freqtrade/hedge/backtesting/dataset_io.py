from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from freqtrade.hedge.simulation.exchange import BarEvent, FundingEvent, SignalEvent

from .contracts import BacktestDataset
from .dataset import build_dataset
from .decimal_utils import json_value, to_decimal


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a JSON boolean")
    return value

def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("timestamp must be an ISO-8601 string")
    if not value.strip():
        raise ValueError("timestamp must be a non-empty ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return parsed.astimezone(UTC)


def load_dataset_csv(
    path: Path,
    *,
    timeframe: str,
    dataset_id: str | None = None,
    default_symbol: str | None = None,
) -> BacktestDataset:
    events = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "open", "high", "low", "close"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("CSV dataset missing columns: " + ", ".join(sorted(missing)))
        for row_number, row in enumerate(reader, 2):
            ts = _timestamp(row["timestamp"])
            symbol = (row.get("symbol") or default_symbol or "").strip()
            if not symbol:
                raise ValueError(f"CSV row {row_number} has no symbol")
            long_signal = to_decimal(row.get("long_signal") or "0", field="long_signal")
            short_signal = to_decimal(row.get("short_signal") or "0", field="short_signal")
            target_net_raw = row.get("target_net")
            events.append(
                SignalEvent(
                    timestamp=ts,
                    symbol=symbol,
                    long_signal=long_signal,
                    short_signal=short_signal,
                    target_net=(
                        to_decimal(target_net_raw)
                        if target_net_raw not in (None, "")
                        else None
                    ),
                    model_version=(row.get("model_version") or "csv"),
                    reason=(row.get("reason") or "CSV_IMPORT"),
                )
            )
            events.append(
                BarEvent(
                    timestamp=ts,
                    symbol=symbol,
                    open=to_decimal(row["open"], field="open"),
                    high=to_decimal(row["high"], field="high"),
                    low=to_decimal(row["low"], field="low"),
                    close=to_decimal(row["close"], field="close"),
                    volume=(
                        to_decimal(row["volume"], field="volume")
                        if row.get("volume") not in (None, "")
                        else None
                    ),
                )
            )
            if row.get("funding_rate") not in (None, ""):
                mark = row.get("mark_price") or row["close"]
                events.append(
                    FundingEvent(
                        timestamp=ts,
                        symbol=symbol,
                        rate=to_decimal(row["funding_rate"], field="funding_rate"),
                        mark_price=to_decimal(mark, field="mark_price"),
                    )
                )
    events.sort(
        key=lambda item: (
            item.timestamp,
            0 if isinstance(item, SignalEvent) else 1 if isinstance(item, FundingEvent) else 2,
        )
    )
    return build_dataset(
        events=events,
        dataset_id=dataset_id or path.stem,
        timeframe=timeframe,
        metadata={"source": str(path.resolve())},
    )


def save_dataset_json(dataset: BacktestDataset, path: Path) -> None:
    event_payload = []
    for event in dataset.events:
        if isinstance(event, SignalEvent):
            event_payload.append(
                {
                    "kind": "signal",
                    **json_value(event.__dict__ if hasattr(event, "__dict__") else {
                "timestamp": event.timestamp,
                "symbol": event.symbol,
                "long_signal": event.long_signal,
                "short_signal": event.short_signal,
                "target_net": event.target_net,
                "model_version": event.model_version,
                "reason": event.reason,
                "target_net_ratio": event.target_net_ratio,
                "confidence": event.confidence,
                "risk_scale": event.risk_scale,
                "long_exposure_scale": event.long_exposure_scale,
                "short_exposure_scale": event.short_exposure_scale,
                "allow_new_risk": event.allow_new_risk,
                "regime": event.regime,
            })})
        elif isinstance(event, BarEvent):
            event_payload.append({"kind": "bar", **json_value({
                "timestamp": event.timestamp, "symbol": event.symbol, "open": event.open,
                "high": event.high, "low": event.low, "close": event.close, "volume": event.volume,
            })})
        elif isinstance(event, FundingEvent):
            event_payload.append({"kind": "funding", **json_value({
                "timestamp": event.timestamp, "symbol": event.symbol, "rate": event.rate,
                "mark_price": event.mark_price,
            })})
        else:
            raise TypeError(f"unsupported dataset event: {type(event).__name__}")
    payload = {
        "schema_version": "hedge-backtest-dataset-v1",
        "dataset_id": dataset.dataset_id,
        "timeframe": dataset.timeframe,
        "metadata": dataset.metadata,
        "fingerprint": dataset.fingerprint,
        "events": event_payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_dataset_json(path: Path) -> BacktestDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "hedge-backtest-dataset-v1":
        raise ValueError("unsupported dataset JSON schema")
    events = []
    for item in raw["events"]:
        kind = item["kind"]
        common = {"timestamp": _timestamp(item["timestamp"]), "symbol": str(item["symbol"])}
        if kind == "signal":
            events.append(
                SignalEvent(
                    **common,
                    long_signal=to_decimal(item["long_signal"]),
                    short_signal=to_decimal(item["short_signal"]),
                    target_net=(
                        to_decimal(item["target_net"])
                        if item.get("target_net") is not None
                        else None
                    ),
                    model_version=str(item.get("model_version", "json")),
                    reason=str(item.get("reason", "JSON_IMPORT")),
                    target_net_ratio=(
                        to_decimal(item["target_net_ratio"])
                        if item.get("target_net_ratio") is not None
                        else None
                    ),
                    confidence=to_decimal(item.get("confidence", "1")),
                    risk_scale=to_decimal(item.get("risk_scale", "1")),
                    long_exposure_scale=to_decimal(item.get("long_exposure_scale", "1")),
                    short_exposure_scale=to_decimal(item.get("short_exposure_scale", "1")),
                    allow_new_risk=_boolean(
                        item.get("allow_new_risk", True), field="allow_new_risk"
                    ),
                    regime=str(item.get("regime", "UNSPECIFIED")),
                )
            )
        elif kind == "bar":
            events.append(
                BarEvent(
                    **common,
                    open=to_decimal(item["open"]), high=to_decimal(item["high"]),
                    low=to_decimal(item["low"]), close=to_decimal(item["close"]),
                    volume=(to_decimal(item["volume"]) if item.get("volume") is not None else None),
                )
            )
        elif kind == "funding":
            events.append(
                FundingEvent(
                    **common,
                    rate=to_decimal(item["rate"]),
                    mark_price=to_decimal(item["mark_price"]),
                )
            )
        else:
            raise ValueError(f"unknown dataset event kind: {kind}")
    dataset = build_dataset(
        events=events,
        dataset_id=str(raw["dataset_id"]),
        timeframe=str(raw["timeframe"]),
        metadata={str(key): str(value) for key, value in raw.get("metadata", {}).items()},
    )
    expected = raw.get("fingerprint")
    if expected and expected != dataset.fingerprint:
        raise ValueError("dataset JSON fingerprint mismatch")
    return dataset


def load_dataset(
    path: Path, *, timeframe: str | None = None, default_symbol: str | None = None
) -> BacktestDataset:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_dataset_json(path)
    if suffix == ".csv":
        if not timeframe:
            raise ValueError("CSV dataset requires timeframe")
        return load_dataset_csv(path, timeframe=timeframe, default_symbol=default_symbol)
    raise ValueError("dataset must be .json or .csv")
