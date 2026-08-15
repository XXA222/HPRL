from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import ExchangeFactBatch, stable_fingerprint, to_primitive
from .symbol_codec import to_canonical_pair


@dataclass(frozen=True, slots=True)
class ContractEnvelope:
    """Versioned boundary payload for the frozen hedge Contracts package."""

    event_id: str
    event_type: str
    payload_version: int
    contracts_version: str
    correlation_id: str | None
    exchange_time_ms: int | None
    observed_time: str
    payload: dict[str, Any]


def _position_key(
    *, account_id: str, symbol: str, position_side: str
) -> dict[str, str]:
    return {
        "exchange": "binance",
        "account_id": account_id,
        "symbol": to_canonical_pair(symbol),
        "position_side": position_side,
    }


def _fact_payload(fact: Any) -> dict[str, Any]:
    payload = dict(to_primitive(fact))
    symbol = payload.get("symbol")
    side = payload.get("position_side")
    account_id = payload.get("account_id")
    if symbol and side and account_id:
        payload["position_key"] = _position_key(
            account_id=str(account_id),
            symbol=str(symbol),
            position_side=str(side),
        )
        payload["canonical_symbol"] = payload["position_key"]["symbol"]
    return payload


def _envelope(
    *,
    event_type: str,
    fact: Any,
    observed_time: str,
    correlation_id: str | None,
    exchange_time_ms: int | None,
) -> ContractEnvelope:
    payload = _fact_payload(fact)
    event_id = stable_fingerprint(
        {
            "event_type": event_type,
            "correlation_id": correlation_id,
            "payload": payload,
        }
    )
    return ContractEnvelope(
        event_id=event_id,
        event_type=event_type,
        payload_version=int(payload.get("event_version", 1)),
        contracts_version=str(payload.get("contract_version", "2.0")),
        correlation_id=correlation_id,
        exchange_time_ms=exchange_time_ms,
        observed_time=observed_time,
        payload=payload,
    )


def batch_to_contract_envelopes(
    batch: ExchangeFactBatch,
) -> tuple[ContractEnvelope, ...]:
    """Map one atomic ingestion batch to common versioned event envelopes."""

    observed = batch.observed_at.isoformat()
    correlation_id = batch.correlation_id or batch.reconciliation_run_id
    envelopes: list[ContractEnvelope] = []
    collections = (
        ("BalanceSnapshot", batch.balances, None),
        ("PositionSnapshot", batch.positions, "update_time_ms"),
        ("OrderSnapshot", batch.orders, "update_time_ms"),
        ("FillEvent", batch.fills, "event_time_ms"),
        ("AccountEvent", batch.account_events, "event_time_ms"),
        ("ReconciliationDiff", batch.reconciliation_diffs, None),
    )
    if batch.account_snapshot is not None:
        envelopes.append(
            _envelope(
                event_type="AccountSnapshot",
                fact=batch.account_snapshot,
                observed_time=observed,
                correlation_id=correlation_id,
                exchange_time_ms=None,
            )
        )
    for event_type, facts, time_field in collections:
        for fact in facts:
            exchange_time = None
            if time_field is not None:
                exchange_time = int(getattr(fact, time_field))
            envelopes.append(
                _envelope(
                    event_type=event_type,
                    fact=fact,
                    observed_time=observed,
                    correlation_id=correlation_id,
                    exchange_time_ms=exchange_time,
                )
            )
    return tuple(envelopes)
