#!/usr/bin/env python3
"""Durable Binance USD-M read-only observer.  It never sends trading requests."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import ccxt
import websockets
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from freqtrade.hedge.observation import HedgeObservationRecorder
from freqtrade.persistence.base import ModelBase
from freqtrade.persistence.hedge_audit import (
    HedgeAccountEvent,
    HedgeSnapshot,
    migrate_hedge_h3_schema,
)


PAIR = "ETH/USDT:USDT"


def credentials(path: Path) -> tuple[str, str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if len(line.strip()) == 64
    ]
    if len(values) != 2:
        raise RuntimeError("Credential file must contain exactly two 64-character values.")
    return values[0], values[1]


def observe_once(
    database_url: str, credential_path: Path, account_id: str
) -> dict[str, int | bool]:
    key, secret = credentials(credential_path)
    exchange = ccxt.binanceusdm(
        {
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future", "fetchOpenOrders": {"warnWithoutSymbol": False}},
        }
    )
    observed_at_ms = exchange.fetch_time()
    position_mode = exchange.fapiPrivateGetPositionSideDual()
    positions = exchange.fetch_positions(None, params={})
    balances = exchange.fetch_balance(params={})
    orders = exchange.fetch_open_orders(None, params={})
    funding = exchange.fapiPublicGetPremiumIndex({"symbol": "ETHUSDT"})
    engine = create_engine(database_url, future=True)
    ModelBase.metadata.create_all(engine)
    migrate_hedge_h3_schema(engine)
    payload = {
        "position_mode": bool(position_mode.get("dualSidePosition")),
        "position_count": len(positions),
        "open_order_count": len(orders),
        "balance_response": isinstance(balances, dict),
    }
    with Session(engine) as session:
        snapshot = (
            session.query(HedgeSnapshot)
            .filter_by(account_id=account_id, observed_at_ms=observed_at_ms)
            .one_or_none()
        )
        if snapshot is None:
            session.add(
                HedgeSnapshot(
                    account_id=account_id,
                    observed_at_ms=observed_at_ms,
                    source="binance_rest",
                    payload=payload,
                )
            )
        funding_time = int(funding.get("nextFundingTime") or observed_at_ms)
        HedgeObservationRecorder().record_funding(
            session,
            account_id=account_id,
            funding_time_ms=funding_time,
            rate=str(funding.get("lastFundingRate", "")),
        )
        session.commit()
    return payload


def stream_exchange(credential_path: Path) -> ccxt.binanceusdm:
    key, secret = credentials(credential_path)
    return ccxt.binanceusdm({"apiKey": key, "secret": secret, "enableRateLimit": True})


def persist_stream_event(database_url: str, account_id: str, event: dict[str, object]) -> bool:
    event_type = str(event.get("e", "UNKNOWN"))
    event_time = int(event.get("E", 0))
    order = event.get("o") if isinstance(event.get("o"), dict) else {}
    event_id = f"{event_type}:{event_time}:{event.get('T', '')}:{order.get('i', '')}"
    engine = create_engine(database_url, future=True)
    with Session(engine) as session:
        existing = session.query(HedgeAccountEvent).filter_by(
            account_id=account_id, event_id=event_id
        )
        if existing.one_or_none() is not None:
            return False
        session.add(
            HedgeAccountEvent(
                account_id=account_id,
                event_id=event_id,
                event_type=event_type,
                event_time_ms=event_time,
                payload=event,
            )
        )
        session.commit()
    return True


async def observe_user_stream(
    database_url: str, credential_path: Path, account_id: str, duration_seconds: int
) -> int:
    """Receive account events; every disconnect is followed by REST calibration."""
    exchange = stream_exchange(credential_path)
    response = exchange.fapiPrivatePostListenKey()
    listen_key = response.get("listenKey") if isinstance(response, dict) else None
    if not isinstance(listen_key, str) or not listen_key:
        raise RuntimeError("Binance did not return a listen key.")
    received = 0
    try:
        async with websockets.connect(
            f"wss://fstream.binance.com/ws/{listen_key}", ping_interval=20
        ) as socket:
            deadline = time.monotonic() + duration_seconds
            while time.monotonic() < deadline:
                try:
                    message = await asyncio.wait_for(
                        socket.recv(), timeout=max(1, deadline - time.monotonic())
                    )
                except TimeoutError:
                    break
                event = json.loads(message)
                if isinstance(event, dict) and persist_stream_event(
                    database_url, account_id, event
                ):
                    received += 1
    finally:
        try:
            exchange.fapiPrivateDeleteListenKey({"listenKey": listen_key})
        finally:
            observe_once(database_url, credential_path, account_id)
    return received


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, default=Path("api/binance.txt"))
    parser.add_argument(
        "--database-url", default="sqlite:////workspace/user_data/hedge_readonly.sqlite"
    )
    parser.add_argument("--account-id", default="binance-readonly")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--user-stream-seconds", type=int, default=0)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be positive")
    if args.user_stream_seconds < 0:
        raise SystemExit("--user-stream-seconds must not be negative")
    if args.user_stream_seconds:
        received = asyncio.run(
            observe_user_stream(
                args.database_url, args.credentials, args.account_id, args.user_stream_seconds
            )
        )
        print(f"readonly_user_stream=ok events={received}", flush=True)
        return
    while True:
        result = observe_once(args.database_url, args.credentials, args.account_id)
        print("readonly_observation=ok", result, flush=True)
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
