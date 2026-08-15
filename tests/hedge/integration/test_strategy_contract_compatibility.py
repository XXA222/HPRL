from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

from freqtrade.hedge.integration.production_controller import ProductionHedgeController


def _legacy_signal() -> SimpleNamespace:
    close_time = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)
    candle = SimpleNamespace(
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("99"),
        close=Decimal("101"),
        volume=Decimal("10"),
    )
    return SimpleNamespace(
        symbol="BTC/USDT:USDT",
        timeframe="1m",
        candle_close_time=close_time,
        feature_timestamp=close_time - timedelta(seconds=1),
        long_score=Decimal("0.6"),
        short_score=Decimal("0.4"),
        target_net=None,
        model_version="legacy-test",
        reason="LEGACY_SIGNAL",
        candle=candle,
    )


def test_legacy_signal_decision_id_uses_conservative_defaults() -> None:
    legacy = _legacy_signal()
    extended = SimpleNamespace(
        **legacy.__dict__,
        target_net_ratio=None,
        confidence=Decimal("1"),
        risk_scale=Decimal("1"),
        long_exposure_scale=Decimal("1"),
        short_exposure_scale=Decimal("1"),
        allow_new_risk=True,
        regime="UNKNOWN",
        strategy_reason="LEGACY_SIGNAL",
    )

    legacy_id = ProductionHedgeController._decision_id(legacy)
    extended_id = ProductionHedgeController._decision_id(extended)

    assert legacy_id == extended_id
    assert len(legacy_id) == 64
    int(legacy_id, 16)


def test_production_controller_source_has_no_direct_optional_signal_access() -> None:
    source_path = (
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "freqtrade"
        / "hedge"
        / "integration"
        / "production_controller.py"
    )
    source = source_path.read_text(encoding="utf-8")
    for field in (
        "target_net_ratio",
        "confidence",
        "risk_scale",
        "long_exposure_scale",
        "short_exposure_scale",
        "allow_new_risk",
        "regime",
        "strategy_reason",
    ):
        assert f"signal.{field}" not in source
